# Ling3-Flash DCN Data-Parallel Scaling Regression (1 slice → 2 slices)

**Date:** 2026-09-14
**Model:** ling3-flash | **MaxText commit:** `ce38f6bca662754ac663c2b649e9c579403cb76c` | **Branch:** `codex/ling3-flash-dcn-dp-benchmark`
**Hardware:** TPU7x, 4x4x4 per slice (128 devices), 16 hosts/slice, 8 devices/host
**Software:** JAX/JAXLIB 0.11.0, libtpu 0.0.44.1, MegaScale gRPC over eth1/eth2
**Artifacts:** `gs://data_ant/benchmarks/ling3-flash-dcn-dp/dcn-ph-09111521-s1/` and `.../dcn-ph-09111521-s2/`

---

## Executive summary

Scaling from 1 to 2 slices with `dcn_data=2` costs **+0.509 s/step (+14.1%)**.

That regression is almost exactly equal to the cost of the cross-slice gradient all-reduce running **fully exposed** (not overlapped with compute) at **roughly half of one NIC's line rate**.

The root cause is visible directly in the job's `LIBTPU_INIT_ARGS`:

1. Every compute/collective overlap flag is explicitly set to `false`, so the DCN all-reduce cannot be hidden behind backward-pass compute.
2. `--megascale_grpc_enable_multi_nic=true` is absent, so MegaScale is very likely using a single NIC despite `eth1,eth2` being listed.
3. Host-side reduction is left at its default (`true`), and the gradient reduction is split into **347 separate cross-slice all-reduces**, 277 of which are latency-bound and carry only 2.4% of the bytes.

Two independent levers exist: **hide** the all-reduce, and **make it 2–3x faster**. Both are flag-level changes that can be tested in a couple of jobs.

Importantly, the sharding itself is **correct** — this is not the known GSPMD full-mesh degradation.

---

## 1. Measured facts

### 1.1 Step times

Source: `rank-0/metrics.jsonl`, field `perf/step_time_seconds`, median of steps 4–11.

| | 1 slice | 2 slices | Delta |
|---|---:|---:|---:|
| Median step time | 3.6136 s | 4.1226 s | **+0.5090 s (+14.1%)** |
| Global batch size | 256 | 512 | 2x |
| Per-device batch | 2 | 2 | unchanged |

Per-step series (steps 4–11):

| Step | 1 slice | 2 slices |
|---:|---:|---:|
| 4 | 3.6190 | 4.0914 |
| 5 | 3.6108 | 4.1154 |
| 6 | 3.6174 | 4.2769 |
| 7 | 3.6112 | 4.1297 |
| 8 | 3.6161 | 4.1129 |
| 9 | 3.6089 | 4.5496 |
| 10 | 3.6187 | 4.1512 |
| 11 | 3.6110 | 4.0744 |

**Secondary observation — jitter.** The 1-slice run is extremely stable (spread of 10 ms across 8 steps). The 2-slice run has a spread of 475 ms, with an obvious outlier at step 9. Cross-slice synchronization is introducing straggler sensitivity that did not exist before. Worth tracking separately from the mean regression.

### 1.2 Cross-slice traffic

Source: `xla_dump/module_0289.jit_train_step.cl_954665169.host_transfers.txt` (2-slice module).

The 2-slice module contains **347 distinct MegaScale `ALL_REDUCE` host transfers**. Each has a `DEVICE_TO_HOST` and a `HOST_TO_DEVICE` entry, `host_handler_name: "xla_megascale_runtime"`, and endpoint groups of the form `{1000NN, 2000NN}` — i.e. 128 cross-slice device pairs. **None of these exist in the 1-slice module.**

Total payload: **1.094 GB per device per step**, in bf16.

Size distribution:

| Payload bucket | Op count | Bytes | Share of bytes |
|---|---:|---:|---:|
| < 4 KB | 46 | 0.1 MB | 0.0% |
| 4–64 KB | 104 | 3.1 MB | 0.3% |
| 64 KB – 1 MB | 127 | 23.1 MB | 2.1% |
| 1–16 MB | 70 | 1068.3 MB | 97.6% |
| > 16 MB | 0 | 0 MB | 0.0% |

**80% of the ops carry 2.4% of the bytes.** Those 277 small ops are pure latency.

### 1.3 The sharding is correct

The large ops are uniformly `bf16[7680,8,128]` = 7,864,320 elements = 15.73 MB.

That is exactly `512 experts x 2560 x 768 / 128 FSDP shards` — one MoE expert-weight gradient shard per op.

So the DCN all-reduce operates on the **reduce-scattered 1/128 shard**, not on a replicated full gradient. This rules out the GSPMD degradation documented in the `moe_force_weight_gather_in_gmm` comment in `src/maxtext/configs/models/ling3-flash.yml`.

### 1.4 The intra-slice graph also changed

Collective op counts in `after_optimizations.txt` (`module_0287` for 1 slice, `module_0289` for 2 slices):

| Op | 1 slice | 2 slices | Delta |
|---|---:|---:|---:|
| `reduce-scatter` | 45 | 186 | **+141 (+313%)** |
| `all-gather` | 424 | 711 | **+287 (+68%)** |
| `collective-permute-start` | 1320 | 1936 | **+616 (+47%)** |
| `all-reduce` | 121 | 145 | +24 |
| `all-to-all` | 12 | 12 | 0 |

Enabling `dcn_data=2` did not simply append DCN all-reduces to an otherwise identical graph. XLA re-decomposed the combined 256-device gradient reduction into:

```
reduce-scatter (ICI)  →  all-reduce (DCN, on the 1/128 shard)  →  all-gather (ICI)
```

This is almost certainly driven by the already-set `--xla_tpu_prefer_async_allgather_to_allreduce=true`. The decomposition is correct and desirable — it is why the DCN payload is only 1.094 GB/device instead of the full gradient — but it adds several hundred extra ICI collectives that are not free. ICI at 128 devices is fast, so budget this at tens of ms rather than hundreds. The practical implication: **do not assume the 1-slice and 2-slice compute phases are identical.**

---

## 2. The bandwidth roofline

Per host, per step: 8 devices x 1.094 GB = **8.75 GB egress + 8.75 GB ingress**.

| Scenario | DCN time per step |
|---|---:|
| **Observed regression** | **0.509 s** |
| **Implied effective bandwidth** | **8.75 GB / 0.509 s = 17.2 GB/s = ~137 Gbps** |
| At 1 x 200 Gbps NIC line rate | 350 ms |
| At 2 x 200 Gbps NIC line rate | 175 ms |
| At measured best DCN all-reduce (236 Gbps, from `dcn/RESULTS.md`) | 297 ms |
| Perfectly overlapped with the 3.6 s of compute | ~0 ms |

The observed ~137 Gbps is about 69% of a **single** 200 Gbps NIC. That is exactly the signature of a single-NIC, host-reduction gRPC all-reduce — and it is consistent with the multi-NIC flag being missing.

---

## 3. Root cause: the job's `LIBTPU_INIT_ARGS`

From `rank-0/train.log`, line 21 (identical in both the 1-slice and 2-slice runs):

```
--xla_tpu_enable_async_collective_fusion=false
--xla_tpu_enable_async_collective_fusion_multiple_steps=false
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false
--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false
--xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false
--xla_enable_async_all_gather=true
--xla_enable_async_collective_permute=true
--xla_tpu_prefer_async_allgather_to_allreduce=true
--xla_tpu_enable_all_experimental_scheduler_features=true
--xla_tpu_overlap_compute_collective_tc=false
--xla_tpu_use_tc_device_shape_on_sc=true
--xla_sc_enable_instruction_fusion=false
--xla_sc_disjoint_spmem=false
--xla_sc_disable_megacore_partitioning=true
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true
--xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true
--xla_tpu_enable_sparse_core_reduce_scatter_padding=true
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true
--xla_tpu_scoped_vmem_limit_kib=65536
--xla_tpu_dvfs_p_state=7
--xla_tpu_data_parallel_opt_different_sized_ops=true
--xla_tpu_enable_data_parallel_all_reduce_opt=true
--megascale_coordinator_address=dcn-ph-09111521-s2-worker-0-0.dcn-ph-09111521-s2:8081
--megascale_slice_id=0
--megascale_num_slices=2
--megascale_transport_type=grpc
--megascale_port=8081
--megascale_use_insecure_grpc
--megascale_grpc_interface_prefixes=eth1,eth2,lo
```

### Problem A — All compute/collective overlap is disabled

`--xla_tpu_enable_async_collective_fusion*=false` and `--xla_tpu_overlap_compute_collective_tc=false` mean XLA will not schedule the (very long) DCN all-reduce concurrently with TensorCore work.

**Design rationale (SparseCore offload)**: This configuration was adopted because the training pipeline relies on SparseCore collective offload for intra-slice collectives:
```
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true
```
Historically, SparseCore offload compiler passes conflict with async collective fusion passes, so `--xla_tpu_enable_async_collective_fusion=false` and `--xla_tpu_overlap_compute_collective_tc=false` were intentionally set to `false`. On **1 slice**, this is optimal because all collectives are intra-slice ICI (AG/RS) and handled smoothly by SparseCore.

**The 2-slice architectural conflict**: SparseCore **only offloads intra-slice ICI collectives**. It cannot execute cross-slice DCN collectives (which traverse PCIe, Host DRAM, and the MegaScale gRPC network stack).
Because TensorCore compute/collective overlap is disabled globally, XLA compiles the DCN all-reduces as strictly synchronous operations on TensorCore (evidenced by 0 `all-reduce-start` ops in HLO).
Consequently, on 2 slices, the DCN all-reduce enjoys neither SparseCore offload nor TensorCore overlap — it runs completely exposed, directly causing the ~500 ms regression.

**Key question**: Can we decouple DCN overlap from ICI SparseCore offload? (e.g. keeping SC offload for intra-slice AG/RS while allowing DCN all-reduce overlap via `--xla_tpu_dcn_max_overlap_estimation=32` or selective async AR).

**Verification.** Two independent checks confirm this is real and not a misread:

1. The banner is emitted by each job's own launcher, immediately below its own `Output Dir: .../dcn-ph-09111521-s2` line — so the flags shown are the ones the 2-slice job ran with. Diffing the two flag strings with the `--megascale_*` arguments stripped shows they are **byte-identical**; the only differences are `megascale_num_slices` (1 vs 2) and the coordinator address.

2. More conclusively, the flags demonstrably took effect in the compiled module. Counting synchronous versus async (`-start`) collective forms in the 2-slice `after_optimizations.txt`:

| Collective | Sync form | Async `-start` form |
|---|---:|---:|
| `all-reduce` | 145 | **0** |
| `all-gather` | 711 | 0 |
| `reduce-scatter` | 186 | 0 |
| `collective-permute` | 0 | 1936 |

**There is not a single `all-reduce-start` in the 2-slice module.** Every all-reduce compiled to the synchronous form. The only collective that compiled to an async form is `collective-permute`, which is exactly the one async flag set to `true` (`--xla_enable_async_collective_permute=true`). That is a clean positive control: the flag string is being honored end-to-end.

One honest caveat: a synchronous HLO all-reduce does not strictly prove zero overlap at runtime, because the MegaScale runtime may internally pipeline the device-to-host, network, and host-to-device stages of a single op. What it does prove is that **the XLA scheduler placed no compute in parallel with these all-reduces**, which is the dominant effect and the thing E2/E3 would change.

### Problem B — Transport bandwidth and Multi-NIC configuration

`--megascale_grpc_interface_prefixes=eth1,eth2,lo` is set, but `--megascale_grpc_enable_multi_nic=true` is absent, as is `--megascale_grpc_dynamic_lb=true`.

**Team feedback on default behavior**:
1. Prior benchmarks (e.g., aolemila's scripts in `google_support`) did not explicitly pass `enable_multi_nic=true` yet observed high bandwidth.
2. In libtpu 0.0.44, `--megascale_grpc_enable_multi_nic` may already default to `true`.

**What the measured numbers tell us**:
- Observed effective bandwidth is **~137 Gbps** (17.2 GB/s).
- Line rate of dual 200 Gbps NICs is 400 Gbps, and isolated microbenchmarks on the same hardware achieve **~236 Gbps** (`dcn/RESULTS.md`).
- **If multi-NIC is already default `true`**: The ~137 Gbps bottleneck is NOT single-NIC serialization, but rather:
  1. **Op fragmentation (H3)**: 277 small (<1 MB) latency-bound ops saturating gRPC/load-balancer queues.
  2. **Host-reduction memory bandwidth (H4)**: Host CPU summation of 8.75 GB/step on DRAM.
  3. **Missing gRPC tuning flags**: Flags established in `dcn/RESULTS.md` (`chaotic_good`, `event_engine_allocator`, disabling TCP recv zerocopy, RPC coalescing) that prevent buffer stagnation.
- **Verification via E14**: Checking `/proc/net/dev` RX/TX byte deltas on `eth1` and `eth2` across a step window will definitively verify multi-NIC status in 5 minutes without code changes. If both interfaces are active, H2 is ruled out and the focus shifts entirely to H1 (overlap), H3 (op fusion), and H4 (host reduction).

Additionally, `lo` in the prefix list is risky — if the load balancer ever selects loopback for a cross-host peer it will misbehave.

None of the known-good gRPC flags previously established in `dcn/RESULTS.md` ("Required Borg gRPC Flags") are present:

```
--grpc_filter_insecure_rpc=false
--psp_grpc_clusters_enabled=
--megascale_grpc_use_chaotic_good=true
--grpc_enable_tcp_recv_zerocopy=false
--megascale_grpc_use_event_engine_allocator=true
--grpc_enable_rpc_receive_coalescing=true
--megascale_grpc_enable_multi_nic=true
--megascale_grpc_dynamic_lb=true
```

### Problem C — Host reduction plus 347 uncombined ops

`--xla_tpu_use_megascale_host_reduction` is unset, so it defaults to `true`. This is consistent with the 347 host-transfer entries found in the HLO. Every op costs PCIe device-to-host, a gRPC round trip, CPU-side summation of 8.75 GB per host, and host-to-device.

On top of that, 277 of the 347 ops carry only 2.4% of the bytes. At even 0.5 ms of fixed cost each, that is roughly 140 ms/step of overhead moving essentially no data.

---

## 4. Ranked hypotheses

| # | Hypothesis | Evidence | Est. cost | Effort to test |
|---|---|---|---:|---|
| H1 | DCN all-reduce is fully exposed (no overlap) | Overlap flags all `false`; regression ≈ full AR time | 300–500 ms | Low (flags) |
| H2 | Single-NIC transport | `enable_multi_nic` absent; 137 Gbps ≈ 69% of one NIC | 150–300 ms | Low (flags) |
| H3 | 277 tiny, uncombined all-reduces | 80% of ops carry 2.4% of bytes | 50–150 ms | Low (flags) |
| H4 | Host-reduction path (PCIe + CPU sum of 8.75 GB/host) | Default `true`; 347 host transfers | 0–150 ms | Low (flags) |
| H5 | `scan_layers=true` prevents per-layer AR emission | Monolithic backward while-loop | Blocks H1 fix | High (recompile) |
| H6 | Extra ICI work from RS→AR→AG decomposition | RS +313%, AG +68% | tens of ms | Medium |

**Separate issue — profiling overhead.** The XPlane profile shows 4.8798 s for the 2-slice `jit_train_step` versus a 4.1226 s steady state (+18%), while the 1-slice profile has almost no overhead. With 347 DCN ops x 2 transfers x 32 hosts, the MegaScale/gRPC trace event volume is enormous and exists only in the 2-slice run. **Treat the profiled step time as unusable for regression accounting** and continue using the `metrics.jsonl` medians.

---

## 5. Experiment plan

All experiments are 2-slice A/B tests against the 4.1226 s baseline. Always use the median of steps 4–11 from `metrics.jsonl`, never the profiled step.

### Tier 1 — Flag-only, one job each (~1 hour total)

| # | Change | Tests | Expected |
|---|---|---|---|
| E1 | `--megascale_grpc_enable_multi_nic=true --megascale_grpc_dynamic_lb=true`, remove `lo` from `interface_prefixes` | H2 | −150 to −300 ms |
| E2 | Set all four `async_collective_fusion` flags to `true` plus `--xla_tpu_overlap_compute_collective_tc=true` | H1 | −200 to −500 ms |
| E3 | Add `--xla_tpu_dcn_max_overlap_estimation=32` (already used in the v5p gpt3_175b config) | H1 | −100 ms+, stacks with E2 |
| E4 | `--xla_tpu_use_megascale_host_reduction=false` | H4 | −0 to −150 ms |
| E5 | Add the full known-good gRPC flag set from `dcn/RESULTS.md` | H2 | stacks with E1 |
| E6 | `--xla_tpu_all_reduce_combine_threshold_bytes=1073741824` | H3 | −50 to −150 ms |

Notes:
- Run **E1 + E5 together** first. These are pure transport changes with no compiler impact, so they are low risk.
- Then run **E2 + E3** on top. Re-A/B these on 1 slice as well — they were presumably disabled for a reason, and you do not want to trade ICI performance for DCN performance.
- For E6, verify the effect by re-dumping the HLO and re-counting `host_transfers` entries. The count should drop well below 347.
- For E4, prior data is mixed: Borg testing showed it neutral, but the customer's GKE testing showed 162 → 250 Gbps. This run is GKE-style, so it is worth one job.

### Tier 2 — Establish the roofline (run in parallel; cheap)

**E7 — Microbenchmark the exact shape.** The existing `dcn/distributed_runner.py` harness already performs 2-slice all-reduce sweeps. Run it on the same 16-host x 8-device/host topology with a bf16 payload of 1.094 GB/device, **solo** (no `ppermute` beforehand — the poisoning effect is documented in `RESULTS.md`). This gives the hard floor for how fast this all-reduce can possibly be, and immediately partitions the 0.509 s into "bandwidth" versus "scheduling".

**E8 — Op-count sensitivity.** Same harness, but issue 347 all-reduces summing to 1.094 GB using the measured size distribution (70 x 15.7 MB plus 277 small) versus a single 1.094 GB all-reduce. The delta is the pure per-op overhead and directly sizes H3.

**E15 — Isolate the extra ICI work.** Run a **1-slice** job with `--xla_tpu_prefer_async_allgather_to_allreduce=false` versus `=true`, and diff both step time and collective counts. If the flag alone moves 1-slice step time materially, part of the apparent "DCN regression" is really an ICI schedule change and should be attributed separately.

### Tier 3 — Model and framework changes

| # | Change | Rationale |
|---|---|---|
| E9 | `gradient_accumulation_steps=2` at 2 slices | The DCN all-reduce fires once per N micro-batches, amortizing the fixed ~0.5 s. Good "does this scale at all" sanity check and a legitimate production lever if the token budget can grow. |
| E10 | `use_qk_clip: false` (temporarily) | MuonClip requires a global per-head max-logit reduction — a likely source of many of the 60 tiny F32 cross-slice all-reduces. Measure the delta, then consider making the reduction cross-slice-lazy (every K steps). |
| E11 | `routed_bias_update_rate: 0`, and reduce per-step `Router/bias_mean/layer_*` metric logging | The aux-loss-free bias update needs global expert counts (the S32 `[1,4,128]` all-reduces), and `metrics.jsonl` shows 42 per-layer router stats being pulled every step. Cheap on 1 slice; a cross-slice sync on 2. |
| E12 | `scan_layers: false` (diagnostic only) | With scan, the backward pass is a single while-loop, so XLA cannot emit per-layer gradient all-reduces that overlap with the remaining layers' backward. Unrolling would let E2/E3 actually find overlap opportunities. Expensive to compile — run once to establish the ceiling. |

### Tier 4 — Measurement hygiene

**E13 — Re-profile with reduced host tracing.** Use `host_tracer_level=1`, or profile only rank 0 with MegaScale tracing suppressed, and confirm the profiled step converges toward 4.12 s. This answers whether the extra 18% is real or an artifact.

**E14 — Per-NIC byte counters.** Capture `/proc/net/dev` deltas across a step window on a few hosts. This is the definitive check for whether `eth1` and `eth2` are both carrying traffic. The same technique is already used in `dcn/RESULTS.md`.

---

## 6. Recommended order

1. **E14** — roughly 5 minutes, no new job required if you can exec into a running pod. Confirm whether only one of `eth1`/`eth2` is moving bytes. If so, H2 is confirmed and E1 is a ~2x win on the DCN leg.
2. **E1 + E5** in one job.
3. **E2 + E3** in a second job, plus a 1-slice control with the same flags.
4. **E7** in parallel, to nail the roofline.

**Success criteria.** If E1 + E2 + E3 brings the 2-slice step time to roughly **3.7–3.8 s**, that is about 97% data-parallel scaling efficiency and the investigation is complete. If it stalls near 4.0 s, the remaining cost is the 277-op latency tail, and the next moves are E6, E8, E10, and E11.

---

## Appendix: How these numbers were derived

| Quantity | Source | Method |
|---|---|---|
| Step times | `rank-{0}/metrics.jsonl` | Median of `perf/step_time_seconds` for steps 4–11 |
| Hosts per slice | GCS listing of `rank-*` prefixes | 16 for s1, 32 for s2 |
| Devices per slice | `rank-0/runtime.json` | `"slices": {"0": 128, "1": 128}` |
| Devices per host | derived | 128 devices / 16 hosts = 8 |
| Cross-slice payload | `xla_dump/module_0289...host_transfers.txt` | Parsed each `host_transfers` block; summed `element_type` x product of `dimensions` for all `DEVICE_TO_HOST` entries with `transfer_type: ALL_REDUCE` |
| Op count (347) | same | Count of unique instruction names in `key:` fields |
| Collective counts | `xla_dump/...after_optimizations.txt` | `grep -o -E "(all-gather\|reduce-scatter\|all-reduce\|collective-permute-start\|all-to-all)\("` piped to `sort \| uniq -c` |
| Launch flags | `rank-0/train.log` line 21 | Literal `LIBTPU_INIT_ARGS` echo |
| Reference DCN bandwidths | `dcn/RESULTS.md` | Prior 2-slice microbenchmark campaign |
