# Minimal ViT while-body LayerNorm compiler reproduction

This removes MaxText, Flax, Tokamax, attention kernels, collectives, and custom
HLO transforms from `exp-gyw26smde7`. It retains the shape and compiler
structure around the failing operation:

```text
BF16 patch convolution (hidden=1152)
  -> rematerialized lax.scan / while
  -> pre-attention LayerNorm
  -> value_and_grad
  -> f32[1152] parameter-gradient reduce_sum
```

The Falcon manifest pins and verifies all three packages before importing JAX:

- `jax==0.11.0`
- `jaxlib==0.11.0`
- `libtpu==0.0.48.dev20260910+nightly`

It also sets the original flag verbatim:

```text
XLA_FLAGS=--xla_backend_extra_options=xla_disable_while_loop_copies=true
```

Run on one v7x `2x2x1` slice:

```bash
./vit_while_repro/falcon/run.sh
```

Override the pinned libtpu build while keeping JAX/JAXLIB at 0.11.0:

```bash
VIT_REPRO_LIBTPU_VERSION=0.0.50.dev20260928+nightly \
VIT_REPRO_RUN_ID=vit-while-layernorm-nightly-20260928 \
./vit_while_repro/falcon/run.sh
```

For a same-version control with the option omitted:

```bash
VIT_REPRO_XLA_FLAGS= \
VIT_REPRO_RUN_ID=vit-while-layernorm-control \
./vit_while_repro/falcon/run.sh
```

The wrapper prints the Falcon experiment ID and stores submit, collect, and log
envelopes under `vit_while_repro/results/<run-id>/`. A compiler abort is an
expected reproduction outcome, so the Falcon experiment will be `FAILED` with
exit code 134; inspect `logs.json` for the XLA check failure and metadata path.

See [RESULTS.md](RESULTS.md) for the confirmed failing run and same-version
control experiment.
