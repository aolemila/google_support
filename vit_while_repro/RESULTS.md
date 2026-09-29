# Reproduction results

Date: 2026-09-29

Both runs used one v7x `2x2x1` slice on
`gke-tpu-train-us-central1-2-prod` and verified these installed versions
before importing JAX:

```text
jax=0.11.0
jaxlib=0.11.0
libtpu=0.0.48.dev20260910+nightly
```

## Reproduction

- Experiment: `exp-ja0xo9r900`
- Artifact: `art-o98dzdwvnu` (`FAILED`, because the compiler aborted)
- `XLA_FLAGS=--xla_backend_extra_options=xla_disable_while_loop_copies=true`
- Result: lowering succeeded; TPU compilation aborted with exit code 134.

The minimized failing instruction is:

```text
%fusion.119 = bf16[2,256,1152] fusion(%while.15), kind=kLoop,
metadata={op_name="jit(loss)/transpose(jvp())/convert_element_type"}
```

The check and stack match the original experiment:

```text
RemoveInstruction(instruction_to_merge) ... IsSafelyRemovable(instruction)
xla::HloFusionInstruction::MergeFusionInstructionIntoMultiOutput()
xla::MultiOutputFusion::Fuse()
xla::jellyfish::TpuMultiOutputFusion::Fuse()
```

The metadata moved from the original LayerNorm `reduce_sum` to a conversion of
the while result after minimization. This indicates that the named reduction
was where the invalid multi-output fusion became observable in the full graph,
not a requirement for triggering the compiler invariant violation.

## Same-version control

- Experiment: `exp-gusqan9dhc`
- Artifact: `art-6r79j137jd` (`SUCCEEDED`)
- `XLA_FLAGS` omitted
- Result: compiled and executed successfully.
- Compile time: 2.586 seconds
- Loss: 1.021391749382019

The sole intentional difference between the two runs was the
`xla_disable_while_loop_copies` backend option. MaxText, Flax, Tokamax, custom
HLO transforms, collectives, and 512-device AOT compilation are not required
to reproduce the crash.

## Latest nightly validation

Validation date: 2026-09-29 12:28 CST (04:28 UTC)

Based on [primatrix/maxtext#1192](https://github.com/primatrix/maxtext/issues/1192),
nightly wheels are published to the
[official libtpu wheel index](https://storage.googleapis.com/libtpu-wheels/index.html)
after the daily build passes. At validation time, the newest Python 3.12 wheel
in that index was:

```text
libtpu-0.0.50.dev20260928+nightly-cp312-cp312-manylinux_2_31_x86_64.whl
```

The probe kept the original JAX versions and problematic backend option:

```text
jax=0.11.0
jaxlib=0.11.0
libtpu=0.0.50.dev20260928+nightly
XLA_FLAGS=--xla_backend_extra_options=xla_disable_while_loop_copies=true
```

- Experiment: `exp-eurrffkslp`
- Artifact: `art-4jthsrqs1f` (`FAILED`, because the compiler aborted)
- Hardware: one v7x `2x2x1` slice
- Result: lowering succeeded; TPU compilation aborted with exit code 134.

The failing instruction remained the same minimized while-result fusion:

```text
%fusion.119 = bf16[2,256,1152] fusion(%while.15), kind=kLoop,
metadata={op_name="jit(loss)/transpose(jvp())/convert_element_type"}
```

The stack again entered
`HloFusionInstruction::MergeFusionInstructionIntoMultiOutput`,
`MultiOutputFusion::Fuse`, and `TpuMultiOutputFusion::Fuse`, then failed the
same `RemoveInstruction` / `IsSafelyRemovable` check. Therefore the issue is
still reproducible in the latest available 2026-09-28 nightly; upgrading from
the 2026-09-10 nightly does not make this workload compile successfully.
