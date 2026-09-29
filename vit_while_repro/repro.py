#!/usr/bin/env python3
"""Minimal TPU compiler probe for a ViT scan-body LayerNorm reduction.

The original failure is in the backward pass of a rematerialized ViT block:

  vision_encoder/vit_blocks/while/body/.../preattn_layernorm/reduce_sum

This probe deliberately keeps only the compiler-relevant structure: a BF16
patch convolution, a rematerialized lax.scan (which lowers to while), and a
pre-attention LayerNorm whose parameter gradient reduces to f32[hidden].
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
from jax import lax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--hidden", type=int, default=1152)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def layer_norm(x: jax.Array, scale: jax.Array, bias: jax.Array) -> jax.Array:
    # Spell this out so the backward pass contains the same f32[hidden]
    # parameter-gradient reduction seen in the original failure.
    with jax.named_scope("preattn_layernorm"):
        xf = x.astype(jnp.float32)
        mean = jnp.mean(xf, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
        normalized = (xf - mean) * lax.rsqrt(variance + 1e-6)
        return (normalized * scale + bias).astype(jnp.bfloat16)


def make_loss(args: argparse.Namespace):
    def vit_block(tokens: jax.Array, layer_params: tuple[jax.Array, ...]):
        scale, bias, proj = layer_params
        with jax.named_scope("vit_block"):
            normalized = layer_norm(tokens, scale, bias)
            # A small residual projection is enough to retain a realistic
            # multi-output fusion opportunity without pulling in attention.
            projected = jnp.einsum("bsh,hk->bsk", normalized, proj)
            return tokens + jnp.tanh(projected).astype(tokens.dtype), None

    remat_block = jax.checkpoint(vit_block)

    def loss(params: dict[str, jax.Array], images: jax.Array) -> jax.Array:
        with jax.named_scope("vision_encoder"):
            patches = lax.conv_general_dilated(
                images,
                params["patch_kernel"],
                window_strides=(args.patch_size, args.patch_size),
                padding="VALID",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            tokens = patches.reshape(args.batch, -1, args.hidden)
            with jax.named_scope("vit_blocks"):
                tokens, _ = lax.scan(
                    remat_block,
                    tokens,
                    (params["scale"], params["bias"], params["proj"]),
                )
        return jnp.mean(jnp.square(tokens.astype(jnp.float32)))

    return loss


def main() -> None:
    args = parse_args()
    key = jax.random.key(0)
    image_key, kernel_key, proj_key = jax.random.split(key, 3)
    tokens_per_side = args.image_size // args.patch_size
    if tokens_per_side * args.patch_size != args.image_size:
        raise ValueError("image-size must be divisible by patch-size")

    images = jax.random.normal(
        image_key, (args.batch, args.image_size, args.image_size, 3), jnp.bfloat16
    )
    params = {
        "patch_kernel": jax.random.normal(
            kernel_key,
            (args.patch_size, args.patch_size, 3, args.hidden),
            jnp.bfloat16,
        ) * jnp.bfloat16(0.02),
        "scale": jnp.ones((args.layers, args.hidden), jnp.float32),
        "bias": jnp.zeros((args.layers, args.hidden), jnp.float32),
        "proj": jax.random.normal(
            proj_key, (args.layers, args.hidden, args.hidden), jnp.bfloat16
        ) * jnp.bfloat16(args.hidden**-0.5),
    }

    loss = make_loss(args)
    compiled_step = jax.jit(jax.value_and_grad(loss))
    started = time.monotonic()
    lowered = compiled_step.lower(params, images)
    lowered_at = time.monotonic()
    print("LOWERING_SUCCEEDED", flush=True)
    executable = lowered.compile()
    compiled_at = time.monotonic()
    print("COMPILATION_SUCCEEDED", flush=True)
    value, grads = executable(params, images)
    jax.block_until_ready((value, grads))
    finished = time.monotonic()

    result = {
        "variant": "conv_scan_remat_layernorm_grad",
        "batch": args.batch,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "tokens": tokens_per_side**2,
        "hidden": args.hidden,
        "layers": args.layers,
        "dtype": "bfloat16",
        "lower_time_s": lowered_at - started,
        "compile_time_s": compiled_at - lowered_at,
        "execute_time_s": finished - compiled_at,
        "loss": float(value),
    }
    print("VIT_WHILE_REPRO_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
