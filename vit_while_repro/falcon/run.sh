#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cluster="${VIT_REPRO_FALCON_CLUSTER:-gke-tpu-train-us-central1-2-prod}"
run_id="${VIT_REPRO_RUN_ID:-vit-while-layernorm-$(date -u +%Y%m%dT%H%M%SZ)}"
state_dir="${VIT_REPRO_STATE_DIR:-$repo_root/vit_while_repro/results/$run_id}"
timeout_seconds="${VIT_REPRO_TIMEOUT_SECONDS:-3600}"
xla_flags="${VIT_REPRO_XLA_FLAGS---xla_backend_extra_options=xla_disable_while_loop_copies=true}"
libtpu_version="${VIT_REPRO_LIBTPU_VERSION:-0.0.48.dev20260910+nightly}"
mkdir -p "$state_dir"

for command in falcon jq tar sed; do
  command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 2; }
done

sed -e "s/__NAME__/${run_id}/g" \
  -e "s/__CLUSTER__/${cluster}/g" \
  -e "s|__XLA_FLAGS__|${xla_flags}|g" \
  -e "s|__LIBTPU_VERSION__|${libtpu_version}|g" \
  "$repo_root/vit_while_repro/falcon/holder.yaml" > "$state_dir/holder.yaml"

response="$(falcon workflow profile submit -f "$state_dir/holder.yaml" --output json)"
printf '%s\n' "$response" | tee "$state_dir/submit.json"
exp_id="$(printf '%s\n' "$response" | jq -er '.ids.exp_id')"
printf '%s\n' "$exp_id" > "$state_dir/exp_id"

deadline=$(( $(date +%s) + timeout_seconds ))
until falcon exp logs "$exp_id" --container task --tail 100 2>/dev/null \
  | grep -q '^VIT_WHILE_REPRO_HOLDER_READY$'; do
  status="$(falcon exp get "$exp_id" --output json | jq -r '.status // "unknown"')"
  case "$status" in failed|aborted|deleted|succeeded)
    falcon exp logs "$exp_id" --tail 500 || true
    echo "$exp_id became $status before holder readiness" >&2
    exit 1
  esac
  [ "$(date +%s)" -lt "$deadline" ] || { echo "timeout waiting for holder" >&2; exit 1; }
  sleep 15
done

bundle="$(mktemp -t vit-while-repro.XXXXXX.tar.gz)"
trap 'rm -f "$bundle"' EXIT
COPYFILE_DISABLE=1 tar -C "$repo_root" --exclude='results' -czf "$bundle" vit_while_repro
falcon exp cp "$bundle" "$exp_id:/tmp/vit-while-repro.tar.gz"
falcon exp exec "$exp_id" --rank 0 -- mkdir -p /tmp/google_support
falcon exp exec "$exp_id" --rank 0 -- tar -xzf /tmp/vit-while-repro.tar.gz -C /tmp/google_support
falcon exp exec "$exp_id" --rank 0 --no-wait -- \
  bash /tmp/google_support/vit_while_repro/falcon/remote_run.sh

set +e
falcon workflow profile collect "$exp_id" --timeout "${timeout_seconds}s" --output json \
  | tee "$state_dir/collect.json"
collect_rc=${PIPESTATUS[0]}
set -e
falcon exp logs "$exp_id" --output json --tail 2000 | tee "$state_dir/logs.json"

echo "experiment: $exp_id"
echo "results: $state_dir"
exit "$collect_rc"
