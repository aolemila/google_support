#!/usr/bin/env bash
set -euo pipefail

status_file=/tmp/vit-repro-status
done_file=/tmp/vit-repro-done
log_file=/tmp/vit-repro.log
metrics_file=/tmp/vit-repro-metrics.jsonl
rm -f "$status_file" "$done_file" "$log_file" "$metrics_file"
exec > >(tee "$log_file") 2>&1

finish() {
  rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$status_file"
  touch "$done_file"
  exit "$rc"
}
trap finish EXIT

test -f /tmp/vit-repro-ready
/tmp/vit-while-repro-venv/bin/python \
  /tmp/google_support/vit_while_repro/repro.py \
  --output "$metrics_file"
