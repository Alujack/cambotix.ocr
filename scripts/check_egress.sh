#!/usr/bin/env bash
# Run a command under strace and fail if it opens any IPv4/IPv6 connection or
# sends a DNS query. Python-level socket mocks cannot see native libraries
# (e.g. onnxruntime's C++ telemetry), so offline guarantees are checked here
# at the system-call level. Linux only; requires strace.
#
#   scripts/check_egress.sh python benchmarks/evaluate.py --dataset ... --model ...
set -euo pipefail
trace="$(mktemp)"
trap 'rm -f "$trace"' EXIT
strace -f -qq -e trace=connect,sendto,sendmmsg -o "$trace" "$@" >/dev/null
if grep -E 'sa_family=AF_INET6?|htons\(53\)' "$trace"; then
  echo "FAIL: outbound network activity detected (see lines above)" >&2
  exit 1
fi
echo "OK: no outbound network activity"
