#!/usr/bin/env bash
# 30-second Polaris TDX smoke test — no GPU, reuses saved attestation from eval box.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
[ -f .env.eval ] && set -a && source .env.eval && set +a
export SSH_KEY="${SSH_KEY:-$HOME/.ssh/speedy}"
HOST="${EVAL_SSH_HOST:?}"; PORT="${EVAL_SSH_PORT:-22}"
ATTEST="${POLARIS_SMOKE_ATTEST:-/tmp/polaris_attestation.json}"

echo "[$(date -u +%FT%TZ)] Polaris TDX smoke test"
ssh -i "$SSH_KEY" -o BatchMode=yes -p "$PORT" "root@${HOST}" \
  cat /tmp/polaris_attestation.json > "$ATTEST"
export POLARIS_SMOKE_ATTEST="$ATTEST"

python3 <<'PY'
import json, hashlib, os, sys
sys.path.insert(0, os.getcwd())
a = json.load(open(os.environ.get("POLARIS_SMOKE_ATTEST", "/tmp/polaris_attestation.json")))
from eval.polaris.client import PolarisClient
from eval.polaris.receipt import build_polaris_receipt
pub = next(l.strip() for l in open("eval/polaris/sparkinfer_eval.pub")
           if l.strip() and not l.startswith("#"))
nonce = hashlib.sha256((
    a.get("code", {}).get("commit", "") +
    a.get("references", {}).get("model_sha256", "") +
    a.get("references", {}).get("eval_seed", "")
).encode()).hexdigest()[:64]
print(f">> endpoint: {os.environ.get('POLARIS_API_BASE', 'https://polaris.computer')}/v1/attest")
resp = PolarisClient(os.environ["POLARIS_API_KEY"]).attest_scoring(
    a.get("measurements", {}), nonce, pub)
receipt = build_polaris_receipt(resp, a)
intel = resp.get("tee_attestation", {}).get("verification", {}).get("intel_verified")
if intel is None:
    intel = resp.get("verification", {}).get("intel_verified")
out = "/tmp/polaris_smoke_receipt.json"
json.dump(receipt, open(out, "w"), indent=2)
print(f">> Polaris OK: intel_verified={intel} receipt_id={receipt['receipt_id'][:16]}")
print(f">> receipt: {out}")
if not intel:
    sys.exit(1)
PY
