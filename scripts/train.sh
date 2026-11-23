#!/bin/sh
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cd "$ROOT"

make -s
scripts/build_dataset.py --seed 42
(cd "$WORK" && "$ROOT/my_torch_generator" "$ROOT/config/best.conf" 1)
./my_torch_analyzer --train --save my_torch_network_basic.nn "$WORK/best_1.nn" dataset/splits/train.txt

python3 - my_torch_network_basic.nn my_torch_network_full.nn <<'EOF'
import json
import sys

with open(sys.argv[1]) as f:
    network = json.load(f)
network["config"]["output"] = "full"
with open(sys.argv[2], "w") as f:
    json.dump(network, f)
EOF
