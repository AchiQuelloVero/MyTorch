#!/bin/sh

python3 --version || exit 84
if ! python3 -c "import numpy; print('numpy', numpy.__version__)" 2>/dev/null; then
    echo "numpy is not available" >&2
    exit 84
fi
