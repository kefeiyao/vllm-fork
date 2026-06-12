#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v icpx >/dev/null 2>&1; then
    echo "ERROR: icpx not found. Please source oneAPI setvars first."
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python interpreter not found: $PYTHON_BIN"
    exit 1
fi

if ! "$PYTHON_BIN" -c "import torch, torch.utils.cpp_extension" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_BIN cannot import torch.utils.cpp_extension"
    echo "Hint: set PYTHON_BIN to a Python with PyTorch installed."
    exit 1
fi

echo "Using Python interpreter: $PYTHON_BIN"

# Get torch include/lib paths from the container's environment
TORCH_INCLUDES=$($PYTHON_BIN -c "
import torch.utils.cpp_extension as ext
for p in ext.include_paths():
    print(f'-I{p}', end=' ')
")
TORCH_LIBS=$($PYTHON_BIN -c "
import torch.utils.cpp_extension as ext
for p in ext.library_paths():
    print(f'-L{p}', end=' ')
")
PYTHON_INCLUDE=$($PYTHON_BIN -c "import sysconfig; print('-I' + sysconfig.get_path('include'))")

echo "Building qk_rms_norm SYCL extension..."
icpx -fsycl -shared -fPIC -O3 -std=c++17 \
    -DTORCH_EXTENSION_NAME=qk_rms_norm_ops \
    $TORCH_INCLUDES \
    $PYTHON_INCLUDE \
    qk_rms_norm.cpp \
    -o qk_rms_norm_ops.so \
    $TORCH_LIBS \
    -ltorch -ltorch_cpu -lc10 -lc10_xpu -ltorch_python

echo "✓ Built: $SCRIPT_DIR/qk_rms_norm_ops.so"
ls -la qk_rms_norm_ops.so
