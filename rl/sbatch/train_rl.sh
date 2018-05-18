#!/usr/bin/env bash
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Load appropriate modules
source activate deep

# Execute Python application in unbuffered mode.
export PYTHONUNBUFFERED=1
exec $@