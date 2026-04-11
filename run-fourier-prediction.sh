#!/bin/bash
# Run Fourier series prediction using ROCm GPU acceleration

echo "Running GPU-accelerated Fourier series prediction..."
echo "Using ROCm PyTorch Docker container with Radeon 8060S iGPU"
echo ""

docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    -v /projects/powerball:/workspace \
    -w /workspace \
    -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    -e HIP_VISIBLE_DEVICES=0 \
    -e PYTORCH_ROCM_ARCH=gfx1151 \
    rocm/pytorch:latest \
    bash -c "pip install -q requests PyPDF2 && python3 predict_powerball_fourier.py"

echo ""
echo "Results saved to:"
echo "  - powerball_games_only.csv (filtered data)"
echo "  - powerball_predictions.csv (predictions)"
