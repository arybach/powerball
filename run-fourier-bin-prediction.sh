#!/bin/bash
# Docker wrapper for ROCm PyTorch Fourier Bin Analysis
# Uses rocm/pytorch:latest for AMD GPU acceleration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR=${1:-$(pwd)}

echo "======================================================================="
echo "FOURIER BIN PREDICTION WITH ROCM PYTORCH DOCKER"
echo "======================================================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check if we can run Docker
if ! docker ps &> /dev/null; then
    echo "❌ Cannot run Docker. Please check Docker service and permissions."
    exit 1
fi

echo "🚀 Using ROCm PyTorch Docker image: rocm/pytorch:latest"
echo "📦 Installing dependencies and running analysis..."

# Run analysis directly in Docker using the existing Python script
docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v "$SCRIPT_DIR":/workspace \
    -w /workspace \
    rocm/pytorch:latest \
    bash -c "
        echo '📦 Installing Python dependencies...'
        pip install pandas requests PyPDF2 --quiet
        
        echo '🔍 Checking ROCm/GPU availability...'
        python3 -c "import torch; print(f'Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}'); print(f'GPU Available: {torch.cuda.is_available()}')"
        
        echo '🧮 Running Fourier Bin Analysis...'
        python3 predict_powerball_fourier_bins.py
        
        echo '✅ Fourier Bin Analysis completed'
    " 2>&1

    # Check if predictions were generated
    if [ -f "$SCRIPT_DIR/powerball_predictions.csv" ]; then
        echo "✅ Predictions generated successfully"
        
        # Copy results to results directory if specified
        if [ "$RESULTS_DIR" != "$SCRIPT_DIR" ]; then
            cp "$SCRIPT_DIR/powerball_predictions.csv" "$RESULTS_DIR/fourier_predictions.csv" 2>/dev/null || true
            cp "$SCRIPT_DIR/fourier_bin_predictions.csv" "$RESULTS_DIR/" 2>/dev/null || true
            cp "$SCRIPT_DIR/powerball_games_only.csv" "$RESULTS_DIR/" 2>/dev/null || true
            echo "📁 Results copied to $RESULTS_DIR"
        fi
else
    echo "⚠️ No predictions file generated"
fi

echo "======================================================================="
echo "FOURIER BIN ANALYSIS COMPLETE"
echo "======================================================================="