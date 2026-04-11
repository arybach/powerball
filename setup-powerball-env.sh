#!/bin/bash
# Setup virtual environment for Powerball PDF parser

set -e

echo "Setting up Python virtual environment for Powerball parser..."

# Check if venv exists
if [ -d "venv-powerball" ]; then
    echo "Virtual environment already exists at venv-powerball/"
    echo "To recreate, run: rm -rf venv-powerball && ./setup-powerball-env.sh"
    exit 0
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv-powerball

# Activate and install dependencies
echo "Installing dependencies..."
source venv-powerball/bin/activate
pip install --upgrade pip
pip install -r requirements-powerball.txt

echo ""
echo "✓ Setup complete!"
echo ""
echo "To use the parser:"
echo "  source venv-powerball/bin/activate"
echo "  python parse_powerball_pdf.py"
echo ""
echo "Or run directly:"
echo "  ./run-powerball-parser.sh"
