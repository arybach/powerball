#!/bin/bash
# Run Powerball PDF parser in virtual environment

# Activate virtual environment
source venv-powerball/bin/activate

# Run parser
python3 parse_powerball_pdf.py

# Deactivate is automatic when script ends
