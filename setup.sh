#!/bin/bash
# Setup script for VU_RG project
# Clones PFNs and installs all dependencies

set -e

echo "=== Installing Python dependencies ==="
python3 -m pip install -r requirements.txt

echo ""
echo "=== Cloning and installing PFNs ==="
if [ -d "PFNs" ]; then
    echo "PFNs directory already exists, pulling latest changes..."
    cd PFNs && git pull && cd ..
else
    git clone https://github.com/automl/PFNs.git
fi
cd PFNs && python3 -m pip install -e . -q && cd ..

echo ""
echo "=== Setup complete ==="
echo "Run notebooks from the project root: jupyter notebook"
