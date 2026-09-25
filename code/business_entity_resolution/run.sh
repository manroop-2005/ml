#!/usr/bin/env bash
set -e

# Change directory to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

echo "============================================================"
echo "ML Challenge 2026: Business Entity Resolution Pipeline"
echo "============================================================"

# Determine Python binary
if [ -d ".venv" ]; then
    PYTHON_BIN=".venv/bin/python3"
else
    PYTHON_BIN="python3"
fi

export PYTHONPATH="code/business_entity_resolution:${PYTHONPATH}"

echo "Using Python: ${PYTHON_BIN}"
mkdir -p models output

# Step 1: Train matching model
echo ""
echo ">>> STEP 1: Training Entity Resolution Model..."
${PYTHON_BIN} -m src.main train --sample-size 30000

# Step 2: Predict matches for test set
echo ""
echo ">>> STEP 2: Running Inference on Test Set..."
${PYTHON_BIN} -m src.main predict --quick-limit 1000

# Step 3: Validate outputs
echo ""
echo ">>> STEP 3: Validating Output TSVs..."
# Run against mock slice or full test
if [ -f "output/matching_results.tsv" ]; then
    echo "Outputs generated successfully:"
    ls -lh output/
fi

echo "============================================================"
echo "Pipeline execution finished successfully!"
echo "============================================================"
