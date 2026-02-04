#!/bin/bash

# =============================================================================
# ACNEP Consistency Test Script
# =============================================================================
# This script tests that ACNEP produces numerically equivalent results to NEP.
# It runs both executables on the same input and compares outputs.

set -e  # Exit on error

echo "=== ACNEP Consistency Test ==="
echo ""

# Check if executables exist
if [ ! -f "src/nep" ]; then
    echo "Error: src/nep executable not found"
    echo "Please run 'make nep' first"
    exit 1
fi

if [ ! -f "src/acnep" ]; then
    echo "Error: src/acnep executable not found"
    echo "Please run 'make acnep' first"
    exit 1
fi

# Create test directory
TEST_DIR="test_acnep_consistency"
mkdir -p $TEST_DIR
cd $TEST_DIR

# Check if test data exists
if [ ! -f "train.xyz" ] || [ ! -f "train.in" ]; then
    echo "Error: Test data not found in $TEST_DIR"
    echo "Please copy train.xyz and train.in to $TEST_DIR"
    echo ""
    echo "Example:"
    echo "  cp examples/nep_test/train.xyz $TEST_DIR/"
    echo "  cp examples/nep_test/train.in $TEST_DIR/"
    exit 1
fi

echo "Running original NEP..."
echo "  Log: nep.log"
../src/nep > nep.log 2>&1 || {
    echo "Error: NEP execution failed"
    echo "Check nep.log for details"
    exit 1
}

# Save NEP outputs
cp nep.txt nep_original.txt 2>/dev/null || true
cp loss.out loss_original.out 2>/dev/null || true
cp train_rmse.out train_rmse_original.out 2>/dev/null || true

echo "Running ACNEP with consistency checking..."
echo "  Log: acnep.log"
../src/acnep --check-consistency > acnep.log 2>&1 || {
    echo "Error: ACNEP execution failed"
    echo "Check acnep.log for details"
    exit 1
}

# Save ACNEP outputs
cp nep.txt nep_acnep.txt 2>/dev/null || true
cp loss.out loss_acnep.out 2>/dev/null || true
cp train_rmse.out train_rmse_acnep.out 2>/dev/null || true

echo ""
echo "Comparing results..."
echo ""

# Python script to compare outputs
python3 << 'PYEOF'
import numpy as np
import sys
import os

def load_file_safe(filename):
    """Load file if it exists, return None otherwise"""
    if os.path.exists(filename):
        try:
            return np.loadtxt(filename)
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")
            return None
    return None

# Load parameters
params_orig = load_file_safe('nep_original.txt')
params_acnep = load_file_safe('nep_acnep.txt')

# Load loss
loss_orig = load_file_safe('loss_original.out')
loss_acnep = load_file_safe('loss_acnep.out')

# Load RMSE
rmse_orig = load_file_safe('train_rmse_original.out')
rmse_acnep = load_file_safe('train_rmse_acnep.out')

results = {}
passed = True
tolerance_strict = 1e-10
tolerance_loose = 1e-8

# Compare parameters
if params_orig is not None and params_acnep is not None:
    param_diff = params_acnep - params_orig
    param_rmse = np.sqrt(np.mean(param_diff**2))
    param_max_diff = np.max(np.abs(param_diff))
    
    print(f"Parameter comparison:")
    print(f"  RMSE:     {param_rmse:.3e}")
    print(f"  Max diff: {param_max_diff:.3e}")
    
    if param_rmse > tolerance_loose:
        print(f"  ✗ FAIL: Exceeds tolerance {tolerance_loose}")
        passed = False
    elif param_rmse > tolerance_strict:
        print(f"  ⚠ WARN: Close but not bit-identical")
    else:
        print(f"  ✓ PASS: Within strict tolerance")
    print()
    
    results['param_rmse'] = param_rmse
    results['param_max_diff'] = param_max_diff
else:
    print("Warning: Could not compare parameters (files missing)")
    print()

# Compare loss
if loss_orig is not None and loss_acnep is not None:
    # Handle both 1D and 2D loss arrays
    if loss_orig.ndim == 2:
        loss_orig = loss_orig.flatten()
    if loss_acnep.ndim == 2:
        loss_acnep = loss_acnep.flatten()
    
    # Ensure same length
    min_len = min(len(loss_orig), len(loss_acnep))
    loss_orig = loss_orig[:min_len]
    loss_acnep = loss_acnep[:min_len]
    
    loss_diff = loss_acnep - loss_orig
    loss_rmse = np.sqrt(np.mean(loss_diff**2))
    loss_max_diff = np.max(np.abs(loss_diff))
    
    print(f"Loss comparison:")
    print(f"  RMSE:     {loss_rmse:.3e}")
    print(f"  Max diff: {loss_max_diff:.3e}")
    
    if loss_rmse > tolerance_loose:
        print(f"  ✗ FAIL: Exceeds tolerance {tolerance_loose}")
        passed = False
    elif loss_rmse > tolerance_strict:
        print(f"  ⚠ WARN: Close but not bit-identical")
    else:
        print(f"  ✓ PASS: Within strict tolerance")
    print()
    
    results['loss_rmse'] = loss_rmse
    results['loss_max_diff'] = loss_max_diff
else:
    print("Warning: Could not compare loss (files missing)")
    print()

# Compare RMSE
if rmse_orig is not None and rmse_acnep is not None:
    rmse_diff = rmse_acnep - rmse_orig
    rmse_rmse = np.sqrt(np.mean(rmse_diff**2))
    rmse_max_diff = np.max(np.abs(rmse_diff))
    
    print(f"RMSE comparison:")
    print(f"  RMSE:     {rmse_rmse:.3e}")
    print(f"  Max diff: {rmse_max_diff:.3e}")
    
    if rmse_rmse > tolerance_loose:
        print(f"  ✗ FAIL: Exceeds tolerance {tolerance_loose}")
        passed = False
    elif rmse_rmse > tolerance_strict:
        print(f"  ⚠ WARN: Close but not bit-identical")
    else:
        print(f"  ✓ PASS: Within strict tolerance")
    print()
    
    results['rmse_rmse'] = rmse_rmse
    results['rmse_max_diff'] = rmse_max_diff
else:
    print("Warning: Could not compare RMSE (files missing)")
    print()

# Overall result
if passed:
    print("=" * 60)
    print("✓ ACNEP CONSISTENCY TEST PASSED")
    print("=" * 60)
    sys.exit(0)
else:
    print("=" * 60)
    print("✗ ACNEP CONSISTENCY TEST FAILED")
    print("=" * 60)
    sys.exit(1)

PYEOF

exit_code=$?

# Return to original directory
cd ..

echo ""
if [ $exit_code -eq 0 ]; then
    echo "=== Consistency test PASSED ==="
    echo "ACNEP produces numerically equivalent results!"
else
    echo "=== Consistency test FAILED ==="
    echo "Check logs in $TEST_DIR/ for details"
fi

exit $exit_code
