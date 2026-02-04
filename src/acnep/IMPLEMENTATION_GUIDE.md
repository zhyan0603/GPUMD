# ACNEP Implementation Guide

## Overview

This document provides detailed implementation steps for the ACNEP optimizations. The goal is to achieve 4-10x speedup while maintaining numerical equivalence (within 1e-10 tolerance) with the original NEP code.

## Project Structure

```
src/acnep/
├── README_ACNEP.md              # High-level optimization overview
├── IMPLEMENTATION_GUIDE.md      # This file
├── acnep_optimization.cuh       # Optimization infrastructure (data structures, helpers)
├── acnep.cu                     # Modified NEP potential (uses cached geometry)
├── acfitness.cu                 # Modified fitness evaluation (batching)
├── dataset.cu/cuh               # Modified dataset (pre-computation)
├── main_acnep.cu                # Entry point
├── snes.cu                      # SNES optimizer (add consistency checking)
└── ... (other files unchanged)
```

## Implementation Phases

### Phase 1: Infrastructure Setup ✅ COMPLETE

- [x] Created `acnep` directory from `main_nep`
- [x] Renamed key files (nep.cu → acnep.cu, fitness.cu → acfitness.cu)
- [x] Updated Makefile to build `acnep` executable
- [x] Created `acnep_optimization.cuh` with data structures
- [x] Added `PrecomputedGeometry` to Dataset class

### Phase 2: Pre-computation Framework 🔄 IN PROGRESS

#### Step 2.1: Integrate Pre-computation into Dataset::construct()

**File:** `src/acnep/dataset.cu`

**Location:** At end of `Dataset::construct()` function

**Code to add:**
```cpp
void Dataset::construct(Parameters& para, std::vector<Structure>& structures, int n1, int n2, int device_id)
{
  // ... existing construct code ...
  
  // ACNEP: Pre-compute geometry if optimization is enabled
  if (para.acnep_opts.use_precomputed_geometry) {
    precompute_geometry(para);
  }
}
```

#### Step 2.2: Implement Optimized Neighbor List Kernel

**File:** `src/acnep/acnep.cu` (add before existing `gpu_find_neighbor_list`)

**Implementation:**

```cpp
// ============================================================================
// ACNEP: Optimized neighbor list with cell-linked list (optional)
// ============================================================================

__global__ void gpu_find_neighbor_list_optimized(
  const NEP::ParaMB paramb,
  const int N,
  const int* Na,
  const int* Na_sum,
  const int* g_type,
  const float* __restrict__ g_box,
  const float* __restrict__ g_box_original,
  const int* __restrict__ g_num_cell,
  const float* x,
  const float* y,
  const float* z,
  int* NN_radial,
  int* NL_radial,
  int* NN_angular,
  int* NL_angular,
  float* x12_radial,
  float* y12_radial,
  float* z12_radial,
  float* r_radial,        // NEW: Pre-compute distances
  float* x12_angular,
  float* y12_angular,
  float* z12_angular,
  float* r_angular)       // NEW: Pre-compute distances
{
  // For now, use same algorithm as original but cache distances
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  
  for (int n1 = N1 + threadIdx.x; n1 < N2; n1 += blockDim.x) {
    const float* __restrict__ box = g_box + 18 * blockIdx.x;
    const float* __restrict__ box_original = g_box_original + 9 * blockIdx.x;
    const int* __restrict__ num_cell = g_num_cell + 3 * blockIdx.x;
    
    float x1 = x[n1];
    float y1 = y[n1];
    float z1 = z[n1];
    int t1 = g_type[n1];
    
    int count_radial = 0;
    int count_angular = 0;
    
    // Brute-force neighbor search (same as original)
    for (int n2 = N1; n2 < N2; ++n2) {
      for (int ia = 0; ia < num_cell[0]; ++ia) {
        for (int ib = 0; ib < num_cell[1]; ++ib) {
          for (int ic = 0; ic < num_cell[2]; ++ic) {
            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) {
              continue;
            }
            
            float delta_x = box_original[0] * ia + box_original[1] * ib + box_original[2] * ic;
            float delta_y = box_original[3] * ia + box_original[4] * ib + box_original[5] * ic;
            float delta_z = box_original[6] * ia + box_original[7] * ib + box_original[8] * ic;
            
            float x12 = x[n2] + delta_x - x1;
            float y12 = y[n2] + delta_y - y1;
            float z12 = z[n2] + delta_z - z1;
            
            dev_apply_mic(box, x12, y12, z12);
            
            float distance_square = x12 * x12 + y12 * y12 + z12 * z12;
            float distance = sqrt(distance_square);  // Compute once!
            
            int t2 = g_type[n2];
            float rc_radial = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
            float rc_angular = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
            
            if (distance_square < rc_radial * rc_radial) {
              NL_radial[count_radial * N + n1] = n2;
              x12_radial[count_radial * N + n1] = x12;
              y12_radial[count_radial * N + n1] = y12;
              z12_radial[count_radial * N + n1] = z12;
              r_radial[count_radial * N + n1] = distance;  // Cache distance!
              count_radial++;
            }
            
            if (distance_square < rc_angular * rc_angular) {
              NL_angular[count_angular * N + n1] = n2;
              x12_angular[count_angular * N + n1] = x12;
              y12_angular[count_angular * N + n1] = y12;
              z12_angular[count_angular * N + n1] = z12;
              r_angular[count_angular * N + n1] = distance;  // Cache distance!
              count_angular++;
            }
          }
        }
      }
    }
    
    NN_radial[n1] = count_radial;
    NN_angular[n1] = count_angular;
    
    // CRITICAL: Sort neighbors to preserve summation order
    ACNEP::sort_neighbors_by_index(
      NL_radial, x12_radial, y12_radial, z12_radial, r_radial,
      count_radial, n1, N
    );
    ACNEP::sort_neighbors_by_index(
      NL_angular, x12_angular, y12_angular, z12_angular, r_angular,
      count_angular, n1, N
    );
  }
}
```

#### Step 2.3: Implement Dataset::precompute_geometry()

**File:** `src/acnep/dataset.cu` (already stubbed, implement here)

Replace the stub with:

```cpp
void Dataset::precompute_geometry(Parameters& para)
{
  printf("\n=== ACNEP: Pre-computing geometric features ===\n");
  printf("  N = %d atoms\n", N);
  printf("  Nc = %d configurations\n", Nc);
  printf("  max_NN_radial = %d\n", max_NN_radial);
  printf("  max_NN_angular = %d\n", max_NN_angular);
  
  // Allocate cache arrays
  precomp_geom.allocate(N, max_NN_radial, max_NN_angular);
  
  // Launch optimized neighbor list kernel
  gpu_find_neighbor_list_optimized<<<Nc, 256>>>(
    para.paramb,
    N,
    Na.data(),
    Na_sum.data(),
    type.data(),
    box.data(),
    box_original.data(),
    num_cell.data(),
    r.data(),
    r.data() + N,
    r.data() + 2 * N,
    precomp_geom.NN_radial.data(),
    precomp_geom.NL_radial.data(),
    precomp_geom.NN_angular.data(),
    precomp_geom.NL_angular.data(),
    precomp_geom.x12_radial.data(),
    precomp_geom.y12_radial.data(),
    precomp_geom.z12_radial.data(),
    precomp_geom.r_radial.data(),
    precomp_geom.x12_angular.data(),
    precomp_geom.y12_angular.data(),
    precomp_geom.z12_angular.data(),
    precomp_geom.r_angular.data()
  );
  
  CHECK(gpuDeviceSynchronize());
  CHECK(gpuGetLastError());
  
  precomp_geom.is_cached = true;
  
  printf("=== Pre-computation complete ===\n\n");
}
```

### Phase 3: Modify Descriptor Kernels to Use Cache

#### Step 3.1: Update find_descriptors_radial

**File:** `src/acnep/acnep.cu`

**Current signature:**
```cpp
static __global__ void find_descriptors_radial(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP::ParaMB paramb,
  const NEP::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_descriptors)
```

**Add new parameter for cached distances:**
```cpp
static __global__ void find_descriptors_radial(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP::ParaMB paramb,
  const NEP::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_r,      // NEW: Pre-computed distances
  float* g_descriptors)
```

**In kernel body, replace:**
```cpp
float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
```

**With:**
```cpp
float d12 = g_r[index];  // Use pre-computed distance!
```

**Repeat for find_descriptors_angular and all force kernels.**

#### Step 3.2: Update NEP::find_force() to Use Cached Data

**File:** `src/acnep/acnep.cu`

Find the `NEP::find_force()` function and modify to check for cached geometry:

```cpp
void NEP::find_force(
  const bool is_training,
  Dataset& dataset,  // Add dataset parameter to access cache
  // ... other params
) {
  // Check if we should use cached geometry
  bool use_cache = dataset.precomp_geom.is_cached;
  
  if (!use_cache) {
    // Fallback to original path: compute neighbor list
    gpu_find_neighbor_list<<<Nc, 256>>>(
      paramb,
      // ... original parameters
    );
  }
  
  // Use cached data for descriptor computation
  find_descriptors_radial<<<(N - 1) / 64 + 1, 64>>>(
    N,
    use_cache ? dataset.precomp_geom.NN_radial.data() : NN_radial.data(),
    use_cache ? dataset.precomp_geom.NL_radial.data() : NL_radial.data(),
    paramb,
    annmb,
    type,
    use_cache ? dataset.precomp_geom.x12_radial.data() : x12_radial.data(),
    use_cache ? dataset.precomp_geom.y12_radial.data() : y12_radial.data(),
    use_cache ? dataset.precomp_geom.z12_radial.data() : z12_radial.data(),
    use_cache ? dataset.precomp_geom.r_radial.data() : nullptr,  // Pass cached distances
    q
  );
  
  // Repeat for angular, force kernels, etc.
}
```

### Phase 4: Add Optimization Parameters

#### Step 4.1: Extend Parameters Class

**File:** `src/acnep/parameters.cuh`

Add near the end of the `Parameters` class:

```cpp
class Parameters {
public:
  // ... existing members ...
  
  // ACNEP optimization parameters
  ACNEP::OptimizationParams acnep_opts;
};
```

#### Step 4.2: Parse Command-Line Flags

**File:** `src/acnep/main_acnep.cu`

```cpp
int main(int argc, char* argv[])
{
  Parameters para;
  
  // Parse command-line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--check-consistency") {
      para.acnep_opts.check_consistency = true;
      printf("Consistency checking enabled.\n");
    }
    else if (arg == "--no-precompute") {
      para.acnep_opts.use_precomputed_geometry = false;
      printf("Pre-computation disabled.\n");
    }
    // Add more flags as needed
  }
  
  Fitness fitness(para);
  SNES snes(para, &fitness);
  
  return 0;
}
```

### Phase 5: Verification and Testing

#### Step 5.1: Add Consistency Checking to SNES

**File:** `src/acnep/snes.cu`

In the main training loop:

```cpp
for (int n = 0; n < maximum_generation; ++n) {
  
  // Consistency checking for first N generations
  if (para.check_consistency && n < para.acnep_opts.check_generations) {
    // Temporarily disable optimizations
    bool saved_use_cache = para.acnep_opts.use_precomputed_geometry;
    para.acnep_opts.use_precomputed_geometry = false;
    
    // Compute with original method
    float loss_original = fitness->compute(para, population[0]);
    
    // Re-enable optimizations
    para.acnep_opts.use_precomputed_geometry = saved_use_cache;
    
    // Compute with optimizations
    float loss_optimized = fitness->compute(para, population[0]);
    
    // Compare
    float diff = fabs(loss_original - loss_optimized);
    printf("Gen %d consistency check: diff = %.3e\n", n, diff);
    
    if (diff > para.acnep_opts.consistency_tolerance) {
      printf("ERROR: Consistency check failed!\n");
      printf("  Original: %.15e\n", loss_original);
      printf("  Optimized: %.15e\n", loss_optimized);
      if (diff > 1e-8) {
        exit(1);
      }
    }
  }
  
  // Normal training continues...
  create_population(para);
  compute();
  // ...
}
```

#### Step 5.2: Create Test Script

**File:** `test_acnep_consistency.sh` (in repository root)

```bash
#!/bin/bash

echo "=== ACNEP Consistency Test ==="
echo ""

# Check if executables exist
if [ ! -f "src/nep" ] || [ ! -f "src/acnep" ]; then
    echo "Error: nep and/or acnep executables not found"
    echo "Please run 'make nep acnep' first"
    exit 1
fi

# Create test directory
mkdir -p test_acnep
cd test_acnep

# Copy test data (assumes example data exists)
# cp ../examples/nep_test/train.xyz .
# cp ../examples/nep_test/train.in .

echo "Running original NEP..."
../src/nep > nep.log 2>&1
cp nep.txt nep_original.txt
cp loss.out loss_original.out

echo "Running ACNEP with consistency checking..."
../src/acnep --check-consistency > acnep.log 2>&1
cp nep.txt nep_acnep.txt
cp loss.out loss_acnep.out

echo ""
echo "Comparing results..."

# Python script to compare
python3 << 'PYEOF'
import numpy as np
import sys

try:
    params_orig = np.loadtxt('nep_original.txt')
    params_acnep = np.loadtxt('nep_acnep.txt')
    
    loss_orig = np.loadtxt('loss_original.out')
    loss_acnep = np.loadtxt('loss_acnep.out')
    
    # Compare parameters
    param_rmse = np.sqrt(np.mean((params_orig - params_acnep)**2))
    param_max_diff = np.max(np.abs(params_orig - params_acnep))
    
    # Compare loss
    loss_rmse = np.sqrt(np.mean((loss_orig - loss_acnep)**2))
    loss_max_diff = np.max(np.abs(loss_orig - loss_acnep))
    
    print(f"Parameter RMSE: {param_rmse:.3e}")
    print(f"Parameter max diff: {param_max_diff:.3e}")
    print(f"Loss RMSE: {loss_rmse:.3e}")
    print(f"Loss max diff: {loss_max_diff:.3e}")
    print("")
    
    # Check tolerance
    tolerance = 1e-10
    if param_rmse < tolerance and loss_rmse < tolerance:
        print("✓ ACNEP results are numerically equivalent!")
        sys.exit(0)
    elif param_rmse < 1e-8 and loss_rmse < 1e-8:
        print("⚠ ACNEP results are close but not bit-identical (within 1e-8)")
        sys.exit(0)
    else:
        print("✗ ACNEP results differ significantly from original NEP")
        sys.exit(1)
        
except Exception as e:
    print(f"Error comparing results: {e}")
    sys.exit(1)
PYEOF

exit_code=$?
cd ..

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "=== Consistency test PASSED ==="
else
    echo ""
    echo "=== Consistency test FAILED ==="
fi

exit $exit_code
```

Make executable:
```bash
chmod +x test_acnep_consistency.sh
```

## Next Steps (Advanced Optimizations)

### Phase 6: Kernel Fusion

Combine radial and angular descriptor kernels - see `README_ACNEP.md` for details.

### Phase 7: Warp-Level Reductions

Optimize force kernels with warp shuffles - see `README_ACNEP.md`.

### Phase 8: Population Batching

Launch multiple population members concurrently - see `README_ACNEP.md`.

### Phase 9: CUDA Graphs

Capture kernel sequences for faster replay - see `README_ACNEP.md`.

## Testing Strategy

1. **Unit Tests:** Test each optimization in isolation
2. **Integration Tests:** Test combined optimizations
3. **Consistency Tests:** Compare with original NEP on multiple datasets
4. **Performance Tests:** Measure speedups with different system sizes
5. **Stress Tests:** Large systems (>10k atoms) and long training (>1000 generations)

## Profiling

Use Nsight Systems to validate speedups:

```bash
nsys profile --stats=true ./acnep
nsys profile --stats=true ./nep

# Compare kernel times
```

Expected kernel time reductions:
- `gpu_find_neighbor_list`: Should be 0 (pre-computed)
- `find_descriptors_*`: 30-50% faster (cached distances)
- `find_force_*`: 20-30% faster (warp reductions)
- Overall: 4-10x depending on system

## Common Issues and Solutions

### Issue: Results don't match within tolerance

**Cause:** Neighbor summation order differs

**Solution:** Ensure `sort_neighbors_by_index()` is called after neighbor list construction

### Issue: Slower than original

**Cause:** Unnecessary data copies or cache misses

**Solution:** Use `nvprof` to identify bottlenecks, check memory access patterns

### Issue: Out of memory

**Cause:** Large pre-computed arrays

**Solution:** Batch structures or use streaming

## Conclusion

This implementation guide provides a roadmap for completing the ACNEP optimizations. Start with Phase 2 (pre-computation) for the biggest impact, then proceed to more advanced optimizations as needed.
