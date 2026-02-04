/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
ACNEP Optimization Infrastructure
Defines data structures and helper functions for optimized NEP training
------------------------------------------------------------------------------*/

#pragma once

#include "utilities/gpu_vector.cuh"

namespace ACNEP {

// ============================================================================
// Pre-computed Geometry Cache
// ============================================================================
// This structure stores all geometric data (neighbor lists, displacement
// vectors, distances) that remain constant across PSO generations.
// Pre-computing once at startup eliminates the most expensive kernel from
// the training loop.

struct PrecomputedGeometry {
    // Radial neighbor data
    GPU_Vector<int> NN_radial;      // [N] Number of radial neighbors per atom
    GPU_Vector<int> NL_radial;      // [N * max_NN] Neighbor indices
    GPU_Vector<float> x12_radial;   // [N * max_NN] Displacement x components
    GPU_Vector<float> y12_radial;   // [N * max_NN] Displacement y components
    GPU_Vector<float> z12_radial;   // [N * max_NN] Displacement z components
    GPU_Vector<float> r_radial;     // [N * max_NN] Pre-computed distances (NEW!)

    // Angular neighbor data  
    GPU_Vector<int> NN_angular;
    GPU_Vector<int> NL_angular;
    GPU_Vector<float> x12_angular;
    GPU_Vector<float> y12_angular;
    GPU_Vector<float> z12_angular;
    GPU_Vector<float> r_angular;    // [N * max_NN] Pre-computed distances (NEW!)

    bool is_cached;  // Flag to indicate if cache is valid

    PrecomputedGeometry() : is_cached(false) {}

    // Allocate cache arrays
    void allocate(int N, int max_NN_radial, int max_NN_angular) {
        // Radial
        NN_radial.resize(N);
        NL_radial.resize(N * max_NN_radial);
        x12_radial.resize(N * max_NN_radial);
        y12_radial.resize(N * max_NN_radial);
        z12_radial.resize(N * max_NN_radial);
        r_radial.resize(N * max_NN_radial);  // NEW: cache distances

        // Angular
        NN_angular.resize(N);
        NL_angular.resize(N * max_NN_angular);
        x12_angular.resize(N * max_NN_angular);
        y12_angular.resize(N * max_NN_angular);
        z12_angular.resize(N * max_NN_angular);
        r_angular.resize(N * max_NN_angular);  // NEW: cache distances
    }

    void clear() {
        is_cached = false;
    }
};

// ============================================================================
// Cell-Linked List Data Structures
// ============================================================================
// For optimized neighbor finding with O(N * neighbors) complexity instead
// of O(N² * images). Bins atoms into 3D grid cells.

struct CellList {
    GPU_Vector<int> cell_count;     // [num_cells] Number of atoms per cell
    GPU_Vector<int> cell_offset;    // [num_cells + 1] Cumulative sum of cell_count
    GPU_Vector<int> cell_atoms;     // [N] Atom indices sorted by cell
    GPU_Vector<int> atom_cell;      // [N] Cell index for each atom
    
    int nx, ny, nz;                 // Number of cells in each dimension
    int num_cells;                  // Total number of cells
    float cell_size;                // Cell size (should be >= rc_max)

    CellList() : nx(0), ny(0), nz(0), num_cells(0), cell_size(0.0f) {}

    void allocate(int N, int nx_, int ny_, int nz_) {
        nx = nx_;
        ny = ny_;
        nz = nz_;
        num_cells = nx * ny * nz;
        
        cell_count.resize(num_cells, 0);
        cell_offset.resize(num_cells + 1, 0);
        cell_atoms.resize(N);
        atom_cell.resize(N);
    }
};

// ============================================================================
// Optimization Flags and Parameters
// ============================================================================

struct OptimizationParams {
    // Enable/disable specific optimizations
    bool use_precomputed_geometry;   // Pre-compute neighbor lists
    bool use_cell_list;              // Use cell-linked list for neighbor finding
    bool use_fused_descriptors;      // Fuse radial/angular descriptor kernels
    bool use_warp_reductions;        // Use warp-level reductions in force kernels
    bool use_population_batching;    // Batch population evaluations
    bool use_cuda_graphs;            // Capture kernel sequences as CUDA graphs

    // Consistency checking
    bool check_consistency;          // Run original and optimized paths in parallel
    int check_generations;           // Number of generations to check (default: 10)
    float consistency_tolerance;     // Maximum allowed difference (default: 1e-10)

    // Population batching parameters
    int population_batch_size;       // Number of population members per batch (default: 4)

    // Constructor with defaults
    OptimizationParams() :
        use_precomputed_geometry(true),
        use_cell_list(true),
        use_fused_descriptors(true),
        use_warp_reductions(true),
        use_population_batching(true),
        use_cuda_graphs(true),
        check_consistency(true),
        check_generations(10),
        consistency_tolerance(1e-10f),
        population_batch_size(4)
    {}
};

// ============================================================================
// Performance Metrics
// ============================================================================

struct PerformanceMetrics {
    float time_neighbor_list_ms;
    float time_descriptors_ms;
    float time_force_ms;
    float time_total_per_generation_ms;
    int num_structures;
    int num_atoms_total;
    
    void reset() {
        time_neighbor_list_ms = 0.0f;
        time_descriptors_ms = 0.0f;
        time_force_ms = 0.0f;
        time_total_per_generation_ms = 0.0f;
    }
    
    void print_summary() const {
        printf("\n=== ACNEP Performance Metrics ===\n");
        printf("Neighbor list time:  %.3f ms\n", time_neighbor_list_ms);
        printf("Descriptor time:     %.3f ms\n", time_descriptors_ms);
        printf("Force time:          %.3f ms\n", time_force_ms);
        printf("Total per gen:       %.3f ms\n", time_total_per_generation_ms);
        printf("Structures:          %d\n", num_structures);
        printf("Total atoms:         %d\n", num_atoms_total);
        printf("=================================\n\n");
    }
};

// ============================================================================
// Helper Functions
// ============================================================================

// Get 3D cell index from position
__device__ inline int get_cell_index_3d(
    float x, float y, float z,
    const float* box,
    float cell_size,
    int nx, int ny, int nz
) {
    // Apply PBC to get fractional coordinates
    float sx = x / box[0];
    float sy = y / box[4];
    float sz = z / box[8];
    
    // Map to cell indices
    int cx = int(sx * nx) % nx;
    int cy = int(sy * ny) % ny;
    int cz = int(sz * nz) % nz;
    
    // Handle negative indices (PBC wrap-around)
    if (cx < 0) cx += nx;
    if (cy < 0) cy += ny;
    if (cz < 0) cz += nz;
    
    return cx + cy * nx + cz * nx * ny;
}

// Get neighboring cell index with periodic boundary conditions
__device__ inline int get_neighbor_cell_periodic(
    int cell_idx, int dx, int dy, int dz,
    int nx, int ny, int nz
) {
    int cx = cell_idx % nx;
    int cy = (cell_idx / nx) % ny;
    int cz = cell_idx / (nx * ny);
    
    cx = (cx + dx + nx) % nx;
    cy = (cy + dy + ny) % ny;
    cz = (cz + dz + nz) % nz;
    
    return cx + cy * nx + cz * nx * ny;
}

// Sort neighbors by index to preserve summation order (for floating-point equivalence)
// This is critical for maintaining bit-identical results!
__device__ inline void sort_neighbors_by_index(
    int* NL,          // Neighbor list array [max_NN * N]
    float* x12,       // Displacement vectors (will be reordered)
    float* y12,
    float* z12,
    float* r,         // Distances (will be reordered)
    int count,        // Number of neighbors
    int n1,           // Atom index
    int N             // Total number of atoms
) {
    // Insertion sort (small neighbor count, typically < 100)
    for (int i = 1; i < count; ++i) {
        int key_idx = NL[i * N + n1];
        float key_x = x12[i * N + n1];
        float key_y = y12[i * N + n1];
        float key_z = z12[i * N + n1];
        float key_r = r[i * N + n1];
        
        int j = i - 1;
        while (j >= 0 && NL[j * N + n1] > key_idx) {
            NL[(j + 1) * N + n1] = NL[j * N + n1];
            x12[(j + 1) * N + n1] = x12[j * N + n1];
            y12[(j + 1) * N + n1] = y12[j * N + n1];
            z12[(j + 1) * N + n1] = z12[j * N + n1];
            r[(j + 1) * N + n1] = r[j * N + n1];
            j--;
        }
        NL[(j + 1) * N + n1] = key_idx;
        x12[(j + 1) * N + n1] = key_x;
        y12[(j + 1) * N + n1] = key_y;
        z12[(j + 1) * N + n1] = key_z;
        r[(j + 1) * N + n1] = key_r;
    }
}

// Warp-level reduction for float (uses shuffle instructions)
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

} // namespace ACNEP
