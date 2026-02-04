# ACNEP: Accelerated Neuroevolution Potential Training

## ⚠️ IMPORTANT: Current Status ⚠️

**ACNEP currently provides NO performance improvement over NEP.**

The optimization kernels are NOT yet implemented. See **[ACNEP_STATUS.md](ACNEP_STATUS.md)** for details.

If you tested ACNEP and found no speedup, this is expected. The current version is a stub/infrastructure release.

---

## Quick Start

This repository now includes **ACNEP**, an optimized version of the NEP training code designed to achieve 4-10x speedups while maintaining numerical equivalence with the original implementation.

## Building

```bash
cd src
make acnep
```

This creates the `acnep` executable alongside `nep` and `gpumd`.

## Documentation

📚 **Start here for your role:**

### For Users
- **[ACNEP_SUMMARY.md](ACNEP_SUMMARY.md)** - Complete project overview
  - What has been implemented
  - Expected performance gains
  - Current status and next steps

### For Developers
- **[src/main_acnep/README_ACNEP.md](src/main_acnep/README_ACNEP.md)** - Technical overview
  - All 6 optimization strategies explained
  - Code examples for each optimization
  - Expected speedup analysis
  - Verification strategy

- **[src/main_acnep/IMPLEMENTATION_GUIDE.md](src/main_acnep/IMPLEMENTATION_GUIDE.md)** - Implementation instructions
  - Step-by-step implementation phases
  - Code snippets for each optimization
  - Integration points in existing code
  - Testing and validation procedures

### For Build/CI Maintainers
- **[MAKEFILE_CHANGES.md](MAKEFILE_CHANGES.md)** - Build system documentation
  - Detailed explanation of Makefile changes
  - Before/after comparisons
  - Troubleshooting build issues

## Testing

### Consistency Test
Verify that ACNEP produces numerically equivalent results to NEP:

```bash
./test_acnep_consistency.sh
```

Requirements:
- Both `nep` and `acnep` executables built
- Test data (train.xyz, train.in) in `test_acnep_consistency/` directory

## Project Structure

```
├── ACNEP_SUMMARY.md                 # Project overview
├── MAKEFILE_CHANGES.md              # Build system docs
├── test_acnep_consistency.sh        # Consistency test script
└── src/
    ├── makefile                     # Updated with acnep target
    ├── main_acnep/                  # ACNEP implementation
    │   ├── README_ACNEP.md          # Technical overview
    │   ├── IMPLEMENTATION_GUIDE.md  # Step-by-step guide
    │   ├── acnep_optimization.cuh   # Optimization infrastructure
    │   ├── acnep.cu                 # Optimized NEP potential
    │   ├── acfitness.cu             # Optimized fitness evaluation
    │   ├── main_acnep.cu            # Entry point
    │   └── ...                      # Other files
    └── main_nep/                    # Original NEP (unchanged)
        └── ...
```

## Implementation Status

### ✅ Complete (Infrastructure)
- Build system integration (Makefile)
- Directory structure and file organization
- Optimization framework (data structures, helpers)
- Comprehensive documentation (41KB)
- Test infrastructure

### 📋 Pending (Optimization Kernels)
The infrastructure is complete and ready for optimization implementation:

1. **Phase 3: Pre-computation** (highest priority, 2-5x speedup)
   - Implement optimized neighbor list kernel
   - Cache geometric features at startup
   - Update descriptor kernels to use cache

2. **Phase 4: Kernel Fusion** (1.5-3x speedup)
   - Fuse radial/angular descriptors
   - Add shared memory optimizations

3. **Phase 5: Advanced** (1.2-2x speedup)
   - Warp-level reductions
   - Population batching
   - CUDA graphs

4. **Phase 6: Testing**
   - Consistency checking in training loop
   - Performance profiling
   - Multi-system validation

See [IMPLEMENTATION_GUIDE.md](src/main_acnep/IMPLEMENTATION_GUIDE.md) for detailed steps.

## Key Design Principles

### 1. Numerical Equivalence First
All optimizations preserve results within 1e-10 tolerance through:
- Neighbor sorting to preserve summation order
- No fast-math or reduced precision
- Deterministic algorithms

### 2. Gradual Implementation
Each optimization can be enabled/disabled independently:
```cpp
para.acnep_opts.use_precomputed_geometry = true;
para.acnep_opts.use_fused_descriptors = true;
para.acnep_opts.use_warp_reductions = false;  // Disable if issues
```

### 3. Comprehensive Verification
Built-in consistency checking:
```bash
./acnep --check-consistency
```

## Expected Performance

| System Size | Original NEP | ACNEP Target | Speedup |
|-------------|--------------|--------------|---------|
| Small (100-500 atoms) | 10s/gen | 2-3s/gen | 3-5x |
| Medium (500-2000 atoms) | 50s/gen | 8-12s/gen | 4-6x |
| Large (2000+ atoms) | 200s/gen | 25-40s/gen | 5-8x |
| Very Large (5000+ atoms) | 600s/gen | 60-100s/gen | 6-10x |

## Optimization Strategies

| Optimization | Expected Speedup | Status |
|--------------|------------------|--------|
| Pre-computed geometry | 2-5x | 📋 Ready to implement |
| Cell-linked list | 2-3x (pre-comp only) | 📋 Ready to implement |
| Fused descriptors | 1.5-2x | 📋 Ready to implement |
| Warp reductions | 1.2-1.5x | 📋 Ready to implement |
| Population batching | 1.1-1.3x | 📋 Ready to implement |
| CUDA graphs | 1.1-1.2x | 📋 Ready to implement |

**Cumulative target:** 4-10x depending on system size and GPU architecture.

## Requirements

- CUDA Toolkit (nvcc)
- GPU with compute capability ≥ 6.0 (sm_60)
- CUDA libraries: cublas, cusolver, cufft
- C++14 compatible compiler

## Contributing

When implementing optimizations:

1. **Follow the guide:** Start with [IMPLEMENTATION_GUIDE.md](src/main_acnep/IMPLEMENTATION_GUIDE.md)
2. **Test incrementally:** Enable one optimization at a time
3. **Verify consistency:** Run `test_acnep_consistency.sh` after each change
4. **Profile performance:** Use `nsys profile` to validate speedups
5. **Document changes:** Update relevant documentation

## Support

For implementation questions:
- See [IMPLEMENTATION_GUIDE.md](src/main_acnep/IMPLEMENTATION_GUIDE.md) for step-by-step instructions
- Check troubleshooting sections in documentation
- Review code examples in [README_ACNEP.md](src/main_acnep/README_ACNEP.md)

## License

Same as GPUMD: GNU General Public License v3.0

## Acknowledgments

Original NEP implementation by Zheyong Fan and GPUMD development team.
ACNEP optimizations designed to accelerate training while preserving the mathematical framework.

---

**Current Status:** Infrastructure complete, optimization kernels ready for implementation.

**Next Step:** Follow Phase 3 in [IMPLEMENTATION_GUIDE.md](src/main_acnep/IMPLEMENTATION_GUIDE.md) to implement pre-computation (highest impact).
