# ACNEP Current Status - Important Notice

## ⚠️ NO PERFORMANCE IMPROVEMENT YET ⚠️

If you compiled ACNEP and found **no speed improvement**, this is **completely expected**!

### Current Status

**ACNEP is currently in STUB/INFRASTRUCTURE phase:**

- ✅ Build system working (can compile `acnep` executable)
- ✅ Directory structure created
- ✅ Data structures defined
- ✅ Documentation written
- ❌ **Optimization kernels NOT implemented**
- ❌ **Pre-computation NOT active**
- ❌ **No speedup available yet**

### What You're Running

When you run `./acnep`, you are running code that is **functionally identical to NEP**. The only differences are:

1. Different startup messages (shows "ACNEP" instead of "NEP")
2. Allocates cache structures (but doesn't use them)
3. Prints status messages about optimization state

**Result: Same speed as NEP, same results, no optimization.**

### Why Is This Happening?

The previous implementation created the **infrastructure** for optimizations:

- Data structures for caching (`PrecomputedGeometry`)
- Helper functions for optimizations
- Documentation and guides
- Build configuration

But did NOT implement the **actual optimization code**:

- GPU neighbor list computation is NOT pre-cached
- Descriptor kernels still recompute everything
- No kernel fusion
- No warp-level optimizations
- No population batching

### What Needs To Be Done

To get actual speedup, the following must be implemented:

#### Phase 1: Basic Pre-computation (2-5x speedup expected)

**File:** `src/main_acnep/acnep.cu`

Add the optimized neighbor list kernel:
```cpp
__global__ void gpu_find_neighbor_list_optimized(
  // ... parameters ...
) {
  // Compute neighbor lists ONCE at startup
  // Store in precomp_geom arrays
  // Cache distances to avoid sqrt() later
}
```

**File:** `src/main_acnep/dataset.cu`

Update `precompute_geometry()` to:
```cpp
void Dataset::precompute_geometry(Parameters& para) {
  // Launch the optimized kernel
  gpu_find_neighbor_list_optimized<<<Nc, 256>>>(...);
  
  // Mark cache as valid
  precomp_geom.is_cached = true;
}
```

**File:** `src/main_acnep/acnep.cu`

Update descriptor kernels to use cached data:
```cpp
__global__ void find_descriptors_radial(...) {
  // Replace: float d12 = sqrt(x12*x12 + y12*y12 + z12*z12);
  // With:    float d12 = g_r[index];  // Use cached distance!
}
```

#### Phase 2: Advanced Optimizations (additional 2-4x)

- Fuse radial/angular descriptor kernels
- Add warp-level reductions
- Implement population batching
- Use CUDA graphs

**See:** `src/main_acnep/IMPLEMENTATION_GUIDE.md` for detailed instructions.

### How To Proceed

If you want to help implement optimizations:

1. **Read the guide**: `src/main_acnep/IMPLEMENTATION_GUIDE.md`
2. **Start with Phase 3**: Pre-computation (biggest impact)
3. **Test incrementally**: Enable one optimization at a time
4. **Verify consistency**: Use `test_acnep_consistency.sh`
5. **Measure speedup**: Compare with `nep` using same inputs

### Estimated Implementation Time

- **Basic pre-computation**: 2-3 days (experienced CUDA developer)
- **All optimizations**: 2-3 weeks
- **Minimum viable**: 1 week (pre-computation + basic testing)

### Why Release Infrastructure First?

The infrastructure-first approach allows:

1. **Build system verification**: Ensure compilation works
2. **Community contribution**: Others can implement optimizations
3. **Clear documentation**: Guides are ready for implementation
4. **Modular development**: Each optimization can be added incrementally

### Questions?

- **Q: When will optimizations be implemented?**  
  A: The infrastructure is ready for contributors. Implementation timeline depends on availability of developers with CUDA expertise and GPU hardware.

- **Q: Should I use ACNEP now?**  
  A: No. Use the original `nep` executable. ACNEP currently provides no benefit.

- **Q: Can I contribute?**  
  A: Yes! Follow the IMPLEMENTATION_GUIDE.md. Start with Phase 3 (pre-computation).

- **Q: How can I tell if optimizations are active?**  
  A: Look for the startup message. It will say "WARNING: Optimizations are currently in development!" if stubs are active. When optimizations are implemented, this message will change to show which optimizations are enabled.

### Summary

**Current State:**
```
ACNEP = NEP (no difference in performance)
```

**After Phase 1 Implementation:**
```
ACNEP = 2-5x faster than NEP
```

**After All Optimizations:**
```
ACNEP = 4-10x faster than NEP (system dependent)
```

---

**For implementation help, see:**
- `IMPLEMENTATION_GUIDE.md` - Step-by-step instructions
- `README_ACNEP.md` - Technical overview
- `ACNEP_SUMMARY.md` - Project status and architecture

**Last Updated:** 2026-02-04
