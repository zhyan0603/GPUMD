# Makefile Changes for ACNEP

## Overview

The Makefile has been updated to build a new `acnep` executable alongside the existing `nep` and `gpumd` executables. This document explains the changes made.

## Changes Made

### 1. Added ACNEP Source Files

**Location:** Lines 42-48 in `src/makefile`

**Before:**
```makefile
SOURCES_NEP =                     \
	$(wildcard main_nep/*.cu)     \
	$(wildcard utilities/*.cu)
SOURCES_GNEP =                     \
	$(wildcard main_gnep/*.cu)     \
	$(wildcard utilities/*.cu)
```

**After:**
```makefile
SOURCES_NEP =                     \
	$(wildcard main_nep/*.cu)     \
	$(wildcard utilities/*.cu)
SOURCES_ACNEP =                   \
	$(wildcard main_acnep/*.cu)        \
	$(wildcard utilities/*.cu)
SOURCES_GNEP =                     \
	$(wildcard main_gnep/*.cu)     \
	$(wildcard utilities/*.cu)
```

**Explanation:** Added `SOURCES_ACNEP` variable that includes all `.cu` files in the `acnep/` directory plus shared utilities.

### 2. Added ACNEP Object Files

**Location:** Lines 53-62 in `src/makefile`

**Before:**
```makefile
ifdef OS # For Windows with the cl.exe compiler
OBJ_GPUMD = $(SOURCES_GPUMD:.cu=.obj)
OBJ_NEP = $(SOURCES_NEP:.cu=.obj)
OBJ_GNEP = $(SOURCES_GNEP:.cu=.obj)
else
OBJ_GPUMD = $(SOURCES_GPUMD:.cu=.o)
OBJ_NEP = $(SOURCES_NEP:.cu=.o)
OBJ_GNEP = $(SOURCES_GNEP:.cu=.o)
endif
```

**After:**
```makefile
ifdef OS # For Windows with the cl.exe compiler
OBJ_GPUMD = $(SOURCES_GPUMD:.cu=.obj)
OBJ_NEP = $(SOURCES_NEP:.cu=.obj)
OBJ_ACNEP = $(SOURCES_ACNEP:.cu=.obj)
OBJ_GNEP = $(SOURCES_GNEP:.cu=.obj)
else
OBJ_GPUMD = $(SOURCES_GPUMD:.cu=.o)
OBJ_NEP = $(SOURCES_NEP:.cu=.o)
OBJ_ACNEP = $(SOURCES_ACNEP:.cu=.o)
OBJ_GNEP = $(SOURCES_GNEP:.cu=.o)
endif
```

**Explanation:** Added `OBJ_ACNEP` variable for both Windows (.obj) and Linux (.o) builds.

### 3. Added ACNEP Headers

**Location:** Lines 67-79 in `src/makefile`

**Before:**
```makefile
HEADERS =                         \
	$(wildcard utilities/*.cuh)   \
	$(wildcard main_gpumd/*.cuh)  \
	$(wildcard integrate/*.cuh)   \
	$(wildcard mc/*.cuh)          \
	$(wildcard minimize/*.cuh)    \
	$(wildcard force/*.cuh)       \
	$(wildcard measure/*.cuh)     \
	$(wildcard model/*.cuh)       \
	$(wildcard phonon/*.cuh)      \
	$(wildcard main_nep/*.cuh)    \
	$(wildcard main_gnep/*.cuh)
```

**After:**
```makefile
HEADERS =                         \
	$(wildcard utilities/*.cuh)   \
	$(wildcard main_gpumd/*.cuh)  \
	$(wildcard integrate/*.cuh)   \
	$(wildcard mc/*.cuh)          \
	$(wildcard minimize/*.cuh)    \
	$(wildcard force/*.cuh)       \
	$(wildcard measure/*.cuh)     \
	$(wildcard model/*.cuh)       \
	$(wildcard phonon/*.cuh)      \
	$(wildcard main_nep/*.cuh)    \
	$(wildcard main_acnep/*.cuh)       \
	$(wildcard main_gnep/*.cuh)
```

**Explanation:** Added `$(wildcard main_acnep/*.cuh)` to include ACNEP header files in dependency tracking.

### 4. Added ACNEP Target

**Location:** Lines 84-106 in `src/makefile`

**Before:**
```makefile
all: gpumd nep 
gpumd: $(OBJ_GPUMD)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo =================================================
	@echo The gpumd executable is successfully compiled!
	@echo =================================================
nep: $(OBJ_NEP)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo =================================================
	@echo The nep executable is successfully compiled!
	@echo =================================================
gnep: $(OBJ_GNEP)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo =================================================
	@echo The gnep executable is successfully compiled!
	@echo =================================================
```

**After:**
```makefile
all: gpumd nep acnep
gpumd: $(OBJ_GPUMD)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo =================================================
	@echo The gpumd executable is successfully compiled!
	@echo =================================================
nep: $(OBJ_NEP)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo =================================================
	@echo The nep executable is successfully compiled!
	@echo =================================================
acnep: $(OBJ_ACNEP)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo =================================================
	@echo The acnep executable is successfully compiled!
	@echo =================================================
gnep: $(OBJ_GNEP)
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)
	@echo =================================================
	@echo The gnep executable is successfully compiled!
	@echo =================================================
```

**Explanation:** 
- Added `acnep` to the `all` target so it builds by default
- Added `acnep` target that links `$(OBJ_ACNEP)` with required libraries
- Uses same compiler flags and libraries as NEP

### 5. Added ACNEP Compilation Rules (Windows)

**Location:** Lines 135-138 in `src/makefile`

**Before:**
```makefile
main_nep/%.obj: main_nep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_gnep/%.obj: main_gnep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
```

**After:**
```makefile
main_nep/%.obj: main_nep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_acnep/%.obj: main_acnep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_gnep/%.obj: main_gnep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
```

**Explanation:** Added pattern rule for compiling `.cu` files in `acnep/` directory to `.obj` files on Windows.

### 6. Added ACNEP Compilation Rules (Linux)

**Location:** Lines 160-163 in `src/makefile`

**Before:**
```makefile
main_nep/%.o: main_nep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_gnep/%.o: main_gnep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
```

**After:**
```makefile
main_nep/%.o: main_nep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_acnep/%.o: main_acnep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
main_gnep/%.o: main_gnep/%.cu $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
```

**Explanation:** Added pattern rule for compiling `.cu` files in `acnep/` directory to `.o` files on Linux.

### 7. Updated Clean Target

**Location:** Lines 169-173 in `src/makefile`

**Before:**
```makefile
clean:
ifdef OS
	del /s *.obj *.exp *.lib *.exe
else
	rm -f */*.o gpumd nep gnep
endif
```

**After:**
```makefile
clean:
ifdef OS
	del /s *.obj *.exp *.lib *.exe
else
	rm -f */*.o gpumd nep acnep gnep
endif
```

**Explanation:** Added `acnep` to the list of executables to remove when running `make clean`.

## Building ACNEP

### Build All Executables
```bash
cd src
make
```

This will now build `gpumd`, `nep`, and `acnep` by default.

### Build ACNEP Only
```bash
cd src
make acnep
```

### Clean Build Artifacts
```bash
cd src
make clean
```

This removes all object files and executables including `acnep`.

## File Structure

```
src/
├── makefile              # Updated with ACNEP targets
├── acnep/               # New directory
│   ├── acnep.cu         # Optimized NEP potential
│   ├── acnep.cuh
│   ├── acfitness.cu     # Optimized fitness evaluation
│   ├── acfitness.cuh
│   ├── main_acnep.cu    # Entry point
│   └── ...              # Other files
├── main_nep/            # Original NEP code (unchanged)
│   ├── nep.cu
│   ├── fitness.cu
│   └── ...
└── utilities/           # Shared utilities
    └── ...
```

## Compiler Flags

ACNEP uses the same compiler flags as NEP:

- **Linux:** `-std=c++14 -O3 -arch=sm_60`
- **Windows:** `-O3 -arch=sm_60`
- **Libraries:** `-lcublas -lcusolver -lcufft`

**Important:** Do NOT add `-use_fast_math` or change precision flags, as this would break numerical equivalence.

## Testing the Build

After building, verify the executable exists:

```bash
ls -lh src/acnep
# Should show the acnep executable with appropriate size and permissions
```

## Troubleshooting

### Error: `nvcc: command not found`
**Solution:** Install CUDA Toolkit and ensure `nvcc` is in your PATH.

### Error: `No such file or directory: main_acnep/*.cu`
**Solution:** Ensure the `src/main_acnep/` directory exists and contains `.cu` files.

### Error: Undefined references during linking
**Solution:** Check that all required CUDA libraries are installed (`cublas`, `cusolver`, `cufft`).

## Future Enhancements

Potential Makefile improvements:

1. **Debug Build:** Add a debug target with `-g -G` flags
2. **Architecture Detection:** Auto-detect GPU architecture
3. **Parallel Compilation:** Add `-j` flag support for faster builds
4. **Install Target:** Add `make install` to copy executables to system path
5. **Dependency Generation:** Auto-generate header dependencies

## Summary

The Makefile changes are minimal and follow the existing pattern for `nep` and `gnep`. The new `acnep` target:

1. Compiles all `.cu` files in `acnep/` directory
2. Links with the same libraries as NEP
3. Produces a standalone `acnep` executable
4. Integrates cleanly with existing build system

No changes were made to the original NEP build process, ensuring backward compatibility.
