# Menura Tutorial Series

**Source code:** Menura v1 (Zenodo: https://zenodo.org/records/6517018)

These tutorials are grounded in the actual Menura source files.  Every claim about file formats, build steps, parameter names, and output structure refers to code found in the Zenodo release.

---

## What Menura is

Menura is a 3-D hybrid Particle-in-Cell (PIC) plasma simulation code:

- **Ions** are treated as macro-particles (kinetic)
- **Electrons** are treated as a massless fluid
- **Fields** (E, B) are solved self-consistently using the CAM (Current Advance Method) predictor-corrector scheme
- **Parallelism:** CUDA (one GPU per MPI rank) + MPI (2-D rank grid in y–z)
- **Build:** GNU Make (`mpic++` + `nvcc`)
- **Output:** NumPy `.npy` files via the bundled `cnpy` library

The default configuration simulates a **magnetized planetary body** in the solar wind (obstacle + dipole).  For a cometary simulation, set `solid_body_cst = false`, `dipole_cst = false`, `inject_pla_cst = true` in `parameters.h` and recompile.

---

## Important: no runtime input file

**All parameters are compile-time constants** in `menura/src/parameters.h`.  To change any parameter — grid size, species, time step, physics flags — edit that file and run `make clean && make`.

---

## Tutorial parts

### Part A: C++ Basics
Background C++ and CUDA concepts used throughout Menura.

| File | Topic |
|------|-------|
| `A_cpp_basics/01_variables_and_constants.md` | Types, `#define`, `const`, feature flags in `parameters.h` |
| `A_cpp_basics/02_arrays_and_memory.md` | Multi-dimensional arrays, ghost cells, SoA layout of `particles` |
| `A_cpp_basics/03_structs_and_classes.md` | All major structs: `simu_fields`, `particles`, `simu_B_field`, `probes`, etc. |
| `A_cpp_basics/04_pointers_and_cuda_memory.md` | Host/device copies, `cudaMalloc`, `cudaMemcpy`, GPU assignment |
| `A_cpp_basics/05_functions_and_cuda_kernels.md` | Function qualifiers, kernel launch syntax, Boris pusher, `sortie()` |

### Part B: Menura User Guide
Step-by-step instructions for building, configuring, running, and analyzing a Menura simulation.

| File | Topic |
|------|-------|
| `B_menura_guide/01_installation_and_build.md` | Prerequisites, `make`, GPU arch flags, pre-run directory setup |
| `B_menura_guide/02_code_structure_walkthrough.md` | Source file roles, execution flow, MPI decomposition, CAM algorithm |
| `B_menura_guide/03_configuration_parameters.md` | All parameters in `parameters.h`; planetary vs. cometary setup |
| `B_menura_guide/04_running_and_monitoring.md` | Local and SLURM launch, output file list, restart procedure |
| `B_menura_guide/05_analyzing_outputs.md` | Reading `.npy` files, `menura_utils.py` classes, assembling MPI tiles |

### Part C: Jupyter Notebooks
Working Python notebooks for common analysis tasks.

| File | Topic |
|------|-------|
| `C_notebooks/01_loading_and_visualizing_fields.ipynb` | Load field snapshots, assemble MPI tiles, midplane plots |
| `C_notebooks/02_house_keeping_diagnostics.ipynb` | Energy evolution, particle counts, B statistics, throughput |
| `C_notebooks/03_particle_data_and_distributions.ipynb` | Particle positions, 1-D and 2-D VDFs, shell analysis |

---

## Quick-start recipe

```bash
# 1. Edit parameters
vim menura-master/menura/src/parameters.h

# 2. Build
cd menura-master/menura
make clean && make

# 3. Create output directories
mkdir -p products/HK products/probes

# 4. Run (adjust NP = mpi_nb_proc_y × mpi_nb_proc_z)
mpirun -np 16 ./menura

# 5. Analyze
cd ../..
jupyter notebook menura-master/menura_tutorials/C_notebooks/
```

---

## Known dependency issue: `menura_utils.py`

`analysis/menura_utils.py` imports `own_tools` and `own_colours` — personal utility packages not included in the Zenodo release.  The notebooks in Part C avoid this dependency by reading `products/parameters.txt` directly.  To use `menura_utils.py` directly, create stub files:

```bash
touch analysis/own_tools.py analysis/own_colours.py
```

or copy only the `menura_param`, `menura_grid`, and `menura_probes` class definitions into your own script.
