# B-01 · Installation and Building Menura

Menura is built with **GNU Make**, using `mpic++` for the CPU code and `nvcc` for the CUDA kernels.  There is no CMake, no configure script, and no package manager step.

---

## Prerequisites

| Tool | Version guidance | Purpose |
|------|-----------------|---------|
| CUDA Toolkit | 10.x or later (tested with sm_35/61/70 targets) | `nvcc` compiler + CUDA runtime |
| MPI implementation | OpenMPI ≥ 3 or MPICH | `mpic++` wrapper compiler |
| zlib | system package (`libz-dev` / `zlib-devel`) | required by the bundled `cnpy` library |
| Python 3 + NumPy | for analysis only | reading `.npy` output files |

The GPU targets hard-coded in the Makefile are Kepler (sm_35 / K80), Pascal (sm_61 / P100), and Volta (sm_70 / V100).  If your GPU is Ampere (A100, sm_80) or later, you must add the appropriate `-gencode` flag — see below.

---

## Repository layout

```
menura-master/
├── analysis/          Python analysis scripts
│   ├── menura_utils.py
│   ├── plt_fields.py
│   └── plt_HK.py
├── docs/              Doxygen + Sphinx HTML documentation
├── gallery/           Example output images and GIFs
├── menura/
│   ├── src/           All C++ and CUDA source files
│   │   ├── parameters.h       ← edit this to configure the simulation
│   │   ├── structures.h
│   │   ├── menura.cu          ← main entry point
│   │   ├── functions_i.cpp/h  ← initialisation routines
│   │   ├── functions_o.cpp/h  ← output routines
│   │   ├── utils.h
│   │   ├── kernels_*.cuh      ← CUDA kernels
│   │   └── cnpy.cpp/h         ← bundled NumPy I/O library
│   ├── obj/           Object files (empty, auto-created by make)
│   ├── inputs/        Input data (empty; populated by Phase 1 run)
│   ├── Makefile
│   └── go_menura      Example SLURM job script
└── notebooks/         Jupyter notebooks
```

---

## Building

All build commands are run from the `menura/` directory:

```bash
cd menura-master/menura
make
```

The Makefile compiles four source files:

| Source | Compiler | Output object |
|--------|----------|---------------|
| `src/menura.cu` | `nvcc` | `obj/menura.o` |
| `src/functions_i.cpp` | `mpic++` | `obj/functions_i.o` |
| `src/functions_o.cpp` | `mpic++` | `obj/functions_o.o` |
| `src/cnpy.cpp` | `mpic++` | `obj/cnpy.o` |

Then `mpic++` links them together with `-lcuda -lcudart -lz` to produce the executable `menura`.

### GPU architecture flags

The Makefile contains:

```makefile
NFLAGS=-gencode arch=compute_35,code=sm_35 \
       -gencode arch=compute_61,code=sm_61 \
       -gencode arch=compute_70,code=sm_70
```

If your GPU is not Kepler/Pascal/Volta, add the correct target.  For Ampere (A100, RTX 3xxx):

```makefile
NFLAGS=... -gencode arch=compute_80,code=sm_80
```

For Ada Lovelace (RTX 4xxx):

```makefile
NFLAGS=... -gencode arch=compute_89,code=sm_89
```

### Clean build

```bash
make clean   # removes obj/*.o
make         # rebuilds everything
```

Note: `make clean` removes `*.x` files and `obj/*.o` but does **not** remove the `menura` executable itself.  If you need a truly clean state:

```bash
make clean && rm -f menura && make
```

---

## Pre-run directory setup

Before executing `menura`, you **must** create two output directories:

```bash
mkdir -p products/HK
mkdir -p products/probes
```

The code checks for these at startup (`pre_checks()` in `utils.h`) and exits with an error message if they are missing:

```
| Cannot access ./products/HK
| Cannot access ./products/probes
```

These directories must exist relative to the working directory from which you launch the executable, i.e. relative to wherever `mpirun` / `srun` is invoked.

---

## Running

### Local (small test, 1 GPU)

```bash
cd menura-master/menura
mkdir -p products/HK products/probes
mpirun -np 1 ./menura
```

For a single-process test you may need to set `mpi_nb_proc_y_cst = 1` and `mpi_nb_proc_z_cst = 1` in `parameters.h` and rebuild.

### HPC cluster (SLURM)

The file `go_menura` is an example SLURM script (written for the Swedish SNIC allocation on Tetralith/Kebnekaise with K80 GPUs):

```bash
#!/bin/bash
#SBATCH -A SNIC-xxxx-xx-xxx
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:k80:1,gpuexcl
#SBATCH -t 5:00:00

srun leo          # NOTE: the executable was called "leo" in older versions
```

Adapt this to your cluster:
- Replace the account, GPU type, and time limit.
- Change `srun leo` to `srun ./menura` (the executable is now named `menura`).
- `--ntasks` must equal `mpi_nb_proc_y_cst × mpi_nb_proc_z_cst` from `parameters.h`.

Menura identifies which GPU to use via `SLURM_LOCALID` (set automatically by SLURM when using `srun`):

```cpp
// utils.h
char* local_rank_env = getenv("SLURM_LOCALID");
int   local_rank     = atoi(local_rank_env);
cudaSetDevice(local_rank);
```

If not using SLURM (e.g. on a workstation with a single GPU), `SLURM_LOCALID` will not be set and all processes will use device 0, which is fine for single-GPU runs.

---

## Verifying a successful build

```bash
ls -lh menura          # executable should exist and be ≥ 1 MB
./menura --help        # not implemented; it will start and fail pre_checks if dirs are missing
```

A successful startup prints:

```
._____________________________________________________________________________________
|
| Initial checks.
| All good.
| Setting device number 0
|_____________________________________________________________________________________
```
