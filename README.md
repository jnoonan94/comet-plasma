# Cometary Plasma Modeling

Research project investigating cometary plasma environments using the
[Menura](https://zenodo.org/records/6517018) 3-D hybrid PIC simulation code.

## Folder layout

```
comet_plasma_modeling/
├── menura_source/          Menura v1 source code (Zenodo release 6517018)
│   ├── menura/             Build directory — edit src/parameters.h, then `make`
│   │   ├── src/            C++/CUDA source files
│   │   └── Makefile
│   ├── analysis/           Python analysis scripts (menura_utils.py, plt_*.py)
│   ├── notebooks/          Physics validation notebooks (MHD modes, CAM, etc.)
│   ├── docs/               Doxygen reference documentation
│   └── gallery/            Algorithm diagrams and simulation animation GIFs
│
└── tutorials/              Step-by-step learning materials
    ├── README.md           Tutorial index and quick-start recipe
    ├── A_cpp_basics/       C++ and CUDA background (5 guides)
    ├── B_menura_guide/     Installation, configuration, running, analysis (5 guides)
    └── C_notebooks/        Working analysis notebooks (3 Jupyter notebooks)
```

## Quick start

```bash
# 1. Configure — all parameters are compile-time constants
vim menura_source/menura/src/parameters.h

# 2. Build
cd menura_source/menura
make clean && make

# 3. Create output directories (required before first run)
mkdir -p products/HK products/probes

# 4. Run  (NP = mpi_nb_proc_y × mpi_nb_proc_z, default 4×4=16)
mpirun -np 16 ./menura

# 5. Analyse
jupyter notebook ../../../tutorials/C_notebooks/
```

## Key reference

Menura source code: https://zenodo.org/records/6517018
