# B-04 · Running and Monitoring a Simulation

This tutorial covers how to launch Menura on both a local workstation and an HPC cluster, and how to monitor its progress and health.

---

## Pre-run checklist

Before every run, verify:

```bash
# 1. Executable exists
ls -lh menura-master/menura/menura

# 2. Output directories exist (relative to launch directory)
ls -d menura-master/menura/products/HK
ls -d menura-master/menura/products/probes

# 3. MPI task count matches parameters.h
#    mpi_nb_proc_y_cst × mpi_nb_proc_z_cst = NP
grep "mpi_nb_proc_[yz]_cst" menura-master/menura/src/parameters.h
```

If the output directories are missing, create them:

```bash
cd menura-master/menura
mkdir -p products/HK products/probes
```

Menura prints a startup message if all checks pass:

```
._____________________________________________________________________________________
|
| Initial checks.
| All good.
| Setting device number 0
| Setting device number 1
...
|_____________________________________________________________________________________
```

If a check fails, `sortie()` is called, which prints the error and exits cleanly.

---

## Running locally

For a single-GPU test (requires `mpi_nb_proc_y_cst = 1`, `mpi_nb_proc_z_cst = 1` in `parameters.h`):

```bash
cd menura-master/menura
mpirun -np 1 ./menura
```

For 4 MPI processes (4 GPUs, `mpi_nb_proc_y_cst × mpi_nb_proc_z_cst = 4`):

```bash
mpirun -np 4 ./menura
```

Standard output goes to the terminal.  You can redirect it to a log file:

```bash
mpirun -np 4 ./menura 2>&1 | tee run.log
```

---

## Running on an HPC cluster (SLURM)

The included `go_menura` script is a template.  Here is a modernized version:

```bash
#!/bin/bash
#SBATCH --job-name=menura
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --ntasks=16                    # = mpi_nb_proc_y × mpi_nb_proc_z
#SBATCH --ntasks-per-node=4            # = GPUs per node
#SBATCH --gres=gpu:4                   # request 4 GPUs per node
#SBATCH --time=05:00:00
#SBATCH --output=menura_%j.out

cd $SLURM_SUBMIT_DIR
mkdir -p products/HK products/probes

srun ./menura
```

Key points:
- `--ntasks` must equal `mpi_nb_proc_y_cst × mpi_nb_proc_z_cst` from `parameters.h`.
- `SLURM_LOCALID` is set automatically by `srun`, so `initialisation_cuda()` can assign GPUs correctly.
- The original `go_menura` uses `srun leo` — the executable is now named `menura`, so use `srun ./menura`.

Submit with:
```bash
sbatch go_menura
```

---

## Standard output format

Menura prints progress to stdout.  Key messages to watch for:

```
| Saving HK 50          ← HK diagnostics saved at iteration 50
| Saving fields 100     ← field snapshot saved at iteration 100
| Saving fields 200
...
| Device 0 Off.         ← clean exit via sortie()
|_____________________________________________________________________________________
```

Error messages follow the pattern:

```
| check_status, tag N, error at iteration NNN, rank y R, rank z Z: <CUDA error string>
| check_status, tag N, it NNN, available/free memory (Mb): TOTAL FREE
```

---

## Monitoring memory usage

GPU memory can be checked at run-time by uncommenting `check_memory_device` calls in `menura.cu`, or after the run by looking for the memory lines in the log.  The utility function signature:

```cpp
// utils.h
void check_memory_device(int rank, int tag);
```

It prints, for each rank:

```
| Tag N, MPI rank R, Device D, memory: free=XXXX, total=YYYY
```

(values in MB)

---

## Output files written during a run

Menura writes output into the `products/` directory (relative to the working directory).  Files are written in NumPy `.npy` format via the bundled `cnpy` library.

### House-keeping files (`products/HK/`)

Written every `rate_save_t_cst` iterations, appended throughout the run:

| File | Shape | Contents |
|------|-------|---------|
| `energy_elec_rank_Y_Z.npy` | `[len_save_t_cst]` | Electric field energy |
| `energy_mag_rank_Y_Z.npy` | `[len_save_t_cst]` | Magnetic field energy |
| `energy_kin_rank_Y_Z.npy` | `[len_save_t_cst]` | Kinetic energy |
| `rms_B_rank_Y_Z.npy` | `[len_save_t_cst]` | RMS of B fluctuations |
| `B_mean_rank_Y_Z.npy` | `[len_save_t_cst, 3]` | Mean B-field components |
| `B_var_rank_Y_Z.npy` | `[len_save_t_cst, 3]` | Variance of B |
| `active_part_0_rank_Y_Z.npy` | `[len_save_t_cst]` | Active SW particle count |
| `active_part_1_rank_Y_Z.npy` | `[len_save_t_cst]` | Active cometary/planetary particle count |
| `simu_time.npy` | `[len_save_t_cst]` | Simulation time (rank 0 only) |
| `run_time.npy` | `[len_save_t_cst]` | Wall-clock time (rank 0 only) |

### Field snapshots (`products/`)

Written every `rate_save_field_cst` iterations, one file per snapshot per rank:

| Filename pattern | Contents |
|-----------------|---------|
| `dens_it{NNN}_rank_{Y}_{Z}.npy` | Total ion density |
| `dens_spec0_it{NNN}_rank_{Y}_{Z}.npy` | Solar wind density |
| `dens_spec1_it{NNN}_rank_{Y}_{Z}.npy` | Cometary/planetary density |
| `curr_it{NNN}_rank_{Y}_{Z}.npy` | Ion current vector |
| `B_it{NNN}_rank_{Y}_{Z}.npy` | Magnetic field vector |

Where `NNN` is the zero-padded iteration number and `Y`, `Z` are the MPI rank indices.

### Particle files

Written every `rate_save_particles_cst` iterations:

```
particles_FILE_it{NNN}_rank_{Y}_{Z}.npy   shape: [N_active, 7]
```

Columns: `rx, ry, rz, vx, vy, vz, ID` (3D).  For large particle counts the file is split into chunks of 100 M particles.

### Probe file

```
products/probes/probes_rank_{Y}_{Z}.npy
```

Contains the time series at all virtual probe locations.

---

## Restarting a simulation

1. In `parameters.h`, set:
   ```cpp
   #define restart_cst true
   const int idx_it_restart_cst = 10000;              // iteration to restart from
   const std::string path_inputs_restart_cst = "../run_031/products";  // path to saved files
   ```
2. Rebuild: `make clean && make`
3. Ensure the products from the previous run are accessible at the path specified.
4. Run normally: `mpirun -np N ./menura`

---

## Exercise

1. A run with the default parameters writes field snapshots every 100 iterations for 1000 iterations.  How many field snapshot files are written per MPI rank?  How many total files are written across 16 ranks?
2. Why is it important to have `SLURM_LOCALID` set before calling `cudaSetDevice`?  What happens if it is not set?  (Hint: look at `initialisation_cuda()` in `utils.h`.)
3. Open the `go_menura` file.  What SNIC project code does it reference, and what does `gpuexcl` mean?
