# B-02 · Code Structure Walkthrough

This tutorial walks through the Menura source tree and explains what each file does, following the execution flow from program start to the main solver loop.

---

## Source file overview

All source files are in `menura/src/`.

| File | Language | Role |
|------|----------|------|
| `parameters.h` | C++ | **All compile-time constants** — grid size, physics flags, time limits |
| `structures.h` | C++ | All data structure definitions (`simu_fields`, `particles`, etc.) |
| `menura.cu` | CUDA C++ | **Main program** — `main()` function, solver loop |
| `functions_i.cpp/h` | C++ | Initialisation routines (grid, fields, particles, B-field) |
| `functions_o.cpp/h` | C++ | Output routines (writes `.npy` files via cnpy) |
| `utils.h` | C++ | Utility functions: pre-checks, CUDA init, error handling, `sortie()` |
| `kernels_particles.cuh` | CUDA | Boris pusher: position and velocity advance |
| `kernels_particles_to_grid.cuh` | CUDA | Moment deposition (charge, current) |
| `kernels_fields.cuh` | CUDA | Ohm's law, Faraday's law, field smoothing |
| `kernels_HK.cuh` | CUDA | House-keeping diagnostics (energy, RMS) |
| `kernels_DF.cuh` | CUDA | Distribution function (optional, `DF_cst`) |
| `kernels_tricks.cuh` | CUDA | Boundary conditions, grid state checks |
| `kernels_calls.h` | C++ | Forward declarations for all kernel functions |
| `cnpy.cpp/h` | C++ | Bundled library: read/write NumPy `.npy` files |

---

## Execution flow

### 1. `main()` entry — `menura.cu`

The program begins in `main()`.  MPI is initialized first:

```cpp
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
```

Then CUDA is set up (one GPU per rank, using `SLURM_LOCALID`):

```cpp
initialisation_cuda();   // utils.h
```

Then the output directories are checked:

```cpp
pre_checks();            // utils.h
```

### 2. Memory allocation

All structs are allocated on both host and device:

```cpp
particles*    pa_h = new particles;
particles*    pa_d;
cudaMalloc((void**)&pa_d, sizeof(particles));
// ... repeated for fields, Ba, Bb, grid, probes, HK, mom, rms, traj, ...
```

Two B-field structs are allocated because the CAM predictor-corrector scheme needs two time levels:

```cpp
simu_B_field* Ba_h = new simu_B_field;  // time level n
simu_B_field* Ba_d;
simu_B_field* Bb_h = new simu_B_field;  // time level n+1 (predicted)
simu_B_field* Bb_d;
```

### 3. Initialisation

Initialisation functions (from `functions_i.h`) populate the host structs:

```cpp
init_house_keeping(mom, traj, rms, simu_param_h, HK_h, sta_sol_h);
init_grid(grid_h, simu_param_h);
init_probes(probes_h, simu_param_h, grid_h);
init_B(Ba_h, simu_param_h);
init_B(Bb_h, simu_param_h);
init_part_node(pa_h, grid_h, simu_param_h);
```

For a restart (`restart_cst = true`), the init calls are replaced by load calls:

```cpp
load_B_field(Ba_h, tank_h, simu_param_h, false, path_inputs_str_cst, idx_load);
load_fields(fields_h, tank_h, simu_param_h, false, path_inputs_str_cst, idx_load);
load_particles(pa_h, grid_h, tank_h, simu_param_h, comm, false,
               path_inputs_str_cst, idx_load, old_file);
```

After initialisation, all host data is copied to the device:

```cpp
cudaMemcpy(pa_d,     pa_h,     sizeof(*pa_h), cudaMemcpyHostToDevice);
cudaMemcpy(fields_d, fields_h, sizeof(*fields_h), cudaMemcpyHostToDevice);
// ...
```

### 4. Main solver loop

The solver loop runs for `nb_it_max_cst` iterations.  Each iteration:

1. **Particle shift** (every `nb_it_per_shift_cst` iterations): shift particle/field data along x, inject new solar wind particles, impose B-field boundary conditions.

2. **Moment deposit 1** — deposit mass and current at positions r^n:
   ```
   moments_mapping(pa_d, fields_d, ..., false, 1)
   ```

3. **Position advance** — Boris integrator, half-step:
   ```
   boris_pos_k<<< ... >>>(pa_d, simu_param_d, grid_d)
   ```

4. **MPI particle exchange** — send particles that have crossed MPI boundaries:
   ```
   comm_part(pa_h, pa_d, fields_h, fields_d, ...)
   ```

5. **Moment deposit 2** — deposit at new positions r^{n+1}:
   ```
   moments_mapping(pa_d, fields_d, ..., true, 2)
   ```

6. **Average moments**:
   ```
   average_moments_k<<< ... >>>(fields_d, ...)
   ```

7. **EM field update** — Ohm + Faraday subcycled + CAM predictor-corrector:
   ```
   ohm(fields_d, Ba_d, ...)
   faraday(Ba_d, fields_d, ...)     ← repeated nb_sub_cycles times
   predict_correct_k<<< ... >>>(Ba_d, Bb_d, ...)
   boundaries_EB(...)
   ohm(...)        ← E^{n+1} estimate
   current_advance_k<<< ... >>>()   ← CAM ion current
   ohm(...)        ← final E^{n+1}
   ```

8. **Velocity advance** — Boris integrator, full step:
   ```
   boris_vel_k<<< ... >>>(pa_d, fields_d, Ba_d, Bb_d, ...)
   ```

9. **Probe recording** and synchronization:
   ```
   push_probes_k<<< ... >>>()
   probe_fields_k<<< ... >>>()
   cudaDeviceSynchronize()
   MPI_Barrier(comm)
   ```

10. **I/O** (conditional on iteration counters):
    - Every `rate_save_t_cst` iterations: copy HK/moments from device → host, append to HK `.npy` files.
    - Every `rate_save_field_cst` iterations: copy fields + B from device → host, write field snapshot `.npy` files.
    - Every `rate_save_particles_cst` iterations: copy particles, write particle `.npy` file.

11. **Error checks**:
    ```
    check_grids(B, fields, sta_sol_h, sta_sol_d, ...)
    check_status(simu_param_h, sta_sol_h, sta_sol_d, comm, idx_it)
    ```

---

## The CAM algorithm

The Current Advance Method (CAM) is the core EM solver.  It uses a predictor-corrector approach:

- **Predict**: advance B from n → n+1 using the current Ohm's law E estimate.
- **Correct**: use B^{n+1} to compute a better E, then update the ion current (`current_advance_k`), then compute the final E^{n+1}.

The parameter `nb_sub_cycles = 11` controls how many Faraday sub-cycles are taken per main iteration (set in `parameters.h`).

---

## MPI decomposition

The simulation domain is divided across MPI processes in the **y and z directions** (not x).  The decomposition dimensions are set in `parameters.h`:

```cpp
const int mpi_nb_proc_y_cst = 4;   // MPI ranks in y
const int mpi_nb_proc_z_cst = 4;   // MPI ranks in z
```

Total MPI processes = `mpi_nb_proc_y_cst × mpi_nb_proc_z_cst = 16`.

Each rank owns a y–z slab of the full 3D grid.  Ghost cells (4 layers each side) are exchanged with `comm_fields` and `comm_part`.  Output files include the rank in their name: `dens_it100_rank_2_3.npy` is the density snapshot at iteration 100 for the rank at (y=2, z=3) in the process grid.

---

## Exercise

1. In which file is `MPI_Init` called?  What would happen if you ran `menura` without MPI (`./menura` instead of `mpirun ./menura`)?
2. What does `nb_it_per_shift_cst` control?  Find it in `parameters.h` and relate it to `nb_cell_per_shift_cst`.
3. During output, why is `cudaMemcpy(..., cudaMemcpyDeviceToHost)` needed before calling `output_fields`?
