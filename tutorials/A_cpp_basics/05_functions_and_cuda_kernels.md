# A-05 · Functions and CUDA Kernels

Menura's physics is split between regular C++ functions (called from the CPU) and CUDA kernels (executed on the GPU).  Understanding the distinction is essential for reading the main loop in `menura.cu`.

---

## C++ functions

A function has a return type, a name, and a parameter list:

```cpp
// utils.h — a utility function with default arguments
void check_status(simu_param* sP, state_solver* sta_sol_h,
                  state_solver* sta_sol_d, MPI_Comm comm,
                  int idx_it, int tag=0);
```

The `tag=0` is a *default argument* — you can omit it at the call site.  The `void` return type means the function returns nothing.

### Function declarations vs. definitions

Menura separates declarations (in `.h` header files) from definitions (in `.cpp`/`.cu` files):

```cpp
// functions_o.h — declaration
void output_fields(simu_fields* fields, int indIt, simu_param* sP, bool full=true);

// functions_o.cpp — definition
void output_fields(simu_fields* fields, int indIt, simu_param* sP, bool full) {
    // ... actual code using cnpy to write .npy files ...
}
```

This allows a `.cu` file to `#include "functions_o.h"` and call `output_fields` without seeing its implementation.

---

## CUDA kernel qualifiers

CUDA adds three *function qualifiers* that control where a function runs and where it can be called from:

| Qualifier | Runs on | Called from |
|-----------|---------|-------------|
| `__global__` | GPU | CPU (host) only |
| `__device__` | GPU | GPU only |
| `__host__` | CPU | CPU only (default if omitted) |

Menura's field and particle kernels are declared in `kernels_calls.h` and defined in `.cuh` files:

```cpp
__global__ void boris_pos_k(particles* pa, simu_param* sP, simu_grid* grid);
__global__ void boris_vel_k(particles* pa, simu_fields* fields,
                            simu_B_field* Ba, simu_B_field* Bb,
                            simu_param* sP);
__global__ void moments_mapping_k(particles* pa, simu_fields* fields,
                                  simu_param* sP, bool second_call,
                                  int step_id);
```

---

## Kernel launch syntax

```cpp
kernel_name<<< gridDim, blockDim >>>(arg1, arg2, ...);
```

- `blockDim` — threads per block (Menura uses `tpb_cst = 256` throughout)
- `gridDim` — number of blocks; chosen to cover all work items

For a loop over `nb_nodes_cst` grid nodes:

```cpp
// menura.cu
check_grids_k<<< 1 + nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(B, fields, sta_sol_d, ...);
```

`1 + nb_nodes_tot_cst/tpb_cst` ensures enough blocks even if `nb_nodes_tot_cst` is not a multiple of `tpb_cst` (integer division rounds down, so `+1` pads).

Inside the kernel each thread computes its global index:

```cpp
__global__ void boris_pos_k(particles* pa, ...) {
    int ip = blockIdx.x * blockDim.x + threadIdx.x; // particle index
    if (ip >= pool_size_cst) return;                 // guard against overflow
    if (!pa->active[ip]) return;                     // skip inactive slots
    // ... update pa->rx[ip], pa->ry[ip], pa->rz[ip] ...
}
```

---

## The main physics functions

Looking at `functions_i.h` and `functions_o.h`, the solver uses three families of functions:

### Initialisation (CPU, `functions_i.h`)

```cpp
void init_grid      (simu_grid* grid,   simu_param* sP);
void init_fields    (simu_fields* fields, simu_param* sP);
void init_B         (simu_B_field* B,   simu_param* sP);
void init_part_node (particles* p, simu_grid* grid, simu_param* sP);
void init_probes    (probes* prob, simu_param* sP, simu_grid* grid);
```

These run once before the main loop and fill host-side structs.

### Solver kernels (GPU, `kernels_*.cuh`)

The main loop calls CUDA kernels via wrappers.  Examples from the solver sequence in `menura.cu`:

```cpp
moments_mapping(pa_d, fields_d, simu_param_d, false, 1);  // deposit moments
boris_pos_k<<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(pa_d, simu_param_d, grid_d);
comm_part(pa_h, pa_d, fields_h, fields_d, ...);           // MPI exchange
moments_mapping(pa_d, fields_d, simu_param_d, true, 2);
average_moments_k<<< ... >>>(fields_d, ...);
ohm(fields_d, Ba_d, simu_param_d);
faraday(Ba_d, fields_d, simu_param_d);
boris_vel_k<<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(pa_d, fields_d, Ba_d, Bb_d, simu_param_d);
```

### Output (CPU, `functions_o.h`)

```cpp
void output_fields   (simu_fields* fields, int indIt, simu_param* sP);
void output_B_field  (simu_B_field* B,     int indIt, simu_param* sP);
void output_HK       (house_keeping* HK, trajectory* traj, moments* mom,
                      root_mean_sqr* rms, simu_param* sP, int idx_save);
void output_probes   (probes* prob, simu_param* sP);
void output_particles(particles* p, int indIt, simu_param* sP);
```

These run after `cudaMemcpy` has transferred the device data to the host, and write `.npy` files using the `cnpy` library.

---

## `sortie()` — clean exit

```cpp
// utils.h
void sortie() {
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "| Device " << id << " Off." << std::endl;
    MPI_Finalize();
    exit(EXIT_FAILURE);
}
```

Called whenever a fatal error is detected (failed pre-check, CUDA error, etc.).  `MPI_Finalize()` must be called before exiting to release MPI resources; without it, other MPI processes may hang.

---

## Exercise

1. Kernel `boris_pos_k` is launched with `<<< 1+pool_size_cst/tpb_cst, tpb_cst >>>`.  If `pool_size_cst = 2000000` and `tpb_cst = 256`, how many thread blocks are launched?  How many total threads?
2. Why does the kernel begin with `if (ip >= pool_size_cst) return;`?
3. Find `output_fields` in `functions_o.cpp`.  What cnpy function is called to write an array?  What is the output filename pattern?
4. Why must `MPI_Finalize()` be called before `exit()` in `sortie()`?
