# A-04 · Pointers and CUDA Memory Management

Menura runs on a hybrid CPU+GPU system.  The CPU (host) controls program flow and I/O; the GPU (device) performs the computationally expensive kernel launches.  Managing memory on two separate address spaces is a central challenge of CUDA programming.

---

## Pointer basics

A pointer is a variable that stores a memory address:

```cpp
float* p;          // p can hold the address of a float
float x = 3.14f;
p = &x;            // & is the "address-of" operator
float val = *p;    // * dereferences: val == 3.14
```

For structs, the arrow operator `->` combines deref and member access:

```cpp
particles* pa = new particles;
pa->rx[0] = 1.5f;   // equivalent to (*pa).rx[0] = 1.5f
```

---

## Host allocation

Menura allocates host structs with `new`:

```cpp
// menura.cu (simplified)
particles*   pa_h    = new particles;
simu_fields* fields_h= new simu_fields;
simu_B_field* Ba_h   = new simu_B_field;
simu_B_field* Bb_h   = new simu_B_field;
```

`new` allocates on the heap.  The object persists until explicitly freed with `delete pa_h;` (or until the process exits).

---

## Device allocation

GPU memory lives in a separate physical memory space (VRAM).  `cudaMalloc` allocates it:

```cpp
particles*   pa_d;
simu_fields* fields_d;
simu_B_field* Ba_d;
simu_B_field* Bb_d;

cudaMalloc((void**)&pa_d,     sizeof(particles));
cudaMalloc((void**)&fields_d, sizeof(simu_fields));
cudaMalloc((void**)&Ba_d,     sizeof(simu_B_field));
cudaMalloc((void**)&Bb_d,     sizeof(simu_B_field));
```

The convention `(void**)&pa_d` casts the address of the pointer to the generic pointer-to-pointer type that `cudaMalloc` requires.

---

## Copying data between host and device

`cudaMemcpy` transfers data between host and device:

```cpp
// Copy host → device (before solver loop)
cudaMemcpy(pa_d, pa_h, sizeof(*pa_h), cudaMemcpyHostToDevice);

// Copy device → host (for output)
cudaMemcpy(fields_h, fields_d, sizeof(*fields_h), cudaMemcpyDeviceToHost);
```

`sizeof(*pa_h)` is the total byte size of the entire struct — including all its embedded arrays.  Because the struct contains only value types (plain arrays), this single `memcpy` correctly transfers everything.

---

## GPU assignment to MPI ranks

Each MPI rank drives exactly one GPU.  Menura reads the `SLURM_LOCALID` environment variable (set by the SLURM scheduler) to determine which GPU the current process should use:

```cpp
// utils.h — initialisation_cuda()
char* local_rank_env = getenv("SLURM_LOCALID");
int   local_rank     = atoi(local_rank_env);
cudaSetDevice(local_rank);
```

`cudaSetDevice` must be called before any CUDA API call.  After this, all `cudaMalloc`, `cudaMemcpy`, and kernel launches in that process go to the specified GPU.

This is also why the SLURM job script (`go_menura`) requests one GPU per MPI task:

```bash
#SBATCH --gres=gpu:k80:1,gpuexcl
```

---

## Checking CUDA errors

After every significant CUDA operation Menura checks for errors:

```cpp
// utils.h — check_status()
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    sta_sol_h->break_solver = true;
    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
}
MPI_Allreduce(&sta_sol_h->break_solver, &sta_sol_h->break_solver,
              1, MPI_C_BOOL, MPI_LOR, comm);
if (sta_sol_h->break_solver) { /* log memory info and exit */ }
```

The `MPI_Allreduce` with `MPI_LOR` (logical OR) ensures that if *any* rank hit an error, *all* ranks know about it and exit cleanly rather than hanging waiting for a rank that died.

`check_memory_device()` can also be called at tagged checkpoints to print free/total GPU memory for each rank:

```cpp
// utils.h
void check_memory_device(int rank, int tag) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout << "Tag " << tag << ", rank " << rank
              << ", free=" << free/1000000 << " MB" << std::endl;
}
```

---

## `cudaDeviceSynchronize`

CUDA kernel launches are asynchronous: the CPU continues while the GPU is still running.  `cudaDeviceSynchronize()` blocks the CPU until all pending GPU operations on the current device complete:

```cpp
// menura.cu — main loop
cudaDeviceSynchronize();
MPI_Barrier(comm);
```

This pair appears at the end of each iteration to ensure all ranks are synchronized before the next I/O or MPI communication step.

---

## Exercise

1. What would happen if `cudaSetDevice` was called with an invalid device number?
2. Why is `MPI_Barrier` called immediately after `cudaDeviceSynchronize`?
3. In `check_status()`, why does `MPI_Allreduce` use `MPI_LOR` instead of `MPI_SUM`?
4. If `sizeof(particles)` is approximately 800 MB (with a large `pool_size_cst`), how much total device memory does Menura need just for the particle struct (remember `pa_d` only — one copy)?
