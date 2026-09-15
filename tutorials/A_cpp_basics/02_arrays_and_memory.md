# A-02 · Arrays and Memory Layout

Plasma simulations are data-intensive.  Understanding how Menura lays out its arrays is essential for reading and post-processing outputs.

---

## Stack vs. heap

C++ arrays can live on the *stack* (automatic storage, size known at compile time) or the *heap* (dynamic allocation with `new`/`malloc`, or `cudaMalloc` for GPU memory).

All of Menura's large data structures — fields, particles, B-field — use **compile-time sizes** (constants from `parameters.h`) and are allocated on the heap as whole structs via `cudaMalloc` (device) and `new` (host):

```cpp
// menura.cu — simplified
particles* pa_h = new particles;          // host copy
particles* pa_d;
cudaMalloc((void**)&pa_d, sizeof(particles)); // device copy
```

---

## Multi-dimensional C arrays

C++ stores multi-dimensional arrays in **row-major** (C) order: the last index varies fastest in memory.

```cpp
// structures.h — 3D electric field
float E[3][len_x_cst+4][len_y_cst+4][len_z_cst+4];
```

This is a contiguous block of `3 × (len_x+4) × (len_y+4) × (len_z+4)` floats.  The first index is the component (0=x, 1=y, 2=z), and the remaining three are spatial indices.

### The "+4" ghost cells

Every spatial dimension is padded by 4 extra cells on each side (`+4` total).  These *ghost cells* hold values communicated from neighbouring MPI ranks (or periodic copies) so that stencil operations at the domain boundary don't require special-casing inside the kernel.

With the default parameters:

```
len_x_cst = 400  →  E[3][404][104][104]
len_y_cst = 100
len_z_cst = 100
```

---

## The `particles` struct: Structure of Arrays (SoA)

Many codes store one particle as a struct (`{float rx, ry, rz, vx, vy, vz; ...}`).  Menura uses the opposite layout — a **Structure of Arrays** — where each physical quantity is a separate flat array:

```cpp
// structures.h
struct particles
{
  float rx     [pool_size_cst]; // x-positions of all particles
  float ry     [pool_size_cst];
  float rz     [pool_size_cst]; // only in 3D (#if NB_DIM==3)
  float vx     [pool_size_cst];
  float vy     [pool_size_cst];
  float vz     [pool_size_cst];
  bool  active [pool_size_cst]; // true = slot is occupied
  int   ID     [pool_size_cst]; // species ID (0 = solar wind, 1 = cometary/planetary)
};
```

`pool_size_cst` is the total number of particle slots per MPI process (active + free).  On the GPU, threads accessing `rx[i], rx[i+1], rx[i+2]...` in adjacent slots benefit from *coalesced memory access* — the reads hit the same cache line, which is essential for throughput.

If you loaded the particle output file (shape `[N, 7]`), the seven columns correspond to: `rx, ry, rz, vx, vy, vz, ID`.

---

## The `simu_grid` struct

```cpp
// structures.h
struct simu_grid
{
  float xGrid[len_x_cst+4]; // x-coordinates of all nodes (including ghosts)
  float yGrid[len_y_cst+4];
  float zGrid[len_z_cst+4]; // 3D only
  float xMin; float xMax;   // particle domain bounds
  float yMin; float yMax;
  float zMin; float zMax;
  float centre_x;           // obstacle centre position
  float centre_y;
  float centre_z;
  float rsq[len_x_cst+4][len_y_cst+4][len_z_cst+4]; // |r - r_obstacle|^2
};
```

The `xGrid` array gives the physical coordinate (in units of `d_i`) of each grid node along x.  Because the grid is regular, `xGrid[i+1] - xGrid[i]` is a constant equal to `dx`.

---

## Host and device copies

Menura maintains two copies of every large struct: one on the host (`*_h`) and one on the device (`*_d`):

```cpp
particles* pa_h;   // CPU — used for I/O
particles* pa_d;   // GPU — used during the solver loop
simu_fields* fields_h;
simu_fields* fields_d;
simu_B_field* Ba_h;   // B at time level n
simu_B_field* Ba_d;
simu_B_field* Bb_h;   // B at time level n+1 (CAM predictor)
simu_B_field* Bb_d;
```

Data is transferred with `cudaMemcpy` only when needed — typically when writing output or when the host must inspect a value:

```cpp
cudaMemcpy(fields_h, fields_d, sizeof(*fields_h), cudaMemcpyDeviceToHost);
```

---

## Exercise

1. Given `len_x_cst=400`, `len_y_cst=100`, `len_z_cst=100`, what is the total number of floats in `simu_fields.E`?  How many megabytes is that?
2. Explain why SoA is preferred over AoS for GPU computation.
3. In the output file `dens_it100_rank_0_0.npy`, what does "rank_0_0" signify?
