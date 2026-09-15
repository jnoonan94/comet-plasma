# A-03 · Structs and Classes in Menura

C++ `struct` and `class` bundle related data (and optionally functions) into a single named type.  Menura uses structs extensively to group the many arrays that make up the simulation state.

---

## `struct` basics

```cpp
struct simu_grid
{
  float xGrid[len_x_cst+4];
  float yGrid[len_y_cst+4];
  float zGrid[len_z_cst+4];
  float xMin; float xMax;
  float yMin; float yMax;
  float zMin; float zMax;
  float centre_x;
  float centre_y;
  float centre_z;
  float rsq[len_x_cst+4][len_y_cst+4][len_z_cst+4];
};
```

*All members are public by default* in a `struct`.  You access them with dot notation:

```cpp
simu_grid* grid_h = new simu_grid;
grid_h->xMin = 0.0f;      // pointer → member
float x0 = grid_h->xGrid[2];
```

---

## Key structs in Menura

### `simu_fields` — all electromagnetic fields except B

This is the largest struct.  In 3D it contains arrays like:

```cpp
float E    [3][len_x_cst+4][len_y_cst+4][len_z_cst+4]; // electric field
float Ji   [3][len_x_cst+4][len_y_cst+4][len_z_cst+4]; // ion current
float J_tot[3][len_x_cst+4][len_y_cst+4][len_z_cst+4]; // total current = curl(B)
float density [len_x_cst+4][len_y_cst+4][len_z_cst+4]; // ion number density
float pres    [len_x_cst+4][len_y_cst+4][len_z_cst+4]; // thermal pressure
int region_ID [len_x_cst+4][len_y_cst+4][len_z_cst+4]; // 0=plasma, 1=vacuum, 2=solid
```

It also includes MPI communication buffers:

```cpp
float buff_send_1d_y [4*(len_x_cst+4)*(len_z_cst+4)];
float buff_reci_1d_y [4*(len_x_cst+4)*(len_z_cst+4)];
float buff_send_1d_z [4*(len_x_cst+4)*(len_y_cst+4)];
float buff_reci_1d_z [4*(len_x_cst+4)*(len_y_cst+4)];
```

### `simu_B_field` — the magnetic field

```cpp
struct simu_B_field
{
  float B[3][len_x_cst+4][len_y_cst+4][len_z_cst+4];
  // if dipole_cst is true:
  float phi  [len_x_cst+4][len_y_cst+4][len_z_cst+4]; // magnetic scalar potential
  float B_dip[3][len_x_cst+4][len_y_cst+4][len_z_cst+4]; // permanent dipole field
};
```

Menura allocates **two** instances: `Ba` (time level n) and `Bb` (time level n+1), which are swapped in the CAM predictor-corrector scheme.

### `particles` — all macro-particles

```cpp
struct particles
{
  float rx[pool_size_cst];
  float ry[pool_size_cst];
  float rz[pool_size_cst];    // 3D only
  float vx[pool_size_cst];
  float vy[pool_size_cst];
  float vz[pool_size_cst];
  bool  active[pool_size_cst];
  int   ID[pool_size_cst];    // 0 = solar wind, 1 = cometary/planetary
};
```

### `house_keeping` — diagnostics

```cpp
struct house_keeping
{
  // energy arrays, one entry per save iteration
  float energy_elec[nb_it_save_cst];
  float energy_mag [nb_it_save_cst];
  float energy_kin [nb_it_save_cst];
  ...
};
```

### `probes` — virtual point-measurement sensors

```cpp
struct probes
{
  float B      [nb_probes_cst][3][nb_it_max_cst];
  float E      [nb_probes_cst][3][nb_it_max_cst];
  float density[nb_probes_cst]   [nb_it_max_cst];
  float Ji     [nb_probes_cst][3][nb_it_max_cst];
  float J_tot  [nb_probes_cst][3][nb_it_max_cst];
  float rx[nb_probes_cst];  // probe positions
  float ry[nb_probes_cst];
  float rz[nb_probes_cst];
  bool  active[nb_probes_cst];
};
```

Probes record field values at fixed spatial positions every iteration without requiring a field snapshot to be written to disk.

### `state_solver` — error tracking

```cpp
struct state_solver
{
  int   error_ID;
  bool  break_solver;
  std::string cuda_error_str;
  std::string error_str2[20];
};
```

`check_status()` in `utils.h` calls `MPI_Allreduce` to propagate `break_solver = true` across all ranks if any rank detects a CUDA error, which causes a clean exit.

### `simu_tank` — Phase 1 snapshot for injection

`simu_tank` stores a snapshot of fields and ordered particles from Phase 1 (uniform solar wind run).  During Phase 2 (the full simulation), the code reads from this tank slice-by-slice to inject solar wind particles and boundary conditions at the inflow face.

---

## Class with constructor: `simu_B_field_2`

`structures.h` also defines a `class` with a constructor for a heap-allocated alternative B-field storage:

```cpp
class simu_B_field_2
{
public:
  float*** B;
  simu_B_field_2(simu_param* sP) {
    B = new float**[3];
    for (int h=0; h<3; h++) {
      B[h] = new float*[sP->yLen];
      for (int i=0; i<sP->xLen+4; i++)
        B[h][i] = new float[sP->yLen];
    }
  }
};
```

This demonstrates the alternative pattern for dynamic allocation when the sizes are not known at compile time — not used in the main solver, but illustrative of how the `new` operator works for multi-dimensional arrays.

---

## Passing structs to CUDA kernels

CUDA kernels receive pointers to device-side structs.  Inside the kernel, member access is identical to host code:

```cpp
__global__ void some_kernel(simu_fields* fields, simu_B_field* B) {
  int node = blockIdx.x * blockDim.x + threadIdx.x;
  float bx = B->B[0][i][j][k];
  float ey = fields->E[1][i][j][k];
}
```

The pointer `fields` must point to GPU memory allocated with `cudaMalloc`.

---

## Exercise

1. Look at `structures.h`: what fields does `simu_param` contain (not shown above)?  What is its role?
2. Why does `simu_fields` contain *two* density arrays (`density` and `density_b`)?  (Hint: look at the solver-loop comments in `menura.cu`.)
3. What does `active[i] = false` mean for a particle in the `particles` struct?
