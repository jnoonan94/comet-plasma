# A-01 · Variables, Types, and Compile-Time Constants

In C++ every value has a *type* that tells the compiler how many bytes to allocate and how to interpret the bits.  Menura uses plain C++ types throughout its source files.

---

## Fundamental types used in Menura

| Type | Size | What it holds in Menura |
|------|------|-------------------------|
| `int` | 4 bytes | iteration counters, grid indices, MPI rank IDs |
| `float` | 4 bytes | all physical field values (E, B, position, velocity) |
| `bool` | 1 byte | flags like `active[i]` (is this particle slot occupied?) |
| `size_t` | 8 bytes (64-bit) | byte counts for `cudaMemGetInfo` |

Menura uses `float` (32-bit) rather than `double` (64-bit) for all particle and field arrays because GPU throughput for 32-bit arithmetic is much higher than for 64-bit.

---

## Compile-time constants: `#define` and `const`

Menura has **no text input file**.  Every parameter — grid size, species count, time-step limit — is a compile-time constant declared in `menura/src/parameters.h`.  To change a parameter you edit that file and recompile.

There are two syntaxes used in `parameters.h`:

### 1. Preprocessor `#define`

```cpp
// parameters.h
#define NB_DIM 3
```

`#define` performs a textual substitution before the compiler sees the code.  `NB_DIM` is not a variable; it has no address, no type, and no scope.  Menura uses it to switch between 2-D and 3-D code paths:

```cpp
// structures.h — conditional compilation
#if NB_DIM==3
    float rz[pool_size_cst];   // z-position only exists in 3D
#endif
```

### 2. `const` variables

```cpp
// parameters.h
const int len_x_cst = 400;   // grid nodes along x
const int len_y_cst = 100;   // grid nodes along y
const int len_z_cst = 100;   // grid nodes along z
const int nb_it_max_cst = 1000; // maximum number of iterations
const int tpb_cst = 256;     // CUDA threads per block
```

Unlike `#define`, a `const` variable has a type.  The compiler can catch type errors, and debuggers can display the value.  In Menura all compile-time integer constants follow the naming convention `*_cst`.

---

## Feature flags

`parameters.h` also contains boolean constants that turn physics features on or off:

```cpp
const bool obstacle_cst   = true;   // include a solid body obstacle
const bool solid_body_cst = true;   // treat obstacle as solid (absorbing)
const bool dipole_cst     = true;   // add a permanent magnetic dipole
const bool inject_pla_cst = false;  // inject cometary (planetary) ions
const bool decay_turb_cst = false;  // decaying turbulence initial condition
const bool periodic_yz_cst= false;  // periodic boundaries in y and z
const bool restart_cst    = false;  // restart from a saved snapshot
```

These are evaluated at compile time: a `#if dipole_cst` block is compiled in (or out) entirely.  Changing a flag requires a full rebuild (`make clean && make`).

---

## Runtime parameters inside `simu_param`

Some values are computed from the compile-time constants at run-time and stored in the `simu_param` struct.  These include normalized physical quantities derived from the SI reference values that appear at the top of `parameters.h`:

```cpp
// parameters.h — SI reference values (examples)
const float n0_SI = 3e6;    // background ion number density [m^-3]
const float B0_SI = 3e-9;   // background magnetic field [T]
const float u0    = 1000;   // solar wind bulk speed [m/s]
```

The normalization (ion inertial length `d_i`, ion cyclotron frequency `Omega_i`, etc.) is applied in `init_house_keeping` (declared in `functions_i.h`) to populate `simu_param` at the start of each run.

---

## Exercise

Open `menura/src/parameters.h`.  Find and record:
1. The value of `pool_size_cst` — how many particle slots are allocated per MPI process?
2. The value of `rate_save_field_cst` — how often (in iterations) are field snapshots written?
3. What is `path_inputs_str_cst`?  What directory does it point to, and when is it used?
