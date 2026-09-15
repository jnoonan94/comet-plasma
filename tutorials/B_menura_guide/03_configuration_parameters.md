# B-03 · Configuring a Simulation via `parameters.h`

**Menura has no runtime input file.**  Every parameter is a compile-time constant or struct member default in `menura/src/parameters.h`.  To change any parameter you edit this file and rebuild.

---

## Workflow

```
1. Edit menura/src/parameters.h
2. cd menura-master/menura
3. make clean && make
4. mkdir -p products/HK products/probes   (if not already present)
5. mpirun -np <N> ./menura
```

`make` (without `clean`) will only recompile files whose dependencies changed, but because many `.cuh` kernel files include `parameters.h`, a `make clean` rebuild is safer when changing physics flags.

---

## Grid and resolution

```cpp
#define NB_DIM 3                  // 2 or 3 spatial dimensions

const int len_x_cst = 400;       // grid nodes along x (solar wind direction)
const int len_y_cst = 100;       // grid nodes along y
const int len_z_cst = 100;       // grid nodes along z (3D only)

const int mpi_nb_proc_y_cst = 4; // MPI ranks in y (must divide len_y evenly)
const int mpi_nb_proc_z_cst = 4; // MPI ranks in z (must divide len_z evenly)
```

Within `simu_param`, the physical spacing and time step are:

```cpp
struct simu_param {
  float dX = 1.25;  // node spacing in units of ion inertial length d_i
  float dt = 0.1;   // time step in units of 1/Omega_ci (inverse ion gyrofrequency)
  ...
};
```

So the physical domain size is `len_x_cst * dX` ion inertial lengths.  With the defaults: 400 × 1.25 = 500 d_i along x.

---

## Time control

```cpp
const int nb_it_max_cst           = 1000;  // total iterations to run
const int rate_save_t_cst         = 10;    // save HK diagnostics every N iterations
const int rate_save_field_cst     = 100;   // save field snapshots every N iterations
const int rate_save_particles_cst = 4000;  // save particle data every N iterations
```

The simulation ends after exactly `nb_it_max_cst` iterations.  For a restart, set `restart_cst = true` and `idx_it_restart_cst` to the saved iteration number.

---

## Particles per cell

```cpp
const int nb_part_node_cst = 40;  // macro-particles per grid node at initialization
```

This sets the statistical resolution — more particles per cell reduces noise in moment estimates but increases memory and compute cost.  The total pool size is:

```cpp
pool_size_cst = 1.2 * nb_part_node_cst * len_x_cst * len_y_cst * len_z_cst
```

(The 1.2 factor provides 20 % headroom for particle injection and MPI buffer transfers.)

---

## Physics flags

These `#define` switches turn entire code sections on or off at compile time:

```cpp
#define solve_EB_cst    true  // solve E and B self-consistently (always true for production)
#define obstacle_cst    true  // include a solid body obstacle in the domain
#define solid_body_cst  true  // obstacle absorbs particles (solid body boundary)
#define dipole_cst      true  // permanent magnetic dipole at obstacle centre
#define inject_pla_cst  false // inject cometary/planetary secondary ions
#define ionosphere_cst  false // inject ionospheric ions (planet only)
#define decay_turb_cst  false // decaying turbulence initial condition
#define periodic_yz_cst false // periodic y–z boundaries (for turbulence studies)
#define ORF_cst         false // obstacle reference frame (obstacle moves, domain fixed)
#define restart_cst     false // restart from a saved snapshot
#define DF_cst          false // track and save a 3D distribution function
#define yee_cst         false // use Yee staggered grid (experimental)
```

### Planetary body simulation (default)

The defaults (`obstacle_cst=true`, `solid_body_cst=true`, `dipole_cst=true`, `inject_pla_cst=false`) configure a **magnetized planetary body** — for example a Mars-like or Mercury-like body.

### Cometary simulation

To simulate a comet in the solar wind, change:

```cpp
#define obstacle_cst    true   // keep, comet is the "obstacle"
#define solid_body_cst  false  // comet is not solid — particles pass through
#define dipole_cst      false  // comets have no intrinsic dipole
#define inject_pla_cst  true   // inject cometary ions (photo-ionization product)
```

Also adjust the cometary production parameters in `simu_param`:

```cpp
float Q    = 5.e26;   // outgassing rate [molecules/s]
float nu_i = 2.e-7;   // photo-ionization frequency [s^-1]
float u0   = 1000.;   // neutral outflow speed [m/s]
float r_obs = 30;     // obstacle radius in d_i (used for the exosphere source region)
```

---

## Background plasma parameters

These are inside the `simu_param` struct and are edited directly in `parameters.h`:

```cpp
struct simu_param {
  // --- background solar wind (SI values used for normalization) ---
  float n0_SI  = 3.e6;    // upstream ion number density [m^-3]
  float B0_SI  = 3.e-9;   // upstream magnetic field strength [T]
  float beta   = 1.;      // plasma beta = 2μ₀nkT/B²

  // --- field orientation ---
  float B0_x = 0.;        // upstream B along x (radial, 0 = perpendicular IMF)
  float B0_y = 1.;        // upstream B along y (normalized)
  float B0_z = 0.;

  // --- resistivity / diffusion ---
  float eta_hyp_res = 1.0; // hyper-resistivity coefficient
  float eta_res_obs = 1.0; // resistivity inside the obstacle region

  // --- grid and time (normalized) ---
  float dX = 1.25;         // cell size [d_i]
  float dt  = 0.1;         // time step [1/Ω_ci]
```

Derived quantities (Alfvén speed `v_A`, ion inertial length `d_i`, cyclotron frequency `omega_ci`, thermal speed `v_thi`, etc.) are computed automatically from the SI values using C++ member initializers — you do not set them directly.

---

## Dipole moment

```cpp
float dip_mom_SI[3] = {0., -1.e13, 0.};  // magnetic dipole moment [T·m³]
float r_obs = 30;                          // obstacle radius [d_i]
```

The dipole is centered at (`centre_x`, `centre_y`, `centre_z`) expressed as fractions of the domain length (default 0.5 = domain centre).

---

## Shifting parameters

The simulation uses a "shifting" technique to keep the obstacle at a fixed position in the frame while the solar wind flows through:

```cpp
const int nb_it_per_shift_cst   = 2;  // shift occurs every N iterations
const int nb_cell_per_shift_cst = 1;  // shift by N cells per shift event
```

The constraint `nb_cell_per_shift_cst <= nb_it_per_shift_cst` is enforced by `pre_checks()`.  This ensures cometary ions cannot cross more than one cell during the interval between shifts.

---

## Output thinning

For large 3D runs it is common to save field snapshots at a reduced spatial resolution.  `parameters.h` defines thinning factors:

```cpp
const int rate_save_x_cst = 2;  // save every 2nd x node
const int rate_save_y_cst = 2;  // save every 2nd y node
```

The `menura_utils.py` analysis script reads these via the `parameters.txt` file that Menura writes at startup.

---

## Exercise

1. What is the total simulation time in units of `1/Ω_ci` for the default parameters?  In seconds (using `omega_ci` computed from `n0_SI` and `B0_SI`)?
2. How would you set up a 2D test run with 1 MPI process and a smaller domain (100 × 50 nodes)?  List every line in `parameters.h` you would change.
3. The default `eta_hyp_res = 1.0` applies hyper-resistivity in the obstacle region.  Setting it to 0 would make the obstacle resistivity-free.  Why might that cause numerical problems?
