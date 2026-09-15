# B-05 · Analyzing Simulation Outputs

Menura writes all output as NumPy `.npy` files using the bundled `cnpy` C++ library.  Analysis is done in Python with NumPy and the utilities in `analysis/menura_utils.py`.

---

## Output file locations

All files are written relative to the working directory in which `menura` is launched:

```
products/
├── HK/
│   ├── energy_elec_rank_0_0.npy
│   ├── energy_mag_rank_0_0.npy
│   ├── energy_kin_rank_0_0.npy
│   ├── rms_B_rank_0_0.npy
│   ├── B_mean_rank_0_0.npy
│   ├── B_var_rank_0_0.npy
│   ├── simu_time.npy              (rank 0 only)
│   ├── run_time.npy               (rank 0 only)
│   ├── active_part_0_rank_0_0.npy
│   └── active_part_1_rank_0_0.npy
├── probes/
│   └── probes_B_ID_0.npy, probes_E_ID_0.npy, ...
├── parameters.txt                 (written at startup, read by menura_utils.py)
├── dens_it100_rank_0_0.npy        (field snapshots)
├── dens_it200_rank_0_0.npy
├── B_it100_rank_0_0.npy
└── particles_FILE_it4000_rank_0_0.npy
```

The `rank_Y_Z` suffix identifies the MPI decomposition position.  For 16 processes (4×4), files appear as `rank_0_0` through `rank_3_3`.

---

## Reading `.npy` files directly

Any `.npy` file can be loaded with NumPy:

```python
import numpy as np

# House-keeping time series
energy_mag = np.load('products/HK/energy_mag_rank_0_0.npy')
simu_time  = np.load('products/HK/simu_time.npy')

# Field snapshot (single rank)
dens = np.load('products/dens_it100_rank_0_0.npy')
B    = np.load('products/B_it100_rank_0_0.npy')
```

The shapes depend on the parameters in use. For the defaults:
- HK arrays: shape `[len_save_t_cst]` = `[101]` (one entry per saved iteration + initial)
- Density snapshots: shape `[len_x_cst+4, len_y_cst//mpi_nb_proc_y + 4, len_z_cst//mpi_nb_proc_z + 4]`
- B-field snapshots: shape `[3, ...]` (x, y, z components)

---

## The `menura_utils.py` module

`analysis/menura_utils.py` provides helper classes for loading and working with Menura data.

> **Note:** `menura_utils.py` imports `own_tools` and `own_colours`, which are personal utility packages belonging to the code's authors and are not included in the Zenodo release. The import will fail at the top of the file. You have two options:
>
> 1. **Copy the relevant classes into your own script** — `menura_param`, `menura_grid`, and `menura_probes` do not use `own_tools` or `own_colours` internally, so they work fine if you define them yourself after removing the problematic imports.
> 2. **Create stub modules** so the import succeeds:
>    ```python
>    # In your analysis directory, create own_tools.py and own_colours.py
>    # as empty files or minimal stubs
>    ```

### `menura_param` — read simulation parameters

This class reads `products/parameters.txt` (written by Menura at startup) and exposes all parameters as attributes:

```python
from menura_utils import menura_param

mp = menura_param('/path/to/run/directory')

# All parameters.h values become attributes:
print(mp.len_x_cst)       # 400
print(mp.dX)              # 1.25
print(mp.dt)              # 0.1
print(mp.nb_it_max_cst)   # 1000
print(mp.mpi_nb_proc_tot) # 16
print(mp.d_i)             # ion inertial length in metres

# Derived time/space step for saved output:
print(mp.dt_low)          # dt * rate_save_t_cst
print(mp.dx_low)          # dX * rate_save_x_cst
```

`menura_param.print_physical_parameters()` prints a formatted table of all physical scales.

### `menura_grid` — coordinate arrays

```python
from menura_utils import menura_param, menura_grid

mp = menura_param('/path/to/run/directory')
mg = menura_grid('/path/to/run/directory', mp)

print(mg.grid_x.shape)     # (len_x_cst + 4,) — includes ghost cells
print(mg.grid_y.shape)     # (len_y_cst,)
print(mg.grid_t_low.shape) # time array for HK data
```

### `menura_probes` — virtual sensor time series

```python
from menura_utils import menura_probes

probes = menura_probes(local_path='/path/to/run', remote_path='', it=0)
# probes.B shape: (3, nb_probes_cst, nb_it_max_cst)
# probes.E shape: (3, nb_probes_cst, nb_it_max_cst)
# probes.dens shape: (nb_probes_cst, nb_it_max_cst)
```

---

## Assembling the full domain from MPI tiles

Each MPI rank writes a tile of the full domain.  To reconstruct the global field you must load and concatenate all rank files:

```python
import numpy as np

def load_field(products_path, field_name, it, nb_proc_y, nb_proc_z):
    """Load a field snapshot and assemble from MPI tiles."""
    tiles_y = []
    for ry in range(nb_proc_y):
        tiles_z = []
        for rz in range(nb_proc_z):
            fn = f'{products_path}/{field_name}_it{it:d}_rank_{ry}_{rz}.npy'
            tile = np.load(fn)
            tiles_z.append(tile)
        tiles_y.append(np.concatenate(tiles_z, axis=-1))  # concatenate along z
    return np.concatenate(tiles_y, axis=-2)                # concatenate along y

# Example: load density at iteration 100 from a 4×4 MPI run
dens = load_field('products', 'dens', 100, nb_proc_y=4, nb_proc_z=4)
print(dens.shape)  # roughly (404, 104, 104) including ghost cells
```

> **Ghost cells:** Each tile includes 4 ghost cells on each boundary (the "+4" in the array dimensions).  Strip them before stitching if you want the physical domain only:
> ```python
> tile_interior = tile[2:-2, 2:-2, 2:-2]   # remove 2-cell ghost layer on each side
> ```

---

## Plotting a field slice

```python
import numpy as np
import matplotlib.pyplot as plt

# Load parameters
products = 'products'
it = 100
ny, nz = 4, 4

# Load and assemble density
dens = load_field(products, 'dens', it, ny, nz)

# Remove ghost cells on x; take midplane in z
dens_xy = dens[2:-2, 2:-2, dens.shape[2]//2]

plt.figure(figsize=(10, 3))
plt.imshow(dens_xy.T, origin='lower', aspect='auto', cmap='plasma')
plt.colorbar(label='density (normalised)')
plt.xlabel('x [d_i]')
plt.ylabel('y [d_i]')
plt.title(f'Ion density — iteration {it}')
plt.tight_layout()
plt.savefig('dens_xy.png', dpi=150)
plt.show()
```

---

## Plotting house-keeping diagnostics

```python
import numpy as np
import matplotlib.pyplot as plt

e_mag = np.load('products/HK/energy_mag_rank_0_0.npy')
e_kin = np.load('products/HK/energy_kin_rank_0_0.npy')
t     = np.load('products/HK/simu_time.npy')

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, e_mag, label='Magnetic energy')
ax.plot(t, e_kin, label='Kinetic energy')
ax.set_xlabel('Time [1/Ω_ci]')
ax.set_ylabel('Energy (normalised)')
ax.legend()
ax.set_title('Energy evolution')
plt.tight_layout()
plt.savefig('energies.png', dpi=150)
```

---

## Loading particle data

```python
import numpy as np

# Particle output: shape [N_active, 7] — columns: rx, ry, rz, vx, vy, vz, ID
pa = np.load('products/particles_FILE_it4000_rank_0_0.npy')
rx, ry, rz = pa[:, 0], pa[:, 1], pa[:, 2]
vx, vy, vz = pa[:, 3], pa[:, 4], pa[:, 5]
species_id  = pa[:, 6].astype(int)

# Select only species 1 (cometary/planetary ions)
mask = (species_id == 1)
plt.scatter(rx[mask], ry[mask], s=1, alpha=0.3)
```

---

## The `parameters.txt` file

Menura writes a text file at startup that `menura_param` reads.  It lists every parameter from `simu_param` as `name,value` pairs, one per line.  If you need to read parameters without using `menura_utils.py`:

```python
import numpy as np
p = np.recfromtxt('products/parameters.txt')
params = {t[0].decode('UTF-8'): float(t[1]) for t in p}
print(params['dX'])      # 1.25
print(params['dt'])      # 0.1
```

---

## Exercise

1. Write a Python function that loads all HK energy files across all ranks and sums them to get the global energy time series.
2. What is the physical time corresponding to iteration 500 in a default run (in seconds)?  Use `t = it * dt / omega_ci`.
3. The particle file has shape `[N, 7]`.  What does column 6 contain, and what values can it take?
4. How would you find the number of active cometary ions as a function of time?  (Hint: look at `active_part_1_rank_Y_Z.npy`.)
