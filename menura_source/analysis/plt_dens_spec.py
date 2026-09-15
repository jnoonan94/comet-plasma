import numpy as np
import matplotlib.pyplot as plt
import own_tools as oT
import own_colours as oC
from leo_utils import *

path = '../leo_mpi'

save_plots = False
if save_plots:
    indices = np.arange(0, 1001, 10, dtype=int) # [600]
else:
    if len(sys.argv)>1:
        indices = [sys.argv[1]]
    else:
        indices = [1000]# np.linspace(0, 1000, 10, dtype=int)

lp = leo_param(f'{path}/src', file_name='parameters.h', print_param=False)
lp.print_physical_parameters()
lg = leo_grid(f'{path}/products', lp)

for i in indices:

    dens     = np.load(f'{path}/products/dens_it{i}_rank0.npy')
    dens_sw  = np.load(f'{path}/products/dens_spec0_it{i}_rank0.npy')
    dens_com = np.load(f'{path}/products/dens_spec1_it{i}_rank0.npy')

    if 1:
        fig, AX = plt.subplots(1, 2, figsize=(18, 14))
        for ax in AX:
            ax.set_aspect('equal')

        p0 = AX[0].pcolormesh(lg.grid_x, lg.grid_y, np.log10(dens_sw).T,
                              vmin=-1, vmax=1,
                         cmap=oC.bwr_2, rasterized=True)

        p1 = AX[1].pcolormesh(lg.grid_x, lg.grid_y, np.log10(dens_com).T,
                              vmin=-6, vmax=0,
                         cmap=oC.wbr_1, rasterized=True)

        t = np.linspace(-np.pi/2, 1.1*np.pi/2, 100)
        R = lp.Z_pla*lp.v0_com/lp.v_A
        drift = - (t-t[0])*R
        # AX[1].plot(R*np.cos(t) + lg.grid_x[int(lg.len_x/2)] + drift, R*np.sin(t) + R + lg.grid_y[int(lg.len_y/2)], '--k')

        # posAx = AX[0].get_position()
        # cax = fig.add_axes([posAx.x1*1., posAx.y0, 0.02, .3])
        # cb = fig.colorbar(p0, cax=cax, orientation='vertical')
        # cb.set_label('n SW', rotation=0, ha='left')
        # posAx = AX[1].get_position()
        # cax = fig.add_axes([posAx.x1*1., posAx.y0, 0.02, .3])
        # cb = fig.colorbar(p1, cax=cax, orientation='vertical')
        # cb.set_label('n com', rotation=0, ha='left')

        oT.set_spines(AX)

        if save_plots:
            plt.savefig(f'{path}/products/plots/dens_spec_{i:05d}.png')
            plt.close()
            # convert -delay 10 -resize x500 -loop 0 *.png leoTurbul.gif
        else:
            plt.show()

    if 1:

        fig, ax = plt.subplots(1, 1, figsize=(14, 14))

        ax.set_aspect('equal')

        ax.pcolormesh(lg.edges_x, lg.edges_y, (dens).T,
                      vmin=.7, vmax=1.3,
                         cmap=oC.bwr_2, rasterized=True)

        oT.set_spines(ax)
        plt.tight_layout()

        if save_plots:
            plt.savefig(f'{path}/products/plots/dens_{i:05d}.png')
            plt.close()
            # convert -delay 10 -resize x500 -loop 0 *.png leoTurbul.gif
        else:
            plt.show()
