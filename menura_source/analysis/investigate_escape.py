import sys
import os

import numpy as np
import matplotlib.pyplot as plt

import own_tools as oT
import own_colours as oC

from menura_utils import *

r''' Plot house-keepings. '''

it = 6000

run_ID = '016'
path = f'/home/etienneb/Models/Menura/Data/run_{run_ID}'
path_remote = f'/gpfswork/rech/fuz/ued64ot/run_{run_ID}'
remote_label = 'jean-zay'
mh_tur = menura_HK(path, path_remote, ask_scp=False, remote_label=remote_label)
md_tur = menura_data(path, path_remote, it, ask_scp=False, print_param=True, remote_label=remote_label)

run_ID = '017'
path = f'/home/etienneb/Models/Menura/Data/run_{run_ID}'
path_remote = f'/gpfswork/rech/fuz/ued64ot/run_{run_ID}'
remote_label = 'jean-zay'
mh_lam = menura_HK(path, path_remote, ask_scp=False, remote_label=remote_label)
md_lam = menura_data(path, path_remote, it, ask_scp=False, print_param=False, remote_label=remote_label)


if 0:   ## Number of macro-particles.
    fig, AX = plt.subplots(2, 1, figsize=(16, 14), sharex=True)

    AX[0].plot(mh_tur.time, np.sum(mh_tur.active_part_0, axis=0), '--', c=oC.rgb[0], label='nb part sw turbulent')
    AX[0].plot(mh_lam.time, np.sum(mh_lam.active_part_0, axis=0), '--', c=oC.rgb[2], label='nb part sw laminar')

    AX[1].plot(mh_tur.time, np.sum(mh_tur.active_part_1, axis=0), c=oC.rgb[0], label='nb part pla turbulent')
    AX[1].plot(mh_lam.time, np.sum(mh_lam.active_part_1, axis=0), c=oC.rgb[2], label='nb part pla laminar')

    ax2 = AX[0].twiny()
    ax2.plot(mh_tur.iterations, mh_tur.active_part_0[0], alpha=0)
    ax2.set_xlim([0, mh_tur.iterations[-1]])


    oT.set_spines(AX)
    plt.tight_layout()
    plt.show()


dens_pla_tur = md_tur.recompose_field(md_tur.dens_spec1)
dens_pla_lam = md_lam.recompose_field(md_lam.dens_spec1)

ratio_densities = dens_pla_tur/dens_pla_lam

distance_to_nucleus = md_tur.grid_xy_box - np.array([500*5/8, 250])[:, None, None]
distance_to_nucleus = np.sqrt(distance_to_nucleus[0]**2 + distance_to_nucleus[1]**2)

radii = np.linspace(0, 40, 400)
ratio = np.zeros_like(radii)
nb_nodes = np.zeros_like(radii)
for i, r in enumerate(radii):
    ratio[i] = np.mean(ratio_densities[distance_to_nucleus<r])
    nb_nodes[i] = np.sum(distance_to_nucleus<r)

print((it+300)%3*6.6666*.05)
sys.exit()

print(np.mean(ratio_densities[distance_to_nucleus<1]))
print(np.sum(distance_to_nucleus<1))
print(np.mean(ratio_densities[distance_to_nucleus<2]))
print(np.sum(distance_to_nucleus<2))

fig, ax = plt.subplots(figsize=(18, 14))

ax.plot(radii, ratio, c=oC.rgb[0])

ax.axhline(1, color='k', lw=1)

oT.set_spines(ax)
plt.tight_layout()
plt.show()


vmin = np.amax(np.log10(dens_pla_tur))-5
vmax = np.amax(np.log10(dens_pla_tur))

fig, AX = plt.subplots(1, 3, figsize=(22, 14), sharex=True, sharey=True)
for ax in AX:
    ax.set_aspect('equal')

AX[0].imshow(np.log10(dens_pla_lam).T,
          vmin=vmin, vmax=vmax,
          extent=(md_tur.edges_x_box[0], md_tur.edges_x_box[-1], md_tur.edges_y_box[0], md_tur.edges_y_box[-1]),
          origin='lower', interpolation='bilinear',
          cmap=oC.wbr_1, rasterized=True)

AX[1].imshow(np.log10(dens_pla_tur).T,
          vmin=vmin, vmax=vmax,
          extent=(md_tur.edges_x_box[0], md_tur.edges_x_box[-1], md_tur.edges_y_box[0], md_tur.edges_y_box[-1]),
          origin='lower', interpolation='bilinear',
          cmap=oC.wbr_1, rasterized=True)

AX[2].imshow(np.log10(ratio_densities).T,
          vmin=-1, vmax=1,
          extent=(md_tur.edges_x_box[0], md_tur.edges_x_box[-1], md_tur.edges_y_box[0], md_tur.edges_y_box[-1]),
          origin='lower', interpolation='bilinear',
          cmap=oC.bwr_2, rasterized=True)

t = np.linspace(0, 2*np.pi, 100)
for ax in AX:
    ax.set_xlim([250, 350])
    ax.set_ylim([200, 300])
    ax.plot([500*5/8], [250], 'wx', ms=6)
    for radius in [2]:#radii:
        ax.plot(radius*np.cos(t)+500*5/8, radius*np.sin(t)+250, 'w', lw=1)

oT.set_spines(AX)
plt.tight_layout()
plt.show()
