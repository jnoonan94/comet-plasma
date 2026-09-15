import sys

import numpy as np
import matplotlib.pyplot as plt

import own_colours as oC
import own_tools as oT

import aidapy.tools.vdf_utils as vdfu
import aidapy.tools.vdf_plot as vplt
sys.path.append('/home/etienneb/Documents/hvm_vdf')
from hvm_utils import *

from menura_utils import *

if len(sys.argv)>1:
    idx_it = int(sys.argv[1])
else:
    idx_it = 0


path = '/home/etienneb/Models/Menura/menura'
DF0 = np.load(f'{path}/products/DF_it0_rank0.npy')
DF = np.load(f'{path}/products/DF_it{idx_it}_rank0.npy')
print(np.amin(DF))

if 1:
    idx_mid_x = int(DF.shape[0]/2.)
    idx_mid_v = int(DF.shape[1]/2.)

    fig, AX = plt.subplots(1, 3, figsize=(20, 14))
    for ax in AX:
        ax.set_aspect('equal')

    AX[0].pcolormesh(np.log10(DF[idx_mid_x, :, :, idx_mid_v]).T,
                     # vmin=-7, vmax=0,
                     cmap=oC.bwr_2)
    AX[1].pcolormesh(np.log10(DF[idx_mid_x, :, idx_mid_v]).T,
                     vmin=-7, vmax=0,
                     cmap=oC.bwr_2)
    AX[2].pcolormesh(np.log10(DF[idx_mid_x, idx_mid_v]).T,
                     vmin=-7, vmax=0,
                     cmap=oC.bwr_2)

    plt.tight_layout()
    plt.show()

    # for idx_it in np.arange(100):
    #
    #     DF = np.load(f'{path}/products/DF_it{idx_it}_rank0.npy')
    #
    #     idx_mid_x = int(DF.shape[0]/2.)
    #     idx_mid_v = int(DF.shape[1]/2.)
    #
    #     fig, AX = plt.subplots(1, 3, figsize=(20, 14))
    #     for ax in AX:
    #         ax.set_aspect('equal')
    #
    #     AX[0].pcolormesh(np.log10(DF[idx_mid_x, :, :, idx_mid_v]).T,
    #                      vmin=-7, vmax=0,
    #                      cmap=oC.bwr_2)
    #     AX[1].pcolormesh(np.log10(DF[idx_mid_x, :, idx_mid_v]).T,
    #                      vmin=-7, vmax=0,
    #                      cmap=oC.bwr_2)
    #     AX[2].pcolormesh(np.log10(DF[idx_mid_x, idx_mid_v]).T,
    #                      vmin=-7, vmax=0,
    #                      cmap=oC.bwr_2)
    #     # plt.show()
    #     plt.tight_layout()
    #     plt.savefig(f'/home/etienneb/Desktop/plots_tmp/{idx_it}.png')

E = np.load(f'{path}/products/E_DF_it{idx_it}_rank0.npy')
B = np.load(f'{path}/products/B_DF_it{idx_it}_rank0.npy')

grid_x = np.load(f'{path}/products/grid_x_rank0.npy')
grid_v = np.load(f'{path}/products/vGrid.npy')
len_x = grid_x.shape
dx = grid_x[1]-grid_x[0]
edges_x = np.append(grid_x-dx/2, grid_x[-1]+dx/2)
x_max = grid_x[-1]
len_v = grid_v.size
dv = grid_v[1]-grid_v[0]
edges_v = np.append(grid_v-dv/2, grid_v[-1]+dv/2)
v_max = grid_v[-1]
nb_distrib = DF.shape[0]
grid_cart = np.mgrid[-v_max:v_max:len_v*1j,-v_max:v_max:len_v*1j,-v_max:v_max:len_v*1j]

mp = menura_param(f'{path}', file_name='parameters.h')

# print(np.amax(DF))
# sys.exit()
if 1:   ## E and B components along the main direction x.

    fig, AX = plt.subplots(2, 1, figsize=(14, 18))

    AX[0].plot(grid_x, E[0], c=oC.rgb[0], label='Ex')
    AX[0].plot(grid_x, E[1], c=oC.rgb[1], label='Ey')
    AX[0].plot(grid_x, E[2], c=oC.rgb[2], label='Ez')

    AX[1].plot(grid_x, B[0]-1, c=oC.rgb[0], label='Bx-1')
    AX[1].plot(grid_x, B[1], c=oC.rgb[1], label='By')
    AX[1].plot(grid_x, B[2], c=oC.rgb[2], label='Bz')

    for ax in AX:
        ax.legend()

    oT.set_spines(AX)
    plt.tight_layout()
    plt.show()

if 1:   ## Cuts in (x, y=z=0), (x, x=z=0) and (x, x=y=0)
    # a0 = np.log10(DF)
    a0 = DF#[2:-2, 2:-2, 2:-2]
    # idx_mid_v = int(DF.shape[1]/2.)-2

    fig, AX = plt.subplots(3, 1, figsize=(14,12))

    AX[0].pcolormesh(grid_x, grid_v, a0[:, :, idx_mid_v, idx_mid_v].T,
                     cmap=oC.bwr_2, rasterized=True)
    AX[1].pcolormesh(grid_x, grid_v, a0[:, idx_mid_v, :, idx_mid_v].T,
                     cmap=oC.bwr_2, rasterized=True)
    AX[2].pcolormesh(grid_x, grid_v, a0[:, idx_mid_v, idx_mid_v, :].T,
                     cmap=oC.bwr_2, rasterized=True)

    # AX[0].contourf(grid_x*1e-3, grid_v*1e-3, a0[:, :, idx_mid_v, idx_mid_v].T, np.linspace(0, 1, 40),
    #                  cmap=oC.bwr_2, zorder=20)
    # AX[1].contourf(grid_x*1e-3, grid_v*1e-3, a0[:, idx_mid_v, :, idx_mid_v].T, np.linspace(0, 1, 40),
    #                  cmap=oC.bwr_2, zorder=20)
    # AX[2].contourf(grid_x*1e-3, grid_v*1e-3, a0[:, idx_mid_v, idx_mid_v, :].T, np.linspace(0, 1, 40),
    #                  cmap=oC.bwr_2, zorder=20)

    a1 = a0[2:-2, 2:-2, idx_mid_v, idx_mid_v]
    # AX[0].contour(grid_x*1e-3, grid_v*1e-3, a1.T, 20, c='k')
    ind_max = np.unravel_index(a1.argmax(), a1.shape)

    # AX[0].plot(grid_x[ind_max[0]]*1e-3, grid_v[ind_max[1]]*1e-3, 'wx', ms=24)
    # print(grid_x[ind_max[0]]*1e-3, grid_v[ind_max[1]]*1e-3)
    # ax.plot(DF, 'x')

    oT.set_spines(AX)
    plt.tight_layout()
    plt.show()

if 1:   ## Cuts in v_z=0, v_y=0, v_x=0
    DF[DF==0] = 1

    fig, AX = plt.subplots(1, 3, figsize=(18, 14))
    for ax in AX:
        ax.set_aspect('equal')

    AX[0].pcolormesh(np.log10(DF[25, :, :, 2]).T,
                     cmap=oC.bwr_2)
    AX[1].pcolormesh(np.log10(DF[50, :, 27]).T,
                     cmap=oC.bwr_2)
    AX[2].pcolormesh(np.log10(DF[50, 27]).T,
                     cmap=oC.bwr_2)
    plt.show()

if 1:   ## Along v_x, 1D

    fig, ax = plt.subplots(figsize=(14, 14))

    ax.plot(grid_v, np.sum(DF, axis=(0, 2, 3)), 'x')
    ax.plot(grid_v, np.sum(DF0, axis=(0, 2, 3)), '--')

    ax.axvline(mp.v_s/mp.v_A, c='k')
    ax.axvline(-mp.v_s/mp.v_A, c='k', label='v_sound')
    ax.axvline(mp.omega_wave/mp.k_wave, c='r')
    ax.axvline(-mp.omega_wave/mp.k_wave, c='r', label='v_phase')
    ax.legend()

    ax.set_yscale('log')

    oT.set_spines(ax)
    plt.tight_layout()
    plt.show()

if 1:   ## Along x and v_x, 2D

    fig, ax = plt.subplots(figsize=(18, 14))

    ax.imshow((np.sum(DF, axis=(2, 3))).T,
              origin='lower',
              extent=[edges_x[0], edges_x[-1], edges_v[0], edges_v[-1]],
              interpolation='bilinear',
              cmap=oC.bwr_2 )
    # ax.pcolormesh(DF[:, :, 27, 27].T,
    #               cmap=oC.bwr_2)

    oT.set_spines(ax)
    plt.tight_layout()
    plt.show()

if 1: # Gyro-angle distribution.
    nb_distrib = DF.shape[0]
    resolution = 50
    distrib_scaled_time = np.zeros((nb_distrib, resolution, resolution)) ##
    distrib_equa_time = np.zeros((nb_distrib, resolution, resolution))
    # grid_cart = np.mgrid[-v_max:v_max:51*1j,-v_max:v_max:51*1j,-v_max:v_max:51*1j]
    vdf_obj = vdfu.vdf(v_max=v_max, resolution=resolution, grid_geom='spher')
    B_cut = np.ones((3, nb_distrib))
    B_cut[:2] = 0.

    v_i, R = init_bulk_rotate(grid_cart, DF, B_cut, nb_distrib)

    # resolution = grid_cart.shape[1]
    distrib_equa_time = np.zeros((nb_distrib, resolution, resolution))
    std_time = interp_all_3(nb_distrib, vdf_obj, v_i, B_cut, grid_cart, DF, distrib_scaled_time, distrib_equa_time)
    # print('')
    # print(distrib_equa_time.shape)
    # print(distrib_equa_time[50, :, 25])
    # sys.exit()


    fig, ax = plt.subplots(figsize=(18, 14))

    for i in np.arange(resolution):
        ax.plot(grid_x, (std_time[:, i]))

    plt.show()


    fig, AX = plt.subplots(4, 1, figsize=(13,16), sharex=True)

    resolution = distrib_equa_time.shape[1]

    centers_rho   = vdf_obj.grid_spher[0, :, 0, 0]
    centers_theta = vdf_obj.grid_spher[1, 0, :, 0]
    centers_phi   = vdf_obj.grid_spher[2, 0, 0, :]

    x = grid_x#pos[0]
    y = centers_phi*180./np.pi



    for i, ax in enumerate(AX):
        i_sta = ((2*i+1)*int(resolution/len(AX))-1)/2.
        i_sta = int(i_sta)
        print(i_sta, centers_rho[i_sta])
        # pad = np.nanmean(distrib_scaled_time[:, i_sta:i_sto], axis=(1))
        # pad = np.mean(distrib_equa_time[:, i_sta-2:i_sta+2], axis=1)
        pad = distrib_equa_time[:, i_sta]
        # pad[std_time<1e-6]=0.
        # ax.pcolormesh(x, y, pad.T,
        #               vmin=0, vmax=1,
        #               cmap=oC.bwr_2, rasterized=True)
        # ax.contourf(x, y, pad.T,
        #               np.linspace(0, 1, 40),
        #               cmap=oC.bwr_2, zorder=-20)
        ax.imshow(pad.T,
                  extent=[edges_x[0], edges_x[-1], edges_v[0], edges_v[-1]],
                  origin='lower',
                  interpolation='bilinear',
                  cmap=oC.bwr_2, rasterized=True)
        # ax.text(x[10], y[10], 'scaled pitch-angle distrib, {:.2f} - {:.2f} v_th'.format(v_max*i_sta/reso_interp, v_max*i_sto/reso_interp))
        # ax.text(x[10], y[10], 'scaled gyro-angle distrib, {:.2f} v_th'.format(v_max*i_sta/reso_interp), fontsize=16)
    #
    # AX[0].legend()
    # AX[0].axhline(0., c='k', lw=1)
    for ax in AX:
        ax.set_rasterization_zorder(-10)


    vplt.set_spines(AX)
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'/home/etienneb/Desktop/plots_tmp/{indIt}.png')
    # sys.exit()





idx_mid_v = int(DF.shape[1]/2.)
mid_ind_x = int(DF.shape[0]/2.)

v_max = 4.

vdf_obj = vdfu.vdf(v_max=v_max, resolution=120, grid_geom='spher')
# vdf = np.nanmean(DF[2:-2,2:-2,2:-2,2:-2], axis=0).T
vdf = DF[75]#, 2:-2, 2:-2, 2:-2]
vdf_obj.interpolate_cart_vdf(grid_cart, vdf, interpolate='lin')
vplt.profiles_1d(vdf_obj.vdf_interp, vdf_obj.grid_spher)
# vplt.cart(vdf_obj.vdf_interp, vdf_obj.grid_spher)
vplt.spher(vdf_obj.vdf_interp, vdf_obj.grid_cart, plt_contourf=False, cmap=oC.bwr_2)
vplt.spher_gyro(vdf_obj.vdf_interp, vdf_obj.grid_spher, vdf_obj.grid_cart, cmap=oC.bwr_2)
vplt.gyro(vdf_obj.vdf_interp, vdf_obj.grid_spher, vdf_obj.grid_cart, cmap=oC.bwr_2)
# fig, ax = plt.subplots(figsize=(14,12))
#
# ax.plot(DF[mid_ind_x, :, idx_mid_v, idx_mid_v])
# ax.plot(DF[mid_ind_x, :, idx_mid_v, idx_mid_v], 'x')
#
# oT.set_spines(ax)
# plt.tight_layout()
# plt.show()
