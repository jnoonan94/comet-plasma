import numpy as np
import matplotlib.pyplot as plt

# import own_tools as oT
# import own_colours as oC

from menura_utils import *




import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--run')
parser.add_argument('--it')
args = parser.parse_args()
#
if args.it != None:
    it = int(args.it)
else:
    it = 0

if args.run == None:
    run_ID = ''
    path = '/home/etienneb/Models/Menura/menura'
else:
    try:
        run_ID = f'{int(args.run):03}'
    except:
        run_ID = args.run

    path = f'/home/etienneb/Models/Menura_own/Data/run_{run_ID}'
# path = '/home/etienneb/Models/Menura_own/menura_ORF'
# path = '/home/etienneb/menura/menura_ORF'
# path = '/home/etienneb/Models/Menura/menura_test_particle'

# path = f'/home/etienneb/Models/Menura_own/Data/run_tmp_2'

# path_remote = '/home/b/behare/Private/menura'
# path_remote = '/home/b/behare/Private/menura_ORF'
# path_remote = '/linkhome/rech/genlag01/ued64ot/menura'
path_remote = f'/gpfswork/rech/fuz/ued64ot/run_{run_ID}'
# if path_remote == '/gpfswork/rech/fuz/ued64ot/run_012':
#     sys.exit('NO, change directory stp!')

remote_label = 'jean-zay'
# remote_label = 'kebnekaise'



# for it in np.arange(430, 450):
md = menura_data(path, path_remote, it, ask_scp=False, remote_label=remote_label)
# md.produce_ohm()
if 0:
    fig, AX = plt.subplots(1, 3, figsize=(20, 14), sharex=True, sharey=True)
    for ax in AX:
        ax.set_aspect('equal')

    vmin = -20
    vmax = 20
    p0 = AX[0].imshow(md.E_mot[0].T,
                      vmin=vmin, vmax=vmax,
                      extent=(md.edges_x_box[0], md.edges_x_box[-1], md.edges_y_box[0], md.edges_y_box[-1]),
                      interpolation='bilinear',
                      rasterized=True, cmap=oC.bwr_2, origin='lower')

    AX[1].imshow(md.E_mot[1].T,
                      vmin=vmin, vmax=vmax,
                      extent=(md.edges_x_box[0], md.edges_x_box[-1], md.edges_y_box[0], md.edges_y_box[-1]),
                      interpolation='bilinear',
                      rasterized=True, cmap=oC.bwr_2, origin='lower')

    AX[2].imshow(md.E_mot[2].T,
                      vmin=vmin, vmax=vmax,
                      extent=(md.edges_x_box[0], md.edges_x_box[-1], md.edges_y_box[0], md.edges_y_box[-1]),
                      interpolation='bilinear',
                      rasterized=True, cmap=oC.bwr_2, origin='lower')

    # AX[1].suptitle(E_Hall)

    oT.set_spines(AX)
    plt.tight_layout()
    plt.show()

    print(np.amax(md.E_mot), np.amax(md.E_hal))

    fig, AX = plt.subplots(1, 3, figsize=(20, 14), sharex=True, sharey=True)
    for ax in AX:
        ax.set_aspect('equal')

    # vmin = -10
    # vmax = 10
    p0 = AX[0].imshow(md.E_hal[0].T,
                      vmin=vmin, vmax=vmax,
                      extent=(md.edges_x_box[0], md.edges_x_box[-1], md.edges_y_box[0], md.edges_y_box[-1]),
                      interpolation='bilinear',
                      rasterized=True, cmap=oC.bwr_2, origin='lower')

    AX[1].imshow(md.E_hal[1].T,
                      vmin=vmin, vmax=vmax,
                      extent=(md.edges_x_box[0], md.edges_x_box[-1], md.edges_y_box[0], md.edges_y_box[-1]),
                      interpolation='bilinear',
                      rasterized=True, cmap=oC.bwr_2, origin='lower')

    AX[2].imshow(md.E_hal[2].T,
                      vmin=vmin, vmax=vmax,
                      extent=(md.edges_x_box[0], md.edges_x_box[-1], md.edges_y_box[0], md.edges_y_box[-1]),
                      interpolation='bilinear',
                      rasterized=True, cmap=oC.bwr_2, origin='lower')

    # AX[0].quiver(md.grid_xy_box[0], md.grid_xy_box[1],
    #              md.E_hal[0], md.E_hal[1],
    #              scale=1e3)
    # AX[1].suptitle(E_Hall)

    oT.set_spines(AX)
    plt.tight_layout()
    plt.show()


# x0 = 100; x1 = 500
# y0 = 150; y1 = 450

# x0 = 250; x1 = 350
# y0 = 200; y1 = 300
# x0 = 295; x1 = 315      ## Publication turbulent comet.
# y0 = 237.5; y1 = 257.5  ##   it. 3000, run_016 and _020
x0 = 280; x1 = 330      ## Publication turbulent comet.
y0 = 225; y1 = 275  ##   it. 6000, run_024
# x0 = 225; x1 = 255
# y0 = 240; y1 = 260
# x0 = 280; x1 = 330
# y0 = 237.5; y1 = 267.5
# x0 = 245; x1 = 280
# y0 = 220; y1 = 255
# x0 = 250; x1 = 290
# y0 = 220; y1 = 260
itk_x = (md.grid_x_box>x0) * (md.grid_x_box<x1)
itk_y = (md.grid_x_box>y0) * (md.grid_x_box<y1)
itk = itk_x[:, None]*itk_y[None, :]


dens_com = np.log10(md.load_field('dens_spec_1'))
# md.produce_ohm()


if 0:   ## E_Hall quiver
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect('equal')

    # vmin = -10
    # vmax = 10
    # p0 = ax.imshow(np.sqrt(md.E_hal[0]**2+md.E_hal[1]**2).T,
    #                   vmin=vmin, vmax=vmax,
    #                   extent=(md.edges_x_box[0], md.edges_x_box[-1], md.edges_y_box[0], md.edges_y_box[-1]),
    #                   interpolation='bilinear',
    #                   rasterized=True, cmap=oC.bwr_2, origin='lower')
    p0 = ax.imshow(dens_com.T,
                      vmin=0, vmax=2,
                      extent=(md.edges_x_box[0], md.edges_x_box[-1], md.edges_y_box[0], md.edges_y_box[-1]),
                      interpolation='bilinear',
                      rasterized=True, cmap=oC.wbr_1, origin='lower')

    # ax.quiver(md.grid_xy_box[0,::2,::2], md.grid_xy_box[1,::2,::2],
    #              md.E_hal[0,::2,::2], md.E_hal[1,::2,::2],
    #              scale=.4e3)




    ax.quiver(md.grid_xy_box[0][itk], md.grid_xy_box[1][itk],
                 # md.E_mot[0][itk]+md.E_hal[0][itk]+md.E_amb[0][itk]+md.E_hyp_res[0][itk],
                 # md.E_mot[1][itk]+md.E_hal[1][itk]+md.E_amb[1][itk]+md.E_hyp_res[1][itk],
                 # md.E_mot[0][itk], md.E_mot[1][itk],
                 md.E_hal[0][itk], md.E_hal[1][itk],
                 # md.E_amb[0][itk], md.E_amb[1][itk],
                 pivot='tip',
                 # headwidth=2, headlength=3,
                 # scale=.4e3,
                 headwidth=3, headlength=4,
                 scale=.4e3
                 )
    # E = md.recompose_field(md.E)
    # ax.quiver(md.grid_xy_box[0][itk], md.grid_xy_box[1][itk],
    #              E[0][itk], E[1][itk],
    #              pivot='tip',
    #              # headwidth=2, headlength=3,
    #              # scale=.4e3,
    #              headwidth=3, headlength=4,
    #              scale=.2e3
    #              )

    ax.set_xlim([x0, x1])
    ax.set_ylim([y0, y1])

    oT.set_spines(ax)
    plt.tight_layout()
    plt.show()


if 1:   ## B lines + planetary density
    B = md.load_field('B')
    B = md.stag_vector(B)

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect('equal')

    p0 = ax.imshow(dens_com.T,
                      vmin=0, vmax=2,
                      extent=(md.edges_x[0], md.edges_x[-1], md.edges_y[0], md.edges_y[-1]),
                      interpolation='bilinear',
                      rasterized=True, cmap=oC.wbr_1, origin='lower')
    # p0 = ax.imshow(md.J_tot[2].T,
                      # vmin=-10, vmax=10,
                      # extent=(md.edges_x_box[0], md.edges_x_box[-1], md.edges_y_box[0], md.edges_y_box[-1]),
                      # interpolation='none',
                      # rasterized=True, cmap=oC.bwr_2, origin='lower')
    # p0 = ax.imshow(B[1].T,
    #                   vmin=-10, vmax=10,
    #                   extent=(md.edges_x_box[0], md.edges_x_box[-1], md.edges_y_box[0], md.edges_y_box[-1]),
    #                   interpolation='none',
    #                   rasterized=True, cmap=oC.bwr_2, origin='lower')

    # colours = np.zeros((md.mp.len_x_cst, md.mp.len_x_cst))

    # colours[B[1]>0] = 1
    # colours[0, B[1]<0] = 2
    # c = np.stack((colours[0][itk], colours[1][itk], colours[2][itk])).reshape((3, np.sum(itk_x), np.sum(itk_y)))
    # print(c.shape)
    ax.streamplot(md.grid_x_box[itk_x], md.grid_y_box[itk_y],
                 # md.E_mot[0][itk]+md.E_hal[0][itk]+md.E_amb[0][itk]+md.E_hyp_res[0][itk],
                 # md.E_mot[1][itk]+md.E_hal[1][itk]+md.E_amb[1][itk]+md.E_hyp_res[1][itk],
                 # md.E_mot[0][itk], md.E_mot[1][itk],
                 # B[0,:,:,42][itk].reshape((np.sum(itk_x), np.sum(itk_y))).T, B[1,:,:,42][itk].reshape((np.sum(itk_x), np.sum(itk_y))).T,
                 B[0][itk].reshape((np.sum(itk_x), np.sum(itk_y))).T, B[1][itk].reshape((np.sum(itk_x), np.sum(itk_y))).T,
                 color='k',#colours[itk].reshape((np.sum(itk_x), np.sum(itk_y))).T,
                 # cmap=oC.bwr_2,
                 density=15,
                 linewidth=.5,
                 )
    # E = md.recompose_field(md.E)
    # ax.quiver(md.grid_xy_box[0][itk], md.grid_xy_box[1][itk],
    #              E[0][itk], E[1][itk],
    #              pivot='tip',
    #              # headwidth=2, headlength=3,
    #              # scale=.4e3,
    #              headwidth=3, headlength=4,
    #              scale=.2e3
    #              )
    # plt.title('Bx')
    ax.set_xlim([x0, x1])
    ax.set_ylim([y0, y1])

    oT.set_spines(ax)
    plt.tight_layout()
    plt.show()




if 0:   ## Integrating electrostatic potential.

    E_mot    = np.zeros((md.nb_proc, 3, md.mp.len_x_cst+4, md.mp.len_y_cst+4))#, self.mp.len_z_cst+4))
    E_hal    = np.zeros((md.nb_proc, 3, md.mp.len_x_cst+4, md.mp.len_y_cst+4))#, self.mp.len_z_cst+4))
    E_amb    = np.zeros((md.nb_proc, 3, md.mp.len_x_cst+4, md.mp.len_y_cst+4))#, self.mp.len_z_cst+4))

    for r in range(md.nb_proc):
        try:
            E_mot[r] = np.load(f'{md.local_path}/products/E_mot_it{md.idx_it}_rank{r}.npy')
            E_hal[r] = np.load(f'{md.local_path}/products/E_hal_it{md.idx_it}_rank{r}.npy')
            E_amb[r] = np.load(f'{md.local_path}/products/E_amb_it{md.idx_it}_rank{r}.npy')
        except:
            pass

    len_x = md.mp.len_x_cst
    # E = md.recompose_field(E_mot)
    E = md.recompose_field(E_hal)
    # E = md.recompose_field(md.E)
    V = np.zeros((md.mp.len_x_cst, md.mp.len_x_cst))

    B = md.recompose_field(md.B)


    for i in np.arange(2, md.mp.len_x_cst-2):
        V[len_x-i] = V[len_x-i-1] + E[0, len_x-i]
        # V[i] = V[i+1] + E[0, i]

    fig, AX = plt.subplots(1, 2, figsize=(18, 14), sharex=True, sharey=True)
    for ax in AX:
        ax.set_aspect('equal')
    # print(np.nanmin(V), np.nanmax(V))
    p0 = AX[0].imshow(V.T,
                      vmin=-1.5, vmax=1.,
                      extent=(md.edges_x_box[0], md.edges_x_box[-1], md.edges_y_box[0], md.edges_y_box[-1]),
                      interpolation='bilinear',
                      rasterized=True, cmap=oC.wbr_1, origin='lower')

    # AX[0].streamplot(md.grid_x_box, md.grid_y_box,
    #              # md.E_mot[0][itk]+md.E_hal[0][itk]+md.E_amb[0][itk]+md.E_hyp_res[0][itk],
    #              # md.E_mot[1][itk]+md.E_hal[1][itk]+md.E_amb[1][itk]+md.E_hyp_res[1][itk],
    #              # md.E_mot[0][itk], md.E_mot[1][itk],
    #              B[0].T, B[1].T,
    #              # color=colours[itk].reshape((np.sum(itk_x), np.sum(itk_y))).T,
    #              # cmap=oC.bwr_2,
    #              density=5,
    #              linewidth=.5,
    #              )

    p1 = AX[1].imshow(dens_com.T,
                      vmin=-3, vmax=-1,
                      extent=(md.edges_x_box[0], md.edges_x_box[-1], md.edges_y_box[0], md.edges_y_box[-1]),
                      interpolation='bilinear',
                      rasterized=True, cmap=oC.wbr_1, origin='lower')

    # AX[1].streamplot(md.grid_x_box, md.grid_y_box,
    #              # md.E_mot[0][itk]+md.E_hal[0][itk]+md.E_amb[0][itk]+md.E_hyp_res[0][itk],
    #              # md.E_mot[1][itk]+md.E_hal[1][itk]+md.E_amb[1][itk]+md.E_hyp_res[1][itk],
    #              # md.E_mot[0][itk], md.E_mot[1][itk],
    #              B[0].T, B[1].T,
    #              # color=colours[itk].reshape((np.sum(itk_x), np.sum(itk_y))).T,
    #              # cmap=oC.bwr_2,
    #              density=5,
    #              linewidth=.5,
    #              )



    plt.tight_layout()
    oT.set_spines(AX)
    plt.show()

sys.exit()

# bin_centres, psd = md.produce_omnidirectional_PSD('B_z')
# np.save('res_15.npy', {'bin_centres': bin_centres, 'PSD': psd})
# md.tmp()
# md.plt_time_space()
# md.plt_ohm()


# md.plt_all()
# md.plt_field('Bz', vminmax=[-2., -.0])
# md.produce_spectra()
# md.plt_field_2d_spectrum('B_perp_sqr')
# md.plt_B_perp(streamplot=False)
# md.plt_field_2d_spectrum('ui_perp_sqr')

# md.plt_ohm(save_fig=False)
# md.plt_field('Ex', vminmax=[-10, 10], save_fig=True)
# md.plt_field('Ey', vminmax=[-10, 10], save_fig=True)
# md.plt_B_perp()
# md.plt_v_perp()
# md.plt_E_perp()




e    = 1.602e-19   # Elementary charge. Or so. C.
m_i  = 1.673e-27   # Atomic mass unit. Or so. kg.
m_e  = 9.109e-31   # Atomic mass unit. Or so. kg.
mu0  = 1.257e-6    # Permeability. H/m.
eps0 = 8.85e-12    # Vacuum permittivity. E/m.
k_B  = 1.3806e-23  # Boltzmann ocnstant, m2 kg s-2 K-1
c    = 299792458.  # Speed of light, m/s

# density = md.recompose_field(md.dens)
# # omega_e = np.sqrt(e*e*md.dens*md.mp.n0_no_norm/(eps0*m_e))
# # d_e = c/omega_e
# # md.dens = d_e# (md.mp.dX*md.mp.x0)/d_e
#
# # fig, ax = plt.subplots(figsize=(14,14))
# # ax.set_aspect('equal')
# # c = ax.pcolormesh((d_e/(md.mp.dX*md.mp.x0)).T,
# #               cmap=oC.bwr_2, rasterized=True)
# # plt.colorbar(c)
# # oT.set_spines(ax)
# # plt.tight_layout()
# # plt.show()
#
# # print( md.grid_x_box[np.argmin(density)//2000], md.grid_y_box[np.argmin(density)%2000])
# density_spec0 = md.recompose_field(md.dens_spec0)
# density_spec1 = md.recompose_field(md.dens_spec1)
# idx_x = np.argmin(density_spec0)//2000
# idx_y = np.argmin(density_spec0)%2000
# x_min = md.grid_x_box[idx_x]
# y_min = md.grid_y_box[idx_y]
# print(x_min, y_min)
# print(np.amin(density), np.amin(density_spec0)/md.mp.w_sw)
# print(density_spec0[idx_x, idx_y]/md.mp.w_sw)
# print(density_spec1[idx_x, idx_y]/md.mp.w_pla)
# sys.exit()

# B = np.zeros((3, 2000, 2000))
B = md.recompose_field(md.B)
dens = md.recompose_field(md.dens)
dx = md.mp.dX

dyBz = 1./(12.*dx) * ( B[2, 2:-2, 4:  ]-8*B[2, 2:-2, 3:-1]+8*B[2, 2:-2, 1:-3]-B[2, 2:-2,  :-4] )
dxBz = 1./(12.*dx) * ( B[2, 4:  , 2:-2]-8*B[2, 3:-1, 2:-2]+8*B[2, 1:-3, 2:-2]-B[2,  :-4, 2:-2] )
dxBy = 1./(12.*dx) * ( B[1, 4:  , 2:-2]-8*B[1, 3:-1, 2:-2]+8*B[1, 1:-3, 2:-2]-B[1,  :-4, 2:-2] )
dyBx = 1./(12.*dx) * ( B[0, 2:-2, 4:  ]-8*B[0, 2:-2, 3:-1]+8*B[0, 2:-2, 1:-3]-B[0, 2:-2,  :-4] )

J_tot = np.zeros_like(B)
J_tot[0, 2:-2, 2:-2] = dyBz
J_tot[1, 2:-2, 2:-2] = -dxBz
J_tot[2, 2:-2, 2:-2] = dxBy - dyBx

Ji = np.zeros_like(B)
Ji[0] = md.recompose_field(md.curr[:, 0])
Ji[1] = md.recompose_field(md.curr[:, 1])
Ji[2] = md.recompose_field(md.curr[:, 2])

E_mot = np.zeros_like(B)
E_mot[0] = Ji[1]*B[2] - Ji[2]*B[1]
E_mot[1] = Ji[2]*B[0] - Ji[0]*B[2]
E_mot[2] = Ji[0]*B[1] - Ji[1]*B[0]
E_mot /= dens

E_hal = np.zeros_like(B)
E_hal[0] = J_tot[1]*B[2] - J_tot[2]*B[1]
E_hal[1] = J_tot[2]*B[0] - J_tot[0]*B[2]
E_hal[2] = J_tot[0]*B[1] - J_tot[1]*B[0]


print(np.amin(dens))
print(np.amin(dens[220:260, 240:280]))
dens[dens<.25] = np.nan

fig, ax = plt.subplots(1, 1, figsize=(14, 14))
# for ax in AX:
ax.set_aspect('equal')

ax.pcolormesh(md.grid_x_box, md.grid_y_box, np.log10(dens).T,
                 cmap=oC.bwr_2)
# for ax in AX.flatten():
# ax.set_xlim([220, 260])
# ax.set_ylim([240, 280])

plt.suptitle('Density')

oT.set_spines(ax)
plt.tight_layout()
plt.show()
# sys.exit()
#
fig, AX = plt.subplots(1, 3, figsize=(18, 14))
for ax in AX:
    ax.set_aspect('equal')

AX[0].pcolormesh(md.grid_x_box, md.grid_y_box, B[0].T,
                 cmap=oC.bwr_2)
AX[1].pcolormesh(md.grid_x_box, md.grid_y_box, B[1].T,
                 cmap=oC.bwr_2)
AX[2].pcolormesh(md.grid_x_box, md.grid_y_box, B[2].T,
                 cmap=oC.bwr_2)

# for ax in AX.flatten():
#     ax.set_xlim([220, 260])
#     ax.set_ylim([240, 280])

plt.suptitle('B')

oT.set_spines(AX)
plt.tight_layout()
plt.show()



fig, AX = plt.subplots(1, 3, figsize=(18, 14))
for ax in AX:
    ax.set_aspect('equal')

AX[0].pcolormesh(md.grid_x_box, md.grid_y_box, J_tot[0].T,
                 cmap=oC.bwr_2)
AX[1].pcolormesh(md.grid_x_box, md.grid_y_box, J_tot[1].T,
                 cmap=oC.bwr_2)
AX[2].pcolormesh(md.grid_x_box, md.grid_y_box, J_tot[2].T,
                 cmap=oC.bwr_2)

# for ax in AX.flatten():
#     ax.set_xlim([220, 260])
#     ax.set_ylim([240, 280])

plt.suptitle('J_tot = curl(B)')

oT.set_spines(AX)
plt.tight_layout()
plt.show()



fig, AX = plt.subplots(1, 3, figsize=(18, 14))
for ax in AX:
    ax.set_aspect('equal')

AX[0].pcolormesh(md.grid_x_box, md.grid_y_box, E_hal[0].T,
                 cmap=oC.bwr_2)
AX[1].pcolormesh(md.grid_x_box, md.grid_y_box, E_hal[1].T,
                 cmap=oC.bwr_2)
AX[2].pcolormesh(md.grid_x_box, md.grid_y_box, E_hal[2].T,
                 cmap=oC.bwr_2)

# for ax in AX.flatten():
#     ax.set_xlim([220, 260])
#     ax.set_ylim([240, 280])

plt.suptitle('J_tot x B')
oT.set_spines(AX)
plt.tight_layout()
plt.show()


fig, AX = plt.subplots(1, 3, figsize=(18, 14))
for ax in AX:
    ax.set_aspect('equal')

AX[0].pcolormesh(md.grid_x_box, md.grid_y_box, E_mot[0].T,
                 cmap=oC.bwr_2)
AX[1].pcolormesh(md.grid_x_box, md.grid_y_box, E_mot[1].T,
                 cmap=oC.bwr_2)
AX[2].pcolormesh(md.grid_x_box, md.grid_y_box, E_mot[2].T,
                 cmap=oC.bwr_2)

# for ax in AX.flatten():
#     ax.set_xlim([220, 260])
#     ax.set_ylim([240, 280])

plt.suptitle('E_mot')
oT.set_spines(AX)
plt.tight_layout()
plt.show()



E_hal /= dens[None, :]

# E_hal_s = stag(E_hal)
# E_hal_s = stag_ho(E_hal)
# E_hal = E_hal_s
print(np.arange(md.grid_x_box.size)[(md.grid_x_box>220)][0])
print(np.arange(md.grid_x_box.size)[(md.grid_x_box<260)][-1])
print(np.arange(md.grid_y_box.size)[(md.grid_y_box>240)][0])
print(np.arange(md.grid_y_box.size)[(md.grid_y_box<280)][-1])
# sys.exit()
# E_hal = .5*(E_hal + E_hal_s)
# E_hal[E_hal>75]  /= 10
# E_hal[E_hal<-75] /= 10

if 1:
    fig, AX = plt.subplots(1, 3, figsize=(18, 14))
    for ax in AX:
        ax.set_aspect('equal')

    p = AX[0].pcolormesh(md.grid_x_box, md.grid_y_box, E_hal[0].T,
                     cmap=oC.bwr_2)
    AX[1].pcolormesh(md.grid_x_box, md.grid_y_box, E_hal[1].T,
                     cmap=oC.bwr_2)
    AX[2].pcolormesh(md.grid_x_box, md.grid_y_box, E_hal[2].T,
                     cmap=oC.bwr_2)

    plt.colorbar(p)

    for ax in AX.flatten():
        ax.set_xlim([220, 260])
        ax.set_ylim([240, 280])

    plt.suptitle('E_hal = 1/dens (J_tot x B)')
    oT.set_spines(AX)
    plt.tight_layout()
    plt.show()

# sys.exit()

dyEz = 1./(12.*dx) * ( E_hal[2, 2:-2, 4:  ]-8*E_hal[2, 2:-2, 3:-1]+8*E_hal[2, 2:-2, 1:-3]-E_hal[2, 2:-2,  :-4] )
dxEz = 1./(12.*dx) * ( E_hal[2, 4:  , 2:-2]-8*E_hal[2, 3:-1, 2:-2]+8*E_hal[2, 1:-3, 2:-2]-E_hal[2,  :-4, 2:-2] )
dxEy = 1./(12.*dx) * ( E_hal[1, 4:  , 2:-2]-8*E_hal[1, 3:-1, 2:-2]+8*E_hal[1, 1:-3, 2:-2]-E_hal[1,  :-4, 2:-2] )
dyEx = 1./(12.*dx) * ( E_hal[0, 2:-2, 4:  ]-8*E_hal[0, 2:-2, 3:-1]+8*E_hal[0, 2:-2, 1:-3]-E_hal[0, 2:-2,  :-4] )

curl_E =  np.zeros_like(B)
curl_E[0, 2:-2, 2:-2] = dyEz
curl_E[1, 2:-2, 2:-2] = -dxEz
curl_E[2, 2:-2, 2:-2] = dxEy - dyEx


fig, AX = plt.subplots(1, 3, figsize=(18, 14))
for ax in AX:
    ax.set_aspect('equal')

p = AX[0].pcolormesh(md.grid_x_box, md.grid_y_box, curl_E[0].T,
                 cmap=oC.bwr_2)
AX[1].pcolormesh(md.grid_x_box, md.grid_y_box, curl_E[1].T,
                 cmap=oC.bwr_2)
AX[2].pcolormesh(md.grid_x_box, md.grid_y_box, curl_E[2].T,
                 cmap=oC.bwr_2)

# plt.colorbar(p)

for ax in AX.flatten():
    ax.set_xlim([220, 260])
    ax.set_ylim([240, 280])

plt.suptitle('curl(E_hal)')
oT.set_spines(AX)
plt.tight_layout()
plt.show()
