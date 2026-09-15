# import sys
# import os

import numpy as np
import matplotlib.pyplot as plt

# import own_tools as oT
# import own_colours as oC

from menura_utils import *



it = 20000
run_ID = '030'
path = f'/home/etienneb/Models/Menura_own/Data/run_{run_ID}'
path_remote = f'/gpfswork/rech/fuz/ued64ot/run_{run_ID}'
remote_label = 'jean-zay'#
md = menura_data(path, path_remote, it, remote_label=remote_label,
                 ask_scp=True, print_param=False, full_scp=True )

v0_com = md.mp.v_A*md.mp.dX*md.mp.nb_cell_per_shift_cst/(md.mp.nb_it_per_shift_cst*md.mp.dt)
M = v0_com/md.mp.v_s
mu = np.arcsin(1/M)
print(f'Mach cone semi-angle: {mu*180/np.pi}')
v0_com /= md.mp.v_A
#
# md.plt_field('dens pla')
#
B = md.load_field('B')
B = md.stag_vector(B)
n_sw  = md.load_field('dens_spec_0')
n_pla = md.load_field('dens_spec_1')






# path = '/home/etienneb/Models/Menura/menura_test_particle'
# traj = np.load(f'{path}/products/HK/traj.npy').T
# t = np.linspace(0, md.mp.dt*traj.shape[1], traj.shape[1])
# traj[0, :, :] -= v0_com*t[:, None]


if 0: ## Plot trajectory in physical space.

    field = np.log10(md.recompose_field(md.dens_spec1))
    vmin = np.amax(np.log10(md.dens_spec1))-5
    vmax = np.amax(np.log10(md.dens_spec1))
    cmap = oC.wbr_1

    # field = md.recompose_field(md.B)
    # field = np.log10(np.sqrt(field[0]**2+field[1]**2))
    # vmin = .5
    # vmax = 1.5
    # cmap = oC.bwr_2

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_aspect('equal')
    p0 = ax.imshow(field.T,
                      vmin=vmin, vmax=vmax,
                      extent=(md.edges_x_box[0], md.edges_x_box[-1], md.edges_y_box[0], md.edges_y_box[-1]),
                      interpolation='bilinear',
                      rasterized=True, cmap=cmap, origin='lower')

    R = md.mp.R_gyr_norm
    print('gyro-radius: ', R)
    print('v0_com: ', md.mp.v0, md.mp.v0/md.mp.v_A)
    tt = np.linspace(0, np.pi, 400)
    xx = -R*(tt - np.sin(tt)) + (5./8.*500) + 5
    yy = -R*(1 + np.cos(tt)) + 490
    ax.plot(xx, yy, 'k')
    ax.plot(xx[0], yy[0], 'xk')

    for t in range(100):
        ax.plot((traj[0, :1000, t]-traj[0, 0, t]+5./8.*500+5)%500, traj[1, :1000, t], c=oC.rgb2[0], lw=.5, alpha=.6)
        # ax.plot(traj[0, :, t], traj[1, :, t], c=oC.rgb2[0], lw=.5, alpha=.6)

    # idx_y = int((.9-1.)*md.nb_proc*md.mp.len_y_cst+md.mp.len_y_cst)
    # ax.text(md.edges_x[-1, int(.1*md.mp.len_x_cst)], md.edges_y[-1, idx_y], 'Cometary ion density', ha='center', va='center', fontsize=20)

    # ax.set_xlim([200, 300])
    # ax.set_ylim([200, 300])

    posAx = ax.get_position()
    cax = fig.add_axes([posAx.x1*1., posAx.y0, 0.02, .3])
    cb = fig.colorbar(p0, cax=cax, orientation='vertical')
    # cb.set_label(label, rotation=0, ha='left', fontsize=24)
    #
    oT.set_spines(ax)
    plt.tight_layout()
    plt.show()


if 0:   ## Trajectory in phase space.

    idx_part = 50

    fig, ax = plt.subplots(figsize=(16, 12))

    ax.plot(traj[0, :, idx_part], c=oC.rgb2[0], label='rx')
    ax.plot(traj[1, :, idx_part], c=oC.rgb2[1], label='ry')

    ax.legend(fontsize='20')

    oT.set_spines(ax)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(16, 12))

    ax.plot(traj[2, :, idx_part], c=oC.rgb2[0], label='vx')
    ax.plot(traj[3, :, idx_part], c=oC.rgb2[1], label='vy')
    ax.plot(traj[4, :, :], c=oC.rgb2[2], lw=1)

    ax.legend(fontsize='20')

    oT.set_spines(ax)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(16, 12))

    E_kin = np.sqrt(traj[2]**2+traj[3]**2,+traj[4]**2)

    ax.plot(E_kin, c=oC.rgb2[2],lw=1)
    ax.plot(np.mean(E_kin, axis=1), c=oC.rgb2[0])

    ax.axhline(E_kin[0, 0], c='k')

    oT.set_spines(ax)
    plt.tight_layout()
    plt.show()


if 0:   ## Ji, Ji_z vs Ji_in-plane

    Ji = md.recompose_field(md.curr)
    field = Ji[2]/np.sqrt(Ji[0]**2+Ji[1]**2)

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect('equal')

    p = ax.imshow(field.T,
                  vmin=-1, vmax=1,
                  origin='lower',
                  cmap=oC.bwr_2, interpolation='bilinear')

    plt.colorbar(p)

    oT.set_spines(ax)
    plt.tight_layout()
    plt.show()


if 0:   ## Compute the gyro-radius.

    dt = .005
    t = np.arange(0, traj.shape[1]*dt, dt)
    # x = np.cos(2*np.pi*t)#
    # y = np.sin(2*np.pi*t)#
    x = traj[0]
    y = traj[1]

    x_dot = (x[2:]-x[:-2])/(2*dt)
    y_dot = (y[2:]-y[:-2])/(2*dt)
    x_dotdot = (x[2:]-2*x[1:-1]+x[:-2])/(dt**2)
    y_dotdot = (y[2:]-2*y[1:-1]+y[:-2])/(dt**2)

    R = ( (x_dot**2+y_dot**2)**(3/2) )/(x_dot*y_dotdot - x_dotdot*y_dot)
    R = np.absolute(R)

    itk = (np.absolute(x_dotdot)>1e6)
    itk += (np.absolute(y_dotdot)>1e6)
    itk += (np.absolute(x_dot)>1e3)
    itk += (np.absolute(y_dot)>1e3)
    R[itk] = np.nan



    fig, ax = plt.subplots(figsize=(16, 12))

    ax.plot(R, c=oC.rgb2[2])
    # ax.set_ylim([0, 2])
    # ax.axhline(R_gyr_norm, color='k', lw=1)
    print(np.nanmean(R))

    oT.set_spines(ax)
    plt.tight_layout()
    plt.show()


if 1:   ## Computes the generalised gyro-frequency and gyro-radii.

    fig, ax = plt.subplots(figsize=(14, 14))

    for n_pla in [.001, .01, .1, 1., 10.]:
        # n_pla = .01
        # omega = -B[2]*(n_sw + n_pla*18)/((n_sw + n_pla)*18)
        omega = oT.norm(B)*(n_sw + n_pla*18)/((n_sw + n_pla)*18)
        omega_mean = 1*(1+n_pla*18)/((1+n_pla)*18)

        u_sw = md.load_field('Ji')  ## Choose the right run! i.e. no planetary ions.
        u_sw /= n_sw
        u_com = np.zeros((3, md.mp.len_x_cst+4, md.mp.len_x_cst+4))
        u_com[0] = v0_com
        v_i = n_sw/(n_sw+18*n_pla)*u_sw + (18*n_pla/(n_sw+18*n_pla))[None,:,:]*u_com
        u_sw  -= v_i
        u_com -= v_i

        Bu = B[0]*u_sw[0] + B[1]*u_sw[1] + B[2]*u_sw[2]
        alpha = np.arccos(Bu/(oT.norm(B)*oT.norm(u_sw)))
        Bu_com = B[0]*u_com[0] + B[1]*u_com[1] + B[2]*u_com[2]
        alpha_com = np.arccos(Bu_com/(oT.norm(B)*oT.norm(u_com)))

        v_i_mean = 18*n_pla/(1+18*n_pla)*v0_com

        u_sw_perp = oT.norm(u_sw)*np.sin(alpha)
        R = np.absolute(u_sw_perp)/omega
        # R_mean = u_i_mean/omega_mean
        R_mean = v_i_mean/omega_mean
        # print(np.nanmean(R), R_mean)

        u_com_perp = oT.norm(u_com)*np.sin(alpha_com)
        R_com = np.absolute(u_com_perp)/omega
        R_mean_com = (v0_com-v_i_mean)/omega_mean
        print(n_pla, np.nanmean(R_com), R_mean_com)
        # sys.exit()
        # print(R_mean)
        # print(np.amax(R), np.amin(R), np.mean(R))
        # print(np.sum(R>R_mean), np.sum(R<R_mean))
        # print(np.sum(B[2]>-1), np.sum(B[2]<-1))

        bins = np.linspace(0.7, 1.3, 100)
        centres = .5*(bins[:-1]+bins[1:])
        h, bins = np.histogram(R_com.flatten()/R_mean_com, bins)
        if n_pla==.001:
            ax.plot(centres, h, color=oC.rgb[0], label=f'R_turb/R_lam {n_pla}')
        else:
            ax.plot(centres, h, label=f'R_turb/R_lam {n_pla}')
        print(np.nanmean(R/R_mean))
        ax.axvline(np.nanmean(R/R_mean), c=oC.rgb[0], lw=1)
    # h, b = np.histogram(-B[2].flatten(), bins)
    h, b = np.histogram((oT.norm(B)).flatten(), bins)
    ax.plot(centres, h, color=oC.rgb[2], label='Bz')
    h, b = np.histogram((alpha/np.pi*2).flatten(), bins)
    ax.plot(centres, h, color=oC.rgb[1], label='alpha/pi/2')

    # ax.axvline(np.mean(-B[2]), c=oC.rgb[2], lw=1)
    print(np.nanmean((oT.norm(B)).flatten()))
    print(np.nanmean(alpha/np.pi*2))
    ax.axvline(np.nanmean((oT.norm(B)).flatten()), c=oC.rgb[2], lw=1)
    ax.axvline(np.nanmean(alpha/np.pi*2), c=oC.rgb[1], lw=1)

    ax.legend(fontsize=20)

    oT.set_spines(ax)
    plt.tight_layout()
    plt.show()

    # fig, ax = plt.subplots(figsize=(14, 14))
    # ax.set_aspect('equal')
    #
    # p = ax.imshow(np.sin(alpha).T,
    #               # vmin=np.pi/4, vmax=3*np.pi/4,
    #               extent=(md.edges_x_box[0], md.edges_x_box[-1], md.edges_y_box[0], md.edges_y_box[-1]),
    #               interpolation='bilinear',
    #               cmap=oC.bwr_2, rasterized=True, origin='lower')
    # # p = ax.imshow(B[2].T,
    # #               # vmin=-0.0569-.1, vmax=-0.0569+.1,
    # #               extent=(md.edges_x_box[0], md.edges_x_box[-1], md.edges_y_box[0], md.edges_y_box[-1]),
    # #               interpolation='bilinear',
    # #               cmap=oC.bwr_2, rasterized=True, origin='lower')
    #
    # plt.colorbar(p)
    #
    # oT.set_spines(ax)
    # plt.tight_layout()
    # plt.show()

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect('equal')

    p = ax.imshow(R.T,
                  vmin=0, vmax=2*R_mean,
                  extent=(md.edges_x_box[0], md.edges_x_box[-1], md.edges_y_box[0], md.edges_y_box[-1]),
                  interpolation='bilinear',
                  cmap=oC.bwr_2, rasterized=True, origin='lower')
    # p = ax.imshow(B[2].T,
    #               # vmin=-0.0569-.1, vmax=-0.0569+.1,
    #               extent=(md.edges_x_box[0], md.edges_x_box[-1], md.edges_y_box[0], md.edges_y_box[-1]),
    #               interpolation='bilinear',
    #               cmap=oC.bwr_2, rasterized=True, origin='lower')

    plt.colorbar(p)

    oT.set_spines(ax)
    plt.tight_layout()
    plt.show()


if 0:   ## Computes the angle between the ion current and the magnetic field.

    B  = md.recompose_field(md.B)
    Ji = md.recompose_field(md.curr)
    BJi = B[0]*Ji[0] + B[1]*Ji[1] + B[2]*Ji[2]
    alpha = np.arccos(BJi/(oT.norm(B)*oT.norm(Ji)))
    alpha *= 180/np.pi

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect('equal')

    p = ax.imshow(alpha.T,
                  vmin=0, vmax=180,
                  origin='lower',
                  cmap=oC.bwr_2, interpolation='bilinear')

    plt.colorbar(p)

    oT.set_spines(ax)
    plt.tight_layout()
    plt.show()
