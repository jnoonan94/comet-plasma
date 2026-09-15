import sys
import numpy as np
import matplotlib.pyplot as plt
import own_tools as oT
import own_colours as oC

from scipy.optimize import curve_fit

# from leo_utils import *


def spher(vdf, grid_cart, plt_contourf=False, v_phase=None, cmap='RdBu_r', vlim_norm=None):
    """ Plots the VDF interpolated over a spherical grid-of-interest, cylindrical symmetry.
    Meant for electrons in a magnetic field aligned frame.
    """
    #
    np.seterr(divide='ignore')

    a0 = np.nanmean(vdf, axis=2)
    resolution = a0.shape[0]
    ind_mid = int(resolution/2.)
    #
    vdf_scaled = vdf.copy()
    vdf_scaled -= np.nanmin(vdf_scaled[:, :], axis=(1, 2))[:, None, None]
    vdf_scaled /= np.nanmax(vdf_scaled[:, :], axis=(1, 2))[:, None, None]
    b0 = np.nanmean(vdf_scaled, axis=2)
    levels0 = np.linspace(0, 1, 40)
    #
    distrib_normed = vdf.copy()
    # mm = np.nanmean(distrib_normed[:, ind_mid-2:ind_mid+2], axis=(1, 2))[:, None, None]
    # distrib_normed /= mm
    b1 = np.nanmean(distrib_normed, axis=2)
    mm = np.nanmean(b1[:, ind_mid-2:ind_mid+2], axis=(1))[:, None]
    b1 /= mm
    b1 = np.log10(b1)
    if vlim_norm is None:
        vlim_norm = max(-1.*np.nanmin(b1[b1 != -np.inf]), np.nanmax(b1[b1 != np.inf]))*.01
    levels1 = np.linspace(-vlim_norm, vlim_norm, 40)

    fig, AX = plt.subplots(1, 3, figsize=(13, 9), sharex=True, sharey=True)
    for ax in AX:
        ax.set_aspect('equal')

    x = grid_cart[0, :, :, 0]
    y = grid_cart[2, :, :, 0]

    if plt_contourf:
        m0 = AX[0].contourf(x, y, np.log10(a0), 60, cmap=cmap, zorder=-20)
    else:
        m0 = AX[0].pcolormesh(x, y, np.log10(a0), cmap=cmap,
                              rasterized=True)
    AX[0].contour(x, y, np.log10(a0), 10, colors='k', linewidths=.5)

    if plt_contourf:
        m1 = AX[1].contourf(x, y, b0, levels0, cmap=cmap, zorder=-20)
        m2 = AX[2].contourf(x, y, b1, levels1,
                            cmap=cmap, zorder=-20)
    else:
        m1 = AX[1].pcolormesh(x, y, b0, vmin=0, vmax=1, cmap=cmap,
                              rasterized=True)
        m2 = AX[2].pcolormesh(x, y, b1, vmin=-vlim_norm, vmax=vlim_norm,
                              cmap=cmap, rasterized=True)

    if v_phase is not None:
        for ax in AX:
            ax.axhline(v_phase, c='k')
            ax.axhline(-v_phase, c='k')

    AX[0].set_xlabel('v_perp')
    AX[0].set_ylabel('v_para')
    AX[0].set_title('Interpolated VDF')
    AX[1].set_xlabel('v_perp')
    AX[1].set_title('0-to-1 scaled')
    AX[2].set_xlabel('v_perp')
    AX[2].set_title('Normalised')

    fig.suptitle('Spherical coordinate system, cylindrical representation')

    posAx = AX[0].get_position()
    cax = fig.add_axes([posAx.x1-.02, posAx.y0+.5, .008, 0.2])
    cb = fig.colorbar(m0, cax=cax, orientation='vertical')
    # cb.set_ticks([-13,-16,-19])
    cb.set_label('VDF (s^3/m^6)')
    AX[0].set_title('Original VDF')
    #
    posAx = AX[1].get_position()
    cax = fig.add_axes([posAx.x1-.02, posAx.y0+.5, .008, 0.2])
    cb = fig.colorbar(m1, cax=cax, orientation='vertical')
    cb.set_ticks([0., .5, 1.])
    AX[1].set_title('Scaled VDF')
    #
    posAx = AX[2].get_position()
    cax = fig.add_axes([posAx.x1-.02, posAx.y0+.5, .008, 0.2])
    cb = fig.colorbar(m2, cax=cax, orientation='vertical')
    vlim = .1*np.floor(vlim_norm*10.)
    cb.set_ticks([-vlim, 0., vlim])
    AX[2].set_title('Normalised VDF')

    if plt_contourf:
        for ax in AX:
            ax.set_rasterization_zorder(-10)

    oT.set_spines(AX)
    plt.tight_layout()
    plt.show()


def maxwellian_1D(v, A, v_th, v0):
    return A * np.exp(-((v-v0)/v_th)**2)

#_____
# path = '../v01/products'
centreX      = .5      ;
centreY      = .5      ;
#_____
path = '../run_27'
# path = '../leo_cam_ho_norm/products'
# path = '../Archive/v_turbulent_plain_no_cam/products'
# path = '/run/media/etienneb/EUGENE/Leo/Data/run_16/products'
#_____
# path = '../Data/run_7'
# centreX = .67
# centreY = .5
# centreZ = .25
#_____
# path = '../Data/run_8'
# centreX = .6
# centreY = .5
# centreZ = .4
#_____
# path = '../Data/run_10'
# centreX = .67
# centreY = .5
# centreZ = .3
#_____
# path = '../Data/run_11'
# centreX = .67
# centreY = .5
# centreZ = .3

# v_ref = 39009.




#________________________________________________
## From Smilei
import happi

me = 9.10938356e-31  ## kg
mi = 1.660538e-27  ## kg
e = 1.60217e-19   ## C
mu0  = 1.257e-6 ## Permeab
epsilon0 = 8.854e-12  ## F.m-1
c = 299792458.  ## m.s-1
k_B  = 1.3806e-23


# n0           = 5.e6
# B0           = 4.e-9
#
# omega_pe     = np.sqrt(e**2*n0/(epsilon0*me))
# d_e          = c/omega_pe
# omega_ce      = e*B0/me


# lp = leo_param('../leo_mpi/src')

# lg = leo_grid(f'{path}', lp)

n0_no_norm = 1.5e6#lp.n0_no_norm
B0_no_norm = 1.e-9#lp.B0_no_norm
Ti_inf = 1.e5#lp.Ti_inf
Te_inf = 1.e5#lp.Te_inf
v_A = 19585.4#B0_no_norm/np.sqrt(mu0*mi*n0_no_norm)

gamma_e = 1.
gamma_i = 3.
nb_oscillations = 3.
v_thi  = np.sqrt(2.*k_B*Ti_inf/mi)
v_s = np.sqrt(k_B*(gamma_e*Te_inf+gamma_i*Ti_inf)/mi)/v_A
d_i = 185922.6

# sys.path.append('/home/etienneb/Documents/MMS/Scripts/')
# from class_vdf import vdf


if 0:
    '''Initial and final VDF'''
    p = np.load(f'{path}/products/particles_0_0.npy').T
    h_vx_0, bin_vx = np.histogram(p[2], 200)
    centers_vx = .5*(bin_vx[:-1]+bin_vx[1:])
    p = np.load(f'{path}/products/particles_0_2000.npy').T
    h_vx_1, bin_vx = np.histogram(p[2], bin_vx)

    fig,  ax = plt.subplots(figsize=(14,14))

    ax.plot(centers_vx, h_vx_0, oC.rgb[0], label='Initial')
    ax.plot(centers_vx, h_vx_1, oC.rgb[2], label='Final')

    ax.set_yscale('log')
    ax.legend()

    oT.set_spines(ax)
    plt.show()
    sys.exit()




if len(sys.argv)>1:
    indices = [int(sys.argv[1])]
else:
    indices = [0]#np.arange(4000, 8001, 1000):#



for ind in indices:
    # p = np.load('{}/particles_sorted_{}.npy'.format(path, ind)).T

    # plt_phase_space(path, ind, lp, lg, v_max=3., res_x=512, res_v=256)
    # plt_phase_space2(path, ind, lp, lg, v_max=3., res_x=512, res_v=256)
    # E = np.load(f'{path}/products/E_{ind}.npy')
    # plt_field_spectrum_1d(lg.grid_x, E[0, :, int(lg.len_y/2)], label='Ex')
    # continue

    p = np.load(f'{path}/products/particles_0_{ind}.npy').T
    print(f'\n {p.shape} particles, {np.sum(np.isnan(p[2]))} NaNs!\n')


    #______________________________________________________________________
    if 0:   ## Using aidapy
        import aidapy.tools.vdf_utils as vdfu
        import aidapy.tools.vdf_plot as vplt

        reso = 91
        speed_max = 3
        bins_speed = np.linspace(-speed_max, speed_max, reso)
        centers_speed = .5*(bins_speed[1:]+bins_speed[:-1])
        distrib, bins = np.histogramdd(p[2:].T, (bins_speed, bins_speed, bins_speed))
        vdf0 = np.transpose(distrib, axes=(1,2,0))

        # grid_cart = np.mgrid[-speed_max:speed_max:51*1j,-speed_max:speed_max:51*1j,-speed_max:speed_max:51*1j]
        grid_cart = np.ones((3, reso-1, reso-1, reso-1))
        grid_cart[0] *= centers_speed[:, None, None]
        grid_cart[1] *= centers_speed[None, :, None]
        grid_cart[2] *= centers_speed[None, None, :]
        vdf_obj = vdfu.vdf(v_max=speed_max, resolution=201, grid_geom='spher')
        vdf_obj.interpolate_cart_vdf(grid_cart, vdf0, interpolate='lin')
        vdf_scaled = vdfu.vdf_scaled(vdf_obj.vdf_interp)
        spher(vdf_obj.vdf_interp,
                   vdf_obj.grid_cart,
                   v_phase=v_s,
                   plt_contourf=True, cmap=oC.bwr_2)
    #______________________________________________________________________


    # grid = np.loadtxt('{}/products/grid_rank0.txt'.format(path)).T
    #
    # xGrid = grid[:,0]
    # xGrid = xGrid[~np.isnan(xGrid)]
    # yGrid = grid[:,1]
    # yGrid = yGrid[~np.isnan(yGrid)]
    xGrid = np.load(f'{path}/products/grid_x.npy')
    yGrid = np.load(f'{path}/products/grid_y.npy')

    if 0:
        '''Phase space, x, vx.'''
        vMax = 3.
        resFinal = 128
        lenTot_para = lp.xLen_cst*lp.dX
        binsPar = np.linspace(-vMax, vMax, resFinal+1)
        centersPar = .5*(binsPar[:-1]+binsPar[1:])
        dv = centersPar[1] - centersPar[0]
        binsParPos = np.linspace(0, lenTot_para, resFinal+1)
        h_phase_space, b = np.histogramdd((p[0], p[2]), (binsParPos, binsPar))

        fig, ax = plt.subplots(figsize=(16, 14))

        ax.pcolormesh(binsParPos, binsPar, np.log10(h_phase_space).T,
                       cmap=oC.bwr_2)

        oT.set_spines(ax)
        plt.show()

    if 1:
        '''Basic, robust histograms.'''
        dx = xGrid[1] - xGrid[0]
        dy = yGrid[1] - yGrid[0]
        xEdges = np.append(xGrid-dx/2, xGrid[-1]+dx/2)
        yEdges = np.append(yGrid-dy/2, yGrid[-1]+dy/2)

        bins = np.linspace(-70, 70, 100)

        print(p[2, :10]/v_A)

        h_x, bin_x  = np.histogram(p[0], 200)
        h_y, bin_y  = np.histogram(p[1], 200)
        h_vx, bin_vx = np.histogram(p[2], 200)
        h_vy, bin_vy = np.histogram(p[3], 200)
        h_vz, bin_vz = np.histogram(p[4], 200)
        centers_x = .5*(bin_x[:-1]+bin_x[1:])
        centers_y = .5*(bin_y[:-1]+bin_y[1:])
        centers_vx = .5*(bin_vx[:-1]+bin_vx[1:])
        centers_vy = .5*(bin_vy[:-1]+bin_vy[1:])
        centers_vz = .5*(bin_vz[:-1]+bin_vz[1:])

        try:
            p0 = [np.amax(h_vx), np.var(p[2]), np.average(centers_vx, weights=h_vx)]
            p_m, pcov = curve_fit(maxwellian_1D, centers_vx, h_vx)
            fit_success = True
        except:
            fit_success = False

        fig, AX = plt.subplots(2, 3, figsize=(20,16))

        # AX[0, 0].hist(p[0], 50)
        # AX[0, 1].hist(p[1], 50)
        # AX[1, 0].hist(p[2], 100)#bins)
        # AX[1, 1].hist(p[3], 100)#bins)
        # AX[1, 2].hist(p[4], 100)#bins)

        AX[0, 0].plot(centers_x/d_i, h_x)
        AX[0, 1].plot(centers_y/d_i, h_y)
        AX[1, 0].plot(centers_vx/v_A, np.log10(h_vx))#bins)
        AX[1, 0].plot(centers_vy/v_A, np.log10(h_vy), '--k', lw=.5)#bins)
        # if fit_success:
        #     AX[1, 0].plot(centers_vx/v_A, np.log10(maxwellian_1D(centers_vx, *p_m)),
        #                   '--r', lw=.5, label=f'maxwel 1d  exp( -(v-v0)**2/v_th**2 )\nv_th = {p_m[1]:.2e}\nv0={p_m[2]:.2e}')
        AX[1, 1].plot(centers_vy/v_A, np.log10(h_vy))#bins)
        AX[1, 2].plot(centers_vz/v_A, np.log10(h_vz))#bins)

        AX[1, 0].axvline(v_s, c='k')
        AX[1, 0].axvline(-v_s, c='k', label='v_sound')
        AX[1, 0].legend()
        #
        # h_vx = np.sum(a0, axis=(1, 2))
        # h_vy = np.sum(a0, axis=(0, 2))
        # h_vz = np.sum(a0, axis=(0, 1))
        # AX[1, 0].plot(gridVel_ion, np.log10(h_vx/np.amax(h_vx)))
        # AX[1, 1].plot(gridVel_ion, np.log10(h_vy/np.amax(h_vy)))
        # AX[1, 2].plot(gridVel_ion, np.log10(h_vz/np.amax(h_vz)))


        AX[0, 0].set_xlabel('x')
        AX[0, 1].set_xlabel('y')
        AX[1, 0].set_xlabel('vx')
        AX[1, 1].set_xlabel('vy')
        AX[1, 2].set_xlabel('vz')
        AX[1, 0].set_xlim([-9.5, 9.5])
        AX[1, 1].set_xlim([-9.5, 9.5])
        AX[1, 2].set_xlim([-9.5, 9.5])

        # AX[1,0].set_xlim([-6., 6.])
        # AX[1,1].set_xlim([-6., 6.])
        # AX[1,2].set_xlim([-6., 6.])
        # AX[1,0].set_ylim([0,5000])
        # AX[1,1].set_ylim([0,5000])
        # AX[1,2].set_ylim([0,5000])
        oT.set_spines(AX)
        # plt.tight_layout()
        plt.show()
        # plt.savefig('/Users/etiennebehar/Desktop/positionHybC/{}.png'.format(i))
        # plt.close()
        # sys.exit()







    if 0:   # All particles positions in the physical domain.

        # xGrid = grid[:,0]
        # xGrid = xGrid[~np.isnan(xGrid)]
        # yGrid = grid[:,1]
        # yGrid = yGrid[~np.isnan(yGrid)]

        dx = xGrid[1] - xGrid[0]
        dy = yGrid[1] - yGrid[0]
        xEdges = np.append(xGrid-dx/2, xGrid[-1]+dx/2)
        yEdges = np.append(yGrid-dy/2, yGrid[-1]+dy/2)

        fig, AX = plt.subplots(1, 2, figsize=(20,16))

        for ax in AX:
            ax.set_aspect('equal')

        for x in xEdges:
            AX[0].plot([x, x], [yEdges[0], yEdges[-1]], 'k', lw=.5)
        for y in yEdges:
            AX[0].plot([xEdges[0], xEdges[-1]], [y, y], 'k', lw=.5)

        for x in xGrid:
            for y in yGrid:
                AX[0].plot(x, y, 'ko', ms=2)
        # AX[0].plot([(xGrid[0]-.5*dx)*1e-3,(xGrid[-1]+.5*dx)*1e-3,(xGrid[-1]+.5*dx)*1e-3,(xGrid[0]-.5*dx)*1e-3,(xGrid[0]-.5*dx)*1e-3],
        #             [(yGrid[0]-.5*dy)*1e-3,(yGrid[0]-.5*dy)*1e-3,(yGrid[-1]+.5*dy)*1e-3,(yGrid[-1]+.5*dy)*1e-3,(yGrid[0]-.5*dy)*1e-3],
        #             '--k')
        # AX[0].set_xlim([xEdges[0]*1e-3, xEdges[-1]*1e-3])
        # AX[0].set_ylim([yEdges[0]*1e-3, yEdges[-1]*1e-3])
        #_______________________
        # for x in xEdges:
        #     AX[1].plot([x*1e-3,x*1e-3], [zEdges[0]*1e-3,zEdges[-1]*1e-3], 'k', lw=.5)
        # for z in zEdges:
        #     AX[1].plot([xEdges[0]*1e-3,xEdges[-1]*1e-3], [z*1e-3,z*1e-3], 'k', lw=.5)
        # AX[1].plot([xGrid[0]*1e-3,xGrid[-1]*1e-3,xGrid[-1]*1e-3,xGrid[0]*1e-3,xGrid[0]*1e-3],
        #             [zGrid[0]*1e-3,zGrid[0]*1e-3,zGrid[-1]*1e-3,zGrid[-1]*1e-3,zGrid[0]*1e-3],
        #             '--k')
        # AX[1].set_xlim([xEdges[0]*1e-3, xEdges[-1]*1e-3])
        # AX[1].set_ylim([zEdges[0]*1e-3, zEdges[-1]*1e-3])
        #_______________________
        # for y in yEdges:
        #     AX[2].plot([y*1e-3,y*1e-3], [zEdges[0]*1e-3,zEdges[-1]*1e-3], 'k', lw=.5)
        # for z in zEdges:
        #     AX[2].plot([yEdges[0]*1e-3,yEdges[-1]*1e-3], [z*1e-3,z*1e-3], 'k', lw=.5)
        # AX[2].plot([yGrid[0]*1e-3,yGrid[-1]*1e-3,yGrid[-1]*1e-3,yGrid[0]*1e-3,yGrid[0]*1e-3],
        #             [zGrid[0]*1e-3,zGrid[0]*1e-3,zGrid[-1]*1e-3,zGrid[-1]*1e-3,zGrid[0]*1e-3],
        #             '--k')
        # AX[2].set_xlim([yEdges[0]*1e-3, yEdges[-1]*1e-3])
        # AX[2].set_ylim([zEdges[0]*1e-3, zEdges[-1]*1e-3])

        # itk = p[0]>250
        AX[0].plot(p[0], p[1], '+', c='#144ff7', ms=.8, alpha=1, rasterized=True)
        # AX[1].plot(p[0]*1e-3, p[2 ]*1e-3, '+', c='#144ff7', ms=.8, alpha=1, rasterized=True)
        # AX[2].plot(p[1]*1e-3, p[2]*1e-3, '+', c='#144ff7', ms=.8, alpha=1, rasterized=True)
        # AX[0].plot(p[0]*1e-3, p[1]*1e-3, '+', c='#f7164c', ms=.8, alpha=1, rasterized=True)
        # AX[1].plot(p[0]*1e-3, p[2]*1e-3, '+', c='#f7164c', ms=.8, alpha=1, rasterized=True)
        # AX[2].plot(p[1]*1e-3, p[2]*1e-3, '+', c='#f7164c', ms=.8, alpha=1, rasterized=True)

        AX[0].set_xlabel('X (km)'); AX[0].set_ylabel('Y')
        AX[1].set_xlabel('X'); AX[1].set_ylabel('Z')
        # AX[2].set_xlabel('Y'); AX[2].set_ylabel('Z')

        plt.tight_layout()
        plt.show()


if 0:
    for i in np.arange(0,1001,10):#np.arange(0,250,10):

        grid = np.loadtxt('products/grid.txt').T

        # print(i, np.amin(p[3]),np.amax(p[3]),np.mean(p[3]))
        # continue

        xGrid = grid[:,0]
        xGrid = xGrid[~np.isnan(xGrid)]
        yGrid = grid[:,1]
        yGrid = yGrid[~np.isnan(yGrid)]
        zGrid = grid[:,2]
        zGrid = zGrid[~np.isnan(zGrid)]

        dx = xGrid[1] - xGrid[0]
        dy = yGrid[1] - yGrid[0]
        dz = zGrid[1] - zGrid[0]
        xEdges = np.append(xGrid-dx/2, xGrid[-1]+dx/2)
        yEdges = np.append(yGrid-dy/2, yGrid[-1]+dy/2)
        zEdges = np.append(zGrid-dz/2, zGrid[-1]+dz/2)

        # fig, AX = plt.subplots(2, 3, figsize=(20,16))
        #
        # AX[0,0].hist(p[0]*1e-3, 50)
        # AX[0,1].hist(p[1]*1e-3, 50)
        # AX[0,2].hist(p[2]*1e-3, 50)
        # AX[1,0].hist(p[3]*1e-3, 50)
        # AX[1,1].hist(p[4]*1e-3, 50)
        # AX[1,2].hist(p[5]*1e-3, 50)
        #
        # AX[0,0].set_xlabel('x (km)')
        # AX[0,1].set_xlabel('y')
        # AX[0,2].set_xlabel('z')
        # AX[1,0].set_xlabel('vx (km)')
        # AX[1,1].set_xlabel('vy')
        # AX[1,2].set_xlabel('vz')
        #
        #
        # plt.tight_layout()
        # plt.show()

        fig, AX = plt.subplots(1, 3, figsize=(20,16))

        for ax in AX:
            ax.set_aspect('equal')

        for x in xEdges:
            AX[0].plot([x*1e-3,x*1e-3], [yEdges[0]*1e-3,yEdges[-1]*1e-3], 'k', lw=.5)
        for y in yEdges:
            AX[0].plot([xEdges[0]*1e-3,xEdges[-1]*1e-3], [y*1e-3,y*1e-3], 'k', lw=.5)
        AX[0].plot([xGrid[0]*1e-3,xGrid[-1]*1e-3,xGrid[-1]*1e-3,xGrid[0]*1e-3,xGrid[0]*1e-3],
                    [yGrid[0]*1e-3,yGrid[0]*1e-3,yGrid[-1]*1e-3,yGrid[-1]*1e-3,yGrid[0]*1e-3],
                    '--k')

        AX[0].set_xlim([xEdges[0]*1e-3, xEdges[-1]*1e-3])
        AX[0].set_ylim([yEdges[0]*1e-3, yEdges[-1]*1e-3])
        #_______________________
        for x in xEdges:
            AX[1].plot([x*1e-3,x*1e-3], [zEdges[0]*1e-3,zEdges[-1]*1e-3], 'k', lw=.5)
        for z in zEdges:
            AX[1].plot([xEdges[0]*1e-3,xEdges[-1]*1e-3], [z*1e-3,z*1e-3], 'k', lw=.5)
        AX[1].plot([xGrid[0]*1e-3,xGrid[-1]*1e-3,xGrid[-1]*1e-3,xGrid[0]*1e-3,xGrid[0]*1e-3],
                    [zGrid[0]*1e-3,zGrid[0]*1e-3,zGrid[-1]*1e-3,zGrid[-1]*1e-3,zGrid[0]*1e-3],
                    '--k')
        AX[1].set_xlim([xEdges[0]*1e-3, xEdges[-1]*1e-3])
        AX[1].set_ylim([zEdges[0]*1e-3, zEdges[-1]*1e-3])
        #_______________________
        for y in yEdges:
            AX[2].plot([y*1e-3,y*1e-3], [zEdges[0]*1e-3,zEdges[-1]*1e-3], 'k', lw=.5)
        for z in zEdges:
            AX[2].plot([yEdges[0]*1e-3,yEdges[-1]*1e-3], [z*1e-3,z*1e-3], 'k', lw=.5)
        AX[2].plot([yGrid[0]*1e-3,yGrid[-1]*1e-3,yGrid[-1]*1e-3,yGrid[0]*1e-3,yGrid[0]*1e-3],
                    [zGrid[0]*1e-3,zGrid[0]*1e-3,zGrid[-1]*1e-3,zGrid[-1]*1e-3,zGrid[0]*1e-3],
                    '--k')
        AX[2].set_xlim([yEdges[0]*1e-3, yEdges[-1]*1e-3])
        AX[2].set_ylim([zEdges[0]*1e-3, zEdges[-1]*1e-3])

        itk = (p[2]>zEdges[int(zEdges.size/2)]) * (p[2]<zEdges[int(zEdges.size/2)+1])
        AX[0].plot(p[0][itk]*1e-3, p[1][itk]*1e-3, 'x', c='#09558e', ms=.8)
        AX[0].plot(p[0,256]*1e-3, p[1,256]*1e-3, 'x', c='#FF0033', ms=4)
        itk = (p[1]>yEdges[int(yEdges.size/2)]) * (p[1]<yEdges[int(yEdges.size/2)+1])
        AX[1].plot(p[0][itk]*1e-3, p[2][itk]*1e-3, 'x', c='#09558e', ms=.8)
        AX[1].plot(p[0,256]*1e-3, p[2,256]*1e-3, 'x', c='#FF0033', ms=4)
        itk = (p[0]>xEdges[int(xEdges.size/2)]) * (p[0]<xEdges[int(xEdges.size/2)+1])
        AX[2].plot(p[1][itk]*1e-3, p[2][itk]*1e-3, 'x', c='#09558e', ms=.8)
        AX[2].plot(p[1,256]*1e-3, p[2,256]*1e-3, 'x', c='#FF0033', ms=4)

        AX[0].set_xlabel('X (km)'); AX[0].set_ylabel('Y')
        AX[1].set_xlabel('X'); AX[1].set_ylabel('Z')
        AX[2].set_xlabel('Y'); AX[2].set_ylabel('Z')

        plt.tight_layout()
        plt.savefig('../Products/positionHybC/{}.png'.format(i))
        plt.close()

if 0:
    velMeanX = []
    velMeanY = []
    velMeanZ = []
    enKinTot = []
    rate = 1000
    traj = np.zeros((6,101))
    nbPart = 54000

    for i in np.arange(1001,10000,rate):#np.arange(0,250,10):
        p = np.loadtxt('products/particles_{}.txt'.format(i)).T

        velMeanX.append(np.mean(p[3]))
        velMeanY.append(np.mean(p[4]))
        velMeanZ.append(np.mean(p[5]))
        enKinTot.append(np.sum(.5*1.661e-27*icaT.norm(p[3:])**2)/nbPart)
        # print(p[3:])
        # print(np.sum(.5*1.661e-27*icaT.norm(p[3:])**2)/nbPart)
        # sys.exit()
        traj[:,int(i/rate)] = p[:,0]

    d = np.loadtxt('products/energies.txt').T

    enKinGrid =  d[3,1:]
    time = d[0,1:]
    # nbPart =  d[4,1:]

    plt.plot(velMeanX, label='x')
    plt.plot(velMeanY, label='y')
    plt.plot(velMeanZ, label='z')
    plt.title('Particles.')
    plt.legend()
    plt.show()

    plt.plot(np.linspace(time[0],time[-1],len(enKinTot)), enKinTot, '+')
    plt.plot(time, enKinGrid, c='#42F496')
    # plt.plot([0,len(enKinTot)], [enKinTot[0], enKinTot[-1]], c='#e52598')
    plt.show()



    fig, ax = plt.subplots(figsize=(16,16))
    ax.set_aspect('equal')

    ax.plot(traj[0], traj[2], '+')

    plt.tight_layout()
    plt.show()
