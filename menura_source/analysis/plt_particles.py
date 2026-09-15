import sys
import numpy as np
import matplotlib.pyplot as plt
import own_tools as oT
import own_colours as oC

from scipy.optimize import curve_fit

from menura_utils import *



def maxwellian_1D(v, A, v_th, v0):
    return A * np.exp(-((v-v0)/v_th)**2)




me = 9.10938356e-31  ## kg
mi = 1.660538e-27  ## kg
e = 1.60217e-19   ## C
mu0  = 1.257e-6 ## Permeab
epsilon0 = 8.854e-12  ## F.m-1
c = 299792458.  ## m.s-1
k_B  = 1.3806e-23



# if len(sys.argv)>1:
#     it = int(sys.argv[1])
# else:
#     it = 0
# path = '/home/etienneb/Models/Menura_own/Data/run_tmp'
#
# run_ID = 'tmp'
# path_remote = f'/gpfswork/rech/fuz/ued64ot/run_{run_ID}'
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


sync = input('scp from kebnekaise? (y/[n]) ')
if sync == 'y':
    remote_pswd = input('Jean-Zay password:')
    remote_log = 'ued64ot@jean-zay.idris.fr'
    print(f'scp grids and parameters...')
    os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{path_remote}/products/grid_*_rank* {path}/products')
    os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{path_remote}/products/parameters.txt {path}/products')
    print(f'scp particles iteration nb. {it}...')
    os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{path_remote}/products/particles_0_it{it}* {path}/products')
mp = menura_param(f'{path}')


grid_x = np.load(f'{path}/products/grid_x_rank_0_0.npy')
grid_y = np.load(f'{path}/products/grid_y_rank_0_0.npy')
xGrid = grid_x
yGrid = grid_y
edge_x = np.zeros(mp.len_x_cst+5)
edge_y = np.zeros(mp.len_y_cst+5)

edge_x[ :-1] = grid_x-mp.dX/2.
edge_x[ -1] =  grid_x[ -1]+mp.dX/2
edge_y[ :-1] = grid_y-mp.dX/2.
edge_y[ -1] =  grid_y[ -1]+mp.dX/2
if mp.nb_dim==3:
    grid_z = np.load(f'{path}/products/grid_z_rank_0_0.npy')
    zGrid = grid_z
    edge_z = np.zeros((mp.len_z_cst+5))
    edge_z[ :-1] = grid_z-mp.dX/2.
    edge_z[ -1] =  grid_z[ -1]+mp.dX/2

n0_no_norm = mp.n0_SI
B0_no_norm = mp.B0_SI
Ti_inf = mp.Ti_inf
Te_inf = mp.Te_inf
# v_A = B0_no_norm/np.sqrt(mu0*mi*n0_no_norm)
v_A = 17804.9
gamma_e = 1.
gamma_i = 3.
nb_oscillations = 3.
v_thi  = np.sqrt(2.*k_B*Ti_inf/mi)
v_s = np.sqrt(k_B*(gamma_e*Te_inf+gamma_i*Ti_inf)/mi)/v_A

mp = menura_param(f'{path}', file_name='parameters.h')

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


# if len(sys.argv)>1:
#     indices = [int(sys.argv[1])]
# else:
#     indices = [0]

for ind in [it]:#indices:#np.arange(4000, 8001, 1000):#
    # p = np.load('{}/particles_sorted_{}.npy'.format(path, ind)).T

    # plt_phase_space(path, ind, lp, lg, v_max=3., res_x=512, res_v=256)
    # plt_phase_space2(path, ind, lp, lg, v_max=3., res_x=512, res_v=256)
    # E = np.load(f'{path}/products/E_{ind}.npy')
    # plt_field_spectrum_1d(lg.grid_x, E[0, :, int(lg.len_y/2)], label='Ex')
    # continue

    p = []
    for j in range(mp.mpi_nb_proc_y):
        for k in range(mp.mpi_nb_proc_z):
            r = j*mp.mpi_nb_proc_z + k
            print(r)
            p.append(np.load(f'{path}/products/particles_0_it{ind}_rank_{j}_{k}.npy').T)
            print(np.amin(p[r][1]), np.amax(p[r][1]))
            print(np.load(f'{path}/products/grid_y_rank_{j}_{k}.npy')[-1])
            print(np.amax(p[r][2]))
            print(np.load(f'{path}/products/grid_z_rank_{j}_{k}.npy')[-1])
            print(f'\n {p[r].shape} particles, {np.sum(np.isnan(p[r][2]))} NaNs!\n')
    # p_1 = np.load(f'{path}/products/particles_it{ind}_rank1.npy').T
    # print(f'\n {p_1.shape} particles, {np.sum(np.isnan(p_1[2]))} NaNs!\n')
    # p_2 = np.load(f'{path}/products/particles_it{ind}_rank2.npy').T
    # print(f'\n {p_2.shape} particles, {np.sum(np.isnan(p_2[2]))} NaNs!\n')
    # p_3 = np.load(f'{path}/products/particles_it{ind}_rank3.npy').T
    # print(f'\n {p_3.shape} particles, {np.sum(np.isnan(p_3[2]))} NaNs!\n')
    # print(p)
    # sys.exit()


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

    if 1 and mp.nb_dim==2:
        '''Basic, robust histograms, 2d.'''
        dx = xGrid[1] - xGrid[0]
        dy = yGrid[1] - yGrid[0]
        xEdges = np.append(xGrid-dx/2, xGrid[-1]+dx/2)
        yEdges = np.append(yGrid-dy/2, yGrid[-1]+dy/2)

        bins = np.linspace(-70, 70, 100)

        itk = ((p[0][0]>100) * (p[0][0]<150))


        h_x, bin_x  = np.histogram(p[0][0], 200)
        h_y, bin_y  = np.histogram(p[0][1], 200)
        h_vx, bin_vx = np.histogram(p[0][2], 200)
        h_vy, bin_vy = np.histogram(p[0][3], 200)
        h_vz, bin_vz = np.histogram(p[0][4], 200)
        centers_x = .5*(bin_x[:-1]+bin_x[1:])
        centers_y = .5*(bin_y[:-1]+bin_y[1:])
        centers_vx = .5*(bin_vx[:-1]+bin_vx[1:])
        centers_vy = .5*(bin_vy[:-1]+bin_vy[1:])
        centers_vz = .5*(bin_vz[:-1]+bin_vz[1:])

        try:
            p0 = [np.amax(h_vx), np.var(p[1][2]), np.average(centers_vx, weights=h_vx)]
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

        AX[0, 0].plot(centers_x, h_x)
        AX[0, 1].plot(centers_y, h_y)
        AX[1, 0].plot(centers_vx, np.log10(h_vx))#bins)
        AX[1, 0].plot(centers_vy, np.log10(h_vy), '--k', lw=.5)#bins)
        if fit_success:
            AX[1, 0].plot(centers_vx, np.log10(maxwellian_1D(centers_vx, *p_m)),
                          '--r', lw=.5, label=f'maxwel 1d  exp( -(v-v0)**2/v_th**2 )\nv_th = {p_m[1]:.2e}\nv0={p_m[2]:.2e}')
        AX[1, 1].plot(centers_vy, np.log10(h_vy))#bins)
        AX[1, 2].plot(centers_vz, np.log10(h_vz))#bins)

        AX[1, 0].axvline(mp.v_s/mp.v_A, c='k')
        AX[1, 0].axvline(-mp.v_s/mp.v_A, c='k', label='v_sound')
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

        # AX[1, 0].set_xlim([-9.5, 9.5])
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

    if 1 and mp.nb_dim==3:
        '''3D, Basic, robust histograms.'''
        dx = xGrid[1] - xGrid[0]
        dy = yGrid[1] - yGrid[0]
        dz = zGrid[1] - zGrid[0]
        xEdges = np.append(xGrid-dx/2, xGrid[-1]+dx/2)
        yEdges = np.append(yGrid-dy/2, yGrid[-1]+dy/2)
        zEdges = np.append(zGrid-dz/2, zGrid[-1]+dz/2)

        # bins = np.linspace(-70, 70, 100)

        pp = np.array(p[0])
        print(pp.shape)
        for n in range(1, mp.mpi_nb_proc_tot):
            pp = np.append(pp, p[n], axis=1)
        print(pp)

        h_x, bin_x   = np.histogram(pp[0], 200)
        h_y, bin_y   = np.histogram(pp[1], 200)
        h_z, bin_z   = np.histogram(pp[2], 200)
        h_vx, bin_vx = np.histogram(pp[3], 200)
        h_vy, bin_vy = np.histogram(pp[4], 200)
        h_vz, bin_vz = np.histogram(pp[5], 200)

        centers_x = .5*(bin_x[:-1]+bin_x[1:])
        centers_y = .5*(bin_y[:-1]+bin_y[1:])
        centers_z = .5*(bin_z[:-1]+bin_z[1:])
        centers_vx = .5*(bin_vx[:-1]+bin_vx[1:])
        centers_vy = .5*(bin_vy[:-1]+bin_vy[1:])
        centers_vz = .5*(bin_vz[:-1]+bin_vz[1:])

        try:
            p0 = [np.amax(h_vx), np.var(p[0][2]), np.average(centers_vx, weights=h_vx)]
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

        AX[0, 0].plot(centers_x, h_x)
        AX[0, 1].plot(centers_y, h_y)
        AX[0, 2].plot(centers_z, h_z)
        AX[1, 0].plot(centers_vx, np.log10(h_vx))#bins)
        # AX[1, 0].plot(centers_vy, np.log10(h_vy), '--k', lw=.5)#bins)
        if fit_success:
            AX[1, 0].plot(centers_vx, np.log10(maxwellian_1D(centers_vx, *p_m)),
                          '--r', lw=.5, label=f'maxwel 1d  exp( -(v-v0)**2/v_th**2 )\nv_th = {p_m[1]:.2e}\nv0={p_m[2]:.2e}')
        AX[1, 1].plot(centers_vy, np.log10(h_vy))#bins)
        AX[1, 2].plot(centers_vz, np.log10(h_vz))#bins)

        # AX[1, 0].axvline(v_s, c='k')
        # AX[1, 0].axvline(-v_s, c='k', label='v_sound')
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



    if 0:

        h_x, bin_x  = np.histogramdd((p[0][0], p[0][2]), (100, np.linspace(-4, 4, 100)))

        fig, ax = plt.subplots(figsize=(18, 14))
        ax.imshow((h_x).T,
                  origin='lower',
                  extent=[edge_x[0], edge_x[-1], bin_x[1][0], bin_x[1][-1]],
                  interpolation='bilinear',
                  cmap=oC.bwr_2)

        oT.set_spines(ax)
        plt.tight_layout()
        plt.show()



    if 1 and mp.nb_dim==2:   # All particles positions in the physical domain.

        dx = xGrid[1] - xGrid[0]
        dy = yGrid[1] - yGrid[0]
        xEdges = np.append(xGrid-dx/2, xGrid[-1]+dx/2)
        yEdges = np.append(yGrid-dy/2, yGrid[-1]+dy/2)

        fig, ax = plt.subplots(1, figsize=(20, 16))
        #
        # for ax in AX:
        ax.set_aspect('equal')
        #
        for x in xEdges:
            ax.plot([x, x], [yEdges[0], yEdges[-1]], 'k', lw=.5)
        for y in yEdges:
            ax.plot([xEdges[0], xEdges[-1]], [y, y], 'k', lw=.5)

        for x in xGrid:
            for y in yGrid:
                ax.plot(x, y, 'ko', ms=2)
        ax.plot([(xGrid[0]-.5*dx)*1e-3,(xGrid[-1]+.5*dx)*1e-3,(xGrid[-1]+.5*dx)*1e-3,(xGrid[0]-.5*dx)*1e-3,(xGrid[0]-.5*dx)*1e-3],
                    [(yGrid[0]-.5*dy)*1e-3,(yGrid[0]-.5*dy)*1e-3,(yGrid[-1]+.5*dy)*1e-3,(yGrid[-1]+.5*dy)*1e-3,(yGrid[0]-.5*dy)*1e-3],
                    '--k')
        # ax.set_xlim([xEdges[0]*1e-3, xEdges[-1]*1e-3])
        # ax.set_ylim([yEdges[0]*1e-3, yEdges[-1]*1e-3])

        # _______________________
        # for x in xEdges:
        #     AX[1].plot([x*1e-3,x*1e-3], [zEdges[0]*1e-3,zEdges[-1]*1e-3], 'k', lw=.5)
        # for z in zEdges:
        #     AX[1].plot([xEdges[0]*1e-3,xEdges[-1]*1e-3], [z*1e-3,z*1e-3], 'k', lw=.5)
        # AX[1].plot([xGrid[0]*1e-3,xGrid[-1]*1e-3,xGrid[-1]*1e-3,xGrid[0]*1e-3,xGrid[0]*1e-3],
        #             [zGrid[0]*1e-3,zGrid[0]*1e-3,zGrid[-1]*1e-3,zGrid[-1]*1e-3,zGrid[0]*1e-3],
        #             '--k')
        # AX[1].set_xlim([xEdges[0]*1e-3, xEdges[-1]*1e-3])
        # AX[1].set_ylim([zEdges[0]*1e-3, zEdges[-1]*1e-3])
        # _______________________
        # for y in yEdges:
        #     AX[2].plot([y*1e-3,y*1e-3], [zEdges[0]*1e-3,zEdges[-1]*1e-3], 'k', lw=.5)
        # for z in zEdges:
        #     AX[2].plot([yEdges[0]*1e-3,yEdges[-1]*1e-3], [z*1e-3,z*1e-3], 'k', lw=.5)
        # AX[2].plot([yGrid[0]*1e-3,yGrid[-1]*1e-3,yGrid[-1]*1e-3,yGrid[0]*1e-3,yGrid[0]*1e-3],
        #             [zGrid[0]*1e-3,zGrid[0]*1e-3,zGrid[-1]*1e-3,zGrid[-1]*1e-3,zGrid[0]*1e-3],
        #             '--k')
        # AX[2].set_xlim([yEdges[0]*1e-3, yEdges[-1]*1e-3])
        # AX[2].set_ylim([zEdges[0]*1e-3, zEdges[-1]*1e-3])

        for r in [0]:# range(mp.mpi_nb_proc):
            ax.plot(p[r][0], p[r][1], '+', c=oC.rgb2[0], ms=.8, alpha=1, rasterized=True)
        # ax.plot(p[1][0], p[1][1], '+', c=oC.rgb2[2], ms=.8, alpha=1, rasterized=True)
            # ax.plot(p_1[0], p_1[1], '+', c=oC.rgb2[1], ms=.8, alpha=1, rasterized=True)
            # ax.plot(p_2[0], p_2[1], '+', c=oC.rgb2[2], ms=.8, alpha=1, rasterized=True)
            # ax.plot(p_3[0], p_3[1], '+', c=oC.rgb2[0], ms=.8, alpha=1, rasterized=True)
        # ax.set_xlim([0, 3])
        # ax.set_ylim([14, 17])
        # AX[1].plot(p[0]*1e-3, p[2 ]*1e-3, '+', c='#144ff7', ms=.8, alpha=1, rasterized=True)
        # AX[2].plot(p[1]*1e-3, p[2]*1e-3, '+', c='#144ff7', ms=.8, alpha=1, rasterized=True)
        # AX[0].plot(p[0]*1e-3, p[1]*1e-3, '+', c='#f7164c', ms=.8, alpha=1, rasterized=True)
        # AX[1].plot(p[0]*1e-3, p[2]*1e-3, '+', c='#f7164c', ms=.8, alpha=1, rasterized=True)
        # AX[2].plot(p[1]*1e-3, p[2]*1e-3, '+', c='#f7164c', ms=.8, alpha=1, rasterized=True)

        # ax.set_xlabel('X (km)'); AX[0].set_ylabel('Y')
        # AX[1].set_xlabel('X'); AX[1].set_ylabel('Z')
        # # AX[2].set_xlabel('Y'); AX[2].set_ylabel('Z')

        plt.tight_layout()
        plt.show()


    if 1 and mp.nb_dim==3:   # 3D, all particles positions in the physical domain.



        fig, AX = plt.subplots(1, 3, figsize=(20, 16))
        #
        for ax in AX:
            ax.set_aspect('equal')
        #
        # for rank_y in range(mp.mpi_nb_proc_y):
        #     for rank_z in range(mp.mpi_nb_proc_z):
        #         xGrid = np.load(f'{path}/products/grid_x_rank_{rank_y}_{rank_z}.npy')
        #         yGrid = np.load(f'{path}/products/grid_y_rank_{rank_y}_{rank_z}.npy')
        #         zGrid = np.load(f'{path}/products/grid_z_rank_{rank_y}_{rank_z}.npy')
        #
        #         dx = xGrid[1] - xGrid[0]
        #         dy = yGrid[1] - yGrid[0]
        #         dz = zGrid[1] - zGrid[0]
        #         xEdges = np.append(xGrid-dx/2, xGrid[-1]+dx/2)
        #         yEdges = np.append(yGrid-dy/2, yGrid[-1]+dy/2)
        #         zEdges = np.append(zGrid-dz/2, zGrid[-1]+dz/2)
        #         for x in xEdges:
        #             AX[0].plot([x, x], [yEdges[0], yEdges[-1]], 'k', lw=.5)
        #             AX[1].plot([x, x], [zEdges[0], zEdges[-1]], 'k', lw=.5)
        #         for y in yEdges:
        #             AX[0].plot([xEdges[0], xEdges[-1]], [y, y], 'k', lw=.5)
        #             AX[2].plot([zEdges[0], zEdges[-1]], [y, y], 'k', lw=.5)
        #         for z in zEdges:
        #             AX[1].plot([xEdges[0], xEdges[-1]], [z, z], 'k', lw=.5)
        #             AX[2].plot([z, z], [yEdges[0], yEdges[-1]], 'k', lw=.5)
        #
        #         for x in xGrid:
        #             for y in yGrid:
        #                 AX[0].plot(x, y, 'ko', ms=2)
        #         for x in xGrid:
        #             for z in zGrid:
        #                 AX[1].plot(x, z, 'ko', ms=2)
        #         for y in yGrid:
        #             for z in zGrid:
        #                 AX[2].plot(z, y, 'ko', ms=2)
        #
        #         AX[0].plot([(xGrid[0]-.5*dx)*1e-3,(xGrid[-1]+.5*dx)*1e-3,(xGrid[-1]+.5*dx)*1e-3,(xGrid[0]-.5*dx)*1e-3,(xGrid[0]-.5*dx)*1e-3],
        #                     [(yGrid[0]-.5*dy)*1e-3,(yGrid[0]-.5*dy)*1e-3,(yGrid[-1]+.5*dy)*1e-3,(yGrid[-1]+.5*dy)*1e-3,(yGrid[0]-.5*dy)*1e-3],
        #                     '--k')
        #         AX[1].plot([(xGrid[0]-.5*dx)*1e-3,(xGrid[-1]+.5*dx)*1e-3,(xGrid[-1]+.5*dx)*1e-3,(xGrid[0]-.5*dx)*1e-3,(xGrid[0]-.5*dx)*1e-3],
        #                     [(yGrid[0]-.5*dy)*1e-3,(yGrid[0]-.5*dy)*1e-3,(yGrid[-1]+.5*dy)*1e-3,(yGrid[-1]+.5*dy)*1e-3,(yGrid[0]-.5*dy)*1e-3],
        #                     '--k')
        #         AX[2].plot([(yGrid[0]-.5*dy)*1e-3,(yGrid[0]-.5*dy)*1e-3,(yGrid[-1]+.5*dy)*1e-3,(yGrid[-1]+.5*dy)*1e-3,(yGrid[0]-.5*dy)*1e-3],
        #                     [(zGrid[0]-.5*dz)*1e-3,(zGrid[-1]+.5*dz)*1e-3,(zGrid[-1]+.5*dz)*1e-3,(zGrid[0]-.5*dz)*1e-3,(zGrid[0]-.5*dz)*1e-3],
        #                     '--k')
        #         # ax.set_xlim([xEdges[0]*1e-3, xEdges[-1]*1e-3])
        #         # ax.set_ylim([yEdges[0]*1e-3, yEdges[-1]*1e-3])
        # # _______________________
        # # for x in xEdges:
        # #     AX[1].plot([x*1e-3,x*1e-3], [zEdges[0]*1e-3,zEdges[-1]*1e-3], 'k', lw=.5)
        # # for z in zEdges:
        # #     AX[1].plot([xEdges[0]*1e-3,xEdges[-1]*1e-3], [z*1e-3,z*1e-3], 'k', lw=.5)
        # # AX[1].plot([xGrid[0]*1e-3,xGrid[-1]*1e-3,xGrid[-1]*1e-3,xGrid[0]*1e-3,xGrid[0]*1e-3],
        # #             [zGrid[0]*1e-3,zGrid[0]*1e-3,zGrid[-1]*1e-3,zGrid[-1]*1e-3,zGrid[0]*1e-3],
        # #             '--k')
        # # AX[1].set_xlim([xEdges[0]*1e-3, xEdges[-1]*1e-3])
        # # AX[1].set_ylim([zEdges[0]*1e-3, zEdges[-1]*1e-3])
        # # _______________________
        # # for y in yEdges:
        # #     AX[2].plot([y*1e-3,y*1e-3], [zEdges[0]*1e-3,zEdges[-1]*1e-3], 'k', lw=.5)
        # # for z in zEdges:
        # #     AX[2].plot([yEdges[0]*1e-3,yEdges[-1]*1e-3], [z*1e-3,z*1e-3], 'k', lw=.5)
        # # AX[2].plot([yGrid[0]*1e-3,yGrid[-1]*1e-3,yGrid[-1]*1e-3,yGrid[0]*1e-3,yGrid[0]*1e-3],
        # #             [zGrid[0]*1e-3,zGrid[0]*1e-3,zGrid[-1]*1e-3,zGrid[-1]*1e-3,zGrid[0]*1e-3],
        # #             '--k')
        # # AX[2].set_xlim([yEdges[0]*1e-3, yEdges[-1]*1e-3])
        # # AX[2].set_ylim([zEdges[0]*1e-3, zEdges[-1]*1e-3])

        # for r_y in range(mp.mpi_nb_proc_y):
        #     for r_z in range(mp.mpi_nb_proc_z):
        #         #     continue
        AX[0].plot(p[:, 0], p[:, 1], '+', ms=.8, alpha=1, rasterized=True)#, c=oC.rgb2[r%mp.mpi_nb_proc_tot]
        # AX[1].plot(p[:, 0], p[:, 2], '+', ms=.8, alpha=1, rasterized=True)#, c=oC.rgb2[r%mp.mpi_nb_proc_tot]
        # AX[2].plot(p[:, 2], p[:, 1], '+', ms=.8, alpha=1, rasterized=True)#, c=oC.rgb2[r%mp.mpi_nb_proc_tot]
            # ax.plot(p_1[0], p_1[1], '+', c=oC.rgb2[1], ms=.8, alpha=1, rasterized=True)
            # ax.plot(p_2[0], p_2[1], '+', c=oC.rgb2[2], ms=.8, alpha=1, rasterized=True)
            # ax.plot(p_3[0], p_3[1], '+', c=oC.rgb2[0], ms=.8, alpha=1, rasterized=True)
        # ax.set_xlim([0, 3])
        # ax.set_ylim([14, 17])
        # AX[1].plot(p[0]*1e-3, p[2 ]*1e-3, '+', c='#144ff7', ms=.8, alpha=1, rasterized=True)
        # AX[2].plot(p[1]*1e-3, p[2]*1e-3, '+', c='#144ff7', ms=.8, alpha=1, rasterized=True)
        # AX[0].plot(p[0]*1e-3, p[1]*1e-3, '+', c='#f7164c', ms=.8, alpha=1, rasterized=True)
        # AX[1].plot(p[0]*1e-3, p[2]*1e-3, '+', c='#f7164c', ms=.8, alpha=1, rasterized=True)
        # AX[2].plot(p[1]*1e-3, p[2]*1e-3, '+', c='#f7164c', ms=.8, alpha=1, rasterized=True)

        AX[0].set_xlabel('X')
        AX[0].set_ylabel('Y')
        AX[1].set_xlabel('X')
        AX[1].set_ylabel('Z')
        AX[2].set_xlabel('Z')
        AX[2].set_ylabel('Y')
        # AX[1].set_xlabel('X'); AX[1].set_ylabel('Z')
        # # AX[2].set_xlabel('Y'); AX[2].set_ylabel('Z')

        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots()
        # for r in range(mp.mpi_nb_proc):
        itk = p[:, 4]==0.
        plt.plot(p[:, 0, itk], p[:, 1, itk], 'xb')
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
