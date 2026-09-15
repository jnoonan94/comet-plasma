from own_py_modules import *
import own_tools as oT
import own_colours as oC
import aidapy.tools.vdf_utils as vdfu
import aidapy.tools.vdf_plot as vplt
sys.path.append('/home/etienneb/Documents/hvm_vdf')
from hvm_utils import *




if len(sys.argv)>1:
    it = int(sys.argv[1])
else:
    it = 0


path = '/home/etienneb/Models/Menura/menura/'
# DF0 = np.load(f'{path}/products/DF_it0_rank0.npy')
# DF = np.load(f'{path}/products/DF_it{idx_it}_rank0.npy')
DFi = np.abs(np.load(f'{path}/products/DF_it{it}_rank0.npy'))

# DFi = np.abs(np.load(path+'DFi_999.npy'))
# print(np.sum(DFi))#
# sys.exit()
DFi = DFi[2:-2,2:-2,2:-2,2:-2]
mid_ind_v = int(DFi.shape[1]/2.)
mid_ind_x = int(DFi.shape[0]/2.)


# grid = np.loadtxt(path+'grid.txt')
vGrid = np.load(f'{path}/products/vGrid.npy')#grid[1]
vGrid = vGrid[~np.isnan(vGrid)][2:-2]
vLen = vGrid.size
v_max = vGrid[-1]
xGrid = np.load(f'{path}/products/grid_x_rank0.npy')#grid[0]
xGrid = xGrid[~np.isnan(xGrid)][2:-2]
xLen = xGrid.size
dv = vGrid[1]-vGrid[0]
#
nb_distrib = DFi.shape[0]
grid_cart = np.mgrid[-v_max:v_max:vLen*1j,-v_max:v_max:vLen*1j,-v_max:v_max:vLen*1j]

# J = np.nansum(DFi[0]*grid_cart, axis=(1,2,3))
# print(J)
# n = np.nansum(DFi[0])
# print(n)
# print(J/n)
# # sys.exit()
# # v_i = np.nanmean(grid_cart[None,:]*DFi[:,None], axis=(2,3,4))/np.nanmean(DFi, axis=(1,2,3))[:,None]
# # v_i = v_i.T
# # # print(v_i[:,7])
# # # sys.exit()
# vdf_obj = vdfu.vdf(v_max=v_max, resolution=120, grid_geom='spher')
# # vdf_obj.transform_grid(v=-v_i[:,7])
# # grid_cart[0] -= 1.e4
# # vdf_obj.interpolate_cart_vdf(grid_cart, DFi[7].T, interpolate='lin')
# vdf_obj.interpolate_cart_vdf(grid_cart, np.nanmean(DFi, axis=0).T, interpolate='lin')
# vplt.spher(vdf_obj.vdf_interp, vdf_obj.grid_cart, plt_contourf=False)
# # vplt.gyro(vdf_obj.vdf_interp, vdf_obj.grid_spher, vdf_obj.grid_cart)


# resolution = 80
# distrib_scaled_time = np.zeros((nb_distrib, resolution, resolution)) ##
# vdf_obj = vdfu.vdf(v_max=v_max, resolution=resolution, grid_geom='spher')
# for i in np.arange(nb_distrib):
#     if (i % 10 == 0):
#         print('{}/{}'.format(i, nb_distrib), end='\r')
#     # vdf_obj.transform_grid(v=-v_i[:,i])
#     # vdf_obj.transform_grid(R=R[i])
#     vdf_obj.interpolate_cart_vdf(grid_cart, DFi[i], interpolate='lin')
#     distrib_scaled_time[i] = np.nanmean(vdfu.vdf_scaled(vdf_obj.vdf_interp), axis=2)



# sys.exit()
for indIt in [it]:#np.arange(0, 1501, 200):#

    # DFi = np.abs(np.load(path+'DFi_{}.npy'.format(indIt)))
    DFi = np.load(f'{path}/products/DF_it{indIt}_rank0.npy')
    DFi = DFi[2:-2,2:-2,2:-2,2:-2]
    # DFi = np.swapaxes(DFi, 1, 3)    ## Swap vx and vz. B0 has to be along z for the following.
    ## DFi = np.swapaxes(DFi, 2, 3)    ## Swap vx and vz. B0 has to be along z for the following.

    # grid = np.ones((3, DFi.shape[1], DFi.shape[2], DFi.shape[3]))
    # grid[0] *= vGrid[:, None, None]
    # grid[1] *= vGrid[None, :, None]
    # grid[2] *= vGrid[None, None, :]
    #
    # v_sqr = grid[0]**2 + grid[1]**2 + grid[2]**2
    #
    # v_ix = np.sum(DFi*grid[0], axis=(1, 2, 3))/np.sum(DFi, axis=(1, 2, 3))
    # v_iy = np.sum(DFi*grid[1], axis=(1, 2, 3))/np.sum(DFi, axis=(1, 2, 3))
    # v_iz = np.sum(DFi*grid[2], axis=(1, 2, 3))/np.sum(DFi, axis=(1, 2, 3))
    #
    # E_kin = np.sum(DFi*v_sqr[None, :])
    # print(E_kin)
    # continue()



    if 1: # Gyro-angle distribution.
        nb_distrib = DFi.shape[0]
        resolution = 50
        distrib_scaled_time = np.zeros((nb_distrib, resolution, resolution)) ##
        distrib_equa_time = np.zeros((nb_distrib, resolution, resolution))
        # grid_cart = np.mgrid[-v_max:v_max:51*1j,-v_max:v_max:51*1j,-v_max:v_max:51*1j]
        vdf_obj = vdfu.vdf(v_max=v_max, resolution=resolution, grid_geom='spher')
        B_cut = np.ones((3, nb_distrib))
        B_cut[:2] = 0.

        v_i, R = init_bulk_rotate(grid_cart, DFi, B_cut, nb_distrib)

        # resolution = grid_cart.shape[1]
        distrib_equa_time = np.zeros((nb_distrib, resolution, resolution))
        interp_all_3(nb_distrib, vdf_obj, v_i, B_cut, grid_cart, DFi, distrib_scaled_time, distrib_equa_time)
        print('')

        fig, AX = plt.subplots(4, 1, figsize=(13,16), sharex=True)

        resolution = distrib_equa_time.shape[1]

        centers_rho   = vdf_obj.grid_spher[0, :, 0, 0]
        centers_theta = vdf_obj.grid_spher[1, 0, :, 0]
        centers_phi   = vdf_obj.grid_spher[2, 0, 0, :]

        x = xGrid#pos[0]
        y = centers_phi*180./np.pi



        for i, ax in enumerate(AX):
            i_sta = ((2*i+1)*int(resolution/len(AX))-1)/2.
            i_sta = int(i_sta)
            print(i_sta, centers_rho[i_sta])
            # pad = np.nanmean(distrib_scaled_time[:, i_sta:i_sto], axis=(1))
            # pad = np.mean(distrib_equa_time[:, i_sta-2:i_sta+2], axis=1)
            pad = distrib_equa_time[:, i_sta]
            # ax.pcolormesh(x, y, pad.T,
            #               vmin=0, vmax=1,
            #               cmap=oC.bwr_2, rasterized=True)
            ax.contourf(x, y, pad.T,
                          np.linspace(0, 1, 40),
                          cmap=oC.bwr_2, zorder=-20)
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





    mid_ind_v = int(DFi.shape[1]/2.)
    mid_ind_x = int(DFi.shape[0]/2.)

    v_max = 4.

    vdf_obj = vdfu.vdf(v_max=v_max, resolution=120, grid_geom='spher')
    # vdf = np.nanmean(DFi[2:-2,2:-2,2:-2,2:-2], axis=0).T
    vdf = DFi[25]#, 2:-2, 2:-2, 2:-2]
    vdf_obj.interpolate_cart_vdf(grid_cart, vdf, interpolate='lin')
    vplt.profiles_1d(vdf_obj.vdf_interp, vdf_obj.grid_spher)
    # vplt.cart(vdf_obj.vdf_interp, vdf_obj.grid_spher)
    vplt.spher(vdf_obj.vdf_interp, vdf_obj.grid_cart, plt_contourf=False, cmap=oC.bwr_2)
    vplt.spher_gyro(vdf_obj.vdf_interp, vdf_obj.grid_spher, vdf_obj.grid_cart, cmap=oC.bwr_2)
    vplt.gyro(vdf_obj.vdf_interp, vdf_obj.grid_spher, vdf_obj.grid_cart, cmap=oC.bwr_2)
    # fig, ax = plt.subplots(figsize=(14,12))
    #
    # ax.plot(DFi[mid_ind_x, :, mid_ind_v, mid_ind_v])
    # ax.plot(DFi[mid_ind_x, :, mid_ind_v, mid_ind_v], 'x')
    #
    # oT.set_spines(ax)
    # plt.tight_layout()
    # plt.show()

    # a0 = np.log10(DFi)
    a0 = DFi#[2:-2, 2:-2, 2:-2]
    # mid_ind_v = int(DFi.shape[1]/2.)-2

    fig, AX = plt.subplots(3, 1, figsize=(14,12))

    AX[0].pcolormesh(xGrid*1e-3, vGrid*1e-3, a0[:, :, mid_ind_v, mid_ind_v].T,
                     cmap=oC.bwr_2, rasterized=True)
    AX[1].pcolormesh(xGrid*1e-3, vGrid*1e-3, a0[:, mid_ind_v, :, mid_ind_v].T,
                     cmap=oC.bwr_2, rasterized=True)
    AX[2].pcolormesh(xGrid*1e-3, vGrid*1e-3, a0[:, mid_ind_v, mid_ind_v, :].T,
                     cmap=oC.bwr_2, rasterized=True)

    # AX[0].contourf(xGrid*1e-3, vGrid*1e-3, a0[:, :, mid_ind_v, mid_ind_v].T, np.linspace(0, 1, 40),
    #                  cmap=oC.bwr_2, zorder=20)
    # AX[1].contourf(xGrid*1e-3, vGrid*1e-3, a0[:, mid_ind_v, :, mid_ind_v].T, np.linspace(0, 1, 40),
    #                  cmap=oC.bwr_2, zorder=20)
    # AX[2].contourf(xGrid*1e-3, vGrid*1e-3, a0[:, mid_ind_v, mid_ind_v, :].T, np.linspace(0, 1, 40),
    #                  cmap=oC.bwr_2, zorder=20)

    a1 = a0[2:-2, 2:-2, mid_ind_v, mid_ind_v]
    # AX[0].contour(xGrid*1e-3, vGrid*1e-3, a1.T, 20, c='k')
    ind_max = np.unravel_index(a1.argmax(), a1.shape)

    # AX[0].plot(xGrid[ind_max[0]]*1e-3, vGrid[ind_max[1]]*1e-3, 'wx', ms=24)
    # print(xGrid[ind_max[0]]*1e-3, vGrid[ind_max[1]]*1e-3)
    # ax.plot(DFi, 'x')

    oT.set_spines(AX)
    plt.tight_layout()
    plt.show()
