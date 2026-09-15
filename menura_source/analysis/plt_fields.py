# import sys
import os
try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    user_paths = []
import numpy as np
import matplotlib.pyplot as plt


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




#
# for it in np.arange(300, 801, 10):
#
#     md = menura_data(path, path_remote, it, remote_label=remote_label,
#                      ask_scp=False, print_param=False, full_scp=True )
#     md.plt_field('B_perp_sqr', plane='xy', vminmax=[0, 1], save_fig=True)
# sys.exit()


md = menura_data(path, path_remote, it, remote_label=remote_label,
                 ask_scp=True, print_param=False, full_scp=True )



# md.mask_body_field()

# md.plt_all()
# sys.exit()

# md.plt_field('Bx', plane='xy')
# sys.exit()

#
# md.produce_ohm()
# md.plt_ohm()
# sys.exit()
# md.plt_tomo()

# md.plt_field('d_i', log=True, vminmax=[-.5, .5])
md.plt_field('dens_tot', log=True, plane='xy', vminmax=[-1, 1])
md.plt_field('dens_spec_0', log=True, plane='xy', vminmax=[-1, 1])
# # md.plt_field('dens_spec_0', plane='xz')
md.plt_field('dens_spec_1', plane='xy', log=True, vminmax=[-3, 2])#, xylim=[[290, 340], [220, 270]])
md.plt_field('dens_spec_1', plane='xy', log=True, vminmax=[-1, 2])
# # md.plt_field('dens_spec_1', plane='xz')
# # md.plt_field('region_ID', plane='xy')
# # md.plt_field('region_ID', plane='xz')
# # md.plt_field('region_ID', plane='yz')
# # md.plt_field('B+B_dip', plane='xy')
# # md.plt_field('B_dipx', plane='xy')
# # md.plt_field('B_dipy', plane='xy')
# # md.plt_field('B_dipz', plane='xy')


md.plt_field('B', plane='xy', log=True, vminmax=[-1., 1.])#, xylim=[[290, 340], [220, 270]])
# # md.plt_field('B', plane='xz')
md.plt_field('Bx')
md.plt_field('By')
md.plt_field('Bz')
# md.plt_field('Ex', xylim=[[240, 340], [190, 290]], vminmax=[-20, 20])#, vminmax=[-50, 50])
# md.plt_field('Ey')#, vminmax=[-50, 50])
# md.plt_field('Ez')#, vminmax=[-50, 50])
# md.plt_field('div_B', vminmax=[-1e-2, 1e-2])
# md.plt_field('EdotJ', vminmax=[-50, 50])
# # md.plt_field('E', plane='xy', vminmax=[0, 150])
# # md.plt_field('E', plane='xz', vminmax=[0, 150])
# # md.plt_field('E', plane='yz', vminmax=[0, 150])
# md.plt_field('uix')
# md.plt_field('uiy')
# md.plt_field('uiz')
# md.plt_field('Jx')
# md.plt_field('Jy')
# md.plt_field('Jz', vminmax=[-6, 6], xylim=[[240, 340], [190, 290]], interpolate=True)
# md.plt_field('B_perp_sqr', plane='xy', vminmax=[0, 1.])
# md.plt_field('B_perp_sqr', plane='xz')
# md.plt_field('B_perp_sqr', plane='yz')
md.plt_field('B_perp', log=True, vminmax=[-1.5, 1.], xylim=[[240, 340], [190, 290]])
# md.plt_field('Jiy', plane='xy', log=True, vminmax=[-4, 4], xylim=[[240, 340], [190, 290]])
# md.plt_field('ui_perp_sqr', plane='xy', vminmax=[0, .4])


# md.plt_field_3d(field_label='B_perp')
# md.plt_field_3d(field_label='B')
# md.plt_field_3d(field_label='dens')

# md.plt_B_perp(streamplot=False)
# md.plt_B_perp(streamplot=True)
# md.plt_field('B_perp', vminmax=[-1., .75])
# # md.plt_field('B_perp', vminmax=[-1., .75], filled_contour=True)
#
# md.plt_field('Bz')#, vminmax=[-2., -.0])


# md.plt_field('Ex', vminmax=[-5., 5], save_fig=False, filled_contour=True)
#
# md.produce_spectra()
# md.plt_field_2d_spectrum('B_perp_sqr')
# md.plt_field_2d_spectrum('J_tot_z')
# md.plt_field_2d_spectrum('ui_perp_sqr')

# md.plt_ohm(save_fig=False)
# md.plt_field('Ex', vminmax=[-10, 10], save_fig=True)
# md.plt_field('Ey', vminmax=[-10, 10], save_fig=True)
# md.plt_B_perp()
# md.plt_v_perp()
# md.plt_E_perp()

if 0:   ## Alfven speed.
    B = md.recompose_field(md.B)
    n = md.recompose_field(md.dens_spec0+md.dens_spec1)
    v_A = oT.norm(B)/np.sqrt(n)

    plt.pcolormesh(v_A.T, cmap=oC.bwr_2)
    plt.show()


if 0:   ## Compare energies vs. laminar case.
    B = md.recompose_field(md.B)
    B_lam = np.zeros_like(B)
    B_lam[2] = 1.
    print(np.sum(B[0]**2+B[1]**2+B[2]**2))
    print(np.sum(B_lam[0]**2+B_lam[1]**2+B_lam[2]**2))

    B = md.recompose_field(md.curr)
    print(np.sum(B[0]**2+B[1]**2+B[2]**2))

if 0:

    for it in np.arange(0, 10, 1):
        print(f'{it}', end='\r')
        md = menura_data(path, path_remote, it, ask_scp=False, print_param=False, remote_label=remote_label)
    #     # md.plt_field('Bz', save_fig=True)
        md.plt_all(save_fig=True)
        if 0:   ## Plot global B-field lines in cuts.
            B = md.recompose_field(md.B)

            idx_mid_x = int(md.mp.len_x_cst/2.)
            idx_mid_y = int(md.mp.len_y_cst/2.)
            idx_mid_z = int(md.mp.len_z_cst/2.)

            # fig, AX = plt.subplots(1, 3, figsize=(18, 14))
            # for ax in AX:
            #     ax.set_aspect('equal')
            #
            # AX[0].streamplot(md.grid_x_box, md.grid_y_box, B[0, :, :, idx_mid_z].T, B[1, :, :, idx_mid_z].T,
            #                       density=3,
            #                       # start_points=start_points,
            #                       color='k')#, linewidth=np.log10(B_perp).T)
            # AX[1].streamplot(md.grid_x_box, md.grid_z_box, B[0, :, idx_mid_y].T, B[2, :, idx_mid_y].T,
            #                       density=3,
            #                       # start_points=start_points,
            #                       color='k')#, linewidth=np.log10(B_perp).T)
            # AX[2].streamplot(md.grid_z_box, md.grid_y_box, B[2, idx_mid_x], B[1, idx_mid_x],
            #                       density=3,
            #                       # start_points=start_points,
            #                       color='k')#, linewidth=np.log10(B_perp).T)
            #
            # # AX[0].quiver(md.grid_x_box, md.grid_y_box, B[0, :, :, idx_mid_z].T, B[1, :, :, idx_mid_z].T,
            # #                       # density=3,
            # #                       # start_points=start_points,
            # #                       color='k')#, linewidth=np.log10(B_perp).T)
            # # print(B[0, :, idx_mid_y])
            # # print( B[2, :, idx_mid_y])
            # # AX[1].quiver(md.grid_x_box, md.grid_z_box, B[0, :, idx_mid_y].T, B[2, :, idx_mid_y].T,
            # #                       # density=3,
            # #                       # start_points=start_points,
            # #                       color='k')#, linewidth=np.log10(B_perp).T)
            # # AX[2].quiver(md.grid_z_box, md.grid_y_box, B[2, idx_mid_x], B[1, idx_mid_x],
            # #                       # density=3,
            # #                       # start_points=start_points,
            # #                       color='k')#, linewidth=np.log10(B_perp).T)
            #
            # plt.tight_layout()
            # oT.set_spines(AX)
            # # plt.show()
            # plt.savefig(f'/home/etienneb/Desktop/plot_tmp/{it}.png')

            fig, ax = plt.subplots(figsize=(14, 14))
            ax.set_aspect('equal')

            ax.streamplot(md.grid_x_box, md.grid_y_box, B[0, :, :, idx_mid_z].T, B[1, :, :, idx_mid_z].T,
                                  density=5,
                                  color='k')#, linewidth=np.log10(B_perp).T)
            plt.tight_layout()
            oT.set_spines(ax)
            # plt.show()
            plt.savefig(f'/home/etienneb/Desktop/plot_tmp/{it}.png')
        # md.plt_B_perp(streamplot=False, save_fig=True)

if md.mp.NB_DIM==3:

    B = md.B
    # B = md.stag_vector(B))
    idx_mid_x = int((md.mp.len_x_cst+4)/2.)
    idx_mid_y = int((md.mp.len_y_cst+4)/2.)
    idx_mid_z = int((md.mp.len_z_cst+4)/2.)


    if 0:   ## Slice of B-field amplitude.

        c = oT.norm(B[:, :, :, idx_mid_z]*md.mp.B0_SI*10**9)
        # c = np.log10(c)
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.set_aspect('equal')

        p0 = ax.pcolormesh(md.grid_x_box, md.grid_z_box, c.T,
                           vmin=0, vmax=300,
                            cmap=oC.bwr_2)

        t = np.linspace(0, 2*np.pi, 200)
        r_obs = md.mp.r_obs/md.mp.d_i
        ax.plot(r_obs*np.cos(t)+md.max_x*md.mp.centre_x+md.mp.dX,
                r_obs*np.sin(t)+md.max_y*md.mp.centre_y+md.mp.dX, 'k', lw=2)


        plt.colorbar(p0)

        plt.show()#

    # sys.exit()

    if 0:   ## B lines in (x, y), zoomed and HR
        x0 = .15*md.grid_x_box[-1]; x1 = .85*md.grid_x_box[-1]
        y0 = .15*md.grid_y_box[-1]; y1 = .85*md.grid_y_box[-1]
        itk_x = (md.grid_x_box>x0) * (md.grid_x_box<x1)
        itk_y = (md.grid_y_box>y0) * (md.grid_y_box<y1)
        itk = itk_x[:, None]*itk_y[None, :]
        idx_mid_z = int((md.mp.len_z_cst+4)/2)
        B = md.recompose_field(md.B)
        # B = md.stag_vector(B)

        fig, ax = plt.subplots(figsize=(14, 14))
        ax.set_aspect('equal')

        # p0 = ax.imshow(dens_com.T,
        #                   vmin=0, vmax=2,
        #                   extent=(md.edges_x_box[0], md.edges_x_box[-1], md.edges_y_box[0], md.edges_y_box[-1]),
        #                   interpolation='bilinear',
        #                   rasterized=True, cmap=oC.wbr_1, origin='lower')

        colours = np.zeros((1000,1000))
        # colours[B[1]>0] = 1
        # colours[0, B[1]<0] = 2
        # c = np.stack((colours[0][itk], colours[1][itk], colours[2][itk])).reshape((3, np.sum(itk_x), np.sum(itk_y)))
        # print(c.shape)
        ax.streamplot(md.grid_x_box[itk_x], md.grid_y_box[itk_y],
                     # md.E_mot[0][itk]+md.E_hal[0][itk]+md.E_amb[0][itk]+md.E_hyp_res[0][itk],
                     # md.E_mot[1][itk]+md.E_hal[1][itk]+md.E_amb[1][itk]+md.E_hyp_res[1][itk],
                     # md.E_mot[0][itk], md.E_mot[1][itk],
                     B[0,:,:,idx_mid_z][itk].reshape((np.sum(itk_x), np.sum(itk_y))).T, B[1,:,:,idx_mid_z][itk].reshape((np.sum(itk_x), np.sum(itk_y))).T,
                     # color=colours[itk].reshape((np.sum(itk_x), np.sum(itk_y))).T,
                     # cmap=oC.bwr_2,
                     density=6,
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

        ax.set_xlim([x0, x1])
        ax.set_ylim([y0, y1])

        oT.set_spines(ax)
        plt.tight_layout()
        plt.show()


    if 0:   ## B lines in (x, z), zoomed and HR
        x0 = .15*md.grid_x_box[-1]; x1 = .85*md.grid_x_box[-1]
        z0 = .15*md.grid_y_box[-1]; z1 = .85*md.grid_y_box[-1]
        itk_x = (md.grid_x_box>x0) * (md.grid_x_box<x1)
        itk_z = (md.grid_z_box>z0) * (md.grid_y_box<z1)
        itk = itk_x[:, None]*itk_z[None, :]
        idx_mid_y = int((md.mp.len_y_cst+4)/2)
        B = md.recompose_field(md.B)
        # B = md.stag_vector(B)

        fig, ax = plt.subplots(figsize=(14, 14))
        ax.set_aspect('equal')

        # p0 = ax.imshow(dens_com.T,
        #                   vmin=0, vmax=2,
        #                   extent=(md.edges_x_box[0], md.edges_x_box[-1], md.edges_y_box[0], md.edges_y_box[-1]),
        #                   interpolation='bilinear',
        #                   rasterized=True, cmap=oC.wbr_1, origin='lower')

        colours = np.zeros((1000,1000))
        # colours[B[1]>0] = 1
        # colours[0, B[1]<0] = 2
        # c = np.stack((colours[0][itk], colours[1][itk], colours[2][itk])).reshape((3, np.sum(itk_x), np.sum(itk_y)))
        # print(c.shape)
        ax.streamplot(md.grid_x_box[itk_x], md.grid_z_box[itk_z],
                     # md.E_mot[0][itk]+md.E_hal[0][itk]+md.E_amb[0][itk]+md.E_hyp_res[0][itk],
                     # md.E_mot[1][itk]+md.E_hal[1][itk]+md.E_amb[1][itk]+md.E_hyp_res[1][itk],
                     # md.E_mot[0][itk], md.E_mot[1][itk],
                     B[0,:,idx_mid_y][itk].reshape((np.sum(itk_x), np.sum(itk_z))).T, B[2,:,idx_mid_y][itk].reshape((np.sum(itk_x), np.sum(itk_z))).T,
                     # color=colours[itk].reshape((np.sum(itk_x), np.sum(itk_y))).T,
                     # cmap=oC.bwr_2,
                     density=6,
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

        ax.set_xlim([x0, x1])
        ax.set_ylim([z0, z1])

        oT.set_spines(ax)
        plt.tight_layout()
        plt.show()

    if 0:   ## Plot global B-field lines in cuts.

        fig, AX = plt.subplots(1, 3, figsize=(18, 14))
        for ax in AX:
            ax.set_aspect('equal')

        AX[0].streamplot(md.grid_x_box, md.grid_y_box, B[0, :, :, idx_mid_z].T, B[1, :, :, idx_mid_z].T,
                              density=3,
                              # start_points=start_points,
                              color='k')#, linewidth=np.log10(B_perp).T)
        AX[1].streamplot(md.grid_x_box, md.grid_z_box, B[0, :, idx_mid_y].T, B[2, :, idx_mid_y].T,
                              density=3,
                              # start_points=start_points,
                              color='k')#, linewidth=np.log10(B_perp).T)
        AX[2].streamplot(md.grid_z_box, md.grid_y_box, B[2, idx_mid_x], B[1, idx_mid_x],
                              density=3,
                              # start_points=start_points,
                              color='k')#, linewidth=np.log10(B_perp).T)

        # AX[0].quiver(md.grid_x_box, md.grid_y_box, B[0, :, :, idx_mid_z].T, B[1, :, :, idx_mid_z].T,
        #                       # density=3,
        #                       # start_points=start_points,
        #                       color='k')#, linewidth=np.log10(B_perp).T)
        # print(B[0, :, idx_mid_y])
        # print( B[2, :, idx_mid_y])
        # AX[1].quiver(md.grid_x_box, md.grid_z_box, B[0, :, idx_mid_y].T, B[2, :, idx_mid_y].T,
        #                       # density=3,
        #                       # start_points=start_points,
        #                       color='k')#, linewidth=np.log10(B_perp).T)
        # AX[2].quiver(md.grid_z_box, md.grid_y_box, B[2, idx_mid_x], B[1, idx_mid_x],
        #                       # density=3,
        #                       # start_points=start_points,
        #                       color='k')#, linewidth=np.log10(B_perp).T)

        plt.tight_layout()
        oT.set_spines(AX)
        plt.show()

    md.plt_field_3d(field_label='dens')
    md.plt_ohm()
    # md.plt_field_3d(field_label='B')
    # md.plt_field_3d(field_label='E')
    md.plt_tomo()
    md.plt_field_3d(field_label='B_perp')
    # PSD = md.produce_omnidirectional_PSD_3D(field_label='B_perp')
    # print(PSD.shape)
    # md.plt_field_3d(field_label='dens')
    # md.plt_field_3d(field_label='E')
    # md.plt_field_3d(field_label='dens_pla')
    # md.plt_field_3d(field_label='curr')
