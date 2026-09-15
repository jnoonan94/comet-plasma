# import sys
import os
try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    user_paths = []
print(user_paths)
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
path_remote = f'/gpfswork/rech/fuz/ued64ot/run_{run_ID}'
remote_label = 'jean-zay'





md = menura_data(path, path_remote, it, ask_scp=True, print_param=True,
                 remote_label=remote_label, full_scp=False)
#
# md.plt_all()#

B = md.load_field('B')


dx = md.mp.dX

if md.mp.NB_DIM==2:
    dxBx = 1./(2.*dx) * ( -B[0, 3:-1, 2:-2]+B[0, 1:-3, 2:-2] )
    dyBy = 1./(2.*dx) * ( -B[1, 2:-2, 3:-1]+B[1, 2:-2, 1:-3] )
    dxBx = 1./(12.*dx) * ( B[0, 4:  , 2:-2]-8*B[0, 3:-1, 2:-2]+8*B[0, 1:-3, 2:-2]-B[0,  :-4, 2:-2] )
    dyBy = 1./(12.*dx) * ( B[1, 2:-2, 4:  ]-8*B[1, 2:-2, 3:-1]+8*B[1, 2:-2, 1:-3]-B[1, 2:-2,  :-4] )
    div_B = dxBx + dyBy

    # div_B = md.load_field('div_B')

    curl_E = md.load_field('curl_E')/md.mp.dt


    try:
        print(md.mp.yee_cst)
    except:
        md.mp.yee_cst = False

    if md.mp.yee_cst:
        div_curl_E = curl_E[0, 1:, :-1] - curl_E[0, :-1, :-1] + curl_E[1, :-1, 1:] - curl_E[1, :-1, :-1]
    else:
        dxcurlEx = 1./(12.*dx) * ( curl_E[0, 4:  , 2:-2]-8*curl_E[0, 3:-1, 2:-2]+8*curl_E[0, 1:-3, 2:-2]-curl_E[0,  :-4, 2:-2] )
        dycurlEy = 1./(12.*dx) * ( curl_E[1, 2:-2, 4:  ]-8*curl_E[1, 2:-2, 3:-1]+8*curl_E[1, 2:-2, 1:-3]-curl_E[1, 2:-2,  :-4] )
        div_curl_E = dxcurlEx + dycurlEy
    # print(div_curl_E.shape)

    # sys.exit()



    print(f'Mean of div(B): {np.mean(div_B[4:-4,4:-4])}')
    print(f'Max of abs(div(B)): {np.amax(np.abs(div_B))}')


    # fig, AX = plt.subplots(1, 2, figsize=(14, 14))
    #
    # vmax = np.amax(np.abs(curl_E))*.2
    # p0 = AX[0].imshow(curl_E[2].T,
    #           vmin=-vmax, vmax=vmax,
    #           extent=(md.edges_x[0], md.edges_x[-1], md.edges_y[0], md.edges_y[-1]),
    #           interpolation='none',
    #           rasterized=True, cmap=oC.bwr_2, origin='lower')
    #
    # vmax = np.amax(np.abs(div_curl_E[4:-4, 4:-4]))*.2
    # vmax = 1e-8
    # p1 = AX[1].imshow(div_curl_E[2:-4, 4:-4].T,
    #           vmin=-vmax, vmax=vmax,
    #           extent=(md.edges_x[0], md.edges_x[-1], md.edges_y[0], md.edges_y[-1]),
    #           interpolation='none',
    #           rasterized=True, cmap=oC.bwr_2, origin='lower')
    #
    # plt.suptitle('curl(E)_x and div(curl(E))')
    # plt.colorbar(p0, ax=AX[0], orientation='horizontal')
    # plt.colorbar(p1, ax=AX[1], orientation='horizontal')
    # plt.tight_layout()
    # oT.set_spines(AX)
    # plt.show()



    fig, AX = plt.subplots(1, 2, figsize=(14, 14), sharex=True, sharey=True)

    # p0 = AX[0].imshow(oT.norm(B[:,2:-2,2:-2]).T,
    #           # vmin=-1, vmax=1,
    #           extent=(md.edges_x[0], md.edges_x[-1], md.edges_y[0], md.edges_y[-1]),
    #           interpolation='none',
    #           rasterized=True, cmap=oC.bwr_2, origin='lower')
    vmax = np.amax(np.abs(B[0]))*.8
    p0 = AX[0].imshow(B[0].T,
              vmin=-vmax, vmax=vmax,
              extent=(md.edges_x[0], md.edges_x[-1], md.edges_y[0], md.edges_y[-1]),
              interpolation='none',
              rasterized=True, cmap=oC.bwr_2, origin='lower')
    # vmax = np.amax(np.abs(dxBx))
    # # vmax *= .2
    # p0 = AX[0].imshow(dxBx.T,
    #         vmin=-vmax, vmax=vmax,
    #         extent=(md.edges_x[0], md.edges_x[-1], md.edges_y[0], md.edges_y[-1]),
    #         interpolation='none',
    #         rasterized=True, cmap=oC.bwr_2, origin='lower')

    vmax = np.amax(np.abs(div_B[14:-4, 4:-4]))#*.8
    vmax = 1e-2
    p1 = AX[1].imshow(div_B[14:-4, 4:-4].T,
              vmin=-vmax, vmax=vmax,
              extent=(md.edges_x[0], md.edges_x[-1], md.edges_y[0], md.edges_y[-1]),
              interpolation='none',
              rasterized=True, cmap=oC.bwr_2, origin='lower')

    if md.mp.yee_cst:
        plt.suptitle('Yee mesh, dxBs_x and div(Bs)')
    else:
        plt.suptitle('Not Yee mesh, dxB_x and div(B)')
    plt.colorbar(p0, ax=AX[0], orientation='horizontal')
    plt.colorbar(p1, ax=AX[1], orientation='horizontal')
    plt.tight_layout()
    oT.set_spines(AX)
    plt.show()

if md.mp.NB_DIM==3:
    dxBx = 1./(12.*dx) * ( B[0, 4:  , 2:-2, 2:-2]-8*B[0, 3:-1, 2:-2, 2:-2]+8*B[0, 1:-3, 2:-2, 2:-2]-B[0,  :-4, 2:-2, 2:-2] )
    dyBy = 1./(12.*dx) * ( B[1, 2:-2, 4:  , 2:-2]-8*B[1, 2:-2, 3:-1, 2:-2]+8*B[1, 2:-2, 1:-3, 2:-2]-B[1, 2:-2,  :-4, 2:-2] )
    dzBz = 1./(12.*dx) * ( B[2, 2:-2, 2:-2, 4:  ]-8*B[2, 2:-2, 2:-2, 3:-1]+8*B[2, 2:-2, 2:-2, 1:-3]-B[2, 2:-2, 2:-2,  :-4] )

    div_B = dxBx + dyBy + dzBz
    print(div_B.shape)

    fig, ax = plt.subplots(figsize=(14, 14))

    p = ax.imshow(div_B[:, :, int(md.mp.len_z_cst/2.)],
              # vmin=-1, vmax=1,
              extent=(md.edges_x[0], md.edges_x[-1], md.edges_y[0], md.edges_y[-1]),
              interpolation='bilinear',
              rasterized=True, cmap=oC.bwr_2, origin='lower')
    plt.colorbar(p)
    plt.tight_layout()
    oT.set_spines(ax)
    plt.show()
