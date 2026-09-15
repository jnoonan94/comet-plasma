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
                 ask_scp=False, print_param=False, full_scp=True )



if 0:            ## Highlighting chaotic tail.
    md2 = menura_data(path, path_remote, 8000, remote_label=remote_label,
                     ask_scp=True, print_param=False, full_scp=True )

    B0 = md.load_field('B')
    B1 = md2.load_field('B')

    plt.pcolormesh((B0[0]-B1[0]).T, vmin=-5., vmax=5., cmap=oC.bwr_2)
    plt.show()


if 0:   ## High-pass filtering tests.
    E = md.load_field('B')
    a = E[0]**2 + E[1]**2
    sigma = 4
    from scipy import ndimage
    lowpass = ndimage.gaussian_filter(a, sigma=sigma)
    highpass = a - lowpass
    fig, AX = plt.subplots(1, 2, figsize=(18, 14))
    for ax in AX:
        ax.set_aspect('equal')
    AX[0].pcolormesh(a.T,
                     vmin=-20, vmax=20,
                     cmap=oC.bwr_2)
    AX[1].pcolormesh(highpass.T,
                     vmin=-5., vmax=5.,
                      cmap=oC.bwr_2)

    plt.tight_layout()
    oT.set_spines(AX)
    plt.show()
    sys.exit()


if 1:   ## Dynamic pressure and gyro-radius just upstream of the nose.
    ## Choose the right run! i.e. no planetary ions.
    idx_y = 980
    Ji = md.load_field('Ji')[:, :, idx_y]
    n_sw = md.load_field('dens_tot')[:, idx_y]
    B = md.load_field('B')[:, :, idx_y]
    md.mp.v_obs = 10.

    u_sw = Ji/n_sw


    ram_pres = n_sw*u_sw[0]**2


    n_pla = .1
    omega = oT.norm(B)*(n_sw + n_pla*18)/((n_sw + n_pla)*18)
    omega_mean = 1*(1+n_pla*18)/((1+n_pla)*18)

    # u = Ji/n_
    u_com = np.zeros((3, md.mp.len_x_cst+4))
    u_com[0] = md.mp.v_obs
    v_i =      n_sw/(n_sw+18*n_pla) *u_sw + \
          (18*n_pla/(n_sw+18*n_pla))*u_com

    u_sw  -= v_i
    u_com -= v_i

    Bu = B[0]*u_sw[0] + B[1]*u_sw[1] + B[2]*u_sw[2]
    alpha = np.arccos(Bu/(oT.norm(B)*oT.norm(u_sw)))
    Bu_com = B[0]*u_com[0] + B[1]*u_com[1] + B[2]*u_com[2]
    alpha_com = np.arccos(Bu_com/(oT.norm(B)*oT.norm(u_com)))

    v_i_mean = 18*n_pla/(1+18*n_pla)*md.mp.v_obs

    u_sw_perp = oT.norm(u_sw)*np.sin(alpha)
    R = np.absolute(u_sw_perp)/omega
    # R_mean = u_i_mean/omega_mean
    R_mean = v_i_mean/omega_mean
    # print(np.nanmean(R), R_mean)

    u_com_perp = oT.norm(u_com)*np.sin(alpha_com)
    R_com = np.absolute(u_com_perp)/omega
    R_mean_com = (md.mp.v_obs-v_i_mean)/omega_mean

    fig, AX = plt.subplots(2, 1, figsize=(16, 14))

    AX[0].plot(R[::1])
    AX[1].plot(ram_pres[::1])

    oT.set_spines(AX)
    plt.tight_layout()
    plt.show()
