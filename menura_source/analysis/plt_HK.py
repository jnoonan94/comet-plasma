import sys
import os

import numpy as np
import matplotlib.pyplot as plt

import own_tools as oT
import own_colours as oC

import argparse

from menura_utils import *

r''' Plot house-keepings. '''


parser = argparse.ArgumentParser()
parser.add_argument('--run')
args = parser.parse_args()
#
if args.run == None:
    # sys.exit('Please provide run ID through --run option.')
    run_ID = ''
    path = '/home/etienneb/Models/Menura/menura'
else:
    try:
        run_ID = f'{int(args.run):03}'
    except:
        run_ID = args.run
    path = f'/home/etienneb/Models/Menura_own/Data/run_{run_ID}'

# path_remote = '/linkhome/rech/genlag01/ued64ot/menura'
path_remote = f'/gpfswork/rech/fuz/ued64ot/run_{run_ID}'
# path_remote = '/home/b/behare/Private/menura'

remote_label = 'jean-zay'
# remote_label = 'kebnekaise'

mh = menura_HK(path, path_remote, ask_scp=True, remote_label=remote_label)

sf=False
mh.plt_run_time(save_fig=sf)
mh.plt_active_part(save_fig=sf)
mh.plt_part_comm(save_fig=sf)
mh.plt_div_B(save_fig=sf)
# mh.plt_rms(save_fig=sf)
mh.plt_energies(save_fig=sf)
mh.plt_energy_tot(save_fig=sf)
# mh.plt_moment_1(save_fig=sf)
# mh.plt_part_moment(save_fig=sf)

if 1:

    e    = 1.602e-19
    m_i  = 1.673e-27
    m_e  = 9.109e-31
    mu0  = 1.257e-6
    eps0 = 8.85e-12
    k_B  = 1.3806e-23
    c    = 299792458.


    mh_0   = menura_HK('/home/etienneb/Models/Menura_own/Data/run_020',   '', ask_scp=False, remote_label=remote_label)
    mh_yee = menura_HK('/home/etienneb/Models/Menura_own/Data/run_tmp_2', '', ask_scp=False, remote_label=remote_label)
    mh_1   = menura_HK('/home/etienneb/Models/Menura_own/Data/run_tmp_3', '', ask_scp=False, remote_label=remote_label)
    mh_2   = menura_HK('/home/etienneb/Models/Menura_own/Data/run_tmp',   '', ask_scp=False, remote_label=remote_label)


    div_B_max_0   = np.amax(  mh_0.div_B_max, axis=0)
    div_B_max_1   = np.amax(  mh_1.div_B_max, axis=0)
    div_B_max_yee = np.amax(mh_yee.div_B_max, axis=0)

    div_B_var_0   = np.mean(  mh_0.div_B_var, axis=0)
    div_B_var_1   = np.mean(  mh_1.div_B_var, axis=0)
    div_B_var_yee = np.mean(mh_yee.div_B_var, axis=0)

    W_E = np.sum(mh_0.energy_E, axis=0)
    W_B = np.sum(mh_0.energy_B, axis=0)
    W_k = np.sum(mh_0.energy_K, axis=0)
    W_E *= mh_0.mp.v_A**2 * mh_0.mp.B0_SI**2
    W_B *= mh_0.mp.B0_SI**2
    W_k *= mh_0.mp.v_A**2
    W_E *= eps0/(2.*mh_0.mp.nb_nodes_cst)
    W_B *= 1./(2.*mu0*mh_0.mp.nb_nodes_cst)
    W_k *= m_i*mh_0.mp.n0_SI/(2*mh_0.mp.nb_nodes_cst*mh_0.mp.nb_part_node_cst)
    W_tot_0 =  W_k + W_E + W_B

    W_E = np.sum(mh_1.energy_E, axis=0)
    W_B = np.sum(mh_1.energy_B, axis=0)
    W_k = np.sum(mh_1.energy_K, axis=0)
    W_E *= mh_1.mp.v_A**2 * mh_1.mp.B0_SI**2
    W_B *= mh_1.mp.B0_SI**2
    W_k *= mh_1.mp.v_A**2
    W_E *= eps0/(2.*mh_1.mp.nb_nodes_cst)
    W_B *= 1./(2.*mu0*mh_1.mp.nb_nodes_cst)
    W_k *= m_i*mh_1.mp.n0_SI/(2*mh_1.mp.nb_nodes_cst*mh_1.mp.nb_part_node_cst)
    W_tot_1 =  W_k + W_E + W_B

    W_E = np.sum(mh_2.energy_E, axis=0)
    W_B = np.sum(mh_2.energy_B, axis=0)
    W_k = np.sum(mh_2.energy_K, axis=0)
    W_E *= mh_2.mp.v_A**2 * mh_2.mp.B0_SI**2
    W_B *= mh_2.mp.B0_SI**2
    W_k *= mh_2.mp.v_A**2
    W_E *= eps0/(2.*mh_2.mp.nb_nodes_cst)
    W_B *= 1./(2.*mu0*mh_2.mp.nb_nodes_cst)
    W_k *= m_i*mh_2.mp.n0_SI/(2*mh_2.mp.nb_nodes_cst*mh_2.mp.nb_part_node_cst)
    W_tot_2 =  W_k + W_E + W_B

    W_E = np.sum(mh_yee.energy_E, axis=0)
    W_B = np.sum(mh_yee.energy_B, axis=0)
    W_k = np.sum(mh_yee.energy_K, axis=0)
    W_E *= mh_yee.mp.v_A**2 * mh_yee.mp.B0_SI**2
    W_B *= mh_yee.mp.B0_SI**2
    W_k *= mh_yee.mp.v_A**2
    W_E *= eps0/(2.*mh_yee.mp.nb_nodes_cst)
    W_B *= 1./(2.*mu0*mh_yee.mp.nb_nodes_cst)
    W_k *= m_i*mh_yee.mp.n0_SI/(2*mh_yee.mp.nb_nodes_cst*mh_yee.mp.nb_part_node_cst)
    W_tot_yee =  W_k + W_E + W_B

    itk_1 = div_B_max_1!=0
    itk_yee = div_B_max_yee!=0

    plt.plot(mh_0.time, div_B_var_1/div_B_var_0, 'k')
    plt.title('var(div(B_11sub)) / var(div(B_51sub))')
    plt.show()

    fig, AX = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

    AX[0].plot(mh_0.time, div_B_max_0, 'k', label='max(abs(div(B)))', lw=1)
    AX[0].plot(mh_1.time[itk_1], div_B_max_1[itk_1], c=oC.rgb[0], label='max(abs(div(B))), 51 sub-steps.', lw=1)
    AX[0].plot(mh_yee.time[itk_yee], div_B_max_yee[itk_yee], c=oC.rgb[2], label='max(abs(div(B))), Yee mesh', lw=1)

    AX[1].plot(mh_0.time, div_B_var_0, 'k', label='var(div(B))', lw=1)
    # AX[1].plot([mh_0.time[0], mh_0.time[-1]], [div_B_var_0[0], div_B_var_0[-1]], '--k', lw=.5)
    AX[1].plot(mh_1.time[itk_1], div_B_var_1[itk_1], c=oC.rgb[0], label='var(div(B)), 51 sub-steps', lw=1)
    AX[1].plot(mh_yee.time[itk_yee], div_B_var_yee[itk_yee], c=oC.rgb[2], label='var(div(B)), Yee mesh', lw=1)

    AX[2].plot(mh_0.time, W_tot_0/W_tot_0[0], 'k', lw=1, label='Total energy')
    AX[2].plot(mh_1.time[itk_1], W_tot_1[itk_1]/W_tot_1[0], c=oC.rgb[0], lw=1, label='Total energy, 51 sub-steps')
    AX[2].plot(mh_2.time, W_tot_2/W_tot_2[0], '--k', lw=1, label='Total energy, no initial fluctuations')
    AX[2].plot(mh_yee.time[itk_yee], W_tot_yee[itk_yee]/W_tot_yee[0], c=oC.rgb[2], lw=1, label='Total energy, Yee mesh')

    # AX[1].set_ylim([0, 3e-11])

    for ax in AX:
        ax.legend()

    plt.tight_layout()
    oT.set_spines(AX)
    plt.show()
