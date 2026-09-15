from own_py_modules import *
import own_tools as oT
import own_colours as oC




class menura_param():

    def __init__(self, path):

        self.param_dict = {}

        p = np.recfromtxt(f'{path}/parameters.txt')

        for t in p:
            param_name = t[0].decode('UTF-8')
            value = t[1]
            self.param_dict[param_name] = value
            setattr(self, param_name, value)

        self.mpi_nb_proc = 0
        self.len_x_cst = int(self.len_x_cst)
        self.len_y_cst = int(self.len_y_cst)

        self.e    = 1.602e-19
        self.m_i  = 1.673e-27
        self.m_e  = 9.109e-31
        self.mu0  = 1.257e-6
        self.eps0 = 8.85e-12
        self.k_B  = 1.3806e-23
        self.c    = 299792458.

# it = 400
# os.system('scp behare@kebnekaise.hpc2n.umu.se:/home/b/behare/Private/run_27/products/*.txt ~/Desktop')
# os.system(f'scp behare@kebnekaise.hpc2n.umu.se:/home/b/behare/Private/run_27/products/B_{it}.npy ~/Desktop')
path = '/home/etienneb/Models/Menura/run_27'
path_new = '/home/etienneb/Models/Menura/menura'

mp = menura_param(f'{path}/products')





d = np.loadtxt(f'{path}/products/energies.txt').T

energy_E_new = np.load(f'{path_new}/products/HK/energy_elec_rank0.npy')
energy_B_new = np.load(f'{path_new}/products/HK/energy_mag_rank0.npy')
energy_K_new = np.load(f'{path_new}/products/HK/energy_kin_rank0.npy')
time_new = np.arange(energy_E_new.size)*.025*10

new_vel_part = np.load(f'{path_new}/products/HK/vel_part_rank0.npy')
new_vel_th_part = np.load(f'{path_new}/products/HK/vel_th_part_rank0.npy')


time =   d[0, :]*mp.omega_ci
enMag =  d[1, :]#/mp.B0_no_norm**2
enMag_raw =  d[1, :]#/mp.B0_no_norm**2
enElec = d[2, :]/(mp.B0_no_norm*mp.v_A)**2
enElec[0] = np.nan
energy_E_new[0] = np.nan
enKin =  d[3, :]#/mp.v_A**2
nbPart =  d[4, :]
densTot = d[5, :]
velTot = d[6, :]
vel_part_x_mean = d[7, :]
vel_part_y_mean = d[8, :]
vel_part_z_mean = d[9, :]
try:
    B_x_mean = d[7, 0:]
    B_y_mean = d[8, 0:]
    B_z_mean = d[9, 0:]
    B_mean = np.sqrt(B_x_mean**2 + B_y_mean**2 + B_z_mean**2)
    B_x_var = d[10, 0:]
    B_y_var = d[11, 0:]
    B_z_var = d[12, 0:]
    Ji_x_mean = d[13, 0:]
    Ji_y_mean = d[14, 0:]
    Ji_z_mean = d[15, 0:]
    Ji_mean = np.sqrt(Ji_x_mean**2 + Ji_y_mean**2 + Ji_z_mean**2)
    vel_part_x_mean = d[16, 0:]
    vel_part_y_mean = d[17, 0:]
    vel_part_z_mean = d[18, 0:]
    vel_part_mean = np.sqrt(vel_part_x_mean**2 + vel_part_y_mean**2 + vel_part_z_mean**2)
    vel_th_part_x_mean = d[19, 0:]
    vel_th_part_y_mean = d[20, 0:]
    vel_th_part_z_mean = d[21, 0:]
except:
    pass



d = np.loadtxt(f'{path}/products/trajectory.txt').T
rms = np.loadtxt(f'{path}/products/rms.txt').T

vel_part = np.load(f'{path}/products/HK/vel_part.npy')
vel_th_part = np.load(f'{path}/products/HK/vel_th_part.npy')#/mp.v_A**2


if 0:
    '''Number of particules and density vs. time'''

    fig, ax = plt.subplots(1, 1, figsize=(16, 14), sharex=True)
    ax.plot(nbPart, c=oC.rgb2[0], label='Nb part')

    ax1 = ax.twinx()
    ax1.plot(densTot, c=oC.rgb2[2], label='Density')

    ax.legend()
    ax1.legend(loc=2)
    plt.tight_layout()
    plt.show()

if 0:
    '''Initial and final VDF'''
    p = np.load(f'{path}/particles_0_0.npy').T
    h_vx_0, bin_vx = np.histogram(p[2], 200)
    centers_vx = .5*(bin_vx[:-1]+bin_vx[1:])
    p = np.load(f'{path}/particles_0_2000.npy').T
    h_vx_1, bin_vx = np.histogram(p[2], bin_vx)

    fig,  ax = plt.subplots(figsize=(14,14))

    ax.plot(centers_vx, h_vx_0, oC.rgb2[0], label='Initial VDF (entire box)')
    ax.plot(centers_vx, h_vx_1, oC.rgb2[2], label='Final VDF')

    ax.set_yscale('log')
    ax.legend()

    oT.set_spines(ax)
    plt.show()

if 1:

    enLim = 10000

    fig, AX = plt.subplots(3, 1, figsize=(20, 16), sharex=True)

    # enKin /= 92627.**2
    enTot = enKin + enMag + enElec

    eEMean = enElec[0] #np.mean(enElec)
    eMMean = enMag[0] #np.mean(enMag)
    eKMean = enKin[0] #np.mean(enKin)
    eTMean = enTot[0] #np.mean(enTot)

    # enElec = (enElec - eEMean)/eEMean*100
    # enMag = (enMag - eMMean)/eMMean*100
    # enKin = (enKin - eKMean)/eKMean*100
    # enTot = (enTot - eTMean)/eTMean*100

    AX[0].plot(time, enElec, c=oC.rgb2[0], label='Electric energy (relative)')
    AX[0].plot(time_new, energy_E_new, '--', c=oC.rgb2[0])
    # AX[0].set_yscale('log')
    AX[1].plot(time, enMag, c=oC.rgb2[2], label='Magnetic energy (relative)')
    AX[1].plot(time_new, energy_B_new, '--', c=oC.rgb2[2])

    AX[2].plot(time, enKin, c=oC.rgb2[1], label='Kinetik energy (relative)')
    AX[2].plot(time_new, energy_K_new, '--', c=oC.rgb2[1])

    # axx = AX[2].twinx()
    # axx.plot(time, -enMag, '--', c=oC.rgb2[2], label='Magnetic energy')
    # AX.plot(time, enTot, '--k', label='total')


    # AX[2].plot(time, E_y_mean, label='Ex')
    # AX[2].plot(time, E_z_mean, label='Ex')

    # AX.plot(timeHR, enMagHR, c='b', label='Magnetic')
    # AX.plot(timeHR, enKinHR, c='g', label='Kinetik')

    # AX.plot(time[(velTot<enLim) * (velTot>-enLim)], velTot[(velTot<enLim) * (velTot>-enLim)], c='#DDDDDD', label='kineticGrid')
    # AX.plot(time[(enTot<enLim) * (enTot>-enLim)], enTot[(enTot<enLim) * (enTot>-enLim)], c='#25417f', label='Total')
    # AX[1].plot(time, nbPart, label='nbPart')
    # for ax in AX:
    #     ax.plot([time[0],time[-1]] ,[0,0], 'k', lw=.5)

    axouille = AX[0].twiny()

    axouille.plot(np.arange(np.size(time)), enElec, alpha=0)

    axouille.set_xlabel('Iterations')
    AX[-1].set_xlabel('Time (s)')
    # AX[0].set_ylabel('Relative difference (%)')
    for i, ax in enumerate(AX):
        ax.legend(loc=2)
        # ax.set_xlim([0, time[-1]])
    # axx.legend(loc=1)
    # axouille.set_xlim([0, time.size])
    # AX[2].set_ylim([6.22, 6.39])

    oT.set_spines(AX[1:])
    oT.set_spines([AX[0], axouille], two_xaxis=True)

    plt.tight_layout()
    plt.show()

if 0:

    # plt.plot(Ji_mean[1:]/vel_part_mean[1:])
    # plt.plot([0, 20000], [1, 1])
    # plt.show()

    fig, AX = plt.subplots(4, 1, figsize=(20,16))

    AX[0].plot(d[0, 1:], B_mean[1:], c='#F442CA', label='Mean B')

    AX[1].plot(d[0, :-1], Ji_mean[1:], c='#F442CA', label='Mean Ji')
    # AX[1].plot(d[0, 1:], Ji_mean[1:], 'x', c='#F442CA', label='Mean Ji')
    axxxx = AX[1].twinx()
    AX[1].plot(d[0, :-1], vel_part_mean[:-1], '--', c='#42F496', label='particles overall mean velocity')

    AX[2].plot(d[0, 1:], vel_part_x_mean[1:], c='#F442CA', label='particles v_x mean')
    AX[2].plot(d[0, 1:], vel_part_y_mean[1:], c='#42F496', label='particles v_y mean')
    AX[2].plot(d[0, 1:], vel_part_z_mean[1:], c='#427AF4', label='particles v_z mean')

    AX[3].plot(d[0, 1:], B_z_var[1:], c='#427AF4', label='B_z variance')
    axx = AX[3].twinx()
    axx.plot(time[1:], enMag_raw[1:], '--', c='#F442CA', label='Magnetic energy')


    axouille = AX[0].twiny()
    axouille.plot(np.arange(np.size(d[0])-1), np.sqrt(B_x_mean**2+B_y_mean**2+B_z_mean**2)[1:], alpha=0)
    axouille.set_xlabel('Iterations')

    AX[0].set_xlabel('Time (s)')
    AX[0].set_ylabel('')


    for i, ax in enumerate(AX):
        ax.legend(loc=2)
    axx.legend(loc=1)
    axxxx.legend(loc=1)


    plt.tight_layout()
    plt.show()


if 1:

    fig, AX = plt.subplots(2, 1, figsize=(20,16), sharex=True)
    fig.subplots_adjust(left=0.04, right=0.993, top=0.924, bottom=0.066, hspace=0.09)
    # plt.suptitle('Statistics on *particles*')

    AX[0].plot(time,         vel_part[0],       c='#F442CA', label='v_x mean')
    AX[0].plot(time,         vel_part[1],       c='#42F496', label='v_y mean')
    AX[0].plot(time,         vel_part[2],       c='#427AF4', label='v_z mean')
    AX[0].plot(time_new, new_vel_part[0], '--', c='#F442CA', lw=2, label='CAM v_x mean')
    AX[0].plot(time_new, new_vel_part[1], '--', c='#42F496', lw=2, label='CAM v_y mean')
    AX[0].plot(time_new, new_vel_part[2], '--', c='#427AF4', lw=2, label='CAM v_z mean')

    axouille = AX[0].twiny()
    axouille.plot(time, vel_part_x_mean/mp.v_A, alpha=0)
    axouille.set_xlabel('Iterations')

    AX[-1].set_xlabel('Time (s)')
    AX[0].set_ylabel('')

    AX[1].plot(time,  vel_th_part[0], c='#F442CA', label='v_x**2 mean')
    AX[1].plot(time,  vel_th_part[1], c='#42F496', label='v_y**2 mean')
    AX[1].plot(time,  vel_th_part[2], c='#427AF4', label='v_z**2 mean')
    AX[1].plot(time_new,  new_vel_th_part[0], '--', c='#F442CA', lw=2, label='CAM v_x**2 mean')
    AX[1].plot(time_new,  new_vel_th_part[1], '--', c='#42F496', lw=2, label='CAM v_y**2 mean')
    AX[1].plot(time_new,  new_vel_th_part[2], '--', c='#427AF4', lw=2, label='CAM v_z**2 mean')

    for i, ax in enumerate(AX):
        ax.legend(loc=3)
        # ax.set_xlim([0, time[-1]])

    # axouille.set_xlim([0, time.size])

    # AX[1].set_ylim([2.125, 2.132])
    oT.set_spines(AX[1])
    oT.set_spines([AX[0], axouille], two_xaxis=True)

    # plt.tight_layout()
    plt.show()

if 0:

    nbT = min(time.size, rms.shape[1])

    fig, AX = plt.subplots(2,1, figsize=(20,16))
    print(time.shape, rms.shape)
    AX[0].plot(time[:nbT], rms[1, ], c='#427AF4', label='rms Magnetic')
    AX[0].plot(time[:nbT], rms[2, ], c='#42F496', label='rms Kinetik')
    AX[0].legend()

    AX[1].plot(time[:nbT], rms[3, ], c='#427AF4', label='rms parallel current')
    AX[1].plot(time[:nbT], rms[4, ]*1.9e4, c='#42F496', label='rms parallel vorticity')
    AX[1].legend()

    plt.show()


    # fig, ax = plt.subplots(figsize=(20,16))
    #
    # ax.plot(time, nbPart, c='#F442CA', label='nbPart')
    # ax.plot(time, densTot, c='#427AF4', label='densTot')
    #
    # ax.legend()
    #
    # plt.tight_layout()
    # plt.show()
