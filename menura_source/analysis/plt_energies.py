from own_py_modules import *
import own_tools as oT
import own_colours as oC



path = '/home/etienneb/Models/Menura/menura'
path = '/home/etienneb/Models/Menura'
path_kebne = '/home/b/behare/Private/menura'
# path_kebne = '/home/b/behare/pfs'

sync = input('scp from kebnekaise? (y/[n]) ')
if sync == 'y':
    os.system(f'sshpass -p "N2cph0710" scp behare@kebnekaise.hpc2n.umu.se:{path_kebne}/products/parameters.txt {path}/products')
    os.system(f'scp behare@kebnekaise.hpc2n.umu.se:{path_kebne}/products/HK/* {path}/products/HK')

d = np.loadtxt(f'{path}/products/energies_rank0.txt').T

time =   d[0, 1:]
enMag =  d[1, 1:]
enMag_raw =  d[1, 1:]
enElec = d[2, 1:]
enKin =  d[3, 1:]
nbPart =  d[4, 1:]
densTot = d[5, 1:]
velTot = d[6, 1:]
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
try:
    vel_th_part_x_mean = d[19, 0:]
    vel_th_part_y_mean = d[20, 0:]
    vel_th_part_z_mean = d[21, 0:]
except:
    pass



# d = np.loadtxt(f'{path}/trajectory.txt').T
# rms = np.loadtxt(f'{path}/rms.txt').T

if 1:
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
