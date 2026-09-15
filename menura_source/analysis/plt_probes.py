import sys, os
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

poi = 6*20 + 18 ## Most central probe.
# poi = 5*20 + 2 ## Downstream, within bubbles.
# poi = 6*20
# poi = 99
# poi = 131
mp = menura_probes(path, path_remote, it, remote_label=remote_label,
                   ask_scp=False, print_param=False, poi=poi)#125)
mp_lam = menura_probes(f'/home/etienneb/Models/Menura_own/Data/run_tmp_2', path_remote, it, remote_label=remote_label,
                   ask_scp=False, print_param=False, poi=poi)#125)


mp.plt_probes_positions(show_probe_ID=True)




fig, AX = plt.subplots(2, 1, figsize=(18, 14), sharex=True)

AX[0].plot(mp.time[::2], mp.dens[poi, ::2], c=oC.rgb[0], label='turbulent, density')
AX[0].plot(mp_lam.time[::2], mp_lam.dens[poi, ::2], c=oC.rgb[2], label='laminar, density')

# AX[1].plot(mp.time[::2], mp.E[0, poi, ::2], c=oC.rgb[1], label='Ex')
# AX[1].plot(mp.time[::2], mp.Ji[0, poi, ::2]/mp.dens[poi, ::2], c=oC.rgb[1], label='Jix')
# AX[1].plot(mp.time[::2], mp.J[2, poi, ::2], c=oC.rgb[1], label='Jz')
# AX[1].plot(mp.time[::2],     oT.norm(mp.B[:, poi])[::2],     c=oC.rgb[0], label='|B|, turbulent')
# AX[1].plot(mp.time[::2],     mp.B[1, poi, ::2],     c=oC.rgb[0], label='By, turbulent')
# AX[1].plot(mp_lam.time[::2],     mp_lam.B[2, poi, ::2],     c=oC.rgb[0], label='Bz, laminar')
AX[1].plot(mp.time[::2],     mp.B[2, poi, ::2],     c=oC.rgb[0], label='Bz, turbulent')
AX[1].plot(mp.time[::2],     np.sqrt(mp.B[0, poi, ::2]**2 + mp.B[1, poi, ::2]**2),     c=oC.rgb[0], label='B_perp, turbulent')
# AX[1].plot(mp.time[::2],( mp.dens[poi, ::2] - np.mean(mp.dens[poi, ::2]))/5, c=oC.rgb[1], label='turbulent, density')
# AX[1].plot(mp_lam.time[::2], oT.norm(mp_lam.B[:, poi])[::2], c=oC.rgb[2], label='|B|, laminar')



AX[0].axhline(np.mean(mp.dens[poi, ::2]), c=oC.rgb[0], lw=.5)
AX[0].axhline(np.mean(mp_lam.dens[poi, ::2]), c=oC.rgb[2], lw=.5)

for ax in AX:
    ax.legend()
# plt.title(f'Probe ID {poi} - density')
plt.tight_layout()
oT.set_spines(AX)
plt.show()

sys.exit()






# mp.output_probes()
#
mp.plt_probes_positions()
# sys.exit()
mp.plt_probes()
mp.plt_spectrum(poi)

# rx = np.load(f'{path}/products/probes/probes_positions_x_rank0.npy')
# ry = np.load(f'{path}/products/probes/probes_positions_y_rank0.npy')
#
# E = np.load(f'{path}/products/probes/probes_E_rank0.npy')
# B = np.load(f'{path}/products/probes/probes_B_rank0.npy')
#
# plt.plot(E[0, 19, 2])
# plt.plot(E[1, 19, 2])
# plt.plot(E[2, 19, 2])
# plt.show()
