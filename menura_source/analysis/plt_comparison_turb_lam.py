# import sys
# import os

import numpy as np
import matplotlib.pyplot as plt

# import own_tools as oT
# import own_colours as oC

from menura_utils import *

idx_it = 20000
path = '../../Menura_own/Data/run_031'
md_turb = menura_data(path, '', idx_it, ask_scp=False)
#idx_it = 5000
path = '../../Menura_own/Data/run_033'
md_lami = menura_data(path, '', idx_it, ask_scp=False)

dens_sw_turb = md_turb.load_field('dens_spec_0')
dens_sw_lami = md_lami.load_field('dens_spec_0')
dens_sw_turb = np.log10(dens_sw_turb)
dens_sw_lami = np.log10(dens_sw_lami)
plt_comparison(dens_sw_lami, dens_sw_turb, md_lami, md_turb, vminmax=[-.8, .8])

dens_pla_turb = md_turb.load_field('dens_spec_1')
dens_pla_lami = md_lami.load_field('dens_spec_1')
# plt_comparison(dens_pla_lami, dens_pla_turb, md_lami, md_turb, vminmax=[0, 10], cmap=oC.wbr_1)
dens_pla_turb = np.log10(dens_pla_turb)
dens_pla_lami = np.log10(dens_pla_lami)
plt_comparison(dens_pla_lami, dens_pla_turb, md_lami, md_turb, vminmax=[-3, 1.5], cmap=oC.wbr_1)

dens_pla_turb[np.isinf(dens_pla_turb)] = 0
dens_pla_lami[np.isinf(dens_pla_lami)] = 0
print(np.sum(dens_pla_turb), np.sum(dens_pla_lami))
profile_turb = np.sum(dens_pla_turb, axis=1)
profile_lami = np.sum(dens_pla_lami, axis=1)

plt.plot(np.log10(profile_turb), c=oC.rgb[0])
plt.plot(np.log10(profile_lami), c=oC.rgb[2])
plt.show()
