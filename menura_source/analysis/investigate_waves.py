# import sys
# import os

import numpy as np
import matplotlib.pyplot as plt

import own_tools as oT
# import own_colours as oC

from menura_utils import *

it = 6000
run_ID = '016'
path = f'/home/etienneb/Models/Menura/Data/run_{run_ID}'
path_remote = f'/gpfswork/rech/fuz/ued64ot/run_{run_ID}'
remote_label = 'jean-zay'
# remote_label = 'kebnekaise'









# for it in np.arange(2000, 4000, 100):
#     md = menura_data(path, path_remote, it, ask_scp=False, print_param=False, remote_label=remote_label)
#     md.plt_B_perp(streamplot=False, save_fig=True)
#
#
md = menura_data(path, path_remote, it, ask_scp=False, print_param=True, remote_label=remote_label)

E = md.recompose_field(md.E)
E = md.stag_vector(E)

B = md.recompose_field(md.B)
B = md.stag_vector(B)

n = md.recompose_field(md.dens)
n = md.stag_scalar(n)


reso = 800
line = np.zeros((2, reso))
line[0] = np.linspace(302, 312, reso)   ## x
line[1] = np.linspace(259, 251, reso)   ## y
line_mod = np.sqrt(line[0]**2 + line[1]**2)

sigma = .25

low, high0 = md.low_high_pass_2d(E[0], sigma=sigma)
cut0 = md.interpolate_cut(high0, line)

low, high1 = md.low_high_pass_2d(E[1], sigma=sigma)
cut1 = md.interpolate_cut(high1, line)

low, high2 = md.low_high_pass_2d(E[2], sigma=sigma)
cut2 = md.interpolate_cut(high2, line)

print(np.stack((cut0, cut1, cut2)).shape)
covariance_B = np.cov(np.stack((cut0, cut1, cut2)))
eigVal, eigVec_B = np.linalg.eig(covariance_B)
R_mean_minVar = eigVec_B
#
v_bMinVar_b0 = np.dot(R_mean_minVar, np.eye(3)[np.argmin(eigVal)])
print(v_bMinVar_b0)
# sys.exit()
# theta_minVar_B[i] = np.arccos( np.dot(np.array([0.,0.,1.]), v_bMinVar_b0)/oT.norm(v_bMinVar_b0) )
# phi_minVar_B[i] = np.arctan2(v_bMinVar_b0[1], v_bMinVar_b0[0])#
#

plt.plot(cut0)
plt.plot(cut1)
plt.plot(cut2)
plt.show()


low, high = md.low_high_pass_2d(B[0], sigma=sigma)
cut0 = md.interpolate_cut(high, line)

low, high = md.low_high_pass_2d(B[1], sigma=sigma)
cut1 = md.interpolate_cut(high, line)

low, high = md.low_high_pass_2d(B[2], sigma=sigma)
cut2 = md.interpolate_cut(high, line)

plt.plot(cut0)
plt.plot(cut1)
plt.plot(cut2)
plt.show()

low, high = md.low_high_pass_2d(n, sigma=sigma)
cut0 = md.interpolate_cut(high, line)

plt.plot(cut0)
plt.show()
sys.exit()

low, high = md.low_high_pass_2d(B[1], sigma=.25)

fig, AX = plt.subplots(1, 2, figsize=(18, 14), sharex=True, sharey=True)
for ax in AX:
    ax.set_aspect('equal')

vmin = np.nanmin(high)*.25
vmax = np.nanmax(high)*.25
p = AX[0].imshow(high.T,
                  vmin=vmin, vmax=vmax,
                  extent=(md.edges_x_box[0], md.edges_x_box[-1], md.edges_y_box[0], md.edges_y_box[-1]),
                  interpolation='bilinear',
                  rasterized=True, cmap=oC.bwr_2, origin='lower')
AX[0].text(139, 350, 'High Ez')

vmin = np.nanmin(low)
vmax = np.nanmax(low)
AX[1].imshow(low.T,
                  vmin=vmin, vmax=vmax,
                  extent=(md.edges_x_box[0], md.edges_x_box[-1], md.edges_y_box[0], md.edges_y_box[-1]),
                  interpolation='bilinear',
                  rasterized=True, cmap=oC.bwr_2, origin='lower')
AX[1].text(139, 350, 'Low Ez')

oT.set_spines(AX)
plt.tight_layout()
plt.show()
