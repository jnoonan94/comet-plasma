import sys
import matplotlib.pyplot as plt
import numpy as np

from scipy.fftpack import fft, fft2, fftn, ifft2
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator, LinearNDInterpolator
from scipy.signal import blackman, hamming

import own_tools as oT
import own_colours as oC

from menura_utils import *


# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--run')
# parser.add_argument('--it')
# args = parser.parse_args()
# #
# if args.it != None:
#     it = int(args.it)
# else:
#     it = 0
#
# if args.run == None:
#     run_ID = ''
#     path = '/home/etienneb/Models/Menura/menura'
# else:
#     try:
#         run_ID = f'{int(args.run):03}'
#     except:
#         run_ID = args.run
#     path = f'/home/etienneb/Models/Menura_own/Data/run_{run_ID}'
# #
# path_remote = f'/gpfswork/rech/fuz/ued64ot/run_{run_ID}'
remote_label = 'jean-zay'

path = f'/home/etienneb/Models/Menura_own/Data/run_030'
path_remote = f'/gpfswork/rech/fuz/ued64ot/run_030'
it = 20000

md = menura_data(path, path_remote, it, remote_label=remote_label,
                 ask_scp=False, print_param=False, full_scp=True )

eta_hyp_res = 1e-3#md.mp.eta_hyp_res
dX = md.mp.dX
dt = md.mp.dt ## ??

B_0 = md.load_field('B')[:, 2:-2, 2:-2]

B = md.load_field('B')[:, 2:-2, 2:-2]
J = np.zeros_like(B)
E = np.zeros_like(B)




field_x = B[0]
field_y = B[1]

res = field_x.shape[1]
xy = np.mgrid[:res, :res].astype(float)
xy -= .5*res
r = np.sum(xy**2, axis=0)**.5
itk = r<900
# itk = (r>50) * (r<100)
# itk = r>50
itk = np.fft.fftshift(itk)
#
field_x_f = fft2(field_x)
field_x_f[itk] = 0
field_x_inv_f = (ifft2(field_x_f))
#
field_y_f = fft2(field_y)
field_y_f[itk] = 0
field_y_inv_f = (ifft2(field_y_f))

#
# plt.pcolormesh(field_x)
# plt.show()
#
# plt.pcolormesh(np.log10(np.abs(np.fft.fftshift(field_x_f))))
# plt.show()
#
# plt.pcolormesh(np.real(field_x_inv_f))
# plt.show()

B[0] -= np.real(field_x_inv_f)
B[1] -= np.real(field_y_inv_f)

# B_perp_sqr = B_0[0]**2 + B_0[1]**2



# sys.exit()
#
#
dx = md.mp.dX
# grid_x = self.grid_x_box[2:-2]# self.grid_x[0, 2:-2]
# grid_y = self.grid_y_box[2:-2]#np.zeros(self.mp.len_y_cst*self.nb_proc)
# # for r in range(self.nb_proc):
# #     grid_y[r*self.mp.len_y_cst: (r+1)*self.mp.len_y_cst] = self.grid_y[r, 2:-2]
# #
f_x = np.fft.fftshift(np.fft.fftfreq(md.mp.len_x_cst, d=dx))
f_y = np.fft.fftshift(np.fft.fftfreq(md.nb_proc_tot*md.mp.len_y_cst, d=dx))
f2d = np.ones((md.mp.len_x_cst, md.nb_proc_tot*md.mp.len_y_cst))
f2d = np.sqrt(f_x[:,None]**2+f_y[None,:]**2)
bin_edges = np.linspace(f_x[0], f_x[-1], 1000)
bin_centres = .5*(bin_edges[1:]+bin_edges[:-1])
bin_centres *= 2*np.pi  ## From spatial frequency to k
# #
field_x = B[0]
field_y = B[1]
field = field_x**2 + field_y**2
PSD_x = np.fft.fftshift(fft2(field_x))
PSD_x =  np.abs(PSD_x)**2
PSD_y = np.fft.fftshift(fft2(field_y))
PSD_y =  np.abs(PSD_y)**2
PSD = PSD_x + PSD_y
#
spectrum, bin_edges, fff = scipy.stats.binned_statistic(f2d.flatten(), PSD.flatten(), statistic='sum', bins=bin_edges)
#
field_x = B_0[0]
field_y = B_0[1]
# field = field_x**2 + field_y**2
PSD_x = np.fft.fftshift(fft2(field_x))
PSD_x =  np.abs(PSD_x)**2
PSD_y = np.fft.fftshift(fft2(field_y))
PSD_y =  np.abs(PSD_y)**2
PSD = PSD_x + PSD_y
#
spectrum_0, bin_edges, fff = scipy.stats.binned_statistic(f2d.flatten(), PSD.flatten(), statistic='sum', bins=bin_edges)
#
#
plt.loglog(bin_centres, spectrum)
plt.loglog(bin_centres, spectrum_0, 'r')
plt.show()
# sys.exit()











# for idx_it in range(10):
#
#     if idx_it%10==0:
#         print(idx_it)
#
#     dyBz = -1./(12.*dX) * ( B[2, 2:-2, 4:  ]-8*B[2, 2:-2, 3:-1]+8*B[2, 2:-2, 1:-3]-B[2, 2:-2,  :-4] )
#     dxBz = -1./(12.*dX) * ( B[2, 4:  , 2:-2]-8*B[2, 3:-1, 2:-2]+8*B[2, 1:-3, 2:-2]-B[2,  :-4, 2:-2] )
#     dxBy = -1./(12.*dX) * ( B[1, 4:  , 2:-2]-8*B[1, 3:-1, 2:-2]+8*B[1, 1:-3, 2:-2]-B[1,  :-4, 2:-2] )
#     dyBx = -1./(12.*dX) * ( B[0, 2:-2, 4:  ]-8*B[0, 2:-2, 3:-1]+8*B[0, 2:-2, 1:-3]-B[0, 2:-2,  :-4] )
#
#     J[0, 2:-2, 2:-2] =  dyBz
#     J[1, 2:-2, 2:-2] = -dxBz
#     J[2, 2:-2, 2:-2] =  dxBy - dyBx
#
#     J[0, 0:2]    = J[0, -4:-2]
#     J[0, :, 0:2] = J[0, :, -4:-2]
#     J[1, 0:2]    = J[1, -4:-2]
#     J[1, :, 0:2] = J[1, :, -4:-2]
#     J[2, 0:2]    = J[2, -4:-2]
#     J[2, :, 0:2] = J[2, :, -4:-2]
#
#     d2xJx2 = 1./(12.*dX**2) * ( -J[0, 4:  , 2:-2]+16*J[0, 3:-1, 2:-2]-30*J[0, 2:-2, 2:-2]+16*J[0, 1:-3, 2:-2]-J[0,  :-4, 2:-2] )
#     d2yJx2 = 1./(12.*dX**2) * ( -J[0, 2:-2, 4:  ]+16*J[0, 2:-2, 3:-1]-30*J[0, 2:-2, 2:-2]+16*J[0, 2:-2, 1:-3]-J[0, 2:-2,  :-4] )
#     d2xJy2 = 1./(12.*dX**2) * ( -J[1, 4:  , 2:-2]+16*J[1, 3:-1, 2:-2]-30*J[1, 2:-2, 2:-2]+16*J[1, 1:-3, 2:-2]-J[1,  :-4, 2:-2] )
#     d2yJy2 = 1./(12.*dX**2) * ( -J[1, 2:-2, 4:  ]+16*J[1, 2:-2, 3:-1]-30*J[1, 2:-2, 2:-2]+16*J[1, 2:-2, 1:-3]-J[1, 2:-2,  :-4] )
#     d2xJz2 = 1./(12.*dX**2) * ( -J[2, 4:  , 2:-2]+16*J[2, 3:-1, 2:-2]-30*J[2, 2:-2, 2:-2]+16*J[2, 1:-3, 2:-2]-J[2,  :-4, 2:-2] )
#     d2yJz2 = 1./(12.*dX**2) * ( -J[2, 2:-2, 4:  ]+16*J[2, 2:-2, 3:-1]-30*J[2, 2:-2, 2:-2]+16*J[2, 2:-2, 1:-3]-J[2, 2:-2,  :-4] )
#
#     E[0, 2:-2, 2:-2] = -eta_hyp_res * (d2xJx2 + d2yJx2)
#     E[1, 2:-2, 2:-2] = -eta_hyp_res * (d2xJy2 + d2yJy2)
#     E[2, 2:-2, 2:-2] = -eta_hyp_res * (d2xJz2 + d2yJz2)
#
#     E[0, 0:2]    = E[0, -4:-2]
#     E[0, :, 0:2] = E[0, :, -4:-2]
#     E[1, 0:2]    = E[1, -4:-2]
#     E[1, :, 0:2] = E[1, :, -4:-2]
#     E[2, 0:2]    = E[2, -4:-2]
#     E[2, :, 0:2] = E[2, :, -4:-2]
#
#     dyEz = -1/(12.*dX) * ( E[2, 2:-2, 4:  ]-8*E[2, 2:-2, 3:-1]+8*E[2, 2:-2, 1:-3]-E[2, 2:-2,  :-4] )
#     dxEz = -1/(12.*dX) * ( E[2, 4:  , 2:-2]-8*E[2, 3:-1, 2:-2]+8*E[2, 1:-3, 2:-2]-E[2,  :-4, 2:-2] )
#     dxEy = -1/(12.*dX) * ( E[1, 4:  , 2:-2]-8*E[1, 3:-1, 2:-2]+8*E[1, 1:-3, 2:-2]-E[1,  :-4, 2:-2] )
#     dyEx = -1/(12.*dX) * ( E[0, 2:-2, 4:  ]-8*E[0, 2:-2, 3:-1]+8*E[0, 2:-2, 1:-3]-E[0, 2:-2,  :-4] )
#
#     B[0, 2:-2, 2:-2] +=  - dt * dyEz
#     B[1, 2:-2, 2:-2] +=  - dt * -dxEz
#     B[2, 2:-2, 2:-2] +=  - dt * (dxEy - dyEx)
#
#     B[0, 0:2]    = B[0, -4:-2]
#     B[0, :, 0:2] = B[0, :, -4:-2]
#     B[1, 0:2]    = B[1, -4:-2]
#     B[1, :, 0:2] = B[1, :, -4:-2]
#     B[2, 0:2]    = B[2, -4:-2]
#     B[2, :, 0:2] = B[2, :, -4:-2]



fig, AX = plt.subplots(1, 2, figsize=(18, 16), sharex=True, sharey=True)
for ax in AX: ax.set_aspect('equal')

AX[0].pcolormesh(B_0[0]**2 + B_0[1]**2,
                 # vmin=0, vmax=1,
                 cmap=oC.bwr_2)

AX[1].pcolormesh(B[0]**2 + B[1]**2,
                 # vmin=0, vmax=1,
                 cmap=oC.bwr_2)

oT.set_spines(AX)
plt.tight_layout()
plt.show()
