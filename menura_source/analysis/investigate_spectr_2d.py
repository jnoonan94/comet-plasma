import numpy as np
import matplotlib.pyplot as plt

import own_tools as oT
import own_colours as oC

import scipy.stats
from scipy.fftpack import fft, fft2, fftn

resolution = 400

B = np.zeros((2, resolution, resolution))

k_x = 2
k_y = 2

grid = np.linspace(0, 100, resolution)

delta = np.cos(grid[:, None]*k_x + grid[None, :]*k_y)
B[0] = delta
B[1] = delta
B_perp = np.sqrt(B[0]**2+B[1]**2)

fig, ax = plt.subplots(figsize=(14,14))
ax.set_aspect('equal')

ax.pcolormesh(B[0].T)

plt.show()


fig, ax = plt.subplots(figsize=(14,14))
ax.set_aspect('equal')

ax.pcolormesh(B_perp.T)

plt.show()

len_x = resolution
len_y = resolution
dx = grid[1] - grid[0]
grid_x = grid
grid_y = grid
f_x = np.fft.fftshift(np.fft.fftfreq(len_x, d=dx))
f_y = np.fft.fftshift(np.fft.fftfreq(len_y, d=dx))
f2d = np.ones((len_x, len_y))
f2d = np.sqrt(f_x[:,None]**2+f_y[None,:]**2)
##
bin_edges = np.linspace(f_x[0], f_x[-1], 1000)
# bin_edges = np.linspace(min_k/2, max_k, 1000)
##
bin_centres = .5*(bin_edges[1:]+bin_edges[:-1])
bin_centres *= 2*np.pi  ## From spatial frequency to k
# bin_centres *= d_p

if 1:
    PSD_x = np.fft.fftshift(fft2(B[0]))
    PSD_x =  np.abs(PSD_x)**2#dx**2/(len_x*len_y) *
    PSD_y = np.fft.fftshift(fft2(B[1]))
    PSD_y =  np.abs(PSD_y)**2#dx**2/(len_x*len_y) *
    # PSD = np.sqrt(PSD_x**2+PSD_y**2)
    PSD = PSD_x + PSD_y

else:
    PSD = np.fft.fftshift(fft2(B_perp))
    PSD =  np.abs(PSD)**2#dx**2/(len_x*len_y) *

# grid_k = np.ones((2, f_x.size, f_y.size))
# grid_k[0] = f_x[:, None]*2*np.pi
# grid_k[1] = f_y[None, :]*2*np.pi
# # print(grid_k)
# spectre_obj = Spectrum()
# spectre_obj.interpolate_cart_2D(grid_k, PSD, 'lin')
# spectrum = np.sum(spectre_obj.vdf_interp, axis=1)
# spectrum /= spectrum[1]
# dic[field_label] = spectrum
# dic['bin_centres'] = spectre_obj.grid_spher[0,:,0]

spectrum, bin_edges, fff = scipy.stats.binned_statistic(f2d.flatten(), PSD.flatten(), statistic='sum', bins=bin_edges)
# spectrum /= spectrum[int(spectrum.size/2)+1]

plt.plot(bin_centres, spectrum)
plt.show()
