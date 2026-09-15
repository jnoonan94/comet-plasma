import sys, os

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import own_tools as oT
import own_colours as oC

from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import RegularGridInterpolator

mpl.rcParams['pdf.fonttype'] = 42   ## Save editable font in pdf's.


def fmt(x, pos):
    # a, b = '{:.2e}'.format(x).split('e')
    # b = int(b)
    return '$10^{{{}}}$'.format(x)


def newVec2ConeClock(v):

	cone = np.arccos(-v[0] / oT.norm(v))
	clock = np.arctan2(v[2], v[1])

	return cone,clock


m = 1.6610e-27
q = 1.6022e-19
mu0 = 4*np.pi*1e-7


from matplotlib.colors import LinearSegmentedColormap
cdict_bwr_2 = {'red': ((0.0, .05, .05),
                 (0.25, .22, .22),
                 (0.50, .98, .98),
                 (0.75, .95, .95),
                 (0.90, .70, .70),
                 (1.00, .45, .45)),
         'green': ((0.0, .2, .2),
                   (0.25, .6, .6),
                   (0.50, .98, .98),
                   (0.65, .7, .7),
                   (0.75, .3, .3),
                   (1.00, 0., 0.)),
         'blue': ((0.00, .4, .4),
                  (0.1, .7, .7),
                  (0.25, .9, .9),
                  (0.50, .98, .98),
                  (0.75, .4, .4),
                  (1.00, .1, .1))}
bwr_2 = LinearSegmentedColormap('EB_bwr_2', cdict_bwr_2, 1024)
cdict_wbr_1 = {'red': ((0.0, 1., 1.),
                 (0.20, .71, .71),
                 (0.40, .41, .41),
                 (0.75, 1., 1.),
                 (1.00, .53, .53)),
         'green': ((0.0, 1., 1.),
                   (0.20, 215 / 255, 215 / 255),
                   (0.40, 91 / 255, 91 / 255),
                   (0.75, 0., 0.),
                   (1.00, 0., 0.)),
         'blue': ((0.00, 1., 1.),
                  (0.20, 1., 1.),
                  (0.40, 1., 1.),
                  (0.75, .22, .22),
                  (1.00, .05, .05))}
wbr_1 = LinearSegmentedColormap('EB_wbr_1', cdict_wbr_1, 1024)




#_______________________________________________________________________
## Data
idx_it = 100
path = '../menura/products'
B =  np.load(f'{path}/B_it{idx_it}_rank0.npy')



plotContour = True

grid_x = np.load(f'{path}/grid_x_rank0.npy')
grid_y = np.load(f'{path}/grid_y_rank0.npy')
grid_z = np.load(f'{path}/grid_z_rank0.npy')
dX = grid_x[1] - grid_x[0]

BMag = oT.norm(B)#


field = BMag
vmin = 0 ; vmax = 2
# vmin = np.log10(2.53e-9) ; vmax = -7.5
cmap = wbr_1


len_x = grid_x.size-4
len_y = grid_y.size-4
len_z = grid_z.size-4

## Centre the grids around 0:
grid_x -= int(len_x/2.)*dX
grid_y -= int(len_y/2.)*dX
grid_z -= int(len_z/2.)*dX

indXX = []

# xOI = np.array([-95, -45, 0, 45, 95])
xOI = np.array([-100, -75, -50, -25, 0])

for xxoi in xOI:
    indXX.append((np.abs(grid_x-xxoi)).argmin())
indXX = np.array(indXX)

winDimY = 100
winDimZ = 100
winDim = 100
winDimZMin = -100
winDimZMax = 100
winDimZTot = 100

indOI = ((grid_y>-winDim) * (grid_y<winDim))[:,None] * ((grid_z>-winDim) * (grid_z<winDim))[None,:]
# indZ =   (grid_z>-winDim) * (grid_z<winDim)
indZ =   (grid_z>winDimZMin) * (grid_z<winDimZMax)
indY =   (grid_y>-winDim) * (grid_y<winDim)

y, z = np.meshgrid(grid_y[indY], grid_z[indZ])


Y = [-winDimY, -winDimY, winDimY, winDimY, -winDimY]
# Z = [-winDimZ, winDimZ, winDimZ, -winDimZ, -winDimZ]
Z = [winDimZMin, winDimZMax, winDimZMax, winDimZMin, winDimZMin]
Z2 = [winDimZMin, 8e6, 8e6, winDimZMin, winDimZMin]
YFrame = [-winDimZTot, -winDimZTot, winDimZTot, winDimZTot, -winDimZTot]
ZFrame = [winDimZMin, winDimZMax, winDimZMax, winDimZMin, winDimZMin]
# if run in [2.0, 2.5]:
#     XFrame = [1.*grid_x[0], 1.*grid_x[-1]]
# else:
#     XFrame = [.6*grid_x[0], .6*grid_x[-1]]
XFrame = [.6*grid_x[0], .6*grid_x[-1]]

stride = 1


for indX in indXX:
    print(grid_x[indX]*1e-3, 'km')
print('')

#____________________________________________
## Plot.

if 0:
    eyzy = E[1,52]
    eyzy = eyzy[indY]
    eyzy = eyzy[:,indZ]
    # eyzy = eyzy[::10]
    # eyzy = eyzy[:,::10]
    eyzz = E[2,52]
    eyzz = eyzz[indY]
    eyzz = eyzz[:,indZ]
    # eyzz = eyzz[::10]
    # eyzz = eyzz[:,::10]
    yNew = grid_y[indY]
    # yNew = yNew[::10]
    zNew = grid_z[indZ]
    # zNew = zNew[::10]
    # plt.pcolormesh(yNew, zNew, eyzy.T)
    # plt.show()
    # plt.pcolormesh(yNew, zNew, eyzz.T)
    # plt.show()
    #
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cut = c[52]
    cut = cut[indY]
    cut = cut[:,indZ]
    cut[cut<vmin] = vmin
    cut[cut>vmax] = vmax
    levels = mpl.ticker.MaxNLocator(nbins=21).tick_values(vmin, vmax)

    fig, AX = plt.subplots(1,2)
    for ax in AX: ax.set_aspect('equal')
    cs = AX[0].contourf(grid_y[indY], grid_z[indZ], cut.T, levels=levels, zdir='x', offset=grid_x[int(indX)],
                        vmin=vmin, vmax=vmax, cmap=cmap, extent='both')
    # AX[1].quiver(np.ones_like(eyzy)*yNew[:,None], np.ones_like(eyzy)*zNew[None,:],
    #             eyzy*1e16, eyzz*1e16)
    c = EMag
    # c = np.log10(EMotMag)
    vmin = -3.3 ; vmax = -2.2
    cmap = bwr_2
    cmap = wbr_1
    cut = c[52]
    cut = cut[indY]
    cut = cut[:,indZ]
    cut[cut<vmin] = vmin
    cut[cut>vmax] = vmax
    AX[1].streamplot(yNew, zNew, eyzy.T, eyzz.T, color=cut.T,
                    cmap=wbr_1, density=2)

    plt.tight_layout()
    plt.show()
    # sys.exit()


if 0:
    # E /= np.mean(E[grid_x.size-20:])
    eyzy = E[0,:,int(yLen*.5)]
    eyzy = eyzy[:,indZ]
    eyzz = E[2,:,int(yLen*.5)]
    eyzz = eyzz[:,indZ]
    xNew = grid_x
    zNew = grid_z[indZ]

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cut = c[:,int(yLen*.5)]
    cut = cut[:,indZ]
    cut[cut<vmin] = vmin
    cut[cut>vmax] = vmax
    levels = mpl.ticker.MaxNLocator(nbins=21).tick_values(vmin, vmax)

    fig, AX = plt.subplots(1,2)
    for ax in AX: ax.set_aspect('equal')
    # cs = AX[0].contourf(grid_x, grid_z[indZ], cut.T, levels=levels, zdir='x', offset=grid_x[int(indX)],
    #                     vmin=vmin, vmax=vmax, cmap=cmap, extent='both')
    cs = AX[0].pcolormesh(grid_x, grid_z[indZ], cut.T, cmap=wbr_1, vmin=vmin, vmax=vmax, rasterized=True)
    c = EMag
    vmin = -3.3 ; vmax = -2.2
    cmap = wbr_1
    cut = c[:,int(yLen*.5)]
    cut = cut[:,indZ]
    cut[cut<vmin] = vmin
    cut[cut>vmax] = vmax
    AX[1].streamplot(xNew, zNew, eyzy.T, eyzz.T, color=cut.T,
                    cmap=wbr_1, density=2)

    plt.tight_layout()
    plt.show()
    sys.exit()


fig = plt.figure(figsize=(24,16))
ax = plt.axes(projection=Axes3D.name)
ax.view_init(elev=20., azim=285)
ax.set_facecolor('#d1d133')

import own_colours as oC
for indX in indXX:
    print(indX)

    #___
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    indInd = indX*np.ones_like(grid_x, dtype=bool)[:,None,None] * indY[None,:,None] * indZ[None,None,:]
    cut = field[indX]
    cut = cut[indY]
    cut = cut[:,indZ]


    # eyzy = B[1,25]
    # eyzy = eyzy[indY]
    # eyzy = eyzy[:,indZ]
    # eyzy = eyzy[::10]
    # eyzy = eyzy[:,::10]
    # eyzz = B[2,25]
    # eyzz = eyzz[indY]
    # eyzz = eyzz[:,indZ]
    # eyzz = eyzz[::10]
    # eyzz = eyzz[:,::10]

    if plotContour:
        cut[cut<vmin] = vmin
        cut[cut>vmax] = vmax
        levels = mpl.ticker.MaxNLocator(nbins=21).tick_values(vmin, vmax)
        cs = ax.contourf(cut.T, y, z, levels=levels, zdir='x', offset=grid_x[int(indX)], vmin=vmin, vmax=vmax, cmap=oC.bwr_2, extent='both')
    else:
        colors = cmap(norm(cut.T))
        ax.plot_surface(X, y, z, cstride=stride, rstride=stride, facecolors=colors, shade=False, rasterized=True)
    if 0:#indX== indXX[1]:
        # xx, yy, zz = np.meshgrid(grid_x[indX], grid_y[indY][::10], grid_z[indZ][::10])
        # print(indX)
        ax.quiver(grid_x[indX], np.ones_like(eyzy)*yNew[None,:,None], np.ones_like(eyzy)*zNew[None,None,:], np.zeros_like(grid_x[indX]),
            eyzy*5e8, eyzz*5e8,
            color='k')


    #___
    X = np.ones(5)*grid_x[indX]
    ax.plot(X, Y, Z, 'k', lw=.2)
    # ax.plot(X, Y, Z2, 'k', lw=.2)
    ax.plot(X,YFrame,ZFrame, 'r')# 'none', lw=.2)
    X = np.ones(2)*grid_x[indX]
    ax.plot(X, [-winDimY,winDimY], [0,0], 'k', lw=.2)
    # ax.plot(X, [0,0], [-winDimZ,winDimZ], 'k', lw=.2)
    ax.plot(X, [0,0], [winDimZMin, winDimZMax], 'k', lw=.2)
    # ax.plot(XFrame, [0,0], [0,0], 'r')# c='none')

# plt.colorbar(cs)

# plt.axis('off')
plt.tight_layout()
plt.show()
