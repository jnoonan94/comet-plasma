#!/usr/bin/env python3.8
import sys
sys.path.append('/home/etienneb/ParaView-5.9.0-MPI-Linux-Python3.8-64bit/lib/python3.8/site-packages')
# print(sys.path)
import numpy as np
from paraview.simple import *
from paraview.servermanager import *


from evtk import hl

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
    # sys.exit('Please provide run_ID with --run option.\n')
    path = '/home/etienneb/Models/Menura/menura'
else:
    try:
        run_ID = f'{int(args.run):03}'
    except:
        run_ID = args.run
    path = f'/home/etienneb/Models/Menura_own/Data/run_{run_ID}'
path_remote = f'/gpfswork/rech/fuz/ued64ot/run_{run_ID}'
remote_label = 'jean-zay'


not_yet_rolled = True

for it in [it]:
    print(it)

    md = menura_data(path, path_remote, it, ask_scp=False, print_param=False, remote_label=remote_label)

    E = md.load_field('E')
    B = md.load_field('B')
    Ji = md.load_field('Ji')
    B = md.stag_vector(B)

    dx = md.mp.dX
    dyBz = 1./(12.*dx) * ( B[2, 2:-2, 4:  ]-8*B[2, 2:-2, 3:-1]+8*B[2, 2:-2, 1:-3]-B[2, 2:-2,  :-4] )
    dxBz = 1./(12.*dx) * ( B[2, 4:  , 2:-2]-8*B[2, 3:-1, 2:-2]+8*B[2, 1:-3, 2:-2]-B[2,  :-4, 2:-2] )
    dxBy = 1./(12.*dx) * ( B[1, 4:  , 2:-2]-8*B[1, 3:-1, 2:-2]+8*B[1, 1:-3, 2:-2]-B[1,  :-4, 2:-2] )
    dyBx = 1./(12.*dx) * ( B[0, 2:-2, 4:  ]-8*B[0, 2:-2, 3:-1]+8*B[0, 2:-2, 1:-3]-B[0, 2:-2,  :-4] )

    J_tot = np.zeros((3, md.mp.len_x_cst, md.mp.len_y_cst*md.mp.mpi_nb_proc_y))
    J_tot[0] = dyBz
    J_tot[1] = -dxBz
    J_tot[2] = dxBy - dyBx
    #
    #
    # fig, AX = plt.subplots(1, 2, figsize=(18, 14), sharex=True, sharey=True)
    # for ax in AX:
    #     ax.set_aspect('equal')
    #
    # AX[0].pcolormesh(B[1].T,
    #                   vmin=-1.5, vmax=1.5,
    #                   cmap=oC.bwr_2)
    #
    # AX[1].pcolormesh(B[0].T,
    #                   vmin=-1.5, vmax=1.5,
    #                   cmap=oC.bwr_2)
    #
    # # plt.colorbar(p)
    #
    # oT.set_spines(AX)
    # plt.tight_layout()
    # plt.show()
    #
    # fig, AX = plt.subplots(1, 2, figsize=(20, 14), sharex=True, sharey=True)
    # for ax in AX:
    #     ax.set_aspect('equal')
    #
    # p = AX[1].pcolormesh(J_tot[2].T,
    #                   vmin=-2, vmax=2,
    #                   cmap=oC.bwr_2)
    #
    # # plt.colorbar(p)
    #
    # p = AX[0].pcolormesh(Ji[2].T,
    #                   vmin=-2, vmax=2,
    #                   cmap=oC.bwr_2)
    #
    # # plt.colorbar(p)
    #
    # oT.set_spines(AX)
    # plt.tight_layout()
    # plt.show()
    # sys.exit()


    J2 = np.ones((md.mp.len_x_cst, md.mp.len_y_cst*md.mp.mpi_nb_proc_y, 2, 3))
    J2[:,:,0] = J_tot.T
    J2[:,:,1] = J_tot.T+1

    # E_hal = np.ones((md.mp.len_x_cst, md.mp.len_y_cst*md.mp.mpi_nb_proc_y, 2, 3))
    # E_hal[:,:,0] = md.E_hal.T
    # E_hal[:,:,1] = md.E_hal.T+1


    # B = md.recompose_field(md.B)
    B_perp = np.sqrt(B[0]**2 + B[1]**2)
    # B[0][B_perp<.1] = 0. #.00001
    # B[1][B_perp<.1] = 0. #.00001
    # B[2][B_perp<.1] = 0. #.00001
    B_perp = np.log10(B_perp)
    B_perp = md.stag_scalar(B_perp)
    B_perp = np.stack((B_perp, B_perp)).T

    B2 = np.ones((md.mp.len_x_cst+4, md.mp.len_x_cst+4, 2, 3))
    B2[:,:,0] = B.T
    B2[:,:,1] = B.T+1
    # Bx = np.ones((1000, 1000, 2))
    # print(Bx.flags)
    # Bx = B2[:,:,:,2]
    # print(Bx.flags)
    # # Bx.flags['C_CONTIGUOUS'] = True
    # sys.exit()
    # # B2.flags['C_CONTIGUOUS'] = True


    dens_com = md.load_field('dens_spec_1')
    dens_com = np.log10(dens_com)
    dens_com = np.stack((dens_com, dens_com)).T

    grid_x = np.ones((md.mp.len_x_cst+4, md.mp.len_x_cst+4, 2))*md.grid_x_box[:, None, None]
    grid_y = np.ones((md.mp.len_x_cst+4, md.mp.len_x_cst+4, 2))*md.grid_y_box[None, :, None]
    grid_z = np.ones((md.mp.len_x_cst+4, md.mp.len_x_cst+4, 2))*np.array([0, 1])[None, None, :]

    # b = np.stack((self.grid_xy_box[0].flatten(), self.grid_xy_box[1].flatten(), B_perp.flatten()))

    import random as rnd
    nx, ny, nz = md.mp.len_x_cst+4, md.mp.len_x_cst+4, 2
    lx, ly, lz = 1.0, 1.0, 1.0
    dx, dy, dz = lx/nx, ly/ny, lz/nz
    npoints = nx * ny * nz
    # npoints = (nx + 1) * (ny + 1) * (nz + 1)
    # Coordinates
    X = np.arange(0, lx + 0.1*dx, dx, dtype='float32')
    Y = np.arange(0, ly + 0.1*dy, dy, dtype='float32')
    Z = np.arange(0, lz + 0.1*dz, dz, dtype='float32')
    x = np.zeros((nx, ny, nz))
    y = np.zeros((nx, ny, nz))
    z = np.zeros((nx, ny, nz))
    # We add some random fluctuation to make the grid more interesting
    for k in range(nz ):
        for j in range(ny ):
            for i in range(nx ):
                x[i,j,k] = X[i] #+ (0.5 - rnd.random()) * 0.2 * dx
                y[i,j,k] = Y[j] #+ (0.5 - rnd.random()) * 0.2 * dy
                z[i,j,k] = Z[k] #+ (0.5 - rnd.random()) * 0.2 * dz
    # Variables
    tx = np.random.rand(npoints).reshape( (nx, ny, nz))
    ty = np.random.rand(npoints).reshape( (nx, ny, nz))
    tz = np.random.rand(npoints).reshape( (nx, ny, nz))
    # print(x.shape, tx.flags)
    # print(grid_x.shape,  B2[:,:,:,0].flags)
    # print(grid_x.shape, B2[:,:,:,0])
    # tx = B2[:,:,:,0]
    # hl.imageToVTK("./image",  pointData = {"tx" : tx, "ty": ty, "tz": tz})
    hl.gridToVTK("./B_perp_tmp", grid_x, grid_y, grid_z,
                 pointData={'Bx': B2[:,:,:,0],
                            'By': B2[:,:,:,1],
                            'Bz': B2[:,:,:,2],
                            'B_perp': B_perp,
                            'dens_com': dens_com})
                            # 'Jz': J2[:,:,:,2],
                            # 'E_halx' : E_hal[:,:,:,0],
                            # 'E_haly' : E_hal[:,:,:,1],
                            # 'E_halz' : E_hal[:,:,:,2],

    #______________________________________________________________________________________________________________________________

    LoadPlugin('/home/etienneb/ParaView-5.9.0-MPI-Linux-Python3.8-64bit/bin/../lib/paraview-5.9/plugins/SurfaceLIC/SurfaceLIC.so',
               remote=False, connection=None)

    reader = OpenDataFile(filename="B_perp_tmp.vts")

    calculator = Calculator(reader)
    calculator.SetPropertyWithName(pname='Input',           arg=reader)
    calculator.SetPropertyWithName(pname='Function',        arg='(iHat*By)+(jHat*Bx)')
    # calculator.SetPropertyWithName(pname='Function',        arg='(iHat*E_haly)+(jHat*E_halx)')
    calculator.SetPropertyWithName(pname='ResultArrayName', arg='yo')

    renderView1 = GetActiveViewOrCreate('RenderView')
    renderView1.Update()

    randomVectors1Display = Show(calculator, renderView1)
    ColorBy(randomVectors1Display, ('POINTS', 'RTData'))
    randomVectors1Display.Representation = 'Surface LIC'
    randomVectors1Display.ColorArrayName = ['POINTS', 'B_perp']
    # randomVectors1Display.ColorArrayName = ['POINTS', 'dens_com']
    # randomVectors1Display.RescaleTransferFunctionToDataRange(True)
    randomVectors1Display.SetPropertyWithName(pname='StepSize',        arg=0.2)
    randomVectors1Display.SetPropertyWithName(pname='ColorMode',       arg='Multiply')
    randomVectors1Display.SetPropertyWithName(pname='EnhanceContrast', arg='LIC and Color')

    randomVectors1Display.SetScalarBarVisibility(renderView1, False)
    rTDataLUT = GetColorTransferFunction('RTData')
    with open('bwr_eb_script.xml', 'r') as f: ## for B_perp
    # with open('wbr_eb_script.xml', 'r') as f: ## for dens_com
        data = f.read()
        rTDataLUT.ApplyColorMap(data)

    rTDataLUT.RescaleTransferFunction(-2.2, 1.5) ## for B_perp
    # rTDataLUT.RescaleTransferFunction(-1., 2.) ## for dens_com
    # rTDataLUT.RescaleTransferFunction(-3.36, 1.6)

    # renderView1.Update()
    # renderView1.ViewSize = [2000, 2000] #[width, height]
    renderView1.ViewSize = [2000, 1400] #[width, height]
    renderView1.CenterOfRotation = [250, 250, 0.5]
    renderView1.CameraPosition = [250, 250, -500]
    renderView1.CameraFocalPoint = [250, 250, 0.5]
    camera = GetActiveCamera()
    if not_yet_rolled:
        camera.Roll(-90)
        not_yet_rolled = False
    # Render()
    renderView1.SetPropertyWithName('OrientationAxesVisibility', False)


    #save screenshot
    SaveScreenshot(f'/home/etienneb/Desktop/LIC/{it}.png', CompressionLevel=0)
    # SaveScreenshot(f'/home/etienneb/Desktop/lic_dens_ityo{it}_2.png', CompressionLevel=0)
    # sys.exit()
    # convert -delay 10 -resize x500 -loop 0 *.png leoTurbul.gif
