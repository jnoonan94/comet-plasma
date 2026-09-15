from ownPyModules import *
import ownTools as oT
import ownColors as oC
# plt.switch_backend('Qt5Agg')

from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D




normalised_sim = False
if normalised_sim:
    path = '../vPolar/products'
else:
    path = '../vPolar_publi_whistler/products'
# path = '../vWhistler/products'

# p0 = np.load('{}/particles_0.npy'.format(path)).T



# k_B = 1.3806e-23 ## Boltzmann constant, m2 kg s-2 K-1
#
B0 = 25.4e-9
ne = 16.3e6  ##m-3
# Te = 20.*60.*11604.5
me = 9.10938356e-31  ## kg
# mi = 1.660538e-27  ## kg
e = 1.60217e-19   ## C
epsilon0 = 8.854e-12  ## F.m-1
c = 299792458.  ## m.s-1
omega_ce = e*B0/me
# omega_ci = e*B0/mi
# omega_pi = np.sqrt(e**2*ne/(mi*epsilon0))
omega_pe = np.sqrt(e**2*ne/(me*epsilon0))
T_ce = 2.*np.pi/omega_ce
#
# v_the  = np.sqrt(2*k_B*Te/me)
# rho_ce = v_the/omega_ce
#
# fci = e*B0/(2*np.pi*mi)
fce = omega_ce/(2*np.pi)
# de = c/omega_pe

#____________________________
if normalised_sim:
    f = 1.
else:
    f = 250.
omega = f*2.*np.pi
X     = omega_pe*omega_pe/(omega*omega)
Y     = omega_ce/omega
theta = 20.*np.pi/180.
#____________________________
if normalised_sim:
    kvec  = 1. #.5*1.3 #2.*np.pi
else:
    kvec  = np.sqrt( (1 - (2.*X*(1.-X))/(2.*(1.-X)-Y*Y*np.sin(theta)*np.sin(theta)+Y*np.sqrt(Y*Y*theta**4+4*(1.-X)*(1.-X)*np.cos(theta)*np.cos(theta)) ) )*omega*omega/(c*c) )
kvec_para = kvec*np.cos(theta)
kvec_perp = kvec*np.sin(theta)
#
lamb        = 2.*np.pi/np.abs(kvec)
lamb_perp   = 2.*np.pi/np.abs(kvec_perp)
if normalised_sim:
    lenTot_para = 4.*2*np.pi/np.abs(kvec_para)
    lenTot_perp = 4.*2*np.pi/np.abs(kvec_perp)
else:
    lenTot_para = 1.*2*np.pi/np.abs(kvec_para)
    lenTot_perp = 1.*2*np.pi/np.abs(kvec_perp)
#
v_landau = (omega)/kvec_para
v_cyclo  = (omega-omega_ce)/kvec_para
print(v_landau)
print(v_cyclo)
print(kvec)
print((omega)/kvec)
print((omega-omega_ce)/kvec)
# sys.exit()



#_______________________________________________________________________________
EBFields = np.load('{}/EBData.npy'.format(path)).T
E = EBFields[:3]
B = EBFields[3:6]
timeSimu = EBFields[6]
EBFields_space = np.load('{}/EB_data_space.npy'.format(path)).T
E_space = EBFields_space[:3]
E_gyro = np.arctan2(E_space[1], E_space[0])
E_gyro[E_gyro<0.] += 2.*np.pi
x_grid = EBFields_space[3]

if 1:   ##  Hodogram
    fig, AX = plt.subplots(1,3, figsize=(16,10))
    for ax in AX: ax.set_aspect('equal')
    #
    AX[0].plot(B[0], B[1], oC.rgb[2], lw=1)
    rr = np.ones(400)*np.mean(oT.norm(B))
    ttheta = np.linspace(0,2*np.pi,400)
    AX[0].plot(rr*np.cos(ttheta), rr*np.sin(ttheta), '--k', lw=.5)
    AX[1].plot(B[0], B[2], oC.rgb[2], lw=1)
    AX[2].plot(B[1], B[2], oC.rgb[2], lw=1)
    # norm = plt.Normalize(subTime.min(), subTime.max())
    # lc = LineCollection(segments, cmap='BuRd', norm=norm)
    # # Set the values used for colormapping
    # lc.set_array(subTime)
    # lc.set_linewidth(2)
    # line = ax.add_collection(lc)
    # fig.colorbar(line, ax=ax)
    #
    oT.set_spines(AX)
    plt.tight_layout()
    plt.show()
    # sys.exit()
#
theta     = np.arctan2(B[1], B[0])
theta_dot = theta[2:]-theta[:-2]
vPoyn     = np.cross(E.T,B.T).T



if 1:   ## Wave form.

    fig, AX = plt.subplots(4,1, figsize=(14,12), sharex=True)
    #
    AX[0].plot(timeSimu, B[0], oC.rgb[0], label='B_x')
    AX[0].plot(timeSimu, B[1], oC.rgb[1], label='B_y')
    AX[0].plot(timeSimu, B[2], oC.rgb[2], label='B_z')
    # AX[0].plot(timeSimu[400:-1000], oT.norm(B[:,400:-1000]), 'k', label='B')
    #
    AX[1].plot(timeSimu, E[0], oC.rgb[0], label='E_x')
    AX[1].plot(timeSimu, E[1], oC.rgb[1], label='E_y')
    AX[1].plot(timeSimu, E[2], oC.rgb[2], label='E_z')
    #
    AX[2].plot(timeSimu, vPoyn[0], oC.rgb[0], label='ExB_x')
    AX[2].plot(timeSimu, vPoyn[1], oC.rgb[1], label='ExB_y')
    AX[2].plot(timeSimu, vPoyn[2], oC.rgb[2], label='ExB_z')
    #
    AX[3].plot(timeSimu, theta, label='theta')
    AX[3].plot(timeSimu[1:-1], theta_dot, label='theta_dot')
    #
    for ax in AX:
        ax.legend()

    oT.set_spines(AX)
    plt.tight_layout()
    plt.show()



    fig, ax = plt.subplots(figsize=(14, 12))

    ax.plot(x_grid, E_space[0], c=oC.rgb[0], label='Ex(x)')
    ax.plot(x_grid, np.roll(E_space[0], 400), '--', c=oC.rgb[0], label='Ex(x)')
    ax.plot(x_grid, E_space[1], c=oC.rgb[1], label='Ey(x)')
    ax.plot(x_grid, E_space[2], c=oC.rgb[2], label='Ez(x)')

    ax.legend()
    ax.set_xlabel('x')

    oT.set_spines(AX)
    plt.tight_layout()
    plt.show()
