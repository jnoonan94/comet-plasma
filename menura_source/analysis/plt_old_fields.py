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



class menura_data:

    def __init__(self, local_path, iteration):

        self.local_path = local_path
        # self.remote_path = remote_path
        self.idx_it = iteration


        self.mp = menura_param(f'{local_path}/products')
        # self.mp.print_physical_parameters()
        self.nb_proc = 1#self.mp.mpi_nb_proc

        self.B    = np.zeros((3, self.mp.len_x_cst+2, self.mp.len_y_cst+2))
        self.E    = np.zeros((3, self.mp.len_x_cst+2, self.mp.len_y_cst+2))
        self.dens = np.zeros((self.mp.len_x_cst+2, self.mp.len_y_cst+2))
        self.dens_spec0 = np.zeros((self.mp.len_x_cst+2, self.mp.len_y_cst+2))
        self.dens_spec1 = np.zeros((self.mp.len_x_cst+2, self.mp.len_y_cst+2))
        self.curr = np.zeros((3, self.mp.len_x_cst+2, self.mp.len_y_cst+2))


        self.grid_x = np.zeros((self.mp.len_x_cst+2))
        self.grid_y = np.zeros((self.mp.len_y_cst+2))
        self.edges_x = np.zeros((self.mp.len_x_cst+3))
        self.edges_y = np.zeros((self.mp.len_y_cst+3))

        self.fig = None
        self.AX = None

        self.load_data()

        self.curr /= self.mp.e*self.dens*self.mp.v_A
        self.dens /= self.mp.n0_no_norm
        self.B /= self.mp.B0_no_norm
        self.E /= (self.mp.B0_no_norm*self.mp.v_A)


    def load_data(self):


        self.B = np.load(f'{self.local_path}/products/B_{self.idx_it}.npy')
        # print(self.B[:, 513, 513]/self.mp.B0_no_norm)#
        # sys.exit()
        self.E = np.load(f'{self.local_path}/products/E_{self.idx_it}.npy')
        self.dens = np.load(f'{self.local_path}/products/dens_{self.idx_it}.npy')
        # self.dens_spec0 = np.load(f'{self.local_path}/products/dens_spec0_{self.idx_it}.npy')
        self.curr = np.load(f'{self.local_path}/products/curr_{self.idx_it}.npy')
        self.grid_x = np.load(f'{self.local_path}/products/grid_x.npy')
        self.grid_y = np.load(f'{self.local_path}/products/grid_y.npy')
        self.edges_x[ :-1] = self.grid_x-self.mp.dX/2.
        self.edges_x[ -1] =  self.grid_x[-1]+self.mp.dX/2
        self.edges_y[ :-1] = self.grid_y-self.mp.dX/2.
        self.edges_y[ -1] =  self.grid_y[-1]+self.mp.dX/2


    def plt_all(self):

        # B_perp = (self.B[:,0]**2 + self.B[:,1]**2)
        # v_perp = (self.v[:,0]**2 + self.v[:,1]**2)

        self.fig, self.AX = plt.subplots(3, 4, figsize=(24, 18))
        for ax in self.AX.flatten():
            ax.set_aspect('equal')

        self.fig.canvas.mpl_connect('button_press_event', self.plt_field)


        self.AX[0, 0].pcolormesh(self.edges_x, self.edges_y, self.B[0].T,
                            vmin=np.amin(self.B[:, 0]), vmax=np.amax(self.B[:, 0]),
                            # vmin=-.6, vmax=.6,
                            cmap=oC.bwr_2, rasterized=True)
        self.AX[1, 0].pcolormesh(self.edges_x, self.edges_y, self.B[1].T,
                            vmin=np.amin(self.B[:, 1]), vmax=np.amax(self.B[:, 1]),
                            # vmin=-.6, vmax=.6,
                            cmap=oC.bwr_2, rasterized=True)
        self.AX[2, 0].pcolormesh(self.edges_x, self.edges_y, self.B[2].T,
                            vmin=np.amin(self.B[:, 2]), vmax=np.amax(self.B[:, 2]),
                            # vmin=-.6, vmax=.6,
                            cmap=oC.bwr_2, rasterized=True)

        self.AX[0, 1].pcolormesh(self.edges_x, self.edges_y, self.E[0].T,
                            # vmin=np.amin(self.E[:, 0]), vmax=np.amax(self.E[:, 0]),
                            vmin=-.6, vmax=.6,
                            cmap=oC.bwr_2, rasterized=True)
        self.AX[1, 1].pcolormesh(self.edges_x, self.edges_y, self.E[1].T,
                            # vmin=np.amin(self.E[:, 1]), vmax=np.amax(self.E[:, 1]),
                            vmin=-.6, vmax=.6,
                            cmap=oC.bwr_2, rasterized=True)
        self.AX[2, 1].pcolormesh(self.edges_x, self.edges_y, self.E[2].T,
                            # vmin=np.amin(self.E[:, 2]), vmax=np.amax(self.E[:, 2]),
                            vmin=-.6, vmax=.6,
                            cmap=oC.bwr_2, rasterized=True)

        self.AX[0, 2].pcolormesh(self.edges_x, self.edges_y, self.dens.T,
                            # vmin=np.amin(self.dens), vmax=np.amax(self.dens),
                            vmin=0.9, vmax=1.1,
                            cmap=oC.bwr_2, rasterized=True)
        # self.AX[1, 2].pcolormesh(self.edges_x, self.edges_y, self.dens_spec0.T,
        #                     vmin=np.amin(self.dens_spec0), vmax=np.amax(self.dens_spec0),
        #                     cmap=oC.bwr_2, rasterized=True)
        # if self.mp.inject_pla_cst:
        #     self.AX[2, 2].pcolormesh(self.edges_x, self.edges_y, np.log10(self.dens_spec1).T,
        #                         vmin=np.amax(np.log10(self.dens_spec1))-5, vmax=np.amax(np.log10(self.dens_spec1)),
        #                         cmap=oC.bwr_2, rasterized=True)

        self.AX[0, 3].pcolormesh(self.edges_x, self.edges_y, self.curr[0].T,
                            # vmin=np.nanmin(self.curr[:, 0]), vmax=np.nanmax(self.curr[:, 0]),
                            vmin=-.6, vmax=.6,
                            cmap=oC.bwr_2, rasterized=True)
        self.AX[1, 3].pcolormesh(self.edges_x, self.edges_y, self.curr[1].T,
                            # vmin=np.amin(self.curr[:, 1]), vmax=np.amax(self.curr[:, 1]),
                            vmin=-.6, vmax=.6,
                            cmap=oC.bwr_2, rasterized=True)
        self.AX[2, 3].pcolormesh(self.edges_x, self.edges_y, self.curr[2].T,
                            # vmin=np.amin(self.curr[:, 2]), vmax=np.amax(self.curr[:, 2]),
                            vmin=-.6, vmax=.6,
                            cmap=oC.bwr_2, rasterized=True)

        for ax in self.AX.flatten():
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def plt_all_3d(self):

        self.fig, self.AX = plt.subplots(3, 4, figsize=(24, 18))
        for ax in self.AX.flatten():
            ax.set_aspect('equal')

        # self.fig.canvas.mpl_connect('button_press_event', self.plt_field)
        idx_mid_z = int(self.mp.len_z_cst/2.)

        for r in range(self.nb_proc):
            self.AX[0, 0].pcolormesh(self.edges_x, self.edges_y, self.B[r, 0, :, :, idx_mid_z].T,
                                vmin=np.amin(self.B[:, 0, :, :, idx_mid_z]), vmax=np.amax(self.B[:, 0, :, :, idx_mid_z]),
                                cmap=oC.bwr_2, rasterized=True)
            self.AX[1, 0].pcolormesh(self.edges_x, self.edges_y, self.B[r, 1, :, :, idx_mid_z].T,
                                vmin=np.amin(self.B[:, 1, :, :, idx_mid_z]), vmax=np.amax(self.B[:, 1, :, :, idx_mid_z]),
                                cmap=oC.bwr_2, rasterized=True)
            self.AX[2, 0].pcolormesh(self.edges_x, self.edges_y, self.B[r, 2, :, :, idx_mid_z].T,
                                vmin=np.amin(self.B[:, 2, :, :, idx_mid_z]), vmax=np.amax(self.B[:, 2, :, :, idx_mid_z]),
                                cmap=oC.bwr_2, rasterized=True)

            self.AX[0, 1].pcolormesh(self.edges_x, self.edges_y, self.E[r, 0, :, :, idx_mid_z].T,
                                vmin=np.amin(self.E[:, 0, :, :, idx_mid_z]), vmax=np.amax(self.E[:, 0, :, :, idx_mid_z]),
                                cmap=oC.bwr_2, rasterized=True)
            self.AX[1, 1].pcolormesh(self.edges_x, self.edges_y, self.E[r, 1, :, :, idx_mid_z].T,
                                vmin=np.amin(self.E[:, 1, :, :, idx_mid_z]), vmax=np.amax(self.E[:, 1, :, :, idx_mid_z]),
                                cmap=oC.bwr_2, rasterized=True)
            self.AX[2, 1].pcolormesh(self.edges_x, self.edges_y, self.E[r, 2, :, :, idx_mid_z].T,
                                vmin=np.amin(self.E[:, 2, :, :, idx_mid_z]), vmax=np.amax(self.E[:, 2, :, :, idx_mid_z]),
                                cmap=oC.bwr_2, rasterized=True)

            self.AX[0, 2].pcolormesh(self.edges_x, self.edges_y, self.dens[r, :, :, idx_mid_z].T,
                                vmin=np.amin(self.dens), vmax=np.amax(self.dens),
                                cmap=oC.bwr_2, rasterized=True)
            self.AX[1, 2].pcolormesh(self.edges_x, self.edges_y, self.dens_spec0[r, :, :, idx_mid_z].T,
                                vmin=np.amin(self.dens_spec0), vmax=np.amax(self.dens_spec0),
                                cmap=oC.bwr_2, rasterized=True)
            if self.mp.inject_pla_cst:
                self.AX[2, 2].pcolormesh(self.edges_x, self.edges_y, self.dens_spec1[r, :, :, idx_mid_z].T,
                                    vmin=np.amin(self.dens_spec1), vmax=np.amax(self.dens_spec1),
                                    cmap=oC.bwr_2, rasterized=True)

            self.AX[0, 3].pcolormesh(self.edges_x, self.edges_y, self.curr[r, 0, :, :, idx_mid_z].T,
                                vmin=np.amin(self.curr[:, 0]), vmax=np.amax(self.curr[:, 0]),
                                cmap=oC.bwr_2, rasterized=True)
            self.AX[1, 3].pcolormesh(self.edges_x, self.edges_y, self.curr[r, 1, :, :, idx_mid_z].T,
                                vmin=np.amin(self.curr[:, 1]), vmax=np.amax(self.curr[:, 1]),
                                cmap=oC.bwr_2, rasterized=True)
            self.AX[2, 3].pcolormesh(self.edges_x, self.edges_y, self.curr[r, 2, :, :, idx_mid_z].T,
                                vmin=np.amin(self.curr[:, 2]), vmax=np.amax(self.curr[:, 2]),
                                cmap=oC.bwr_2, rasterized=True)

        for ax in self.AX.flatten():
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def plt_field(self, event):

        plt_it = True
        if   event.inaxes==self.AX[0, 0]:
            field = self.B[0]
            label = 'Bx'
        elif event.inaxes==self.AX[1, 0]:
            field = self.B[1]
            label = 'By'
        elif event.inaxes==self.AX[2, 0]:
            field = self.B[2]
            label = 'Bz'
        elif event.inaxes==self.AX[0, 1]:
            field = self.E[0]
            label = 'Ex'
        elif event.inaxes==self.AX[1, 1]:
            field = self.E[1]
            label = 'Ey'
        elif event.inaxes==self.AX[2, 1]:
            field = self.E[2]
            label = 'Ez'
        elif event.inaxes==self.AX[0, 2]:
            field = self.dens
            label = 'dens tot'
        elif event.inaxes==self.AX[1, 2]:
            field = self.dens_spec0
            label = 'dens sw'
        elif event.inaxes==self.AX[2, 2]:
            field = self.dens_spec1
            label = 'dens pla'
        elif event.inaxes==self.AX[0, 3]:
            field = self.curr[0]
            label = 'Jx'
        elif event.inaxes==self.AX[1, 3]:
            field = self.curr[1]
            label = 'Jy'
        elif event.inaxes==self.AX[2, 3]:
            field = self.curr[2]
            label = 'Jz'
        else:
            plt_it = False

        if plt_it:
            self.fig2, ax = plt.subplots(figsize=(16, 12))
            # self.fig2.canvas.mpl_connect('button_press_event', self.close_fig)
            ax.set_aspect('equal')
            #
            for r in range(self.nb_proc):
                p0 = ax.pcolormesh(self.edges_x, self.edges_y, (field).T,
                                   # vmin=np.nanmin(field), vmax=np.nanmax(field),
                                   vmin=0.85, vmax=1.15,
                                   cmap=oC.bwr_2, rasterized=True)


            idx_y = int((.9-1.)*self.nb_proc*self.mp.len_y_cst+self.mp.len_y_cst)
            # ax.text(self.edges_x[-1, int(.1*self.mp.len_x_cst)], self.edges_y[-1, idx_y], label, ha='center', va='center', fontsize=20)

            posAx = ax.get_position()
            cax = self.fig2.add_axes([posAx.x1*1., posAx.y0, 0.02, .3])
            cb = self.fig2.colorbar(p0, cax=cax, orientation='vertical')
            # cb.set_label(label, rotation=0, ha='left', fontsize=24)
            #
            oT.set_spines(ax)
            plt.tight_layout()
            plt.show()

    def plt_B_perp(self):

        fig, ax = plt.subplots(figsize=(12, 12))

        ax.set_aspect('equal')
        #
        B_perp = (self.B[:,0]**2 + self.B[:,1]**2)
        print('B_x mean: ', np.mean(self.B[:, 0]))
        print('B_y mean: ', np.mean(self.B[:, 1]))
        print('B_z mean: ', np.mean(self.B[:, 2]))

        for r in range(self.nb_proc):
            p0 = ax.pcolormesh(self.edges_x, self.edges_y, B_perp.T,
                               # vmin=np.amin(B_perp), vmax=np.amax(B_perp),
                               vmin=0., vmax=self.mp.fluctu_sqrt_B,
                               cmap=oC.bwr_2, rasterized=True)

        idx_y = int((.9-1.)*self.nb_proc*self.mp.len_y_cst+self.mp.len_y_cst)
        ax.text(self.edges_x[-1, int(.1*self.mp.len_x_cst)], self.edges_y[-1, idx_y], 'B_perp', ha='center', va='center', fontsize=20)

        posAx = ax.get_position()
        cax = fig.add_axes([posAx.x1*1., posAx.y0, 0.02, .3])
        cb = fig.colorbar(p0, cax=cax, orientation='vertical')
        # cb.set_label(label, rotation=0, ha='left', fontsize=24)
        #
        oT.set_spines(ax)
        plt.show()

    def plt_v_perp(self):

        fig, ax = plt.subplots(figsize=(12, 12))

        ax.set_aspect('equal')
        #
        v_i = self.curr/self.dens[:,None]
        v_i_perp = (v_i[:, 0]**2 + v_i[:, 1]**2)
        print('v_x mean: ', np.mean(v_i[:, 0]))
        print('v_y mean: ', np.mean(v_i[:, 1]))
        print('v_z mean: ', np.mean(v_i[:, 2]))

        for r in range(self.nb_proc):
            p0 = ax.pcolormesh(self.edges_x, self.edges_y, v_i_perp.T,
                               vmin=np.amin(v_i_perp), vmax=np.amax(v_i_perp),
                               # vmin=0., vmax=.8,
                               cmap=oC.bwr_2, rasterized=True)

        idx_y = int((.9-1.)*self.nb_proc*self.mp.len_y_cst+self.mp.len_y_cst)
        ax.text(self.edges_x[-1, int(.1*self.mp.len_x_cst)], self.edges_y[-1, idx_y], 'v_i_perp', ha='center', va='center', fontsize=20)

        posAx = ax.get_position()
        cax = fig.add_axes([posAx.x1*1., posAx.y0, 0.02, .3])
        cb = fig.colorbar(p0, cax=cax, orientation='vertical')
        # cb.set_label(label, rotation=0, ha='left', fontsize=24)
        #
        oT.set_spines(ax)
        plt.show()

    def plt_E_perp(self):

        fig, ax = plt.subplots(figsize=(12, 12))

        ax.set_aspect('equal')
        #

        E_perp = (self.E[:, 0]**2 + self.E[:, 1]**2)
        print('E_x mean: ', np.mean(self.E[:, 0]))
        print('E_y mean: ', np.mean(self.E[:, 1]))
        print('E_z mean: ', np.mean(self.E[:, 2]))

        for r in range(self.nb_proc):
            p0 = ax.pcolormesh(self.edges_x, self.edges_y, E_perp.T,
                               # vmin=np.amin(v_i_perp), vmax=np.amax(v_i_perp),
                               # vmin=0., vmax=.8,
                               cmap=oC.bwr_2, rasterized=True)

        idx_y = int((.9-1.)*self.nb_proc*self.mp.len_y_cst+self.mp.len_y_cst)
        ax.text(self.edges_x[-1, int(.1*self.mp.len_x_cst)], self.edges_y[-1, idx_y],
                'E_perp', ha='center', va='center', fontsize=20)

        posAx = ax.get_position()
        cax = fig.add_axes([posAx.x1*1., posAx.y0, 0.02, .3])
        cb = fig.colorbar(p0, cax=cax, orientation='vertical')
        # cb.set_label(label, rotation=0, ha='left', fontsize=24)
        #
        oT.set_spines(ax)
        plt.show()

    def plt_field_3d(self, field_label='B'):

        if field_label == 'B':
            field = self.B
            # field = self.B[:,:,2:-2,2:-2,2:-2]
            # field = np.roll(field, 16, axis=2)
            # field = np.roll(field, 16, axis=3)
            # field = np.roll(field, 16, axis=4)
            # self.edges_x = self.edges_x[:,2:-2]
            # self.edges_y = self.edges_y[:,2:-2]
            # self.edges_z = self.edges_z[:,2:-2]
            vec_dim = 3
        elif field_label == 'E':
            field = self.E
            vec_dim = 3
        elif field_label == 'curr':
            field = self.curr
            vec_dim = 3
        elif field_label == 'dens':
            field = self.dens
            vec_dim = 1


        if vec_dim == 3:
            fig, AX = plt.subplots(3, 3, figsize=(18, 12))
            for ax in AX.flatten():
                ax.set_aspect('equal')

            AX[0, 0].pcolormesh(self.edges_x[0], self.edges_y[0], field[0, 0, :, :, int((self.mp.len_z_cst+4)/2.)].T,
                             cmap=oC.bwr_2, rasterized=True)
            AX[0, 1].pcolormesh(self.edges_x[0], self.edges_z[0], field[0, 0, :, int((self.mp.len_y_cst+2)/2.)].T,
                             cmap=oC.bwr_2, rasterized=True)
            AX[0, 2].pcolormesh(self.edges_y[0], self.edges_z[0], field[0, 0, int((self.mp.len_x_cst+2)/2.)].T,
                             cmap=oC.bwr_2, rasterized=True)

            AX[1, 0].pcolormesh(self.edges_x[0], self.edges_y[0], field[0, 1, :, :, int((self.mp.len_z_cst+4)/2.)].T,
                                cmap=oC.bwr_2, rasterized=True)
            AX[1, 1].pcolormesh(self.edges_x[0], self.edges_z[0], field[0, 1, :, int((self.mp.len_y_cst+2)/2.)].T,
                                cmap=oC.bwr_2, rasterized=True)
            AX[1, 2].pcolormesh(self.edges_y[0], self.edges_z[0], field[0, 1, int((self.mp.len_x_cst+2)/2.)].T,
                                cmap=oC.bwr_2, rasterized=True)

            AX[2, 0].pcolormesh(self.edges_x[0], self.edges_y[0], field[0, 2, :, :, int((self.mp.len_z_cst+4)/2.)].T,
                               cmap=oC.bwr_2, rasterized=True)
            AX[2, 1].pcolormesh(self.edges_x[0], self.edges_z[0], field[0, 2, :, int((self.mp.len_y_cst+2)/2.)].T,
                               cmap=oC.bwr_2, rasterized=True)
            AX[2, 2].pcolormesh(self.edges_y[0], self.edges_z[0], field[0, 2, int((self.mp.len_x_cst+2)/2.)].T,
                               cmap=oC.bwr_2, rasterized=True)

            AX[2, 0].set_xlabel('X', fontsize=14)
            AX[2, 0].set_ylabel('Y', fontsize=14)
            AX[2, 1].set_xlabel('X', fontsize=14)
            AX[2, 1].set_ylabel('Z', fontsize=14)
            AX[2, 2].set_xlabel('Y', fontsize=14)
            AX[2, 2].set_ylabel('Z', fontsize=14)

            plt.suptitle(field_label)

            oT.set_spines(AX)
            plt.tight_layout()
            plt.show()

        elif vec_dim == 1:
            fig, AX = plt.subplots(1, 3, figsize=(18, 12))
            for ax in AX:
                ax.set_aspect('equal')

            p0 = AX[0].pcolormesh(self.edges_x[0], self.edges_y[0], field[0, :, :, int((self.mp.len_z_cst+4)/2.)].T,
                             cmap=oC.bwr_2, rasterized=True)
            AX[1].pcolormesh(self.edges_x[0], self.edges_z[0], field[0, :, int((self.mp.len_z_cst+4)/2.)].T,
                             cmap=oC.bwr_2, rasterized=True)
            AX[2].pcolormesh(self.edges_y[0], self.edges_z[0], field[0, int((self.mp.len_z_cst+4)/2.)].T,
                             cmap=oC.bwr_2, rasterized=True)

            posAx = AX[0].get_position()
            cax = fig.add_axes([posAx.x1*1., posAx.y0, 0.02, .3])
            cb = fig.colorbar(p0, cax=cax, orientation='vertical')

            AX[0].set_xlabel('X', fontsize=14)
            AX[0].set_ylabel('Y', fontsize=14)
            AX[1].set_xlabel('X', fontsize=14)
            AX[1].set_ylabel('Z', fontsize=14)
            AX[2].set_xlabel('Y', fontsize=14)
            AX[2].set_ylabel('Z', fontsize=14)

            plt.suptitle(field_label)

            oT.set_spines(AX)
            plt.tight_layout()
            plt.show()

    def close_fig(self, event):
        plt.close(self.fig2)

it = 0
# os.system('scp behare@kebnekaise.hpc2n.umu.se:/home/b/behare/Private/run_27/products/*.txt ~/Desktop')
# os.system(f'scp behare@kebnekaise.hpc2n.umu.se:/home/b/behare/Private/run_27/products/B_{it}.npy ~/Desktop')
path = '/home/etienneb/Models/Menura/run_27'


if len(sys.argv)>1:
    it = int(sys.argv[1])
else:
    it = 0

md = menura_data(path, it)

md.plt_all()


sys.exit()
q = 1.602e-19

B0 = 1.e-9
v_A = 17804.9



B = np.load(f'{path}/products/B_{it}.npy')

B /= B0

B_perp = B[0]**2 + B[1]**2


fig, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect('equal')
p = ax.pcolormesh(B_perp.T,
                  vmin=0, vmax=.8,
                  cmap=oC.bwr_2, rasterized=True)

posAx = ax.get_position()
cax = fig.add_axes([posAx.x1*1., posAx.y0, 0.02, .3])
cb = fig.colorbar(p, cax=cax, orientation='vertical')

oT.set_spines(ax)
plt.show()




curr = np.load(f'{path}/products/curr_{it}.npy')

dens = np.load(f'{path}/products/dens_{it}.npy')
v = curr/(q*dens[None, :]*v_A)

v_perp = v[0]**2 + v[1]**2


fig, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect('equal')
p = ax.pcolormesh(v_perp.T,
                  # vmin=0, vmax=.8,
                  cmap=oC.bwr_2, rasterized=True)

posAx = ax.get_position()
cax = fig.add_axes([posAx.x1*1., posAx.y0, 0.02, .3])
cb = fig.colorbar(p, cax=cax, orientation='vertical')

oT.set_spines(ax)
plt.show()




E = np.load(f'{path}/products/E_{it}.npy')
E /= B0*v_A

E_perp = E[0]**2 + E[1]**2


fig, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect('equal')
p = ax.pcolormesh(E_perp.T,
                  # vmin=0, vmax=.8,
                  cmap=oC.bwr_2, rasterized=True)

posAx = ax.get_position()
cax = fig.add_axes([posAx.x1*1., posAx.y0, 0.02, .3])
cb = fig.colorbar(p, cax=cax, orientation='vertical')

oT.set_spines(ax)
plt.show()
