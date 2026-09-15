#ifndef __STRUCTURES_H_INCLUDED__   // if x.h hasn't been included yet...
#define __STRUCTURES_H_INCLUDED__


#include "parameters.h"


#include <vector>


//_____________________________
/*! \brief Alternative type of field data structure.
 *
 *  The only way I found for allocating a multi-dimensional array on heap memory
 * without recompiling when changing domain dimension. Not yet used in the code.
 * Method used by smilei for their field variables.
 */
class simu_B_field_2
{
private:

public:

  float*** B;
	//constructors and destructor
	simu_B_field_2(simu_param* sP)
  {
    B = new float**[3];
    for (int h=0; h<3; h++){
      B[h] = new float*[sP->yLen];
      for(int i = 0; i < sP->xLen+4; i++){
        B[h][i] = new float[sP->yLen];
      }
    }

  };
	~simu_B_field_2()
  {
  	delete [] B;
  };

};



//______________________________________________________________________________
/*! \brief Contains the simulation grid with additional informations, such as
 *          obstacle position.
 *
 *  The simulation grid, 2D (resp. 3D), is described by 2 (resp. 3) one dimensional
 * arrays of float containing the spatial coordinate of each node. Since the grid
 * used in Menura is regular, all nodes sharing the same x-index share the same
 * x-coordinate.
 *
 * The bounds for macro-particle positions are defined here as well. No particle
 * should ever be existing outside these bounds, or they will be deposited/mapped
 * elsewhere than their simu-field->counts array, very bad.
 *
 * The centre of the optional obstacle is also stored in simu_grid.
 */
struct simu_grid
{
  float xGrid[len_x_cst+4]; /*!< Grid of node coordinates along x. */
  float yGrid[len_y_cst+4]; /*!< Grid of node coordinates along y. */
  float xMin; float xMax; /*!< Extremal position for particles along x. */
  float yMin; float yMax; /*!< Extremal position for particles along y. */
  float centre_x; /*!< Obstacle position (centre of solid body for instance) along x. */
  float centre_y; /*!< Obstacle position along y. */
  #if NB_DIM==3
    float zGrid[len_z_cst+4]; /*!< Grid of node coordinates along z. */
    float zMin; float zMax; /*!< Extremal position for particles along z. */
    float centre_z; /*!< Obstacle position along z. */
  #endif

  #if NB_DIM==2
    float rsq[len_x_cst+4][len_y_cst+4];
  #elif NB_DIM==3
    float rsq[len_x_cst+4][len_y_cst+4][len_z_cst+4];
  #endif

  #if DF_cst
  float L ;
  float vGrid [len_v_cst+4] ;
  float vMin ; float vMax ;
  #endif
};



//______________________________________________________________________________
/*! \brief Contains all field values used in the solver, at the exception of the
 *         B-field. Conatins field buffers as well.
 *
 *  All fields contained in simu_fields are not followed in time, they are
 * obtained by a combination of the magnetic field and the particles' positions
 * and speed.
 *
 *  MPI buffers are also included in this structure, with fixed size.
 *  The "steps" indicated below refer to the algorithm steps described in the
 * technical publication of the code, also found in the gallery and in the
 * main-loop comments.
 */
struct simu_fields
{
  #if NB_DIM==2
    float E           [3][len_x_cst+4][len_y_cst+4]; /*!< Electric field, crunched by Ohm. */
    float curl_E      [3][len_x_cst+4][len_y_cst+4]; /*!< E-field curl for house-keeping. */
    float density        [len_x_cst+4][len_y_cst+4]; /*!< Density used for step 0 and 3. */
    float density_b      [len_x_cst+4][len_y_cst+4]; /*!< Density used for step 1. */
    float counts         [len_x_cst+4][len_y_cst+4]; /*!< Number of macro-particle per node (a float). */
    float counts_pla     [len_x_cst+4][len_y_cst+4]; /*!< Must be multiplied by wSW/wCom to get a physical density. */
    float Lambda         [len_x_cst+4][len_y_cst+4]; /*!< Almost identical to counts, see CAM. */
    float Ji          [3][len_x_cst+4][len_y_cst+4]; /*!< Ion current for steps 0 and 3. */
    float Ji_b        [3][len_x_cst+4][len_y_cst+4]; /*!< Ion current for step1. */
    float fluxNum     [3][len_x_cst+4][len_y_cst+4]; /*!< Macro-particle flux. */
    float fluxNum_pla [3][len_x_cst+4][len_y_cst+4]; /*!< Must be multiplied by (q*wSW) to get the ion current. */
    float Gamma       [3][len_x_cst+4][len_y_cst+4]; /*!< Almost identical to fluxNum, see CAM. */
    float J_tot       [3][len_x_cst+4][len_y_cst+4]; /*!< Proper total current, curl(B) . */
    float pres           [len_x_cst+4][len_y_cst+4]; /*!< Pressure (physical). */
    #if yee_cst
      float E_s       [3][len_x_cst+4][len_y_cst+4];
      float J_tot_s   [3][len_x_cst+4][len_y_cst+4];
    #endif

    /*! Identifies in which region the node is:
    * - within a nominal plasma: 0
    * - within vacuum: 1
    * - within a solid body: 2
    */
    int region_ID  [len_x_cst+4][len_y_cst+4];

    float vort  [3][len_x_cst+4][len_y_cst+4]; /*!< Vorticity of B-field (obsolete?). */

    /*! For house-keeping purposes, these four fields contain the four Ohm's law terms.
    * Their formulation is obsolete and should perfectly mirror what is done in ohm_k,
    * using the above region_ID.
    */
    float E_mot [3][len_x_cst+4][len_y_cst+4];
    float E_hal [3][len_x_cst+4][len_y_cst+4];
    float E_amb [3][len_x_cst+4][len_y_cst+4];
    float E_res [3][len_x_cst+4][len_y_cst+4];

    /*! The following two fields are used to contain temporary fields, needed
    * when shifting the fields, smoothing the fields, etc. Rename, reduce to 1.
    */
    float smooth [len_x_cst+4][len_y_cst+4];
    float E_stag [3][len_x_cst+4][len_y_cst+4];

    /*! In 2D, only the y-direction is communicated through MPI. These two buffers
    * are used to send and receive the four outter cell layers to the neighbouring
    * process.
    */
    float buff_send_1d_y [4*(len_x_cst+4)];
    float buff_reci_1d_y [4*(len_x_cst+4)];

    bool smooth_field = false;
    int smooth_idx [2] = { 0 };
    // (i, j, idx_it)
    int save_smooth [3][smooth_patch_save_len_cst] = { 0 };
    int idx_save_smooth = 0;

    #if obstacle_cst
      int bound_pos [int(nb_it_max_cst/nb_it_per_shift_cst)][len_y_cst];
    #endif

  #elif NB_DIM==3
    float E           [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];
    float curl_E      [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];
    float density        [len_x_cst+4][len_y_cst+4][len_z_cst+4];
    float density_b      [len_x_cst+4][len_y_cst+4][len_z_cst+4];
    float counts         [len_x_cst+4][len_y_cst+4][len_z_cst+4];  // Number of macro-particle per node (a float).
    float counts_pla     [len_x_cst+4][len_y_cst+4][len_z_cst+4];  // Must be multiplied by wSW/wCom to get a physical density.
    float Lambda         [len_x_cst+4][len_y_cst+4][len_z_cst+4];  // Number of macro-particle per node (a float).
    float Ji          [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];
    float Ji_b        [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];
    float fluxNum     [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];  // Macro-particle flux.
    float fluxNum_pla [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];  // Must be multiplied by (q*wSW) to get the ion current.
    float Gamma       [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];  // See CAM, Bagdonat thesis.
    float J_tot       [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];  // Proper total current (curl(J)/mu0) .
    float pres           [len_x_cst+4][len_y_cst+4][len_z_cst+4];  // Pressure (physical).
    #if yee_cst
      float E_s       [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];
      float J_tot_s   [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];
    #endif

    int region_ID  [len_x_cst+4][len_y_cst+4][len_z_cst+4]; // Identifies in which region the node is,
                                                            //   i.e. within a solid body, vacuum, nominal, etc.

    float vort  [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];  // vorticity.

    float E_mot [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];
    float E_hal [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];
    float E_amb [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];
    float E_res [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];

    float smooth [len_x_cst+4][len_y_cst+4][len_z_cst+4];
    float E_stag [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];

    float buff_send_1d_y [4*(len_x_cst+4)*(len_z_cst+4)];
    float buff_reci_1d_y [4*(len_x_cst+4)*(len_z_cst+4)];
    float buff_send_1d_z [4*(len_x_cst+4)*(len_y_cst+4)];
    float buff_reci_1d_z [4*(len_x_cst+4)*(len_y_cst+4)];

    bool smooth_field = false;
    int smooth_idx [3] = { 0 };
    // (i, j, k, idx_it)
    int save_smooth [4][nb_it_max_cst] = { 0 };
    int idx_save_smooth = 0;

  #endif
};


//______________________________________________________________________________
/*! \brief Contains the magnetic field, as well as the optional dipole field.
 *
 * The B-field is the only field incremented and followed through time. It gets
 * its own structure.
 *
 * When a permanent dipole is simulated, it is contained in this structure as well.
 * In the solar wind reference fram, in which the dipole moves, its description is
 * updated a few times during each time step, using the additional potenital phi.
 */
struct simu_B_field
{
  #if NB_DIM==2
    float B     [3][len_x_cst+4][len_y_cst+4]; /*!< Magnetic field. */
    #if dipole_cst
      /*! Magnetic scalar potential from the magnetic pole limit. */
      float phi            [len_x_cst+4][len_y_cst+4][5];
      /*! Magnetic permanent dipole, taken as divergence of the above potential. */
      float B_dip       [3][len_x_cst+4][len_y_cst+4];
    #endif
    #if yee_cst
      float B_s       [3][len_x_cst+4][len_y_cst+4];
    #endif
  #elif NB_DIM==3
    float B           [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];
    #if dipole_cst
      float phi          [len_x_cst+4][len_y_cst+4][len_z_cst+4];
      float B_dip     [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];
    #endif
    #if yee_cst
      float B_s       [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];
    #endif
  #endif
};

//______________________________________________________________________________
/*! \brief Contains all particles.
 *
 * This is (nominally) by far the largest data structure, containing active as well as
 * free particles, with the least amount of data for each particle: position,
 * velocity, species, and whether this particle is free or not, i.e. active in the
 * domain or available for representing a new particle to be injected in the domain.
 */
struct particles
{
  float rx     [pool_size_cst]; /*!< x position. */
  float ry     [pool_size_cst]; /*!< y position. */
  #if NB_DIM==3
    float rz   [pool_size_cst]; /*!< z position. */
  #endif
  float vx     [pool_size_cst]; /*!< Speed along x. */
  float vy     [pool_size_cst]; /*!< Speed along y. */
  float vz     [pool_size_cst]; /*!< Speed along z. */
  bool  active [pool_size_cst]; /*!< Active or free. */
  int   ID     [pool_size_cst]; /*!< Species. */
};

//_____________________________
#if DF_cst
struct simu_DF {
  float DF [len_x_cst+4][len_v_cst+4][len_v_cst+4][len_v_cst+4] ;
  // float df_e [xLen+2][vLen+2] ;
  float E [3][len_x_cst+4];
  float B [3][len_x_cst+4];
} ;
#endif


//______________________________________________________________________________
/*! \brief Contains all necessary (?) fields for Step 2 from Step 1, decaying
 *         turbulence, as well as Step-1-particle-related indices.
 *
 * These fields are only stored on the host, since they are only introduced slice
 * by slice in the simulation domain on the devices during Step 2.
 *
 * The index of the current injection slice is stored here in idx_x.
 *
 * Additional informations about particles from Step 1, are also stored here.
 * Particles from Step 1 are ordered according to their x-position, distributed in
 * slices corresponding to the injection in the domain during Step 2, as illustrated
 * in the injection algorithm (cf gallery and technical publication). For this
 * purpose, it is only necessary to save the ordered index of the first particle
 * in each slice, given in indFirstPartSlice, as well as the total number of particles
 * in each slice, given in sliceWidth.
 */
struct simu_tank
{
  #if NB_DIM==2
    float B  [3][len_x_cst+4][len_y_cst+4]; /*!< B-field from Step 1. */
    float E  [3][len_x_cst+4][len_y_cst+4]; /*!< E-field from Step 1. */
    float Ji [3][len_x_cst+4][len_y_cst+4]; /*!< Ion current from Step 1. */
    float density [len_x_cst+4][len_y_cst+4]; /*!< Density from Step 1. */
  #elif NB_DIM==3
    float B  [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];
    float E  [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];
    float Ji [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];
    float density [len_x_cst+4][len_y_cst+4][len_z_cst+4];
  #endif
  /*! The current index of the slice to be added, within [0, len_x_cst/nb_cell_per_shift_cst-1]*/
  int idx_x = 0;
  //
  std::vector<size_t> idx_ordered; /*!< Indices of x-ordered Step 1 particles.*/
  int sliceWidth [int(len_x_cst/nb_cell_per_shift_cst)]; /*!< How many particles in each slice.*/
  int indFirstPartSlice [int(len_x_cst/nb_cell_per_shift_cst)]; /*!< Index of the first particle in each slice.*/
};


//______________________________________________________________________________
/*! \brief Used for the injection of fields and particles during Step 2.
 *
 * Only a slice upstream of the domain is concerned here, including the two guard
 * cells and an additional nb_cell_per_shift_cst. The same goes for the particles
 * to be injected, contained in the same spatial slice, as discussed in simu_tank.
 */
struct injector
{
  #if NB_DIM==2
    float B    [3][2+nb_cell_per_shift_cst][len_y_cst+4]; /*!< B-field slice to be injected. */
    float E    [3][2+nb_cell_per_shift_cst][len_y_cst+4]; /*!< E-field slice to be injected. */
    float Ji   [3][2+nb_cell_per_shift_cst][len_y_cst+4]; /*!< Ji slice to be injected. */
    float density [2+nb_cell_per_shift_cst][len_y_cst+4]; /*!< Density slice to be injected. */
  #elif NB_DIM==3
    float B    [3][2+nb_cell_per_shift_cst][len_y_cst+4][len_z_cst+4];
    float E    [3][2+nb_cell_per_shift_cst][len_y_cst+4][len_z_cst+4];
    float Ji   [3][2+nb_cell_per_shift_cst][len_y_cst+4][len_z_cst+4];
    float density [2+nb_cell_per_shift_cst][len_y_cst+4][len_z_cst+4];
  #endif
  float rx [injec_size_cst]; /*!< x-position of particles to be injected. */
  float ry [injec_size_cst]; /*!< y-position of particles to be injected. */
  #if NB_DIM==3
    float rz [injec_size_cst]; /*!< z-position of particles to be injected. */
  #endif
  float vx [injec_size_cst]; /*!< x-speed of particles to be injected. */
  float vy [injec_size_cst]; /*!< y-speed of particles to be injected. */
  float vz [injec_size_cst]; /*!< z-speed of particles to be injected. */
};


//_____________________________
/*! \brief Particle buffer,
 *         managing particles' MPI communications and injection.
 *
 *  This structure contains all required information when particles need to be
 * either injected or communicated from one MPI process to another.
 */
struct buff_part
{
  /*! An array containing indices of free particles, indices corresponding to
  * the particles' pool object.
  */
  int idx_free [buff_size_cst];
  int next_idx = 0; /*!< Gives a single temporary index of a particle. */
  #if NB_DIM==2
    /*! Send buffer, containing pool indices of particles to be communicated out. */
    float buff_send[6*buff_size_cst];
    /*! Receive buffer, containing pool indices of particles to be communicated in. */
    float buff_rece[6*buff_size_cst];
  #elif NB_DIM==3
    /*! Send buffer, containing pool indices of particles to be communicated out. */
    float buff_send[7*buff_size_cst];
    /*! Receive buffer, containing pool indices of particles to be communicated in. */
    float buff_rece[7*buff_size_cst];
  #endif
  int next_idx_comm = 0; /*!< Temporary index. */
  int nb_part_send = 0; /*!< Number of particles to be sent. */
  int nb_part_rece = 0; /*!< Number of particles to be received. */
  #if NB_DIM==2
    int exosphere[2][nb_part_add_pla_cst]; /*!< Contains [i][j] indices */
    /*! Contains the probability to add one particle around this node (flat index).*/
    float exo_proba[len_x_cst*len_y_cst];
  #elif NB_DIM==3
    int exosphere[3][nb_part_add_pla_cst];  // Contains [i][j][k]
    float exo_proba[len_x_cst*len_y_cst*len_z_cst];
  #endif
};
//_____________________________
/*! \brief Particles and fields house-keeping/moments data structure.
 *
 *  This structure contains various fields and particles global informations,
 * mostly related to their moments (plasma moments and statistical moments).
 * The time resolution of these time series is chosen through the constants
 * len_save_t_cst
 */
struct moments
{
  float E_mag          [len_save_t_cst]; /*!< Magnetic energy, sum(B_i^2). */
  float E_elec         [len_save_t_cst]; /*!< Electric energy. */
  float E_kin          [len_save_t_cst]; /*!< Particles kinetic energy. */
  float B_mean      [3][len_save_t_cst]; /*!< B-field mean over the grid. */
  float B_var       [3][len_save_t_cst]; /*!< B-field variance over the grid. */
  float div_B_mean     [len_save_t_cst]; /*!< Mean of div(B) over the grid. */
  float div_B_var      [len_save_t_cst]; /*!< Variacne of div(B) over the grid. */
  float div_B_max      [len_save_t_cst]; /*!< Maximum of abs(div(B)) over the grid. */
  float dens_tot       [len_save_t_cst]; /*!< Total density, sum(dens_tot). */
  float Ji_mean     [3][len_save_t_cst];
  //
  float vel_part    [3][len_save_t_cst];
  float vel_th_part [3][len_save_t_cst];
  float temp_tot       [len_save_t_cst];
  int nb_part_tot      [len_save_t_cst];
};
struct fields_time_space
{
  float E        [3][len_save_t_cst][len_save_x_cst];
  float B        [3][len_save_t_cst][len_save_x_cst];
  float density     [len_save_t_cst][len_save_x_cst];
};
struct root_mean_sqr
{
  float meansqr;
  float mean;
  float B     [3][len_save_t_cst];
  float E     [3][len_save_t_cst];
  float Ji    [3][len_save_t_cst];
  float J_tot [3][len_save_t_cst];
  float vort  [3][len_save_t_cst];
};
struct trajectory
{
  float rx [len_save_t_cst];
  float ry [len_save_t_cst];
  #if NB_DIM==3
    float rz [len_save_t_cst];
  #endif
  float vx [len_save_t_cst];
  float vy [len_save_t_cst];
  float vz [len_save_t_cst];
};
struct probes
{
  #if NB_DIM==2
    float rx   [nb_probes_cst][nb_it_max_cst];
    float ry   [nb_probes_cst][nb_it_max_cst];
    bool active [nb_probes_cst];
    float B     [nb_probes_cst][3][nb_it_max_cst];
    float E     [nb_probes_cst][3][nb_it_max_cst];
    float density  [nb_probes_cst][nb_it_max_cst];
    float Ji    [nb_probes_cst][3][nb_it_max_cst];
    float J_tot [nb_probes_cst][3][nb_it_max_cst];
  #elif NB_DIM==3
    float rx   [nb_probes_cst][nb_it_max_cst];
    float ry   [nb_probes_cst][nb_it_max_cst];
    float rz   [nb_probes_cst][nb_it_max_cst];
    bool active [nb_probes_cst];
    float B     [nb_probes_cst][3][nb_it_max_cst];
    float E     [nb_probes_cst][3][nb_it_max_cst];
    float density  [nb_probes_cst][nb_it_max_cst];
    float Ji    [nb_probes_cst][3][nb_it_max_cst];
    float J_tot [nb_probes_cst][3][nb_it_max_cst];
  #endif
};
//_____________________________
struct house_keeping
{
  int nbTruePart = 0;  // Number of active particles in the box.
  int nbFreeIndTot = 0;  // Number of free indices summed over all stacks.
  int nbPartSW          [len_save_t_cst];
  int nbPartPla         [len_save_t_cst];
  float simuTime        [len_save_t_cst];
  int runTime           [len_save_t_cst];
  int nb_part_comm_up   [len_save_t_cst];
  int nb_part_comm_down [len_save_t_cst];
  int nb_pla_proba      [len_save_t_cst];
  int nb_patch_smooth   [nb_it_max_cst];
};
struct state_solver
{
  bool break_solver = false;
  int error_ID = 0;
  std::string cuda_error_str     = "cudaSuccess";
  std::string error_str2[20] = {std::string("Error ID 1, cudaMalloc of particles."),
                                 std::string("Error ID 2, cudaMalloc of fields."),
                                 std::string("Error ID 3, invalid B-field value."),
                                 std::string("Error ID 4, invalid E-field value."),
                                 std::string("Error ID 5, invalid J_tot value."),
                                 std::string("Error ID 6, invalid fluxNum value."),
                                 std::string("Error ID 7, invalid density value."),
                                 std::string("Error ID 8, invalid new particle index."),
                                 std::string("Error ID 9, too many patches smoothed."),
                                 std::string("Error ID 10, empty."),
                                 std::string("Error ID 11, empty."),
                                 std::string("Error ID 12, empty."),
                                 std::string("Error ID 13, empty."),
                                 std::string("Error ID 14, empty."),
                                 std::string("Error ID 15, empty."),
                                 std::string("Error ID 16, empty."),
                                 std::string("Error ID 17, empty."),
                                 std::string("Error ID 18, empty."),
                                 std::string("Error ID 19, empty."),
                                 std::string("Error ID 20, empty.") };
};

#endif
