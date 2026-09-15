#ifndef __PARAMETERS_H_INCLUDED__   // if x.h hasn't been included yet...
#define __PARAMETERS_H_INCLUDED__

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>

//______________________________________________________________________________
// Physical constants, SI.
const float e    = 1.602e-19 ;  // Elementary charge. Or so. C.
const float m_i  = 1.673e-27 ;  // Atomic mass unit. Or so. kg.
const float m_e  = 9.109e-31 ;  // Atomic mass unit. Or so. kg.
const float mu0  = 1.257e-6  ;  // Permeability. H/m.
const float eps0 = 8.85e-12  ;  // Vacuum permittivity. E/m.
const float k_B  = 1.3806e-23;  // Boltzmann ocnstant, m2 kg s-2 K-1
const float c    = 299792458.;  // Speed of light, m/s
//
const float PI   = 3.14159265359f;
//______________________________________________________________________________


/* Comet/Mars parameters. */

//______________________________________________________________________________
// Dimensionality & spatial resolutions.
#define NB_DIM 2
//
const int len_x_cst        = 2000;
const int len_y_cst        = 50;
#if NB_DIM==3
  const int len_z_cst      = 999;
#endif
//
const int mpi_nb_proc_y_cst = 40;
const int mpi_nb_proc_z_cst = 1;
//
const int nb_part_node_cst = 4000;
//
const int rate_save_x_cst  = 2;  //   so the dt_low and dx_low are well defined.
const int len_save_x_cst   = int(len_x_cst/rate_save_x_cst);
const int rate_save_y_cst  = 2;  //   so the dt_low and dx_low are well defined.
const int len_save_y_cst   = int(len_y_cst/rate_save_y_cst);
//______________________________________________________________________________

//______________________________________________________________________________
// Time resolutions.
const int nb_it_max_cst           = 20000;
//                                   |
//                                   v
const int rate_save_t_cst         = std::min(100, nb_it_max_cst);  // For simplicity, keep it an integer fraction of nb_it_max_cst and len_x_cst
const int len_save_t_cst          = int(nb_it_max_cst/rate_save_t_cst);
const int rate_save_field_cst     = 400;
const int rate_save_particles_cst = 40000;
//______________________________________________________________________________

//______________________________________________________________________________
// Type of simulation.
#define solve_EB_cst    true // If false, E and B are not solved self-consistently.
#define DF_cst          false // Use a distribution function instead of particles.
#define yee_cst         false // Use a Yee mesh (div(B)=0).
//
#define decay_turb_cst  false // Decaying turbulence run.
#define obstacle_cst    true // Adding an obstacle.
#define ORF_cst         false // Obstacle reference frame used.
#define inject_pla_cst  true // Inject planetary ions.
#define inject_turb_cst false // Inject turbulent upstream solar wind.
#define solid_body_cst  false // Solid body in the domain.
#define dipole_cst      false // Permanent magnetic dipole.
#define ionosphere_cst  false
//
#define periodic_yz_cst true // Periodic domain along y and z directions.
#define restart_cst     true // Re-start the simulation at some iteration.
//______________________________________________________________________________

//______________________________________________________________________________
// Shifting fields and particles.
const int nb_it_per_shift_cst   = 2; // nb_cell_per_shift_cst must be <= nb_it_per_shift_cst
const int nb_cell_per_shift_cst = 1; //
//______________________________________________________________________________

//______________________________________________________________________________
// Memory/variable sizes.
#if NB_DIM==2
  const unsigned int nb_nodes_cst     = len_x_cst*len_y_cst;
  const unsigned int nb_nodes_tot_cst = (len_x_cst+4)*(len_y_cst+4);
  const unsigned int pool_size_cst    = static_cast<unsigned int>(3.05*nb_part_node_cst*len_x_cst*len_y_cst);
  const unsigned int tank_size_cst    = static_cast<unsigned int>(1.1*nb_part_node_cst*len_x_cst*len_y_cst);
  const unsigned int buff_size_cst    = static_cast<unsigned int>(4.*(2*nb_cell_per_shift_cst+.5)*nb_part_node_cst*len_y_cst);
  const unsigned int injec_size_cst   = static_cast<unsigned int>((2*nb_cell_per_shift_cst+.5)*nb_part_node_cst*len_y_cst);
#elif NB_DIM==3
  const unsigned int nb_nodes_cst     = len_x_cst*len_y_cst*len_z_cst;
  const unsigned int nb_nodes_tot_cst = (len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4);
  const unsigned int pool_size_cst    = static_cast<unsigned int>(1.01*nb_part_node_cst*len_x_cst*len_y_cst*len_z_cst);
  const unsigned int tank_size_cst    = static_cast<unsigned int>(1.1*nb_part_node_cst*len_x_cst*len_y_cst*len_z_cst);
  const unsigned int buff_size_cst    = static_cast<unsigned int>((2*nb_cell_per_shift_cst+.5)*nb_part_node_cst*len_y_cst*len_z_cst);
  const unsigned int injec_size_cst   = static_cast<unsigned int>((2*nb_cell_per_shift_cst+.5)*nb_part_node_cst*len_y_cst*len_z_cst);
#endif
//


//______________________________________________________________________________
// Other.
const unsigned int nb_part_add_pla_cst = 80000;
//
const int idx_it_inject_cst = 20000;
const std::string path_inputs_str_cst = "../run_030/products";
//
const int idx_it_restart_cst = 20000;
const std::string path_inputs_restart_cst = "../run_033/products";
//
const int seed_val_cst = 0; //time(NULL);
const int tpb_cst = 256; // threads per block
//
#define debug_mode_cst false
//
const int nb_probes_cst = 160;
//
#define smooth_patch_cst true
// How many patches can be saved: N times the number of iterations. Quick solution.
const int smooth_patch_save_len_cst = 100*nb_it_max_cst;
// How many cell wide are the smoothing patches:
const int len_patch_cst = 20;


//______________________________________________________________________________
// Parameters class.
struct simu_param {

  float dX = .25; // Node spacing, in unit of d_i
  float dt = 0.0125; // time step, in unit of 1/omega_ci

  //____________________________________________________________________________
  // Background physical values in SI.
  float n0_SI  = 3.e6;
  float B0_SI  = 3.e-9;
  float v0     = 0.;
  float E0     = -v0*B0_SI;
  float beta   = 1.;
  float Ti_inf = beta*B0_SI*B0_SI/(n0_SI*k_B*2.*mu0);
  float Te_inf = 1.*Ti_inf;

  //____________________________________________________________________________
  // Important things.
  float e_mot       = 1.; // 0. or 1.!
  float e_hal       = 1.; // 0. or 1.!
  float e_amb       = 1.; // 0. or 1.!
  //
  float poly_ind     = 1.;
  //
  float eta_res_norm = 0.;
  float eta_hyp_res  = 2.5e-4;
  float eta_res_vac  = 0.;
  float eta_res_obs  = 0.;
  //
  float eta_sm       = 0.;
  //
  float fluctu_sqrt  = 0.6;
  //
  float B0_x         = 0.; // If Alfven.
  float B0_y         = 0.;
  float B0_z         = -1.; // If magnetosonic.
  //
  float dens_min    = 0.05;  // Minimum density, normalised.
  //
  float E_max_smooth = 100.; // If used, above this value in any electric field component, E and B are locally smoothed.

  //____________________________________________________________________________
  // Background/initial main plasma parameters:
  float omega_i     = std::sqrt(e*e*n0_SI/(eps0*m_i));
  float omega_ci    = e*B0_SI/m_i;
  float v_A         = B0_SI/std::sqrt(mu0*m_i*n0_SI);
  // Fundamental normalising parameters:
  float t0          = 1/omega_ci;
  float x0          = v_A/omega_ci;
  float q0          = e*n0_SI*x0*x0*x0;
  float m0          = m_i*n0_SI*x0*x0*x0;
  float p0          = B0_SI*B0_SI/(2.*mu0);
  // Additional physical parameters.
  float d_i         = c/omega_i;
  float omega_e     = std::sqrt(e*e*n0_SI/(eps0*m_e));
  float omega_ce    = e*B0_SI/m_e;
  float d_e         = c/omega_e;
  float v_thi       = std::sqrt(2.*k_B*Ti_inf/m_i);
  float v_the       = std::sqrt(2.*k_B*Te_inf/m_e);
  float v_s         = std::sqrt(k_B*(Te_inf+3*Ti_inf)/m_i);
  float v_ms        = std::sqrt(c*c*(v_A*v_A+v_s*v_s)/(v_A*v_A+c*c));
  float lambda_D    = std::sqrt(eps0*k_B*Te_inf/(n0_SI*e*e));
  float Beta_p      = 2*mu0*n0_SI*k_B*Ti_inf/(B0_SI*B0_SI);
  float Beta_e      = 2*mu0*n0_SI*k_B*Te_inf/(B0_SI*B0_SI);
  // Obstacle
  float v_obs       = dX*nb_cell_per_shift_cst/(nb_it_per_shift_cst*dt);  // Normalised. Get rid of v0_com...
  float v0_com      = v_A*dX*nb_cell_per_shift_cst/(nb_it_per_shift_cst*dt);  // m/s
  float R_gyr_norm  = Z_pla*v0_com/v_A;
  float R_gyr       = Z_pla*m_i*v0_com/(e*B0_SI);

  //____________________________________________________________________________
  // Obstacle related.
  // Exosphere:
  float Q           = 1.25e27;  // s-1
  float nu_i        = 2.e-7;  // s-1
  float nu_d        = 5.0e-7;  // s-1
  float u0          = 1000.;  // m/s
  // Ionosphere:
  float prod0       = 1.e6; // s-1 m-3 , maximum production rate, at h0.
  float h0          = 5.e5; // m , height of maximum production rate p0.
  float H           = 2.5e5; // m , Scale height.
  //
  #if inject_pla_cst
  int   Z_pla       = 18;
  #elif ionosphere_cst
  int   Z_pla       = 44;
  #else
  int   Z_pla       = 99;
  #endif
  float centre_x    = 5./8.; //3./8.
  float centre_y    = 1./2.;
  #if NB_DIM==3
    float centre_z  = 1./2.;
  #endif
  float dip_mom [3] = {0., 0., 0.};
  float r_obs_SI    = 3.380e6; // m
  float r_obs       = r_obs_SI/d_i;
  float r_obs_sqr   = std::pow(r_obs, 2);

  //____________________________________________________________________________
  // Copying/setting numerical parameters.
  int   xLen        = len_x_cst;  // Number of nodes along x.
  int   yLen        = len_y_cst;  // Number of nodes along y.
  #if NB_DIM==3
    int   zLen      = len_z_cst;  // Number of nodes along z.
  #endif
  float dX_i        = 1./dX;
  int   nb_dim      = NB_DIM;
  //
  float tMax        = dt*nb_it_max_cst; // MEDIOCRE. 400./omega_ci; // Total time, s
  //
  int   nb_sub_cycles = 51;
  float sub_dt        = dt/nb_sub_cycles;
  float sub_dx        = dX/nb_sub_cycles;
  //
  int   rateSaveIt  = std::max(1, int(.05*tMax/dt));  // Save data every rateSaveIt iterations.
  //
  #if NB_DIM==2
    int nb_nodes        = len_x_cst*len_y_cst;  // Total number of nodes.
    int nb_nodes_tot    = (len_x_cst+4)*(len_y_cst+4);  // Total number of nodes.
    int nb_part_add_sw  = len_y_cst*nb_part_node_cst; //int(nbPartNode*yLen*(-.5*v0*dt)/dX); // Number of particles to add per SEMI time step.
  #elif NB_DIM==3
    int nb_nodes        = len_x_cst*len_y_cst*len_z_cst;  // Total number of nodes.
    int nb_nodes_tot    = (len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4);  // Total number of nodes.
    int nb_part_add_sw  = len_y_cst*len_z_cst*nb_part_node_cst; //int(nbPartNode*yLen*(-.5*v0*dt)/dX); // Number of particles to add per SEMI time step.
  #endif
  //
  int nb_part_add_pla = 0;
  float sum_pla_proba = 0;
  //
  float w_sw        = 1./nb_part_node_cst;
  float w_pla       = 1.; //2.358*nu_i*Q*dt*logf(r_max_insert/r_min_insert)/(2*u0*nb_part_add_com*dX*dX);
  //
  int width_smooth_downstream = std::min(40, int(len_x_cst/4.));




  //
  float delta_E = .04;
  float delta_B = .04;
  float nb_wave = 10.;
  float k_wave = 1.25;
  float lambda_wave = 2.*PI/k_wave;
  float omega_wave = 1.75;
  float period_wave = 2.*PI/omega_wave;

  //____________________________________________________________________________
  // MPI related values.
  int mpi_rank_y    = 0;
  int mpi_rank_z    = 0;
  int mpi_nb_proc_y = mpi_nb_proc_y_cst;
  int mpi_nb_proc_z = mpi_nb_proc_z_cst;
  int mpi_nb_proc_tot = mpi_nb_proc_y_cst*mpi_nb_proc_z_cst;
  // mpi_rank_lin is the flatten rank of the process, mpi_rank_y*mpi_nb_proc_z + mpi_rank_z
  int mpi_rank_lin = 0;
  //
  #if NB_DIM==2
    int len_buff_field_y = int(4*(len_x_cst+4));
  #elif NB_DIM==3
    int len_buff_field_y = int(4*(len_x_cst+4)*(len_z_cst+4));
    int len_buff_field_z = int(4*(len_x_cst+4)*(len_y_cst+4));
  #endif
  //
  int nb_part_tank = 0;






  int init_rank_order(int mpi_rank_lin_input, int mpi_nb_proc_tot_input)
  {
    int error_nb_proc = 0;
    mpi_rank_lin = mpi_rank_lin;
    //
    if (mpi_nb_proc_tot_input!=mpi_nb_proc_tot){
      if (mpi_rank_lin==0){
        std::cout << "| The number of processes given to slurm differs from the number set in the parameters." << std::endl;
      }
      error_nb_proc = 1;
    }
    //
    mpi_rank_y = int(mpi_rank_lin_input/mpi_nb_proc_z);
    // mpi_rank_z is 0 for all in 2D case.
    mpi_rank_z = mpi_rank_lin_input - mpi_rank_y*mpi_nb_proc_z_cst;
    mpi_rank_lin = mpi_rank_y*mpi_nb_proc_z + mpi_rank_z;
    //
    return error_nb_proc;

  }


};



#endif
