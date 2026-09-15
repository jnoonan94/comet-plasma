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


/* Mars-like parameters. */

//______________________________________________________________________________
// Dimensionality & spatial resolutions.
#define NB_DIM 2
//
const int len_x_cst        = 2000;
const int len_y_cst        = 100;
#if NB_DIM==3
  const int len_z_cst      = 100;
#endif
const int nb_part_node_cst = 2000;
//
const int rate_save_x_cst  = 2;  //   so the dt_low and dx_low are well defined.
const int len_save_x_cst   = int(len_x_cst/rate_save_x_cst)+1;
const int rate_save_y_cst  = 2;  //   so the dt_low and dx_low are well defined.
const int len_save_y_cst   = int(len_y_cst/rate_save_y_cst)+1;
//______________________________________________________________________________

//______________________________________________________________________________
// Time resolutions.
const int nb_it_max_cst           = 20000;
//                                   |
//                                   v
const int rate_save_t_cst         = 100;  // For simplicity, keep it an integer fraction of nb_it_max_cst and len_x_cst
const int len_save_t_cst          = int(nb_it_max_cst/rate_save_t_cst)+1;
const int rate_save_field_cst     = 1000;
const int rate_save_particles_cst = 100000;
//______________________________________________________________________________

//______________________________________________________________________________
// Type of simulation.
#define solve_EB_cst    true // If false, E and B are not solved self-consistently.
#define DF_cst          false // Use a distribution function instead of particles.
//
#define decay_turb_cst  false // Decaying turbulence run.
#define obstacle_cst    true // Adding an obstacle.
#define ORF_cst         false // Obstacle reference frame used.
#define inject_pla_cst  false // Inject planetary ions.
#define inject_turb_cst true // Inject turbulent upstream solar wind.
#define solid_body_cst  true // Solid body in the domain.
#define dipole_cst      false // Permanent magnetic dipole.
#define ionosphere_cst  true
//
#define periodic_yz_cst false
//______________________________________________________________________________

//______________________________________________________________________________
// Shifting fields and particles, if not orf_cst.
const int nb_it_per_shift_cst   = 2;
const int nb_cell_per_shift_cst = 2;
//______________________________________________________________________________

//______________________________________________________________________________
// Memory/variable sizes.
#if NB_DIM==2
  const unsigned int nb_nodes_cst     = len_x_cst*len_y_cst;
  const unsigned int nb_nodes_tot_cst = (len_x_cst+4)*(len_y_cst+4);
  const unsigned int pool_size_cst    = static_cast<unsigned int>(2.8*nb_part_node_cst*len_x_cst*len_y_cst);
  const unsigned int tank_size_cst    = static_cast<unsigned int>(1.2*nb_part_node_cst*len_x_cst*len_y_cst);
  const unsigned int buff_size_cst    = static_cast<unsigned int>(5.*(2*nb_cell_per_shift_cst+.5)*nb_part_node_cst*len_y_cst);
  const unsigned int injec_size_cst   = static_cast<unsigned int>((2.2*nb_cell_per_shift_cst+.5)*nb_part_node_cst*len_y_cst);
#elif NB_DIM==3
  const unsigned int nb_nodes_cst     = len_x_cst*len_y_cst*len_z_cst;
  const unsigned int nb_nodes_tot_cst = (len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4);
  const unsigned int pool_size_cst    = static_cast<unsigned int>(1.2*nb_part_node_cst*len_x_cst*len_y_cst*len_z_cst);
  const unsigned int tank_size_cst    = static_cast<unsigned int>(1.2*nb_part_node_cst*len_x_cst*len_y_cst*len_z_cst);
  const unsigned int buff_size_cst    = static_cast<unsigned int>((2*nb_cell_per_shift_cst+.5)*nb_part_node_cst*len_y_cst*len_z_cst);
  const unsigned int injec_size_cst   = static_cast<unsigned int>((2*nb_cell_per_shift_cst+.5)*nb_part_node_cst*len_y_cst*len_z_cst);
#endif
//


//______________________________________________________________________________
// Other.
const unsigned int nb_part_add_pla_cst = 200000;
//
const int idx_it_inject_cst = 20000;
//
const int seed_val_cst = 0; //time(NULL);
const int tpb_cst = 256; // threads per block
//
// std::string save_products_str_cst = "products_run_005";
const bool debug_mode_cst = false;

const int nb_probes_x_cst = 20;
const int nb_probes_y_cst = 4;

// How many patches can be saved: N times the number of iterations. Quick solution.
const int smooth_patch_save_len_cst = 100*nb_it_max_cst;


//______________________________________________________________________________
// Parameters class.
struct simu_param {

  float dX = .25; // Node spacing, in unit of d_i
  float dt = .025; // time step, in unit of 1/omega_ci

  //____________________________________________________________________________
  // Background physical values in SI.
  float n0_no_norm  = 3.e6;
  float B0_no_norm  = 3.e-9;
  //
  float B0_x        = 0.; // If Alfven.
  float B0_y        = 0.;
  float B0_z        = -1.; // If magnetosonic.
  //
  float beta        = 1.;
  float Ti_inf      = beta*B0_no_norm*B0_no_norm/(n0_no_norm*k_B*2.*mu0);
  float Te_inf      = 1.*Ti_inf;

  //____________________________________________________________________________
  // Important things.
  float e_mot       = 1.;
  float e_hal       = 1.;
  float e_amb       = 1.;
  //
  float poly_ind     = 1.; //1.;
  //
  float eta_res_norm = 0.; //5.e-4; //.00375; //eta_res*q*n0_no_norm/B0_no_norm;
  float eta_hyp_res  = 1.e-3; //5.e-5; //2.e-4; //2.e-4; //.00375; //eta_res*q*n0_no_norm/B0_no_norm;
  float eta_res_vac  = 1.e-2;
  float eta_res_obs  = 3.e-2;//1.e7*e*n0_no_norm/B0_no_norm;
  //
  float eta_sm       = 0.;
  //
  float fluctu_sqrt = 0.5;

  //____________________________________________________________________________
  // Background/initial main plasma parameters:
  float omega_i     = std::sqrt(e*e*n0_no_norm/(eps0*m_i));
  float omega_ci    = e*B0_no_norm/m_i;
  float v_A         = B0_no_norm/std::sqrt(mu0*m_i*n0_no_norm);
  // Fundamental normalising parameters:
  float t0          = 1/omega_ci;
  float x0          = v_A/omega_ci;
  float q0          = e*n0_no_norm*x0*x0*x0;
  float m0          = m_i*n0_no_norm*x0*x0*x0;
  float p0          = B0_no_norm*B0_no_norm/(2.*mu0);
  // Additional physical parameters.
  float d_i         = c/omega_i;
  float omega_e     = std::sqrt(e*e*n0_no_norm/(eps0*m_e));
  float omega_ce    = e*B0_no_norm/m_e;
  float d_e         = c/omega_e;
  float v_thi       = std::sqrt(2.*k_B*Ti_inf/m_i);
  float v_the       = std::sqrt(2.*k_B*Te_inf/m_e);
  float v_s         = std::sqrt(k_B*(Te_inf+3*Ti_inf)/m_i);
  float v_ms        = std::sqrt(c*c*(v_A*v_A+v_s*v_s)/(v_A*v_A+c*c));
  float lambda_D    = std::sqrt(eps0*k_B*Te_inf/(n0_no_norm*e*e));
  float Beta_p      = 2*mu0*n0_no_norm*k_B*Ti_inf/(B0_no_norm*B0_no_norm);
  float Beta_e      = 2*mu0*n0_no_norm*k_B*Te_inf/(B0_no_norm*B0_no_norm);



  //____________________________________________________________________________
  // Obstacle related.
  // Exosphere:
  float Q           = 5.e26;  // s-1 , neutral outgassing rate.
  float nu_i        = 2.e-7;  // s-1 , ionisation rate.
  float nu_d        = 5.0e-7;  // s-1 , destruction rate.
  float u0          = 1000.;  // m/s , radial neutral speed.
  // Ionosphere:
  float prod0       = 1.e6;//1.e4; // s-1 m-3 , maximum production rate, at h0.
  float h0          = 5.e5; // m , height of maximum production rate p0.
  float H           = 2.5e5; // m , Scale height.
  //
  int   Z_pla       = 44;
  float centre_x    = 5./8.;
  float centre_y    = 1./2.;
  #if NB_DIM==3
    float centre_z  = 1./2.;
  #endif
  float dip_mom [3] = {0., -1.e14, 0.};
  float r_obs_SI    = 3.380e6; // m
  float r_obs       = r_obs_SI/d_i;
  float r_obs_sqr   = std::pow(r_obs, 2);
  float v_obs       = dX*nb_cell_per_shift_cst/(nb_it_per_shift_cst*dt);  // Normalised.
  float R_gyr_norm  = Z_pla*v_obs;
  float R_gyr       = Z_pla*m_i*v_obs*v_A/(e*B0_no_norm);
  #if ORF_cst
    float E0 = v_obs;
  #else
    float E0 = 0.;
  #endif
  //
  float delta_x_inj_ORF = v_obs*dt; // If ORF, the gap to fill with particles every iteration.
  #if NB_DIM==2
    int nb_part_add_ORF = int(nb_part_node_cst*len_y_cst*delta_x_inj_ORF/dX);
  #elif NB_DIM==3
    int nb_part_add_ORF = int(nb_part_node_cst*len_y_cst*len_z_cst*delta_x_inj_ORF/dX);
  #endif

  //____________________________________________________________________________
  // Copying/setting numerical parameters.
  int   xLen        = len_x_cst;  // Number of nodes along x.
  int   yLen        = len_y_cst;  // Number of nodes along y.
  #if NB_DIM==3
    int   zLen        = len_z_cst;  // Number of nodes along z.
  #endif
  float dX_i        = 1./dX;
  int   nb_dim      = NB_DIM;
  //
  float tMax        = dt*nb_it_max_cst; // MEDIOCRE. 400./omega_ci; // Total time, s
  //
  int   nb_sub_cycles = 11;
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
  float dens_min    = 0.05;  // Minimum density, normalised.
  //
  float delta_E = 0.;
  float delta_B = 0.;
  float nb_wave = 3.;
  float lambda_wave = xLen*dX/nb_wave;
  float k_wave = 2.*PI/lambda_wave;
  float omega_wave = k_wave*v_s/v_A;
  float period_wave = 2.*PI/omega_wave;

  //____________________________________________________________________________
  // MPI related values.
  int mpi_rank    = 0;
  int mpi_nb_proc = 0;
  //
  #if NB_DIM==2
    int len_buff_field = int(4*(len_x_cst+4));
  #elif NB_DIM==3
    int len_buff_field = int(4*(len_x_cst+4)*(len_z_cst+4));
  #endif
  //
  int nb_part_tank = 0;




  void printout_parameters(){

    printf("\n._____________________________________________________________________________________\n|\n");
    printf("|                Dimensionality: %iD\n|\n", NB_DIM);
    printf("|       Proton plasma frequency: %9.4f 1/s\n", omega_i);
    printf("|          Proton gyrofrequency: %9.4f 1/s\n", omega_ci);
    printf("|\n");
    printf("|   Proton inertial length d_i : %9.4f km\n", d_i*1e-3);
    printf("| Electron inertial length d_e : %9.4f km\n", d_e*1e-3);
    printf("|        Debye length lambda_D : %9.4f km\n", lambda_D*1e-3);
    printf("|           Gyro-radius (v_th) : %9.4f \n", v_thi/v_A);
    printf("|\n");
    printf("|           Proton temperature : %9.4f K \n", Ti_inf);
    printf("|         Electron temperature : %9.4f K \n", Te_inf);
    printf("|\n");
    printf("|                 Alfven speed : %9.4f km/s\n", v_A*1e-3);
    printf("|                  Sound speed : %9.4f km/s\n", v_s*1e-3);
    printf("|           Magnetosonic speed : %9.4f km/s\n", v_ms*1e-3);
    printf("|            Ion thermal speed : %9.4f km/s\n", v_thi*1e-3);
    printf("|       Electron thermal speed : %9.4f km/s\n", v_the*1e-3);
    printf("|\n");
    printf("|                       Beta_p : %9.4f \n", Beta_p);
    printf("|                       Beta_e : %9.4f \n", Beta_e);
    printf("|\n");
    printf("|                      eta_res : %9.4f \n", eta_res_norm);
    printf("|              eta_res (ohm.m) : %9.4f \n", eta_res_norm*B0_no_norm/(e*n0_no_norm) );
    printf("|                  eta_hyp_res : %9.4f \n", eta_hyp_res);
    printf("|                  eta_res_vac : %9.4f \n", eta_res_vac);
    printf("|          eta_res_vac (ohm.m) : %9.4f \n", eta_res_vac*B0_no_norm/(e*n0_no_norm));
    printf("|                  eta_res_obs : %9.4f \n", eta_res_obs);
    printf("|          eta_res_obs (ohm.m) : %9.4f \n", eta_res_obs*B0_no_norm/(e*n0_no_norm));
    if (decay_turb_cst){
      printf("|\n");
      printf("| Decaying turbulence run, fluctuations sqr %.2f \n", fluctu_sqrt);
    }
    if (obstacle_cst){
      printf("|\n");
      printf("| Obstacle simulation. \n" );
      printf("|         Obstacle speed (km/s): %9.4f km/s \n", v_obs*v_A*1e-3);
      printf("|          Obstacle speed (v_A): %9.4f  \n", v_obs);
      if (solid_body_cst){
        printf("|         Obstacle radius (d_i): %9.4f  \n", r_obs);
        printf("|          Obstacle radius (km): %9.4f  \n", r_obs_SI*1e-3);
      }
    }
    if (inject_pla_cst){
      printf("| Cometary simulation. \n" );
      printf("|              Gyro-radius (km): %9.4f \n", Z_pla*m_i*v_obs*v_A/(e*B0_no_norm) );
      printf("|             Gyro-radius (d_i): %9.4f \n", Z_pla*v_obs);
    }
    if (ionosphere_cst){
      printf("| Ionosphere simulation. \n" );
    }
    printf("|\n|\n");
    printf("| Particle-per-node:             %i \n", nb_part_node_cst);
    printf("| Particle weight spec 0 (sw):   %.2e\n", w_sw);
    if (inject_pla_cst || ionosphere_cst){
      printf("| Particle weight spec 1 (pla):  %.2e\n", w_pla);
    }
    printf("|\n");
    printf("| Particle pool size:            %i\n", pool_size_cst);
    printf("| Particle buffer size:          %i\n", buff_size_cst);
    if (inject_turb_cst){
      printf("| Turbulent injection. \n" );
      printf("| Tank size:                   %i\n", tank_size_cst);
      printf("| Injector size:               %i\n", injec_size_cst);
    }
    printf("|\n");
    printf("| Box length (%i x %i nodes):    %.4e (%.4f proton inertial lengths d_i)\n", len_x_cst, len_y_cst, len_x_cst*dX, len_x_cst*dX);
    printf("|\n");
    printf("| Node spacing: %.3e d_i, %.4f km.\n", dX, dX*d_i);
    printf("| dx >> %.1e (d_e) \n", d_e/d_i);
    printf("|\n");
    printf("| Time step: %.3e (%.4f gyroperiod).\n", dt, dt/(2.*PI));
    printf("| dt < %.1e (CFL) \n", 1/(sqrt(nb_dim)*PI) * (dX*dX) ) ;
    printf("| dt < %.1e (No cell jump at v_thi) \n", dX/(v_thi/v_A));
    printf("| dt < %.1e (No cell jump at v_s) \n", dX/(v_s/v_A));
    printf("| dt < %.1e (No cell jump at v_A) \n", dX);
    printf("| dt < %.1e (vacuum resistivity) \n", dX*dX/(2*eta_res_vac));
    if (ORF_cst){
      printf("| dt < %.1e (No cell jump) \n", dX/v_obs);
    }
    if (solid_body_cst){
      printf("| dt < %.1e (obstacle resistivity) \n", dX*dX/(2*eta_res_obs));
    }
    printf("|\n|_____________________________________________________________________________________\n\n\n");

  }

  void output_parameters(){
    if (mpi_rank==0){
      std::string file_name = "products/parameters.txt"; // + std::to_string(indIt) + "_rank" + std::to_string(sP->mpi_rank) + ".txt";
      std::ofstream thefile;
      thefile.open (file_name);
      thefile << "NB_DIM                " << NB_DIM                 << "\n";
      thefile << "seed_val_cst          " << seed_val_cst           << "\n";
      thefile << "tpb_cst               " << tpb_cst                << "\n";
      thefile << "nb_it_max_cst         " << nb_it_max_cst          << "\n";
      thefile << "rate_save_t_cst       " << rate_save_t_cst        << "\n";
      thefile << "len_save_t_cst        " << len_save_t_cst         << "\n";
      thefile << "len_x_cst             " << len_x_cst              << "\n";
      thefile << "len_y_cst             " << len_y_cst              << "\n";
      #if NB_DIM==3
      thefile << "len_z_cst             " << len_z_cst              << "\n";
      #endif
      thefile << "rate_save_x_cst       " << rate_save_x_cst        << "\n";
      thefile << "len_save_x_cst        " << len_save_x_cst         << "\n";
      thefile << "rate_save_y_cst       " << rate_save_y_cst        << "\n";
      thefile << "len_save_y_cst        " << len_save_y_cst         << "\n";
      thefile << "nb_probes_x_cst       " << nb_probes_x_cst        << "\n";
      thefile << "nb_probes_y_cst       " << nb_probes_y_cst        << "\n";
      thefile << "len_save_y_cst        " << len_save_y_cst         << "\n";
      thefile << "nb_nodes_cst          " << nb_nodes_cst           << "\n";
      thefile << "nb_nodes_tot_cst      " << nb_nodes_tot_cst       << "\n";
      thefile << "nb_part_node_cst      " << nb_part_node_cst       << "\n";
      thefile << "pool_size_cst         " << pool_size_cst          << "\n";
      thefile << "tank_size_cst         " << tank_size_cst          << "\n";
      thefile << "buff_size_cst         " << buff_size_cst          << "\n";
      thefile << "injec_size_cst        " << injec_size_cst         << "\n";
      thefile << "nb_part_add_pla_cst   " << nb_part_add_pla_cst    << "\n";
      thefile << "nb_probes_x_cst       " << nb_probes_x_cst        << "\n";
      thefile << "nb_probes_y_cst       " << nb_probes_y_cst        << "\n";
      #if NB_DIM==3
        thefile << "nb_probes_z_cst       " << nb_probes_z_cst        << "\n";
      #endif
      thefile << "smooth_patch_save_len_cst " << smooth_patch_save_len_cst << "\n";
      thefile << "decay_turb_cst        " << decay_turb_cst         << "\n";
      thefile << "obstacle_cst          " << obstacle_cst           << "\n";
      thefile << "ORF_cst               " << ORF_cst                << "\n";
      thefile << "inject_pla_cst        " << inject_pla_cst         << "\n";
      thefile << "inject_turb_cst       " << inject_turb_cst        << "\n";
      thefile << "solid_body_cst        " << solid_body_cst         << "\n";
      thefile << "dipole_cst            " << dipole_cst             << "\n";
      thefile << "ionosphere_cst        " << ionosphere_cst         << "\n";
      thefile << "idx_it_inject_cst     " << idx_it_inject_cst      << "\n";
      thefile << "nb_it_per_shift_cst   " << nb_it_per_shift_cst    << "\n";
      thefile << "nb_cell_per_shift_cst " << nb_cell_per_shift_cst  << "\n";
      thefile << "n0_no_norm            " << n0_no_norm             << "\n";
      thefile << "B0_no_norm            " << B0_no_norm             << "\n";
      // thefile << "v0                    " << v0                     << "\n";
      thefile << "E0                    " << E0                     << "\n";
      thefile << "Ti_inf                " << Ti_inf                 << "\n";
      thefile << "Te_inf                " << Te_inf                 << "\n";
      thefile << "e_mot                 " << e_mot                  << "\n";
      thefile << "e_hal                 " << e_hal                  << "\n";
      thefile << "e_amb                 " << e_amb                  << "\n";
      thefile << "poly_ind              " << poly_ind               << "\n";
      thefile << "eta_res_norm          " << eta_res_norm           << "\n";
      thefile << "eta_sm                " << eta_sm                 << "\n";
      thefile << "omega_i               " << omega_i                << "\n";
      thefile << "omega_ci              " << omega_ci               << "\n";
      thefile << "v_A                   " << v_A                    << "\n";
      thefile << "t0                    " << t0                     << "\n";
      thefile << "x0                    " << x0                     << "\n";
      thefile << "q0                    " << q0                     << "\n";
      thefile << "m0                    " << m0                     << "\n";
      thefile << "p0                    " << p0                     << "\n";
      thefile << "B0_x                  " << B0_x                   << "\n";
      thefile << "B0_y                  " << B0_y                   << "\n";
      thefile << "B0_z                  " << B0_z                   << "\n";
      thefile << "Q                     " << Q                      << "\n";
      thefile << "nu_i                  " << nu_i                   << "\n";
      thefile << "nu_d                  " << nu_d                   << "\n";
      thefile << "u0                    " << u0                     << "\n";
      thefile << "r_obs                 " << r_obs                  << "\n";
      thefile << "v_obs                 " << v_obs                  << "\n";
      thefile << "centre_x              " << centre_x               << "\n";
      thefile << "centre_y              " << centre_y               << "\n";
      #if NB_DIM==3
      thefile << "centre_z              " << centre_z               << "\n";
      #endif
      thefile << "R_gyr_norm            " << R_gyr_norm             << "\n";
      thefile << "R_gyr                 " << R_gyr                  << "\n";
      thefile << "fluctu_sqrt           " << fluctu_sqrt            << "\n";
      thefile << "d_i                   " << d_i                    << "\n";
      thefile << "omega_e               " << omega_e                << "\n";
      thefile << "omega_ce              " << omega_ce               << "\n";
      thefile << "d_e                   " << d_e                    << "\n";
      thefile << "v_thi                 " << v_thi                  << "\n";
      thefile << "v_the                 " << v_the                  << "\n";
      thefile << "v_s                   " << v_s                    << "\n";
      thefile << "v_ms                  " << v_ms                   << "\n";
      thefile << "lambda_D              " << lambda_D               << "\n";
      thefile << "Beta_p                " << Beta_p                 << "\n";
      thefile << "Beta_e                " << Beta_e                 << "\n";
      thefile << "xLen                  " << xLen                   << "\n";
      thefile << "yLen                  " << yLen                   << "\n";
      thefile << "dX                    " << dX                     << "\n";
      thefile << "dX_i                  " << dX_i                   << "\n";
      thefile << "nb_dim                " << nb_dim                 << "\n";
      thefile << "dt                    " << dt                     << "\n";
      thefile << "tMax                  " << tMax                   << "\n";
      thefile << "nb_wave       " << nb_wave        << "\n";
      thefile << "lambda_wave    " << lambda_wave     << "\n";
      thefile << "k_wave         " << k_wave          << "\n";
      thefile << "omega_wave             " << omega_wave              << "\n";
      thefile << "period_wave            " << period_wave             << "\n";
      thefile << "nb_sub_cycles         " << nb_sub_cycles          << "\n";
      thefile << "sub_dt                " << sub_dt                 << "\n";
      thefile << "rateSaveIt            " << rateSaveIt             << "\n";
      thefile << "nb_nodes              " << nb_nodes               << "\n";
      thefile << "nb_nodes_tot          " << nb_nodes_tot           << "\n";
      thefile << "nb_part_add_sw        " << nb_part_add_sw         << "\n";
      thefile << "nb_part_add_pla       " << nb_part_add_pla        << "\n";
      thefile << "sum_pla_proba         " << sum_pla_proba          << "\n";
      // thefile << "pla_proba_max       " << pla_proba_max        << "\n";
      thefile << "w_sw                  " << w_sw                   << "\n";
      thefile << "w_pla                 " << w_pla                  << "\n";
      thefile << "Z_pla                 " << Z_pla                  << "\n";
      thefile << "dens_min              " << dens_min               << "\n";
      thefile << "mpi_rank              " << mpi_rank               << "\n";
      thefile << "mpi_nb_proc           " << mpi_nb_proc            << "\n";
      thefile << "len_buff_field        " << len_buff_field         << "\n";
      thefile << "nb_part_tank          " << nb_part_tank           << "\n";

      thefile.close();
    }
  }


};



#endif
