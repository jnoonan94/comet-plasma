#ifndef __PARAMETERS_H_INCLUDED__   // if x.h hasn't been included yet...
#define __PARAMETERS_H_INCLUDED__

#include <stdio.h>
#include <iostream>
#include <fstream>

const float e    = 1.602e-19 ;  // Elementary charge. Or so. C.
const float m_i  = 1.673e-27 ;  // Atomic mass unit. Or so. kg.
const float m_e  = 9.109e-31  ;  // Atomic mass unit. Or so. kg.
const float mu0  = 1.257e-6 ;  // Permeability. H/m.
const float eps0 = 8.85e-12 ;  // Vacuum permittivity. E/m.
const float k_B  = 1.3806e-23 ;  // Boltzmann ocnstant, m2 kg s-2 K-1
const float c    = 299792458. ;  // Speed of light, m/s

const float PI   = 3.14159265359f ;

const int seed_val_cst = 0 ; //time(NULL) ;
const int tpb_cst = 256 ; // threads per block

const int nb_it_max_cst   = 4000 ;
const int rate_save_t_cst = std::min(10, nb_it_max_cst) ;  // For simplicity, keep it an integer fraction of nb_it_max_cst and len_x_cst
const int len_save_t_cst  = int(nb_it_max_cst/rate_save_t_cst) ;

const int len_x_cst = 128 ;
const int len_y_cst = 8 ;
const int nb_part_node_cst = 32768 ; //16384 ;

const int rate_save_x_cst  = 1 ;  //   so the dt_low and dx_low are well defined.
const int len_save_x_cst   = int(len_x_cst/rate_save_x_cst) ;
const int rate_save_y_cst  = 1 ;  //   so the dt_low and dx_low are well defined.
const int len_save_y_cst   = int(len_y_cst/rate_save_y_cst) ;
const int nb_nodes_cst     = len_x_cst*len_y_cst ;
const int nb_nodes_tot_cst = (len_x_cst+4)*(len_y_cst+4) ;
const int  pool_size_cst = int(1.1*nb_part_node_cst*len_x_cst*len_y_cst) ;
const int  tank_size_cst = int(1.1*nb_part_node_cst*len_x_cst*len_y_cst) ;
const int  buff_size_cst = int(1.5*nb_part_node_cst*len_y_cst) ;
const int injec_size_cst = int(1.5*nb_part_node_cst*len_y_cst) ;
// const int buff_field_size_cst = int(4*(len_x_cst+4))
const int nb_part_add_pla_cst = 4000 ;

const bool decay_turb_cst    = false ;
const bool obstacle_cst      = false ;
const bool inject_pla_cst    = false ;
const bool inject_turb_cst   = false ;

const int  idx_it_inject_cst = 1000 ;

const int nb_it_per_cell_cst = 3 ;



struct simu_param {

  //____________________________________________________________________________
  // Background physical values in SI.
  float n0_no_norm  = 5.e6 ;
  float B0_no_norm  = 1.8e-9 ;
  float v0          = 0. ;
  float E0          = -v0*B0_no_norm ;
  float Ti_inf      = 1.5e4 ;
  float Te_inf      = 1.e5 ;

  float e_mot       = 1. ;
  float e_hal       = 1. ;
  float e_amb       = 1. ;
  //
  float poly_ind     = 1. ; //1. ;
  float eta_res_norm = 0. ; //.00375 ; //eta_res*q*n0_no_norm/B0_no_norm ;
  float eta_sm       = 0. ;

  //____________________________________________________________________________
  // Background/initial main plasma parameters:
  float omega_i     = std::sqrt(e*e*n0_no_norm/(eps0*m_i)) ;
  float omega_ci    = e*B0_no_norm/m_i ;
  float v_A         = B0_no_norm/std::sqrt(mu0*m_i*n0_no_norm) ;
  // Fundamental normalising parameters:
  float t0          = 1/omega_ci ;
  float x0          = v_A/omega_ci ;
  float q0          = e*n0_no_norm*x0*x0*x0 ;
  float m0          = m_i*n0_no_norm*x0*x0*x0 ;
  float p0          = B0_no_norm*B0_no_norm/(2.*mu0) ;

  float B0_x        = 1. ; // If Alfven.
  float B0_y        = 0. ;
  float B0_z        = 0. ; // If magnetosonic.

  // float p_e0        = n0_no_norm*k_B*Te_inf ;

  float Q           = 6.e26 ;  // s-1
  float nu_i        = 2.e-7 ;  // s-1
  float nu_d        = 5.0e-7 ;  // s-1
  float u0          = 700.    ;  // m/s

  float fluctu_sqrt = .4 ;

  float d_i         = c/omega_i ;
  float omega_e     = std::sqrt(e*e*n0_no_norm/(eps0*m_e)) ;
  float omega_ce    = e*B0_no_norm/m_e ;
  float d_e         = c/omega_e ;
  float v_thi       = std::sqrt(2.*k_B*Ti_inf/m_i) ;
  float v_the       = std::sqrt(2.*k_B*Te_inf/m_e) ;
  float v_s         = std::sqrt(k_B*(Te_inf+3*Ti_inf)/m_i) ;
  float v_ms        = std::sqrt(c*c*(v_A*v_A+v_s*v_s)/(v_A*v_A+c*c)) ;
  float lambda_D    = std::sqrt(eps0*k_B*Te_inf/(n0_no_norm*e*e)) ;

  float Beta_p      = 2*mu0*n0_no_norm*k_B*Ti_inf/(B0_no_norm*B0_no_norm) ;
  float Beta_e      = 2*mu0*n0_no_norm*k_B*Te_inf/(B0_no_norm*B0_no_norm) ;

  int   xLen        = len_x_cst ;  // Number of nodes along x.
  int   yLen        = len_y_cst ;  // Number of nodes along y.
  float dX          = .1 ; // Node spacing, in unit of d_i
  float dX_i        = 1./dX ;
  int   nb_dim      = 2 ;

  // int   nbPartNode  = nb_part_node_cst ;  // Number of particles per node.

  float dt          = 0.002 ;  // time step, in unit of 1/omega_ci
  float tMax        = dt*nb_it_max_cst ; // MEDIOCRE. 400./omega_ci ; // Total time, s

  float v0_com      = v_A*dX/(nb_it_per_cell_cst*dt) ;  // m/s

  float nb_oscillations = 5. ;
  float lambda_oscillation = xLen*dX/nb_oscillations ;
  float k_oscillation = 2.*PI/lambda_oscillation ;
  float omega_iaw = k_oscillation*v_s/v_A ;
  float period_iaw = 2.*PI/omega_iaw ;

  int   nbSubCycles = 11 ;
  float subDt       = dt/nbSubCycles ;

  int   rateSaveIt  = std::max(1, int(.05*tMax/dt)) ;  // Save data every rateSaveIt iterations.
  // int   nbItMax     = int(tMax/dt) ;  // Maximum number of iterations.

  int nb_nodes        = len_x_cst*len_y_cst ;  // Total number of nodes.
  int nb_nodes_tot    = (len_x_cst+4)*(len_y_cst+4) ;  // Total number of nodes.
  int nb_part_add_sw  = len_y_cst*nb_part_node_cst ; //int(nbPartNode*yLen*(-.5*v0*dt)/dX) ; // Number of particles to add per SEMI time step.
  int nb_part_add_pla = 0 ;
  float sum_pla_proba = 0 ;
  // int pla_proba_max   = 0 ;
  // int nb_part_mean    = int(xLen*yLen*nb_part_node_cst) ;   // Maximum number of particles in memory.
  // int nb_part_max     = nb_part_mean + nb_part_add_sw ; //int(1.01*(nbPartMean+2*nbPartAdd)) ; //int(std::floor(1.01*nbPartMean)) ;

  float w_sw        = 1./nb_part_node_cst ;
  float w_pla       = 1e-8 ; //2.358*nu_i*Q*dt*logf(r_max_insert/r_min_insert)/(2*u0*nb_part_add_com*dX*dX) ;
  int Z_pla         = 18 ;

  float dens_min    = 0.05 ;  // Minimum density, normalised.

  int mpi_rank    = 0 ;
  int mpi_nb_proc = 0 ;

  int len_buff_field = int(4*(len_x_cst+4)) ;

  int nb_part_tank = 0 ;


  void printout_parameters(){

    printf("\n._____________________________________________________________________________________\n|\n") ;
    printf("|       Proton plasma frequency: %9.4f 1/s\n", omega_i) ;
    printf("|          Proton gyrofrequency: %9.4f 1/s\n", omega_ci) ;
    printf("|\n") ;
    printf("|   Proton inertial length d_i : %9.4f km\n", d_i*1e-3) ;
    printf("| Electron inertial length d_e : %9.4f km\n", d_e*1e-3) ;
    printf("|        Debye length lambda_D : %9.4f km\n", lambda_D*1e-3) ;
    printf("|           Gyro-radius (v_th) : %9.4f \n", v_thi/v_A) ;
    printf("|\n") ;
    printf("|                 Alfven speed : %9.4f km/s\n", v_A*1e-3) ;
    printf("|                  Sound speed : %9.4f km/s\n", v_s*1e-3) ;
    printf("|           Magnetosonic speed : %9.4f km/s\n", v_ms*1e-3) ;
    printf("|            Ion thermal speed : %9.4f km/s\n", v_thi*1e-3) ;
    printf("|       Electron thermal speed : %9.4f km/s\n", v_the*1e-3) ;
    printf("|\n") ;
    printf("|                       Beta_p : %9.4f \n", Beta_p) ;
    printf("|                       Beta_e : %9.4f \n", Beta_e) ;
    if (obstacle_cst){
      printf("|\n") ;
      printf("|               Obstacle speed : %9.4f m/s \n", v0_com) ;
      printf("|      Gyro-radius, normalised : %9.4f \n", Z_pla*v0_com/v_A) ;
    }
    printf("|\n|\n") ;
    printf("| Particle-per-node:             %i \n", nb_part_node_cst) ;
    printf("| Particle weight spec 0 (sw):   %.2e\n", w_sw) ;
    if (inject_pla_cst){
      printf("| Particle weight spec 1 (pla):  %.2e\n", w_pla) ;
    }
    printf("|\n") ;
    printf("| Particle pool size:            %i\n", pool_size_cst) ;
    printf("| Particle buffer size:          %i\n", buff_size_cst) ;
    if (inject_turb_cst){
      printf("| Tank size:                   %i\n", tank_size_cst) ;
      printf("| Injector size:               %i\n", injec_size_cst) ;
    }
    printf("|\n") ;
    printf("| Box length (%i x %i nodes):    %.4e (%.4f proton inertial lengths d_i)\n", len_x_cst, len_y_cst, len_x_cst*dX, len_x_cst*dX) ;
    printf("|\n") ;
    printf("| Number of iterations for 1 IAW: %9.4f\n", period_iaw/dt) ;
    printf("|\n") ;
    printf("| Node spacing: %.3e (%.4f d_i).\n", dX, dX) ;
    printf("| dx >> %.1e (d_e) \n", d_e/d_i) ;
    printf("|\n") ;
    printf("| Time step: %.3e (%.4f gyroperiod).\n", dt, dt/(2.*PI)) ;
    printf("| dt < %.1e (CFL) \n", 1/(sqrt(nb_dim)*PI) * (dX*dX) )  ;
    printf("| dt < %.1e (No cell jump at v_thi) \n", dX/(v_thi/v_A)) ;
    printf("| dt < %.1e (No cell jump at v_s) \n", dX/(v_s/v_A)) ;
    printf("| dt < %.1e (No cell jump at v_A) \n", dX) ;
    if (v0!=0.){
      printf("| dt < %.1e (No cell jump) \n", dX/(v0/v_A)) ;
    }
    printf("|\n|_____________________________________________________________________________________\n\n\n") ;

  }

  void output_parameters(){
    if (mpi_rank==0){
      std::string file_name = "products/parameters.txt" ; // + std::to_string(indIt) + "_rank" + std::to_string(sP->mpi_rank) + ".txt" ;
      std::ofstream thefile ;
      thefile.open (file_name) ;
      thefile << "seed_val_cst        " << seed_val_cst         << "\n" ;
      thefile << "tpb_cst             " << tpb_cst              << "\n" ;
      thefile << "nb_it_max_cst       " << nb_it_max_cst        << "\n" ;
      thefile << "rate_save_t_cst     " << rate_save_t_cst      << "\n" ;
      thefile << "len_save_t_cst      " << len_save_t_cst       << "\n" ;
      thefile << "len_x_cst           " << len_x_cst            << "\n" ;
      thefile << "len_y_cst           " << len_y_cst            << "\n" ;
      thefile << "rate_save_x_cst     " << rate_save_x_cst      << "\n" ;
      thefile << "len_save_x_cst      " << len_save_x_cst       << "\n" ;
      thefile << "rate_save_y_cst     " << rate_save_y_cst      << "\n" ;
      thefile << "len_save_y_cst      " << len_save_y_cst       << "\n" ;
      thefile << "nb_nodes_cst        " << nb_nodes_cst         << "\n" ;
      thefile << "nb_nodes_tot_cst    " << nb_nodes_tot_cst     << "\n" ;
      thefile << "nb_part_node_cst    " << nb_part_node_cst     << "\n" ;
      thefile << "pool_size_cst       " << pool_size_cst        << "\n" ;
      thefile << "tank_size_cst       " << tank_size_cst        << "\n" ;
      thefile << "buff_size_cst       " << buff_size_cst        << "\n" ;
      thefile << "injec_size_cst      " << injec_size_cst       << "\n" ;
      thefile << "nb_part_add_pla_cst "<< nb_part_add_pla_cst  << "\n" ;
      thefile << "decay_turb_cst      " << decay_turb_cst       << "\n" ;
      thefile << "obstacle_cst        " << obstacle_cst         << "\n" ;
      thefile << "inject_pla_cst      " << inject_pla_cst       << "\n" ;
      thefile << "inject_turb_cst     " << inject_turb_cst      << "\n" ;
      thefile << "idx_it_inject_cst   " << idx_it_inject_cst    << "\n" ;
      thefile << "nb_it_per_cell_cst  " << nb_it_per_cell_cst   << "\n" ;
      thefile << "n0_no_norm          " << n0_no_norm           << "\n" ;
      thefile << "B0_no_norm          " << B0_no_norm           << "\n" ;
      thefile << "v0                  " << v0                   << "\n" ;
      thefile << "E0                  " << E0                   << "\n" ;
      thefile << "Ti_inf              " << Ti_inf               << "\n" ;
      thefile << "Te_inf              " << Te_inf               << "\n" ;
      thefile << "e_mot               " << e_mot                << "\n" ;
      thefile << "e_hal               " << e_hal                << "\n" ;
      thefile << "e_amb               " << e_amb                << "\n" ;
      thefile << "poly_ind            " << poly_ind             << "\n" ;
      thefile << "eta_res_norm        " << eta_res_norm         << "\n" ;
      thefile << "eta_sm              " << eta_sm               << "\n" ;
      thefile << "omega_i             " << omega_i              << "\n" ;
      thefile << "omega_ci            " << omega_ci             << "\n" ;
      thefile << "v_A                 " << v_A                  << "\n" ;
      thefile << "t0                  " << t0                   << "\n" ;
      thefile << "x0                  " << x0                   << "\n" ;
      thefile << "q0                  " << q0                   << "\n" ;
      thefile << "m0                  " << m0                   << "\n" ;
      thefile << "p0                  " << p0                   << "\n" ;
      thefile << "B0_x                " << B0_x                 << "\n" ;
      thefile << "B0_y                " << B0_y                 << "\n" ;
      thefile << "B0_z                " << B0_z                 << "\n" ;
      thefile << "Q                   " << Q                    << "\n" ;
      thefile << "nu_i                " << nu_i                 << "\n" ;
      thefile << "nu_d                " << nu_d                 << "\n" ;
      thefile << "u0                  " << u0                   << "\n" ;
      thefile << "fluctu_sqrt         " << fluctu_sqrt          << "\n" ;
      thefile << "d_i                 " << d_i                  << "\n" ;
      thefile << "omega_e             " << omega_e              << "\n" ;
      thefile << "omega_ce            " << omega_ce             << "\n" ;
      thefile << "d_e                 " << d_e                  << "\n" ;
      thefile << "v_thi               " << v_thi                << "\n" ;
      thefile << "v_the               " << v_the                << "\n" ;
      thefile << "v_s                 " << v_s                  << "\n" ;
      thefile << "v_ms                " << v_ms                 << "\n" ;
      thefile << "lambda_D            " << lambda_D             << "\n" ;
      thefile << "Beta_p              " << Beta_p               << "\n" ;
      thefile << "Beta_e              " << Beta_e               << "\n" ;
      thefile << "xLen                " << xLen                 << "\n" ;
      thefile << "yLen                " << yLen                 << "\n" ;
      thefile << "dX                  " << dX                   << "\n" ;
      thefile << "dX_i                " << dX_i                 << "\n" ;
      thefile << "nb_dim              " << nb_dim               << "\n" ;
      thefile << "dt                  " << dt                   << "\n" ;
      thefile << "tMax                " << tMax                 << "\n" ;
      thefile << "v0_com              " << v0_com               << "\n" ;
      thefile << "nb_oscillations     " << nb_oscillations      << "\n" ;
      thefile << "lambda_oscillation  " << lambda_oscillation   << "\n" ;
      thefile << "k_oscillation       " << k_oscillation        << "\n" ;
      thefile << "omega_iaw           " << omega_iaw            << "\n" ;
      thefile << "period_iaw          " << period_iaw           << "\n" ;
      thefile << "nbSubCycles         " << nbSubCycles          << "\n" ;
      thefile << "subDt               " << subDt                << "\n" ;
      thefile << "rateSaveIt          " << rateSaveIt           << "\n" ;
      thefile << "nb_nodes            " << nb_nodes             << "\n" ;
      thefile << "nb_nodes_tot        " << nb_nodes_tot         << "\n" ;
      thefile << "nb_part_add_sw      " << nb_part_add_sw       << "\n" ;
      thefile << "nb_part_add_pla     " << nb_part_add_pla      << "\n" ;
      thefile << "sum_pla_proba       " << sum_pla_proba        << "\n" ;
      // thefile << "pla_proba_max       " << pla_proba_max        << "\n" ;
      thefile << "w_sw                " << w_sw                 << "\n" ;
      thefile << "w_pla               " << w_pla                << "\n" ;
      thefile << "Z_pla               " << Z_pla                << "\n" ;
      thefile << "dens_min            " << dens_min             << "\n" ;
      thefile << "mpi_rank            " << mpi_rank             << "\n" ;
      thefile << "mpi_nb_proc         " << mpi_nb_proc          << "\n" ;
      thefile << "len_buff_field      " << len_buff_field       << "\n" ;
      thefile << "nb_part_tank        " << nb_part_tank         << "\n" ;

      thefile.close() ;
    }
  }


} ;



#endif
