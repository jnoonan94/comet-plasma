#ifndef __PARAMETERS_H_INCLUDED__   // if x.h hasn't been included yet...
#define __PARAMETERS_H_INCLUDED__

#include <stdio.h>
#include <iostream>

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

const bool inject_cst = false ;

const int nb_it_max_cst   = 6000 ;
const int rate_save_t_cst = 10 ;  // For simplicity, keep it an integer fraction of nb_it_max_cst and xLen_cst
const int len_save_t_cst  = int(nb_it_max_cst/rate_save_t_cst) ;

const int xLen_cst = 64 ;
const int yLen_cst = 8 ;
const int rate_save_x_cst = 1 ;  //   so the dt_low and dx_low are well defined.
const int len_save_x_cst  = int(xLen_cst/rate_save_x_cst) ;
const int rate_save_y_cst = 1 ;  //   so the dt_low and dx_low are well defined.
const int len_save_y_cst  = int(yLen_cst/rate_save_y_cst) ;

const int nb_nodes_cst = xLen_cst*yLen_cst ;
const int nb_nodes_tot_cst = (xLen_cst+4)*(yLen_cst+4) ;
const int nb_part_node_cst = 2048 ; //16384 ;
const int nb_part_max_cst = 1.*int(xLen_cst*yLen_cst*nb_part_node_cst) ;
const int nb_part_add_cst = 5000 ;
const int nb_part_pist_cst = 999 ;
const int nb_part_inj_max_cst = 99 ;
const int nb_stack_cst = 128 ;
const int len_stack_cst = 1+nb_part_add_cst/nb_stack_cst ; // Could be max(nbPartAddSW,nbPartAddCom)

struct simu_param {

  //____________________________________________________________________________
  // Background physical values in SI.
  float n0_no_norm  = 1.e6 ;
  float B0_no_norm  = 1.8e-9 ;
  float v0          = 0. ;
  float E0          = -v0*B0_no_norm ;
  float Ti_inf      = 1.e2 ;
  float Te_inf      = 1.e3 ;

  float e_mot       = 1. ;
  float e_hal       = 1. ;
  float e_amb       = 1. ;
  //
  float poly_ind    = 1. ;
  float eta_res_norm = 0. ; //.00375 ; //eta_res*q*n0_no_norm/B0_no_norm ;
  float eta_sm = 1. ;

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

  float B0_x        = 1. ;
  float B0_y        = 0. ;
  float B0_z        = 0. ;

  // float p_e0        = n0_no_norm*k_B*Te_inf ;

  float Q           = 6.e26 ;  // s-1
  float nu_i        = 2.e-7 ;  // s-1
  float nu_d        = 5.0e-7 ;  // s-1
  float u0          = 700.    ;  // m/s

  float fluctu_sqrt = 0. ;

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

  int   xLen        = xLen_cst ;  // Number of nodes along x.
  int   yLen        = yLen_cst ;  // Number of nodes along y.
  float dX          = .266 ; // Node spacing, in unit of d_i
  float dX_i        = 1./dX ;
  int   nb_dim      = 2 ;

  int   nbPartNode  = nb_part_node_cst ;  // Number of particles per node.

  float dt          = 0.005 ;  // time step, in unit of 1/omega_ci
  float tMax        = dt*nb_it_max_cst ; // MEDIOCRE. 400./omega_ci ; // Total time, s

  float nb_oscillations = 3. ;
  float lambda_oscillation = xLen*dX/nb_oscillations ;
  float k_oscillation = 2.*PI/lambda_oscillation ;
  float omega_iaw = k_oscillation*v_s/v_A ;
  float period_iaw = 2.*PI/omega_iaw ;

  int   nbSubCycles = 11 ;
  float subDt       = dt/nbSubCycles ;

  int   rateSaveIt  = .05*tMax/dt ;  // Save data every rateSaveIt iterations.
  int   nbItMax     = int(tMax/dt) ;  // Maximum number of iterations.

  int nb_nodes      = xLen*yLen ;  // Total number of nodes.
  int nb_nodes_tot  = (xLen+2)*(yLen+2) ;  // Total number of nodes.
  int nbPartAdd     = nb_part_add_cst ; //int(nbPartNode*yLen*(-.5*v0*dt)/dX) ; // Number of particles to add per SEMI time step.
  int nbPartAddCom  = 20000 ; //30000     ;
  int nbPartMean    = int(xLen*yLen*nbPartNode) ;   // Maximum number of particles in memory.
  int nbPartMax     = nbPartMean + nbPartAdd ; //int(1.01*(nbPartMean+2*nbPartAdd)) ; //int(std::floor(1.01*nbPartMean)) ;
  float rMaxInsert   = 50.e6 ; // 26.e6  ;  // Maximum radius of commetary ion insertion.
  float rMinInsert   = 1.e2  ;  // Maximum radius of commetary ion insertion.

  float w_sw        = 1./nbPartNode ;
  float w_pla       = 999. ; // 2.358*nu_i*Q*dt*logf(rMaxInsert/rMinInsert)/(2*u0*nbPartAddCom*dX*dX) ;
  int Z_pla         = 18 ;



  int nbStack       = nb_stack_cst ;

  float dens_min    = 0.05 ;  // Minimum density, normalised.



  void printout_parameters(){

    printf("\n._____________________________________________________________________________________\n|\n") ;
    printf("| Proton plasma frequency:         %9.4f 1/s\n", omega_i) ;
    printf("| Proton gyrofrequency:            %9.4f 1/s\n", omega_ci) ;
    printf("|\n") ;
    printf("| Proton inertial length d_i :     %9.4f km\n", d_i*1e-3) ;
    printf("| Electron inertial length d_e :   %9.4f km\n", d_e*1e-3) ;
    printf("| Debye length lambda_D :          %9.4f km\n", lambda_D*1e-3) ;
    printf("| Gyro-radius (v_th) :             %9.4f \n", v_thi/v_A) ;
    printf("|\n") ;
    printf("| Alfven speed :                   %9.4f km/s\n", v_A*1e-3) ;
    printf("| Sound speed :                    %9.4f km/s\n", v_s*1e-3) ;
    printf("| Magnetosonic speed :             %9.4f km/s\n", v_ms*1e-3) ;
    printf("| Ion thermal speed :              %9.4f km/s\n", v_thi*1e-3) ;
    printf("| Electron thermal speed :         %9.4f km/s\n", v_the*1e-3) ;
    printf("|\n") ;
    printf("| Beta_p :                         %9.4f \n", Beta_p) ;
    printf("| Beta_e :                         %9.4f \n", Beta_e) ;
    printf("|\n|\n") ;
    printf("| Particle-per-node:   %i \n", nb_part_node_cst) ;
    printf("| Number of particles: %i\n", nb_part_max_cst) ;
    printf("| Particle weight:     %.2e\n|\n", w_sw) ;
    printf("| Box length (%i x %i nodes): %.4e (%.4f proton inertial lengths d_i)\n", xLen_cst, yLen_cst, xLen_cst*dX, xLen_cst*dX) ;
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


} ;



#endif
