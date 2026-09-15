#include <cmath>
#include <time.h>

#include "parameters.h"
#include "structures.h"
#include "functions_i.h"
#include "functions_o.h"
#include "cnpy.h"
#include <iostream>
#include <fstream>
#include <algorithm>

#include "mpi.h"

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}





/**
* ... Yo yo ya ...
*/
float normalSample(float mean = 0.0, float standardDeviation = 1.0)
{

  // if (standardDeviation <= 0.0)
  // {
	// std::stringstream os;
  //     os << "Standard deviation must be positive." << "\n"
  //        << "Received standard deviation " << standardDeviation;
  //     throw std::invalid_argument( os.str() );
  // }
    // Use Box-Muller algorithm
  float u1 = float(rand())/float(RAND_MAX); //GetUniform();
  if (u1==0.){
    u1 = 0.000001;
  }
  float u2 = float(rand())/float(RAND_MAX); //GetUniform();
  float r = std::sqrt( -2.0*std::log(u1) );
  float theta = 2.0*PI*u2;
  return mean + standardDeviation*r*std::sin(theta);
}

/**
* Shuffles ransP->r_obsdommly an array initially containing ordered indices (a range).
*/
void shuffleOwn(int *arr, size_t n)
{
    if (n > 1)
    {
        size_t i;
        // srand(time(NULL));
        for (i = 0; i < n - 1; i++)
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = arr[j];
          arr[j] = arr[i];
          arr[i] = t;
        }
    }
}

//__________________________________________________________________________________________________________________


//__________________________________________________________________________________________________________________
//
//  Modules handling grid values and particles.
//
void init_house_keeping(moments* mom, trajectory* traj, root_mean_sqr* rms,
                        simu_param* sP, house_keeping* h_k, state_solver* sta_sol)
{
  for ( int i=0; i<len_save_t_cst; i++){
    mom->E_mag[i] = 0; mom->E_elec[i] = 0; mom->E_kin[i] = 0;
    mom->nb_part_tot[i] = 0; mom->dens_tot[i] = 0;
    mom->vel_part[0][i] = 0; mom->vel_part[1][i] = 0; mom->vel_part[2][i] = 0;
    mom->vel_th_part[0][i] = 0; mom->vel_th_part[1][i] = 0; mom->vel_th_part[2][i] = 0;
    //
    // rms->Brms[i] = 0.; rms->vrms[i] = 0.;
    //
    traj->rx[i] = 0; traj->ry[i] = 0;
    #if NB_DIM==3
      traj->rz[i] = 0;
    #endif
    traj->vx[i] = 0; traj->vy[i] = 0; traj->vz[i] = 0;
  }
}



void init_grid(simu_grid* grid, simu_param* sP)
{

  grid->xMin = 0.; // -dX*float(len_x_cst)*.5;
  grid->yMin = len_y_cst*sP->dX*sP->mpi_rank_y; // -dX*float(len_y_cst)*.5;
  #if NB_DIM==3
    grid->zMin = len_z_cst*sP->dX*sP->mpi_rank_z;
  #endif

  float offset = 2-.5;

  for (int i=0; i<(len_x_cst+4); i++){grid->xGrid[i] = grid->xMin + (i-offset)*sP->dX; }
  for (int j=0; j<(len_y_cst+4); j++){grid->yGrid[j] = grid->yMin + (j-offset)*sP->dX; }
  #if NB_DIM==3
    for (int k=0; k<(len_z_cst+4); k++){grid->zGrid[k] = grid->zMin + (k-offset)*sP->dX; }
  #endif

  grid->xMax = len_x_cst*sP->dX; //grid->xGrid[len_x_cst] + .5*sP->dX;
  grid->yMax = len_y_cst*sP->dX*(sP->mpi_rank_y+1); //grid->yGrid[len_y_cst] + .5*sP->dX;
  #if NB_DIM==3
    grid->zMax = len_z_cst*sP->dX*(sP->mpi_rank_z+1); //grid->yGrid[len_y_cst] + .5*sP->dX;
  #endif

  float max_x_box = grid->xMax;
  float max_y_box = len_y_cst*sP->dX*sP->mpi_nb_proc_y;
  #if NB_DIM==3
    float max_z_box = len_z_cst*sP->dX*sP->mpi_nb_proc_z;
  #endif

  grid->centre_x = sP->centre_x*max_x_box;
  grid->centre_y = sP->centre_y*max_y_box;
  #if NB_DIM==3
    grid->centre_z = sP->centre_z*max_z_box;
  #endif
  for (int i=0; i<len_x_cst+4; i++){
    for (int j=0; j<len_y_cst+4; j++){
      #if NB_DIM==2
        grid->rsq[i][j] = (grid->xGrid[i]-grid->centre_x)*(grid->xGrid[i]-grid->centre_x) +
                          (grid->yGrid[j]-grid->centre_y)*(grid->yGrid[j]-grid->centre_y);
      #elif NB_DIM==3
        for (int k=0; k<len_z_cst+4; k++){
          grid->rsq[i][j][k] = (grid->xGrid[i]-grid->centre_x)*(grid->xGrid[i]-grid->centre_x) +
                               (grid->yGrid[j]-grid->centre_y)*(grid->yGrid[j]-grid->centre_y) +
                               (grid->zGrid[k]-grid->centre_z)*(grid->zGrid[k]-grid->centre_z);
        }
      #endif
    }
  }

  #if DF_cst
    for (int l=0; l<(len_v_cst+4); l++){
      grid->vGrid[l] = (l - int((len_v_cst-1)/2+2)) * sP->dV ;
    }
  #endif

}

void init_fields(simu_fields* fields, simu_param* sP)
{
  #if NB_DIM==2
    for (int i = 0; i < (len_x_cst+4); i++){
      for (int j = 0; j < (len_y_cst+4); j++){
          fields->E[0][i][j] = 0.;
          fields->E[1][i][j] = 0.;
          fields->E[2][i][j] = 0.;
          #if yee_cst
            fields->E_s[0][i][j] = 0.;
            fields->E_s[1][i][j] = 0.;
            fields->E_s[2][i][j] = 0.;
            fields->J_tot_s[0][i][j] = 0.;
            fields->J_tot_s[1][i][j] = 0.;
            fields->J_tot_s[2][i][j] = 0.;
          #endif
          // #if ORF_cst || dipole_cst
          #if ORF_cst
            fields->E[0][i][j] += 0.;
            fields->E[1][i][j] += -sP->e_mot*sP->v_obs*sP->B0_z;
            fields->E[2][i][j] +=  sP->e_mot*sP->v_obs*sP->B0_y;
            #if yee_cst
              fields->E_s[0][i][j] += 0.;
              fields->E_s[1][i][j] += -sP->e_mot*sP->v_obs*sP->B0_z;
              fields->E_s[2][i][j] +=  sP->e_mot*sP->v_obs*sP->B0_y;
            #endif
          #endif
          // #if (dipole_cst && !ORF_cst)
          //   // Surprising one, but fields are solved in ORF when simulating
          //   //  a dipole in SWRF.
          //   fields->E[0][i][j] += 0.;
          //   fields->E[1][i][j] += -sP->v_obs*sP->B0_z;
          //   fields->E[2][i][j] +=  sP->v_obs*sP->B0_y;
          // #endif
          fields->counts[i][j]     = 0.;
          fields->counts_pla[i][j] = 0.;
          fields->density[i][j]    = 0.; //sP->dens_min;
          fields->density_b[i][j]  = 0.; //sP->dens_min;
          fields->pres[i][j]       = sP->n0_SI*k_B*sP->Te_inf/sP->p0; //sP->dens_min;
          fields->Lambda[i][j]     = 0.;
          fields->fluxNum[0][i][j] = 0.;
          fields->fluxNum[1][i][j] = 0.;
          fields->fluxNum[2][i][j] = 0.;
          fields->fluxNum_pla[0][i][j] = 0.;
          fields->fluxNum_pla[1][i][j] = 0.;
          fields->fluxNum_pla[2][i][j] = 0.;
          fields->Gamma[0][i][j] = 0.;
          fields->Gamma[1][i][j] = 0.;
          fields->Gamma[2][i][j] = 0.;
          fields->Ji[0][i][j] = 0.;
          fields->Ji[1][i][j] = 0.;
          fields->Ji[2][i][j] = 0.;
          fields->Ji_b[0][i][j] = 0.;
          fields->Ji_b[1][i][j] = 0.;
          fields->Ji_b[2][i][j] = 0.;
          fields->J_tot[0][i][j] = 0.;
          fields->J_tot[1][i][j] = 0.;
          fields->J_tot[2][i][j] = 0.;
          fields->region_ID[i][j] = 0;
      }
    }
  #elif NB_DIM==3
    for (int i = 0; i < (len_x_cst+4); i++){
      for (int j = 0; j < (len_y_cst+4); j++){
        for (int k = 0; k < (len_z_cst+4); k++){
          fields->E[0][i][j][k] = 0.;
          fields->E[1][i][j][k] = 0.;
          fields->E[2][i][j][k] = 0.;
          #if yee_cst
            fields->E_s[0][i][j][k] = 0.;
            fields->E_s[1][i][j][k] = 0.;
            fields->E_s[2][i][j][k] = 0.;
            fields->J_tot_s[0][i][j][k] = 0.;
            fields->J_tot_s[1][i][j][k] = 0.;
            fields->J_tot_s[2][i][j][k] = 0.;
          #endif
          // #if ORF_cst || dipole_cst
          #if ORF_cst
            fields->E[0][i][j][k] += 0.;
            fields->E[1][i][j][k] += -sP->e_mot*sP->v_obs*sP->B0_z;
            fields->E[2][i][j][k] +=  sP->e_mot*sP->v_obs*sP->B0_y;
            #if yee_cst
              fields->E_s[0][i][j][k] += 0.;
              fields->E_s[1][i][j][k] += -sP->e_mot*sP->v_obs*sP->B0_z;
              fields->E_s[2][i][j][k] +=  sP->e_mot*sP->v_obs*sP->B0_y;
            #endif
          #endif
          // #if (dipole_cst && !ORF_cst)
          //   // Surprising one, but fields are solved in ORF when simulating
          //   //  a dipole in SWRF.
          //   fields->E[0][i][j][k] += 0.;
          //   fields->E[1][i][j][k] += -sP->v_obs*sP->B0_z;
          //   fields->E[2][i][j][k] +=  sP->v_obs*sP->B0_y;
          // #endif
          fields->counts[i][j][k]     = 0.;
          fields->counts_pla[i][j][k] = 0.;
          fields->density[i][j][k]    = 0.; //sP->dens_min;
          fields->density_b[i][j][k]  = 0.; //sP->dens_min;
          fields->pres[i][j][k]       = sP->n0_SI*k_B*sP->Te_inf/sP->p0; //sP->dens_min;
          fields->Lambda[i][j][k]     = 0.;
          fields->fluxNum[0][i][j][k] = 0.;
          fields->fluxNum[1][i][j][k] = 0.;
          fields->fluxNum[2][i][j][k] = 0.;
          fields->fluxNum_pla[0][i][j][k] = 0.;
          fields->fluxNum_pla[1][i][j][k] = 0.;
          fields->fluxNum_pla[2][i][j][k] = 0.;
          fields->Gamma[0][i][j][k] = 0.;
          fields->Gamma[1][i][j][k] = 0.;
          fields->Gamma[2][i][j][k] = 0.;
          fields->Ji[0][i][j][k] = 0.;
          fields->Ji[1][i][j][k] = 0.;
          fields->Ji[2][i][j][k] = 0.;
          fields->Ji_b[0][i][j][k] = 0.;
          fields->Ji_b[1][i][j][k] = 0.;
          fields->Ji_b[2][i][j][k] = 0.;
          fields->J_tot[0][i][j][k] = 0.;
          fields->J_tot[1][i][j][k] = 0.;
          fields->J_tot[2][i][j][k] = 0.;
          fields->region_ID[i][j][k] = 0;
        }
      }
    }
  #endif
}

void init_probes(probes* prob, simu_param* sP, simu_grid* grid)
{

  #if NB_DIM==2

    // Probes at vertices of equilateral triangles.
    // float probes_rx[nb_probes_cst] = {400., 399.567, 399.567, 395.667, 395.670, 356.699, 356.699, 183.494, 183.494};
    // float probes_ry[nb_probes_cst] = {300., 299.75, 300.25, 297.5, 302.5, 275., 325., 175., 425.};

    // Probes probing probed.
    float probes_rx[nb_probes_cst] = {262.07, 238.97, 241.65, 239.06, 264.92, 228.97, 266.51, 258.61, 253.53, 228.43, 258.54, 260.66, 262.97, 260.15, 237.74, 235.71, 243.90, 266.18, 252.72, 259.25, 44.12, 30.62, 29.80, 36.05, 30.35, 54.69, 39.11, 31.51, 45.02, 45.72, 46.16, 32.52, 38.47, 29.98, 37.61, 38.48, 52.17, 32.32, 36.84, 38.83, 253.78, 263.47, 249.45, 254.52, 262.13, 260.83, 245.48, 250.67, 269.47, 238.29, 236.99, 241.87, 258.71, 258.77, 221.51, 244.56, 231.64, 230.68, 268.92, 232.89, 55.84, 32.08, 52.76, 71.50, 38.19, 52.09, 51.12, 35.60, 52.78, 40.57, 42.62, 48.37, 53.53, 34.88, 42.55, 40.14, 29.19, 47.87, 36.89, 43.58, 246.22, 264.98, 265.67, 239.57, 258.61, 238.47, 245.00, 251.49, 230.91, 250.99, 249.92, 248.80, 235.30, 253.86, 239.48, 239.39, 243.51, 227.75, 233.77, 254.56, 46.59, 59.93, 50.08, 36.05, 36.99, 30.77, 45.34, 43.39, 29.24, 55.24, 29.91, 58.59, 39.27, 58.85, 58.16, 37.92, 51.92, 36.04, 41.92, 34.13, 314.65, 313.56, 309.35, 313.60, 309.08, 312.26, 311.08, 310.14, 312.88, 312.69, 310.97, 313.81, 312.46, 313.28, 310.76, 311.86, 310.75, 312.65,     312.50      , 310.86, 479.42, 472.11, 472.83, 470.84, 473.00, 476.12, 470.63, 487.74, 473.39, 484.44, 482.29, 486.67, 477.75, 477.16, 487.52, 486.93, 485.66, 478.01, 481.72, 476.23};
    float probes_ry[nb_probes_cst] = {189.78, 202.22, 184.19, 203.20, 191.24, 186.13, 191.13, 199.31, 202.37, 187.47, 186.27, 208.61, 206.68, 193.68, 184.42, 189.79, 202.16, 196.59, 191.57, 200.30, 90.00, 123.72, 124.00, 92.63, 119.84, 97.29, 115.05, 121.95, 85.37, 127.55, 98.78, 98.01, 130.03, 128.94, 102.73, 136.86, 87.77, 86.41, 111.38, 130.32, 262.03, 275.35, 291.37, 276.99, 279.78, 272.81, 283.13, 278.47, 268.81, 278.42, 266.71, 292.22, 266.17, 267.42, 268.05, 284.00, 281.91, 281.43, 272.29, 275.23, 342.79, 307.03, 318.45, 328.16, 327.66, 342.81, 334.64, 326.74, 321.00, 328.27, 300.19, 312.42, 313.09, 335.62, 312.88, 344.55, 322.54, 345.65, 304.42, 320.73, 231.72, 233.22, 241.62, 237.98, 229.87, 235.53, 231.04, 240.21, 236.35, 241.15, 229.15, 227.38, 229.46, 230.33, 238.85, 236.68, 232.31, 226.28, 227.38, 235.11, 222.90, 217.45, 209.03, 227.77, 204.34, 219.94, 226.54, 230.27, 218.54, 217.37, 228.79, 227.40, 221.58, 216.80, 219.36, 224.28, 231.45, 209.85, 235.61, 210.42, 249.41, 248.79, 252.20, 246.92, 248.25, 247.03, 246.41, 249.99, 247.24, 247.92, 251.00, 249.65, 254.48, 248.42, 251.54, 254.03, 248.10, 251.98,    250.00    , 247.33, 325.13, 274.75, 403.32, 381.44, 158.47, 445.98, 119.53, 47.95, 107.30, 389.28, 23.39, 68.05, 482.21, 440.38, 124.61, 471.77, 250.98, 251.95, 94.31, 76.06};


    for (int p=0; p<nb_probes_cst; p++){
      if (     (probes_rx[p]>grid->xMin) && (probes_rx[p]<=grid->xMax)
            && (probes_ry[p]>grid->yMin) && (probes_ry[p]<=grid->yMax) ){
          for (int t=0; t<nb_it_max_cst; t++){
            prob->rx[p][t] = probes_rx[p];
            prob->ry[p][t] = probes_ry[p];
          }
        prob->active[p] = true;
      }
      else {
        prob->active[p] = false;
      }


      for (int t=0; t<nb_it_max_cst; t++){
        for (int d=0; d<3; d++){
          prob->B[p][d][t] = 0.;
          prob->E[p][d][t] = 0.;
          prob->Ji[p][d][t] = 0.;
          prob->J_tot[p][d][t] = 0.;
        }
        prob->density[p][t] = 0.;
      }
    }

  #elif NB_DIM==3
    printf("Not implemented, init probes 3D.\n");
    sortie();
    // float delta_x = (grid->xMax - grid->xMin)/nb_probes_x_cst;
    // float delta_y = (grid->yMax - grid->yMin)/nb_probes_y_cst;
    // float delta_z = (grid->zMax - grid->zMin)/nb_probes_z_cst;
    //
    // for (int i=0; i<nb_probes_x_cst; i++){
    //   for (int j=0; j<nb_probes_y_cst; j++){
    //     for (int k=0; k<nb_probes_z_cst; k++){
    //
    //       prob->rx[i][j][k] = grid->xMin + (i+.5)*delta_x;
    //       prob->ry[i][j][k] = grid->yMin + (j+.5)*delta_y;
    //       prob->rz[i][j][k] = grid->zMin + (k+.5)*delta_z;
    //
    //       for (int t=0; t<nb_it_max_cst; t++){
    //         for (int d=0; d<3; d++){
    //           prob->B[d][i][j][k][t] = 0.;
    //           prob->E[d][i][j][k][t] = 0.;
    //           prob->Ji[d][i][j][k][t] = 0.;
    //           prob->J_tot[d][i][j][k][t] = 0.;
    //         }
    //         prob->density[i][j][k][t] = 0.;
    //       }
    //     }
    //   }
    // }
  #endif
}

void init_B(simu_B_field* B, simu_param* sP)
{
  #if NB_DIM==2
    for (int i = 0; i < (len_x_cst+4); i++){
      for (int j = 0; j < (len_y_cst+4); j++){
          B->B[0][i][j] = sP->B0_x;
          B->B[1][i][j] = sP->B0_y;
          B->B[2][i][j] = sP->B0_z;
          // B->B[0][i][j] += .23*(float(rand())/float(RAND_MAX)-1);
          // B->B[1][i][j] += .23*(float(rand())/float(RAND_MAX)-1);
          // B->B[2][i][j] += .23*(float(rand())/float(RAND_MAX)-1);
          #if yee_cst
            B->B_s[0][i][j] = sP->B0_x;
            B->B_s[1][i][j] = sP->B0_y;
            B->B_s[2][i][j] = sP->B0_z;
            // B->B_s[0][i][j] += .2*(float(rand())/float(RAND_MAX)-1);
            // B->B_s[1][i][j] += .2*(float(rand())/float(RAND_MAX)-1);
            // B->B_s[2][i][j] += .2*(float(rand())/float(RAND_MAX)-1);
          #endif
      }
    }
  #elif NB_DIM==3
    for (int i = 0; i < (len_x_cst+4); i++){
      for (int j = 0; j < (len_y_cst+4); j++){
        for (int k = 0; k < (len_z_cst+4); k++){
          B->B[0][i][j][k] = sP->B0_x;
          B->B[1][i][j][k] = sP->B0_y;
          B->B[2][i][j][k] = sP->B0_z;
          // B->B[0][i][j][k] += .4*(float(rand())/float(RAND_MAX)-1);
          // B->B[1][i][j][k] += .4*(float(rand())/float(RAND_MAX)-1);
          // B->B[2][i][j][k] += .4*(float(rand())/float(RAND_MAX)-1);
          #if yee_cst
            B->B_s[0][i][j][k] = sP->B0_x;
            B->B_s[1][i][j][k] = sP->B0_y;
            B->B_s[2][i][j][k] = sP->B0_z;
          #endif
        }
      }
    }
  #endif
}

#if DF_cst
void init_DF(simu_DF* DF, simu_grid* grid, simu_param* sP){

  float v_sq ;
  float v_sq2 ;
  float v_sq3 ;
  float v0_x = 0. ; //3.e6 ; //6e6
  float v0_y = 0. ; //1.e6 ;

  for (int i = 0; i < (len_x_cst+4); i++){
    // v_sq2 = grid->xGrid[i]-grid->xGrid[sP->xLen+3]/2. ;
    // v_sq2 *= v_sq2 ;
    // v_sq3 = grid->xGrid[i]-grid->xGrid[sP->xLen+3]/4. ;
    // v_sq3 *= v_sq3 ;
    for (int l = 2; l < (len_v_cst+2); l++){
      for (int m = 2; m < (len_v_cst+2); m++){
        for (int n = 2; n < (len_v_cst+2); n++){
          v_sq =  (grid->vGrid[l]+v0_x)*(grid->vGrid[l]+v0_x)
                + (grid->vGrid[m]+v0_y)*(grid->vGrid[m]+v0_y)
                +  grid->vGrid[n]      * grid->vGrid[n] ;

          DF->DF[i][l][m][n] = 1./sqrt(2) * std::exp(-v_sq/(sP->v_thi/sP->v_A*sP->v_thi/sP->v_A));
          if (DF->DF[i][l][m][n]<0){
            std::cout << "yo pouf " << std::endl;
          }
          //
          // v_sq = (grid->vGrid[l]+3.)*(grid->vGrid[l]+3.)
          //       + (grid->vGrid[m])*(grid->vGrid[m])
          //       +  grid->vGrid[n]      * grid->vGrid[n] ;
          //
          // DF->DF[i][l][m][n] += .5/sqrt(2) * std::exp(-10*v_sq/(sP->v_thi/sP->v_A*sP->v_thi/sP->v_A)) ;

        }
      }
    }
  }
}
#endif

#if dipole_cst
void init_B_dipole(simu_B_field* B, simu_param* sP, simu_grid* grid)
{
  #if NB_DIM==2
    for (int i = 0; i < (len_x_cst+4); i++){
      for (int j = 0; j < (len_y_cst+4); j++){
        for (int k = 0; k < 5; k++){
          float rx = grid->xGrid[i]-grid->centre_x;
          // #if (!ORF_cst)
          //   rx -= nb_cell_per_shift_cst*sP->dX;
          // #endif
          float ry = grid->yGrid[j]-grid->centre_y;
          float rz = (k-2)*sP->dX;
          // float r3_SI = std::pow( (rx*rx + ry*ry + rz*rz) *sP->d_i*sP->d_i, 3./2.);
          float r3_SI = (rx*rx + ry*ry + rz*rz) *sP->d_i*sP->d_i;
          // The magnetic scalar potential from the magnetic pole limit:
          B->phi[i][j][k] = (sP->dip_mom_SI[0]*rx*sP->d_i
                           + sP->dip_mom_SI[1]*ry*sP->d_i
                           + sP->dip_mom_SI[2]*rz*sP->d_i)/r3_SI*1e-6;
        }
      }
    }
    for (int i = 2; i < (len_x_cst+2); i++){
      for (int j = 2; j < (len_y_cst+2); j++){
        // for (int k = 2; k < (len_z_cst+2); k++){
          float dxdphi = 1./(12.*sP->dX*sP->d_i) * ( B->phi[i-2][j  ][2  ]-8*B->phi[i-1][j  ][2  ]+8*B->phi[i+1][j  ][2  ]-B->phi[i+2][j  ][2  ] );
          float dydphi = 1./(12.*sP->dX*sP->d_i) * ( B->phi[i  ][j-2][2  ]-8*B->phi[i  ][j-1][2  ]+8*B->phi[i  ][j+1][2  ]-B->phi[i  ][j+2][2  ] );
          float dzdphi = 1./(12.*sP->dX*sP->d_i) * ( B->phi[i  ][j  ][2-2]-8*B->phi[i  ][j  ][2-1]+8*B->phi[i  ][j  ][2+1]-B->phi[i  ][j  ][2+2] );
          B->B_dip[0][i][j] = dxdphi/sP->B0_SI;
          B->B_dip[1][i][j] = dydphi/sP->B0_SI;
          B->B_dip[2][i][j] = dzdphi/sP->B0_SI;
        // }
      }
    }

    for (int i = 2; i < (len_x_cst+2); i++){
      // for (int k = 2; k < (len_z_cst+2); k++){
        B->B_dip[0][i][0] = B->B_dip[0][i][2];
        B->B_dip[0][i][1] = B->B_dip[0][i][2];
        B->B_dip[1][i][0] = B->B_dip[1][i][2];
        B->B_dip[1][i][1] = B->B_dip[1][i][2];
        B->B_dip[2][i][0] = B->B_dip[2][i][2];
        B->B_dip[2][i][1] = B->B_dip[2][i][2];
        //
        B->B_dip[0][i][len_y_cst+2] = B->B_dip[0][i][len_y_cst+1];
        B->B_dip[0][i][len_y_cst+3] = B->B_dip[0][i][len_y_cst+1];
        B->B_dip[1][i][len_y_cst+2] = B->B_dip[1][i][len_y_cst+1];
        B->B_dip[1][i][len_y_cst+3] = B->B_dip[1][i][len_y_cst+1];
        B->B_dip[2][i][len_y_cst+2] = B->B_dip[2][i][len_y_cst+1];
        B->B_dip[2][i][len_y_cst+3] = B->B_dip[2][i][len_y_cst+1];
      // }
    }
    // for (int i = 2; i < (len_x_cst+2); i++){
    //   for (int j = 2; j < (len_y_cst+2); j++){
    //     B->B_dip[0][i][j][0] = B->B_dip[0][i][j][2];
    //     B->B_dip[0][i][j][1] = B->B_dip[0][i][j][2];
    //     B->B_dip[1][i][j][0] = B->B_dip[1][i][j][2];
    //     B->B_dip[1][i][j][1] = B->B_dip[1][i][j][2];
    //     B->B_dip[2][i][j][0] = B->B_dip[2][i][j][2];
    //     B->B_dip[2][i][j][1] = B->B_dip[2][i][j][2];
    //     //
    //     B->B_dip[0][i][j][len_z_cst+2] = B->B_dip[0][i][j][len_z_cst+1];
    //     B->B_dip[0][i][j][len_z_cst+3] = B->B_dip[0][i][j][len_z_cst+1];
    //     B->B_dip[1][i][j][len_z_cst+2] = B->B_dip[1][i][j][len_z_cst+1];
    //     B->B_dip[1][i][j][len_z_cst+3] = B->B_dip[1][i][j][len_z_cst+1];
    //     B->B_dip[2][i][j][len_z_cst+2] = B->B_dip[2][i][j][len_z_cst+1];
    //     B->B_dip[2][i][j][len_z_cst+3] = B->B_dip[2][i][j][len_z_cst+1];
    //   }
    // }
    for (int j = 2; j < (len_y_cst+2); j++){
      // for (int k = 2; k < (len_z_cst+2); k++){
        B->B_dip[0][0][j] = B->B_dip[0][2][j];
        B->B_dip[0][1][j] = B->B_dip[0][2][j];
        B->B_dip[1][0][j] = B->B_dip[1][2][j];
        B->B_dip[1][1][j] = B->B_dip[1][2][j];
        B->B_dip[2][0][j] = B->B_dip[2][2][j];
        B->B_dip[2][1][j] = B->B_dip[2][2][j];
        //
        B->B_dip[0][len_x_cst+2][j] = B->B_dip[0][len_x_cst+1][j];
        B->B_dip[0][len_x_cst+3][j] = B->B_dip[0][len_x_cst+1][j];
        B->B_dip[1][len_x_cst+2][j] = B->B_dip[1][len_x_cst+1][j];
        B->B_dip[1][len_x_cst+3][j] = B->B_dip[1][len_x_cst+1][j];
        B->B_dip[2][len_x_cst+2][j] = B->B_dip[2][len_x_cst+1][j];
        B->B_dip[2][len_x_cst+3][j] = B->B_dip[2][len_x_cst+1][j];
      // }
    }



  #elif NB_DIM==3
    for (int i = 0; i < (len_x_cst+4); i++){
      for (int j = 0; j < (len_y_cst+4); j++){
        for (int k = 0; k < (len_z_cst+4); k++){

          // float r3 = std::pow(grid->rsq[i][j][k]*sP->d_i*sP->d_i, 3./2.);
          // // The magnetic scalar potential from the magnetic pole limit:
          // B->phi[i][j][k] = (sP->dip_mom_SI[0]*(grid->xGrid[i]-grid->centre_x)*sP->d_i
          //                  + sP->dip_mom_SI[1]*(grid->yGrid[j]-grid->centre_y)*sP->d_i
          //                  + sP->dip_mom_SI[2]*(grid->zGrid[k]-grid->centre_z)*sP->d_i)/r3;
          float rx = grid->xGrid[i]-grid->centre_x;
          // #if (!ORF_cst)
          //   rx -= nb_cell_per_shift_cst*sP->dX;
          // #endif
          float ry = grid->yGrid[j]-grid->centre_y;
          float rz = grid->zGrid[k]-grid->centre_z;
          float r3_SI = std::pow( (rx*rx + ry*ry + rz*rz) *sP->d_i*sP->d_i, 3./2.);
          // The magnetic scalar potential from the magnetic pole limit:
          B->phi[i][j][k] = (sP->dip_mom_SI[0]*rx*sP->d_i
                           + sP->dip_mom_SI[1]*ry*sP->d_i
                           + sP->dip_mom_SI[2]*rz*sP->d_i)/r3_SI;
        }
      }
    }
    for (int i = 2; i < (len_x_cst+2); i++){
      for (int j = 2; j < (len_y_cst+2); j++){
        for (int k = 2; k < (len_z_cst+2); k++){
          float dxdphi = 1./(12.*sP->dX*sP->d_i) * ( B->phi[i-2][j  ][k  ]-8*B->phi[i-1][j  ][k  ]+8*B->phi[i+1][j  ][k  ]-B->phi[i+2][j  ][k  ] );
          float dydphi = 1./(12.*sP->dX*sP->d_i) * ( B->phi[i  ][j-2][k  ]-8*B->phi[i  ][j-1][k  ]+8*B->phi[i  ][j+1][k  ]-B->phi[i  ][j+2][k  ] );
          float dzdphi = 1./(12.*sP->dX*sP->d_i) * ( B->phi[i  ][j  ][k-2]-8*B->phi[i  ][j  ][k-1]+8*B->phi[i  ][j  ][k+1]-B->phi[i  ][j  ][k+2] );
          B->B_dip[0][i][j][k] = dxdphi/sP->B0_SI;
          B->B_dip[1][i][j][k] = dydphi/sP->B0_SI;
          B->B_dip[2][i][j][k] = dzdphi/sP->B0_SI;
        }
      }
    }

    for (int i = 2; i < (len_x_cst+2); i++){
      for (int k = 2; k < (len_z_cst+2); k++){
        B->B_dip[0][i][0][k] = B->B_dip[0][i][2][k];
        B->B_dip[0][i][1][k] = B->B_dip[0][i][2][k];
        B->B_dip[1][i][0][k] = B->B_dip[1][i][2][k];
        B->B_dip[1][i][1][k] = B->B_dip[1][i][2][k];
        B->B_dip[2][i][0][k] = B->B_dip[2][i][2][k];
        B->B_dip[2][i][1][k] = B->B_dip[2][i][2][k];
        //
        B->B_dip[0][i][len_y_cst+2][k] = B->B_dip[0][i][len_y_cst+1][k];
        B->B_dip[0][i][len_y_cst+3][k] = B->B_dip[0][i][len_y_cst+1][k];
        B->B_dip[1][i][len_y_cst+2][k] = B->B_dip[1][i][len_y_cst+1][k];
        B->B_dip[1][i][len_y_cst+3][k] = B->B_dip[1][i][len_y_cst+1][k];
        B->B_dip[2][i][len_y_cst+2][k] = B->B_dip[2][i][len_y_cst+1][k];
        B->B_dip[2][i][len_y_cst+3][k] = B->B_dip[2][i][len_y_cst+1][k];
      }
    }
    for (int i = 2; i < (len_x_cst+2); i++){
      for (int j = 2; j < (len_y_cst+2); j++){
        B->B_dip[0][i][j][0] = B->B_dip[0][i][j][2];
        B->B_dip[0][i][j][1] = B->B_dip[0][i][j][2];
        B->B_dip[1][i][j][0] = B->B_dip[1][i][j][2];
        B->B_dip[1][i][j][1] = B->B_dip[1][i][j][2];
        B->B_dip[2][i][j][0] = B->B_dip[2][i][j][2];
        B->B_dip[2][i][j][1] = B->B_dip[2][i][j][2];
        //
        B->B_dip[0][i][j][len_z_cst+2] = B->B_dip[0][i][j][len_z_cst+1];
        B->B_dip[0][i][j][len_z_cst+3] = B->B_dip[0][i][j][len_z_cst+1];
        B->B_dip[1][i][j][len_z_cst+2] = B->B_dip[1][i][j][len_z_cst+1];
        B->B_dip[1][i][j][len_z_cst+3] = B->B_dip[1][i][j][len_z_cst+1];
        B->B_dip[2][i][j][len_z_cst+2] = B->B_dip[2][i][j][len_z_cst+1];
        B->B_dip[2][i][j][len_z_cst+3] = B->B_dip[2][i][j][len_z_cst+1];
      }
    }
    for (int j = 2; j < (len_y_cst+2); j++){
      for (int k = 2; k < (len_z_cst+2); k++){
        B->B_dip[0][0][j][k] = B->B_dip[0][2][j][k];
        B->B_dip[0][1][j][k] = B->B_dip[0][2][j][k];
        B->B_dip[1][0][j][k] = B->B_dip[1][2][j][k];
        B->B_dip[1][1][j][k] = B->B_dip[1][2][j][k];
        B->B_dip[2][0][j][k] = B->B_dip[2][2][j][k];
        B->B_dip[2][1][j][k] = B->B_dip[2][2][j][k];
        //
        B->B_dip[0][len_x_cst+2][j][k] = B->B_dip[0][len_x_cst+1][j][k];
        B->B_dip[0][len_x_cst+3][j][k] = B->B_dip[0][len_x_cst+1][j][k];
        B->B_dip[1][len_x_cst+2][j][k] = B->B_dip[1][len_x_cst+1][j][k];
        B->B_dip[1][len_x_cst+3][j][k] = B->B_dip[1][len_x_cst+1][j][k];
        B->B_dip[2][len_x_cst+2][j][k] = B->B_dip[2][len_x_cst+1][j][k];
        B->B_dip[2][len_x_cst+3][j][k] = B->B_dip[2][len_x_cst+1][j][k];
      }
    }

    // Calculating now the time-shifted dipole.
    // for (int i = 0; i < (len_x_cst+4); i++){
    //   for (int j = 0; j < (len_y_cst+4); j++){
    //     for (int k = 0; k < (len_z_cst+4); k++){
    //       // std::cout << grid->centre_x << std::endl;
    //       float rx = grid->xGrid[i]-grid->centre_x + sP->v_obs*sP->dt;
    //       // #if (!ORF_cst)
    //       //   rx -= nb_cell_per_shift_cst*sP->dX;
    //       // #endif
    //       float ry = grid->yGrid[j]-grid->centre_y;
    //       float rz = grid->zGrid[k]-grid->centre_z;
    //       float r3_SI = std::pow( (rx*rx + ry*ry + rz*rz) *sP->d_i*sP->d_i, 3./2.);
    //       // The magnetic scalar potential from the magnetic pole limit:
    //       B->phi[i][j][k] = (sP->dip_mom_SI[0]*rx*sP->d_i
    //                        + sP->dip_mom_SI[1]*ry*sP->d_i
    //                        + sP->dip_mom_SI[2]*rz*sP->d_i)/r3_SI;
    //     }
    //   }
    // }
    // for (int i = 2; i < (len_x_cst+2); i++){
    //   for (int j = 2; j < (len_y_cst+2); j++){
    //     for (int k = 2; k < (len_z_cst+2); k++){
    //       float dxdphi = 1./(12.*sP->dX*sP->d_i) * ( B->phi[i-2][j  ][k  ]-8*B->phi[i-1][j  ][k  ]+8*B->phi[i+1][j  ][k  ]-B->phi[i+2][j  ][k  ] );
    //       float dydphi = 1./(12.*sP->dX*sP->d_i) * ( B->phi[i  ][j-2][k  ]-8*B->phi[i  ][j-1][k  ]+8*B->phi[i  ][j+1][k  ]-B->phi[i  ][j+2][k  ] );
    //       float dzdphi = 1./(12.*sP->dX*sP->d_i) * ( B->phi[i  ][j  ][k-2]-8*B->phi[i  ][j  ][k-1]+8*B->phi[i  ][j  ][k+1]-B->phi[i  ][j  ][k+2] );
    //       B->dtdBdip[0][i][j][k] = -(B->B_dip[0][i][j][k] - dxdphi/sP->B0_SI)/sP->dt;
    //       B->dtdBdip[1][i][j][k] = -(B->B_dip[1][i][j][k] - dydphi/sP->B0_SI)/sP->dt;
    //       B->dtdBdip[2][i][j][k] = -(B->B_dip[2][i][j][k] - dzdphi/sP->B0_SI)/sP->dt;
    //       // B->B[0][i][j][k] = B->B_dip[0][i][j][k] - B->dtdBdip[0][i][j][k];
    //       // B->B[1][i][j][k] = B->B_dip[1][i][j][k] - B->dtdBdip[1][i][j][k];
    //       // B->B[2][i][j][k] = B->B_dip[2][i][j][k] - B->dtdBdip[2][i][j][k];
    //     }
    //   }
    // }
  #endif
}
#endif

void init_alfven_fluctuations(simu_B_field* B, simu_param* sP){
  #if NB_DIM==2
    for (int i = 0; i < (len_x_cst+4); i++){
      for (int j = 0; j < (len_y_cst+4); j++){
          B->B[0][i][j] = sP->B0_x;
          B->B[1][i][j] = sP->B0_y;
          B->B[2][i][j] = sP->B0_z;
          B->B[2][i][j] += .05*sin(  2.*2.*PI*i/len_x_cst);
          B->B[2][i][j] += .05*sin(  5.*2.*PI*i/len_x_cst);
          B->B[2][i][j] += .05*sin( 10.*2.*PI*i/len_x_cst);
          B->B[2][i][j] += .05*sin( 20.*2.*PI*i/len_x_cst);
          B->B[2][i][j] += .05*sin( 50.*2.*PI*i/len_x_cst);
          B->B[2][i][j] += .05*sin(150.*2.*PI*i/len_x_cst);
          B->B[2][i][j] += .05*sin(300.*2.*PI*i/len_x_cst);

      }
    }
  #elif NB_DIM==3
    for (int i = 0; i < (len_x_cst+4); i++){
      for (int j = 0; j < (len_y_cst+4); j++){
        for (int k = 0; k < (len_z_cst+4); k++){
          B->B[0][i][j][k] = sP->B0_x;
          B->B[1][i][j][k] = sP->B0_y;
          B->B[2][i][j][k] = sP->B0_z;
          B->B[2][i][j][k] += .05*sin(  2.*2.*PI*i/len_x_cst);
          B->B[2][i][j][k] += .05*sin(  5.*2.*PI*i/len_x_cst);
          B->B[2][i][j][k] += .05*sin( 10.*2.*PI*i/len_x_cst);
          B->B[2][i][j][k] += .05*sin( 20.*2.*PI*i/len_x_cst);
          B->B[2][i][j][k] += .05*sin( 50.*2.*PI*i/len_x_cst);
          B->B[2][i][j][k] += .05*sin(150.*2.*PI*i/len_x_cst);
          B->B[2][i][j][k] += .05*sin(300.*2.*PI*i/len_x_cst);
        }
      }
    }
  #endif
}

/**
 * Initialises the particles "locally".
 *
 * Iterates over each node, distributing nb_part_nodes_cst per node uniformly,
 * using a maxwellian distribution for velocity components.
 *
 * @param
 * @return
 */
void init_part_node(particles* p, simu_grid* grid, simu_param* sP){

  int* indices = new int [pool_size_cst];
  int n;
  /// Simple range.
  for (int i=0; i<pool_size_cst; i++){
    indices[i] = i;
  }
  shuffleOwn(indices, pool_size_cst);

  int idx_lin;
  float vx = 0, vy = 0, vz = 0;
  float sigma = 1./sqrt(2) * sP->v_thi/sP->v_A;
  int idx_shuf;

  #if NB_DIM==2
    for (int i = 0; i < len_x_cst; i++){
      for (int j = 0; j < len_y_cst; j++){
        vx = 0; vy = 0; vz = 0;
        for (int n = 0; n < nb_part_node_cst; n++ ){
          idx_lin = (i*len_y_cst + j)*nb_part_node_cst + n;
          idx_shuf = indices[idx_lin];
          p->rx[idx_shuf] = (float(rand())/float(RAND_MAX) + i) * sP->dX + grid->xMin;
          p->ry[idx_shuf] = (float(rand())/float(RAND_MAX) + j) * sP->dX + grid->yMin;
          p->vx[idx_shuf] = normalSample(sP->v_obs, sigma);
          p->vy[idx_shuf] = normalSample(0., sigma);
          p->vz[idx_shuf] = normalSample(0., sigma);
          p->active[idx_shuf] = true;
          p->ID[idx_shuf] = 0;
          vx += p->vx[idx_shuf];
          vy += p->vy[idx_shuf];
          vz += p->vz[idx_shuf];
          //
          #if solid_body_cst
            if (grid->rsq[i+2][j+2]<(sP->r_obs_sqr)){
              p->active[idx_shuf] = false;
              vx -= p->vx[idx_shuf];
              vy -= p->vy[idx_shuf];
              vz -= p->vz[idx_shuf];
            }
          #endif
        }
        // Note: putting to zero the local ion current as done below reduces greatly the power of
        //    numerical MHD modes.
        for (int n = 0; n < nb_part_node_cst; n++ ){
          idx_lin = (i*len_y_cst + j)*nb_part_node_cst + n;
          idx_shuf = indices[idx_lin];
          p->vx[idx_shuf] -= vx/nb_part_node_cst;
          p->vy[idx_shuf] -= vy/nb_part_node_cst;
          p->vz[idx_shuf] -= vz/nb_part_node_cst;
          #if (ORF_cst && !dipole_cst)
            p->vx[idx_shuf] -= sP->v_obs;
          #endif
        }
      }
    }

    if ( idx_lin!=(nb_part_node_cst*len_x_cst*len_y_cst-1) ){
      std::cout << "\n\n Something wrong in init_part_node() \n\n";
      exit(EXIT_FAILURE);
    }
    for (int n=idx_lin+1; n<pool_size_cst; n++){
      idx_shuf = indices[n];
      p->active[idx_shuf] = false;
    }

  #elif NB_DIM==3
    for (int i = 0; i < len_x_cst; i++){
      for (int j = 0; j < len_y_cst; j++){
        for (int k = 0; k < len_z_cst; k++){
          vx = 0; vy = 0; vz = 0;

          for (int n = 0; n < nb_part_node_cst; n++ ){
            idx_lin = (i*(len_y_cst*len_z_cst) + j*len_z_cst + k)*nb_part_node_cst + n;
            idx_shuf = indices[idx_lin];
            p->rx[idx_shuf] = (float(rand())/float(RAND_MAX) + i) * sP->dX + grid->xMin;
            p->ry[idx_shuf] = (float(rand())/float(RAND_MAX) + j) * sP->dX + grid->yMin;
            p->rz[idx_shuf] = (float(rand())/float(RAND_MAX) + k) * sP->dX + grid->zMin;
            p->vx[idx_shuf] = normalSample(0., sigma);
            p->vy[idx_shuf] = normalSample(0., sigma);
            p->vz[idx_shuf] = normalSample(0., sigma);
            p->active[idx_shuf] = true;
            #if solid_body_cst
              if (grid->rsq[i+2][j+2][k+2]<(sP->r_obs_sqr)){
                p->active[idx_shuf] = false;
              }
            #endif
            p->ID[idx_shuf] = 0;
            vx += p->vx[idx_shuf];
            vy += p->vy[idx_shuf];
            vz += p->vz[idx_shuf];
          }
          // Note: putting to zero the local ion current as done below reduces greatly the power of
          //    numerical MHD modes.
          for (int n = 0; n < nb_part_node_cst; n++ ){
            idx_lin = (i*(len_y_cst*len_z_cst) + j*len_z_cst + k)*nb_part_node_cst + n;
            idx_shuf = indices[idx_lin];
            p->vx[idx_shuf] -= vx/nb_part_node_cst;
            p->vy[idx_shuf] -= vy/nb_part_node_cst;
            p->vz[idx_shuf] -= vz/nb_part_node_cst;

            // #if (ORF_cst || dipole_cst)
            #if ORF_cst
              p->vx[idx_shuf] += -sP->v_obs;
            #endif
          }
        }
      }
    }

    if ( idx_lin!=(nb_part_node_cst*len_x_cst*len_y_cst*len_z_cst-1) ){
      std::cout << "\n\n| Something wrong in init_part_node() \n\n";
      exit(EXIT_FAILURE);
    }
    for (int n=idx_lin+1; n<pool_size_cst; n++){
      idx_shuf = indices[n];
      p->active[idx_shuf] = false;
    }
  #endif
}

void init_part_acoustic(particles* p, simu_grid* grid, simu_param* sP){

  float sigma = 1./sqrt(2) * sP->v_thi/sP->v_A; //std::sqrt(k_B*sP->Ti_inf/m);
  float amplitude = .01;
  float ppn_local;
  int ind_part = 0;

  for (int i=0; i<len_x_cst; i++){

    ppn_local = nb_part_node_cst*(1 + amplitude*sin(sP->nb_wave*2.*PI*i/len_x_cst));

    for (int j=0; j<len_y_cst; j++){
      for (int n=0; n<ppn_local; n++){
        p->rx[ind_part] = grid->xGrid[i+2] + (float(rand())/float(RAND_MAX)-.5)*sP->dX;
        p->ry[ind_part] = grid->yGrid[j+2] + (float(rand())/float(RAND_MAX)-.5)*sP->dX;
        p->vx[ind_part] = normalSample(0., sigma);
        p->vy[ind_part] = normalSample(0., sigma);
        p->vz[ind_part] = normalSample(0., sigma);
        p->ID[ind_part] = 0;
        p->active[ind_part] = true;
        ind_part += 1;
        if (ind_part>pool_size_cst){
          std::cout << "Wrong. " << i << " " << j << "\n";
          sortie();
        }
      }
    }
  }

  for (int n=ind_part+1; n<pool_size_cst; n++){
    p->active[n] = false;
  }
}

void init_part_2_stream(particles* p, simu_grid* grid, simu_param* sP){

  float sigma = 1./sqrt(2) * sP->v_thi/sP->v_A;
  float v_d = 50.*sP->v_thi/sP->v_A;
  int nb_part_mean = (nb_part_node_cst*len_x_cst*len_y_cst);

  for (int n=0; n<int(1./2.*nb_part_mean); n++){
    p->rx[n] = float(rand())/float(RAND_MAX) * sP->dX*len_x_cst + grid->xMin;
    p->ry[n] = float(rand())/float(RAND_MAX) * sP->dX*len_y_cst + grid->yMin;
    p->vx[n] = normalSample(v_d, sigma);
    p->vy[n] = normalSample(0., sigma);
    p->vz[n] = normalSample(0., sigma);
    p->active[n] = true;
  }
  for (int n=int(1./2.*nb_part_mean); n<nb_part_mean; n++){
    p->rx[n] = float(rand())/float(RAND_MAX) * sP->dX*len_x_cst + grid->xMin;
    p->ry[n] = float(rand())/float(RAND_MAX) * sP->dX*len_y_cst + grid->yMin;
    p->vx[n] = normalSample(-v_d, sigma);
    p->vy[n] = normalSample(0., sigma);
    p->vz[n] = normalSample(0., sigma);
    p->active[n] = true;
  }
  for (int n=nb_part_mean; n<pool_size_cst; n++){
    p->active[n] = false;
  }
}

void init_fluctuations(simu_B_field* B, particles* p, simu_grid* grid, simu_param* sP, MPI_Comm comm){

  #if NB_DIM==2
    //__________________________________________________________________________
    // Fields:
    float kx; float ky; float kAmp;
    // float xx; float yy; float alpha;
    float phiB; float phiV;
    float delta;
    int kLen = 9;
    float k [kLen];
    float k0 = (2*PI)/((grid->yMax-grid->yMin)*sP->mpi_nb_proc_y);
    // float kMax = kLen*k0;
    float deltaV [2][len_x_cst+4][len_y_cst+4];
    float dbx, dby;
    float bPerpMax = 0, vMax = 0;
    float bMag, vMag;

    float phiBB[kLen*kLen];  /// Phases along the k-direction.
    float phiVV[kLen*kLen];  ///
    if (sP->mpi_rank_lin==0){
      for (int kk=0; kk<kLen*kLen; kk++){
        phiBB[kk] = float(rand())/float(RAND_MAX);
        phiVV[kk] = float(rand())/float(RAND_MAX);
      }
    }
    MPI_Bcast(&phiBB, kLen*kLen, MPI_FLOAT, 0, comm);
    MPI_Bcast(&phiVV, kLen*kLen, MPI_FLOAT, 0, comm);

    for (int i = 0; i < (len_x_cst+4); i++){
      for (int j = 0; j < (len_y_cst+4); j++){
        deltaV[0][i][j] = 0.;
        deltaV[1][i][j] = 0.;
      }
    }

    for (int ii=0; ii<kLen; ii++){
      k[ii] = (-((kLen-1)/2)+ii)*k0;
    }
    for (int g = 0; g < kLen; g++){
      kx = k[g];
      for (int h = 0; h < kLen; h++){
        ky = k[h];
        if ((kx!=0.) || (ky!=0.)){

          // if ((g==3) && (h==0)){
          //   std::cout << kx << " " << ky << std::endl;

            kAmp = std::sqrt(kx*kx + ky*ky);
            //
            phiB = 2.*PI*phiBB[g*kLen+h]; //float(rand())/float(RAND_MAX)*2*PI;
            phiV = 2.*PI*phiVV[g*kLen+h]; //float(rand())/float(RAND_MAX)*2*PI;
            //
            // alpha = float(rand())/float(RAND_MAX)*2*PI;
            // xx = std::cos(alpha);
            // yy = std::sin(alpha);
            //
            for (int i = 0; i < (len_x_cst+4); i++){
              for (int j = 0; j < (len_y_cst+4); j++){
                // The following definition of each component correspond to
                //   zero divergence!
                delta = std::cos(kx*grid->xGrid[i] + ky*grid->yGrid[j] + phiB); // delta between -1 and 1
                //
                B->B[0][i][j] +=  ky/kAmp * delta;
                B->B[1][i][j] += -kx/kAmp * delta;

                delta = std::cos(kx*grid->xGrid[i] + ky*grid->yGrid[j] + phiV);
                deltaV[0][i][j] += ky/kAmp * delta;
                deltaV[1][i][j] += -kx/kAmp * delta;

                #if yee_cst
                  delta = std::cos(kx*(grid->xGrid[i]-sP->dX/2.) + ky*grid->yGrid[j] + phiB); // delta at staggered grid for B_s[0]
                  B->B_s[0][i][j] += ky/kAmp * delta;
                  //
                  delta = std::cos(kx*grid->xGrid[i] + ky*(grid->yGrid[j]-sP->dX/2.) + phiB); // delta at staggered grid for B_s[1]
                  B->B_s[1][i][j] += -kx/kAmp * delta;
                #endif
              }
            }

          // }

        }
      }
    }

    for (int i = 0; i < (len_x_cst+4); i++){
      for (int j = 0; j < (len_y_cst+4); j++){
        bMag = std::sqrt(B->B[0][i][j]*B->B[0][i][j] + B->B[1][i][j]*B->B[1][i][j]);
        vMag = std::sqrt(deltaV[0][i][j]*deltaV[0][i][j] + deltaV[1][i][j]*deltaV[1][i][j]);
        if (bMag>bPerpMax){
          bPerpMax = bMag;
        }
        if (vMag>vMax){
          vMax = vMag;
        }
      }
    }

    float send_max = bPerpMax;
    float rece_max;
    MPI_Allreduce(&send_max, &rece_max, 1, MPI_FLOAT, MPI_MAX, comm);
    bPerpMax = rece_max;

    send_max = vMax;
    MPI_Allreduce(&send_max, &rece_max, 1, MPI_FLOAT, MPI_MAX, comm);
    vMax = rece_max;


    for (int i = 0; i < (len_x_cst+4); i++){
      for (int j = 0; j < (len_y_cst+4); j++){
        B->B[0][i][j]   *= std::sqrt(sP->fluctu_sqrt)/bPerpMax;
        B->B[1][i][j]   *= std::sqrt(sP->fluctu_sqrt)/bPerpMax;
        B->B[0][i][j]   += sP->B0_x;
        B->B[1][i][j]   += sP->B0_y;
        B->B[2][i][j]   += sP->B0_z;
        #if yee_cst
          B->B[0][i][j] = 0.;
          B->B[1][i][j] = 0.;
          B->B[2][i][j] = 0.;
          // std::cout << std::sqrt(sP->fluctu_sqrt) / bPerpMax << std::endl;
          B->B_s[0][i][j] *= std::sqrt(sP->fluctu_sqrt)/bPerpMax;
          B->B_s[1][i][j] *= std::sqrt(sP->fluctu_sqrt)/bPerpMax;
          B->B_s[0][i][j] += sP->B0_x;
          B->B_s[1][i][j] += sP->B0_y;
          B->B_s[2][i][j] += sP->B0_z;
        #endif
        deltaV[0][i][j] *= std::sqrt(sP->fluctu_sqrt)/vMax;
        deltaV[1][i][j] *= std::sqrt(sP->fluctu_sqrt)/vMax;
      }
    }

    // #if yee_cst
    //   for (int i = 1; i < (len_x_cst+3); i++){
    //     for (int j = 1; j < (len_y_cst+3); j++){
    //
    //       B->B_s[0][i][j] = B->B[0][i  ][j  ] ;//* .5 +
    //                         // B->B[0][i-1][j  ] ;//* .5;
    //       B->B_s[1][i][j] = B->B[1][i  ][j  ] ;//* .5 +
    //                         // B->B[1][i  ][j-1] ;//* .5;
    //       B->B_s[2][i][j] = B->B[2][i  ][j  ];
    //     }
    //   }
    //   for (int i = 0; i < (len_x_cst+4); i++){
    //     for (int j = 0; j < (len_y_cst+4); j++){
    //
    //       //
    //       // B->B[0][i][j] = 0.;//B->B_s[0][i  ][j  ] * .5 +
    //       //                 // B->B_s[0][i+1][j  ] * .5;
    //       // B->B[1][i][j] = 0.;// B->B_s[1][i  ][j  ] * .5 +
    //       //                 // B->B_s[1][i  ][j+1] * .5;
    //       // B->B[2][i][j] = 0.;//B->B_s[2][i][j];
    //     }
    //   }
    // #endif

    //__________________________________________________________________________
    // Particles:
    int* indices = new int [pool_size_cst];
    int n;
    /// Simple range.
    for (int i=0; i<pool_size_cst; i++){
      indices[i] = i;
    }
    shuffleOwn(indices, pool_size_cst);

    float vx = 0, vy = 0, vz = 0;
    float sigma = 1./sqrt(2) * sP->v_thi/sP->v_A;
    unsigned long int idx_lin;
    unsigned long int idx_shuf;

    for (uint i = 0; i < len_x_cst; i++){
      for (uint j = 0; j < len_y_cst; j++){
        vx = 0; vy = 0; vz = 0;
        for (uint n = 0; n < nb_part_node_cst; n++ ){
          idx_lin = (i*uint(len_y_cst) + j)*uint(nb_part_node_cst) + n;
          idx_shuf = idx_lin;// indices[idx_lin];
          p->rx[idx_shuf] = (float(rand())/float(RAND_MAX) + i) * sP->dX  + grid->xMin;
          p->ry[idx_shuf] = (float(rand())/float(RAND_MAX) + j) * sP->dX  + grid->yMin;
          p->vx[idx_shuf] = normalSample(0., sigma);
          p->vy[idx_shuf] = normalSample(0., sigma);
          p->vz[idx_shuf] = normalSample(0., sigma);
          p->active[idx_shuf] = true;
          vx += p->vx[idx_shuf];
          vy += p->vy[idx_shuf];
          vz += p->vz[idx_shuf];
        }
        for (int n = 0; n < nb_part_node_cst; n++ ){
          idx_lin = (i*len_y_cst + j)*nb_part_node_cst + n;
          idx_shuf = idx_lin; //indices[idx_lin];
          p->vx[idx_shuf] -= vx/nb_part_node_cst - deltaV[0][i][j];
          p->vy[idx_shuf] -= vy/nb_part_node_cst - deltaV[1][i][j];
          p->vz[idx_shuf] -= vz/nb_part_node_cst;

        }
      }
    }

    for (uint idx_part=idx_lin; idx_part<pool_size_cst; idx_part++){
      /* Had to be done after changing compiler, for one which doesn't initialise anything to zero or false*/
      p->active[idx_part] = false;
    }



  #elif NB_DIM==3

    //__________________________________________________________________________
    // Fields:
    float kx; float ky; float kz; float kAmp;
    // float xx; float yy; float alpha;
    float phiB; float phiV;
    float delta;
    int kLen = 9;
    int kLen_z = 5;
    float k_mod [kLen];
    float k_mod_z [kLen_z];
    float k0 = (2*PI)/(len_x_cst*sP->dX);
    float k0_z = (2*PI)/(sP->mpi_nb_proc_z*len_z_cst*sP->dX);
    // float kMax = kLen*k0;
    float deltaV [3][len_x_cst+4][len_y_cst+4][len_z_cst+4];
    float dbx, dby, dbz;
    float bPerpMax = 0, vMax = 0;
    float bMag, vMag;

    float phiBB[kLen*kLen*kLen_z];  /// Phases along the k-direction.
    float phiVV[kLen*kLen*kLen_z];  ///
    if (sP->mpi_rank_lin==0){
      for (int kk=0; kk<kLen*kLen*kLen_z; kk++){
        phiBB[kk] = float(rand())/float(RAND_MAX);
        phiVV[kk] = float(rand())/float(RAND_MAX);
      }
    }
    MPI_Bcast(&phiBB, kLen*kLen*kLen_z, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&phiVV, kLen*kLen*kLen_z, MPI_DOUBLE, 0, comm);

    for (int i = 0; i < (len_x_cst+4); i++){
      for (int j = 0; j < (len_y_cst+4); j++){
        for (int k = 0; k < (len_z_cst+4); k++){
          deltaV[0][i][j][k] = 0.;
          deltaV[1][i][j][k] = 0.;
          deltaV[2][i][j][k] = 0.;
        }
      }
    }

    for (int ii=0; ii<kLen; ii++){
      k_mod[ii] = (-4+ii)*k0;
    }

    for (int ii=0; ii<kLen_z; ii++){
      k_mod_z[ii] = (-2+ii)*k0_z;
    }
    for (int g = 0; g < kLen; g++){
      kx = k_mod[g];
      for (int h = 0; h < kLen; h++){
        ky = k_mod[h];
        for (int l = 0; l < kLen_z; l++){
          kz = k_mod_z[l];
          if ((kx!=0.) || (ky!=0.) || (kz!=0.)){ //
            kAmp = std::sqrt(kx*kx + ky*ky + kz*kz);
            // printf("%.4e\n", kAmp);
            //
            phiB = 2.*PI*phiBB[g*kLen*kLen_z + h*kLen_z + l]; //float(rand())/float(RAND_MAX)*2*PI;
            phiV = 2.*PI*phiVV[g*kLen*kLen_z + h*kLen_z + l]; //float(rand())/float(RAND_MAX)*2*PI;
            //
            for (int i = 0; i < (len_x_cst+4); i++){
              for (int j = 0; j < (len_y_cst+4); j++){
                for (int k = 0; k < (len_z_cst+4); k++){
                  // printf("%i %i %i\n",)
                  //
                  delta = std::cos(kx*grid->xGrid[i] + ky*grid->yGrid[j] + kz*grid->zGrid[k] + phiB); // delta between -1 and 1
                  B->B[0][i][j][k] += (ky*sP->B0_z-kz*sP->B0_y)/kAmp * delta;
                  B->B[1][i][j][k] += (kz*sP->B0_x-kx*sP->B0_z)/kAmp * delta;
                  B->B[2][i][j][k] += (kx*sP->B0_y-ky*sP->B0_x)/kAmp * delta;

                  delta = std::cos(kx*grid->xGrid[i] + ky*grid->yGrid[j] + kz*grid->zGrid[k] + phiV);
                  deltaV[0][i][j][k] += (ky*sP->B0_z-kz*sP->B0_y)/kAmp * delta;
                  deltaV[1][i][j][k] += (kz*sP->B0_x-kx*sP->B0_z)/kAmp * delta;
                  deltaV[2][i][j][k] += (kx*sP->B0_y-ky*sP->B0_x)/kAmp * delta;
                }
              }
            }
          }
        }
      }
    }

    for (int i = 0; i < (len_x_cst+4); i++){
      for (int j = 0; j < (len_y_cst+4); j++){
        for (int k = 0; k < (len_z_cst+4); k++){
          bMag = std::sqrt(  B->B[0][i][j][k]*B->B[0][i][j][k]
                           + B->B[1][i][j][k]*B->B[1][i][j][k]
                           + B->B[2][i][j][k]*B->B[2][i][j][k]);
          vMag = std::sqrt(  deltaV[0][i][j][k]*deltaV[0][i][j][k]
                           + deltaV[1][i][j][k]*deltaV[1][i][j][k]
                           + deltaV[2][i][j][k]*deltaV[2][i][j][k]);
          if (bMag>bPerpMax){
            bPerpMax = bMag;
          }
          if (vMag>vMax){
            vMax = vMag;
          }
        }
      }
    }

    float send_max = bPerpMax;
    float rece_max;
    MPI_Allreduce(&send_max, &rece_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    bPerpMax = rece_max;

    send_max = vMax;
    MPI_Allreduce(&send_max, &rece_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    vMax = rece_max;

    for (int i = 0; i < (len_x_cst+4); i++){
      for (int j = 0; j < (len_y_cst+4); j++){
        for (int k = 0; k < (len_z_cst+4); k++){
          B->B[0][i][j][k]   *=  std::sqrt(sP->fluctu_sqrt)/bPerpMax;
          B->B[1][i][j][k]   *=  std::sqrt(sP->fluctu_sqrt)/bPerpMax;
          B->B[2][i][j][k]   *=  std::sqrt(sP->fluctu_sqrt)/bPerpMax;
          // std::cout << B->B[0][i][j][k] << " " << B->B[1][i][j][k] << std::endl;
          B->B[0][i][j][k]   += sP->B0_x;
          B->B[1][i][j][k]   += sP->B0_y;
          B->B[2][i][j][k]   += sP->B0_z;
          deltaV[0][i][j][k] *=  std::sqrt(sP->fluctu_sqrt)/vMax;
          deltaV[1][i][j][k] *=  std::sqrt(sP->fluctu_sqrt)/vMax;
          deltaV[2][i][j][k] *=  std::sqrt(sP->fluctu_sqrt)/vMax;
        }
      }
    }

    #if yee_cst
      for (int i = 0; i < (len_x_cst+4); i++){
        for (int j = 0; j < (len_y_cst+4); j++){
          for (int k = 0; k < (len_z_cst+4); k++){
            B->B_s[0][i][j][k] = B->B[0][i][j][k];
            B->B_s[1][i][j][k] = B->B[1][i][j][k];
            B->B_s[2][i][j][k] = B->B[2][i][j][k];
            B->B_s[0][i][j][k] = B->B[0][i  ][j  ][k  ] * .25 +
                                 B->B[0][i  ][j  ][k-1] * .25 +
                                 B->B[0][i  ][j-1][k  ] * .25 +
                                 B->B[0][i  ][j-1][k-1] * .25;
            B->B_s[1][i][j][k] = B->B[1][i  ][j  ][k  ] * .25 +
                                 B->B[1][i  ][j  ][k-1] * .25 +
                                 B->B[1][i-1][j  ][k  ] * .25 +
                                 B->B[1][i-1][j  ][k-1] * .25;
            B->B_s[2][i][j][k] = B->B[2][i  ][j  ][k  ] * .25 +
                                 B->B[2][i  ][j-1][k  ] * .25 +
                                 B->B[2][i-1][j  ][k  ] * .25 +
                                 B->B[2][i-1][j-1][k  ] * .25;
          }
        }
      }
    #endif


    //__________________________________________________________________________
    // Particles:
    int idx_lin;
    float vx = 0, vy = 0, vz = 0;
    float sigma = 1./sqrt(2) * sP->v_thi/sP->v_A;

    int* indices = new int [pool_size_cst];
    /// Simple range.
    for (int i=0; i<pool_size_cst; i++){
      indices[i] = i;
    }
    shuffleOwn(indices, pool_size_cst);
    int idx_shuf;

    for (int i = 0; i < len_x_cst; i++){
      for (int j = 0; j < len_y_cst; j++){
        for (int k = 0; k < len_z_cst; k++){
          vx = 0; vy = 0; vz = 0;
          for (int n = 0; n < nb_part_node_cst; n++ ){
            idx_lin = (i*(len_y_cst*len_z_cst) + j*len_z_cst + k)*nb_part_node_cst + n;
            idx_shuf = indices[idx_lin];
            p->rx[idx_shuf] = grid->xGrid[i+2] + (float(rand())/float(RAND_MAX) - .5) * sP->dX ;
            p->ry[idx_shuf] = grid->yGrid[j+2] + (float(rand())/float(RAND_MAX) - .5) * sP->dX ;
            p->rz[idx_shuf] = grid->zGrid[k+2] + (float(rand())/float(RAND_MAX) - .5) * sP->dX ;
            p->vx[idx_shuf] = normalSample(sP->v_obs, sigma);
            p->vy[idx_shuf] = normalSample(0., sigma);
            p->vz[idx_shuf] = normalSample(0., sigma);
            p->active[idx_shuf] = true;
            vx += p->vx[idx_shuf];
            vy += p->vy[idx_shuf];
            vz += p->vz[idx_shuf];
          }
          for (int n = 0; n < nb_part_node_cst; n++ ){
            idx_lin = (i*len_y_cst*len_z_cst + j*len_z_cst + k)*nb_part_node_cst + n;
            idx_shuf = indices[idx_lin];
            p->vx[idx_shuf] -= vx/nb_part_node_cst - deltaV[0][i][j][k];
            p->vy[idx_shuf] -= vy/nb_part_node_cst - deltaV[1][i][j][k];
            p->vz[idx_shuf] -= vz/nb_part_node_cst;//- deltaV[2][i][j][k];
          }
        }
      }
    }

    for (uint idx_part=idx_lin; idx_part<pool_size_cst; idx_part++){
      /* Had to be done after changing compiler, for one which doesn't initialise anything to zero or false*/
      p->active[idx_part] = false;
    }

  #endif

}

void init_exosphere(buff_part* b_p, simu_grid* grid, simu_param* sP, MPI_Comm comm){

  #if NB_DIM==2
    int i, j;
    /** profile will first contain a 1/r**2 profile. The sum of this profile (nb_part_tot_physical)
    over the grid is then used together with the number of planetary ions to be
    added (nb_part_add_pla_cst) to normalise this profile (second nested for-loops).
    It then contains floats inidcating the amount of ions to add at each time,
    with the decimals indicating the probability to add (or not) one additional ion.*/
    float profile[len_x_cst][len_y_cst] = { 0. };
    float nb_part_tot_physical = 0.;
    // For grid subdivision:
    float xx = 0.;
    float yy = 0.;
    float rsq = 0.;
    int len_sub = 100;
    float dx = sP->dX/len_sub;
    float r_peak = (1.e4*1.e4)/(sP->d_i*sP->d_i); // r of iono peak.
    /** 1/r**2 profile */
    for (i=0; i<len_x_cst; i++){
      for (j=0; j<len_y_cst; j++){
        if (grid->rsq[i+2][j+2]>(sP->dX*sP->dX)){
          profile[i][j] = 1/grid->rsq[i+2][j+2];
        }
        else{
          for (int ii=0; ii<len_sub; ii++){
            for (int jj=0; jj<len_sub; jj++){
              xx = (grid->xGrid[i+2] - grid->centre_x) - .5*sP->dX + (ii + .5)*dx ;
              yy = (grid->yGrid[j+2] - grid->centre_y) - .5*sP->dX + (jj + .5)*dx ;
              rsq = xx*xx + yy*yy;
              rsq = std::max( rsq, r_peak ); // Ionospheric peak.
              profile[i][j] += 1/rsq;
            }
          }
          profile[i][j] /= (len_sub*len_sub);
          /** To check the convergence of this sub-division, len_sub can be increased
          from 1 to any (e.g. [1, 10, 100, 1000]), and check the local profile evolution: */
          // std::cout << profile[i][j] << " " << 1/grid->rsq[i+2][j+2] << std::endl;
        }

        nb_part_tot_physical += profile[i][j];
      }
    }

    /** Normalisation of the profile ==> sum(profile)=nb_part_add_pla_cst .
    For one MPI process, we need to communicate and use the global MINIMUM value
    calculated locally on each process, which corresponds to the process with
    the most particles to add, i.e. the "most dense". */
    float norm_fact = nb_part_add_pla_cst/nb_part_tot_physical;
    //
    struct {
      float nor_fac;
      int rank;
    } send_min, rece_min;
    //
    send_min.nor_fac = norm_fact;
    send_min.rank = sP->mpi_rank_lin;
    //
    MPI_Allreduce(&send_min, &rece_min, 1, MPI_FLOAT_INT, MPI_MINLOC, comm);
    norm_fact = rece_min.nor_fac;
    int rank_to_copy = rece_min.rank;
    float profile_max = 0.;
    int i_max, j_max, k_max;
    //
    for (i=0; i<len_x_cst; i++){
      for (j=0; j<len_y_cst; j++){
        profile[i][j] *= norm_fact;
        if (profile[i][j]>profile_max){
          profile_max = profile[i][j];
          i_max = i;
          j_max = j;
        }
      }
    }
    // Ignoring previous i_max, j_max for manual definition:
    i_max = sP->centre_x*len_x_cst; // Same x as the centre of the exosphere.
    j_max = len_y_cst*.9;


    /** The creation rate q_i is computed at one single and arbitrary position in the grid,
    using the (normalised) distance to the nucleus, the neutral density n_n.
    This creation rate and the local amount of macro particles to be added at the same
    single and arbitrary position in the grid provide the weight w_pla of the planetary
    ions. */
    rsq = grid->rsq[i_max+2][j_max+2]*sP->d_i*sP->d_i;
    float n_n = ( sP->Q/(4.*PI*sP->u0*rsq) );
    n_n /= sP->n0_SI;
    float q_i = n_n * sP->nu_i*sP->t0; // Normalised.
    sP->w_pla = q_i*sP->dt/profile[i_max][j_max];
    MPI_Bcast( &sP->w_pla, 1, MPI_FLOAT, rank_to_copy, comm);

    /** We now fill the exosphere "model" of the particle buffer, in a 1D array
    indicating for each particle to be added a couple of coordinates (i, j) and
    the additional probability to add one more particle there. */
    int idx_tot = 0;
    int idx_lin = 0;
    float whole, proba;

    for (i=0; i<len_x_cst; i++){
      for (j=0; j<len_y_cst; j++){
        proba = std::modf(profile[i][j], &whole);
        for (int n=0; n<int(whole); n++){
          b_p->exosphere[0][idx_tot] = i+2;
          b_p->exosphere[1][idx_tot] = j+2;
          idx_tot++;
          #if debug_mode_cst
            if (idx_tot>nb_part_add_pla_cst){
              std::cout << "| Out of exosphere (init, rank i j) " << sP->mpi_rank_lin << " " << i << " " << j << std::endl;
            }
            if (b_p->exosphere[1][idx_tot]>len_y_cst+2){
              std::cout << "| Particle out of the domain (exopshere init, rank i j) " << sP->mpi_rank_lin << " " << i << " " << j << std::endl;
            }
          #endif
        }
        b_p->exo_proba[i*len_y_cst+j] = proba;
        sP->sum_pla_proba += proba;
      }
    }
    sP->nb_part_add_pla = idx_tot;

  #elif NB_DIM==3

    int i, j, k;
    /** profile will first contain a 1/r**2 profile. The sum of this profile (nb_part_tot_physical)
    over the grid is then used together with the number of planetary ions to be
    added (nb_part_add_pla_cst) to normalise this profile (second nested for-loops).
    It then contains floats inidcating the amount of ions to add at each time,
    with the decimals indicating the probability to add (or not) one additional ion.*/
    float profile[len_x_cst][len_y_cst][len_z_cst];
    float nb_part_tot_physical;
    float xx = 0.;
    float yy = 0.;
    float zz = 0.;
    float rsq = 0.;
    int len_sub = 20;
    float dx = sP->dX/len_sub;
    float r_peak = (1.e4*1.e4)/(sP->d_i*sP->d_i); // r of iono peak.
    /** 1/r**2 profile */
    for (i=0; i<len_x_cst; i++){
      for (j=0; j<len_y_cst; j++){
        for (k=0; k<len_z_cst; k++){
        // if (grid->rsq[i+2][j+2]>0){
          // profile[i][j][k] = 1/grid->rsq[i+2][j+2][k+2];

          if (grid->rsq[i+2][j+2][k+2]>(sP->dX*sP->dX)){
            profile[i][j][k] = 1/grid->rsq[i+2][j+2][k+2];
          }
          else{
            for (int ii=0; ii<len_sub; ii++){
              for (int jj=0; jj<len_sub; jj++){
                for (int kk=0; kk<len_sub; kk++){
                  xx = (grid->xGrid[i+2] - grid->centre_x) - .5*sP->dX + (ii + .5)*dx ;
                  yy = (grid->yGrid[j+2] - grid->centre_y) - .5*sP->dX + (jj + .5)*dx ;
                  zz = (grid->zGrid[k+2] - grid->centre_z) - .5*sP->dX + (kk + .5)*dx ;
                  rsq = xx*xx + yy*yy + zz*zz;
                  rsq = std::max( rsq, r_peak ); // Ionospheric peak.
                  profile[i][j][k] += 1/rsq;
                }
              }
            }
            profile[i][j][k] /= (len_sub*len_sub);
            /** To check the convergence of this sub-division, len_sub can be increased
            from 1 to any (e.g. [1, 10, 100, 1000]), and check the local profile evolution: */
            // std::cout << profile[i][j] << " " << 1/grid->rsq[i+2][j+2] << std::endl;
          }

          nb_part_tot_physical += profile[i][j][k];
        // }
        }
      }
    }
    /** Normalisation of the profile ==> sum(profile)=nb_part_add_pla_cst .
    For one MPI process, we need to communicate and use the global MINIMUM value
    calculated locally on each process. */
    float norm_fact = nb_part_add_pla_cst/nb_part_tot_physical;
    //
    struct {
      float nor_fac;
      int rank;
    } send_min, rece_min;
    //
    send_min.nor_fac = norm_fact;
    send_min.rank = sP->mpi_rank_lin;
    //
    MPI_Allreduce(&send_min, &rece_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, comm);
    norm_fact = rece_min.nor_fac;
    int rank_to_copy = rece_min.rank;
    float profile_max = 0.;
    int i_max, j_max, k_max;
    //
    for (i=0; i<len_x_cst; i++){
      for (j=0; j<len_y_cst; j++){
        for (k=0; k<len_z_cst; k++){
          profile[i][j][k] *= norm_fact;
          if (profile[i][j][k]>profile_max){
            profile_max = profile[i][j][k];
            i_max = i;
            j_max = j;
            k_max = k;
          }
        }
      }
    }



    /** The creation rate q_i is computed at one single and arbitrary position in the grid,
    using the (normalised) distance to the nucleus, the neutral density n_n.
    This creation rate and the local amount of macro particles to be added at the same
    single and arbitrary position in the grid provide the weight w_pla of the planetary
    ions. */
    rsq = grid->rsq[i_max+2][j_max+2][k_max+2]*sP->d_i*sP->d_i;
    float n_n = ( sP->Q/(4.*PI*sP->u0*rsq) );
    n_n /= sP->n0_SI;
    float q_i = n_n * sP->nu_i*sP->t0;
    sP->w_pla = q_i*sP->dt/profile[i_max][j_max][k_max];
    MPI_Bcast( &sP->w_pla, 1, MPI_DOUBLE, rank_to_copy, comm);
    // std::cout << sP->w_pla << " " << q_i << " " << std::endl;
    // sortie();

    /** We now fill the exosphere "model" of the particle buffer, in a 1D array
    indicating for each particle to be added the coordinates (i, j, k) and
    the additional probability to add one more particle there. */
    int idx_tot = 0;
    int idx_lin = 0;
    float whole, proba;

    for (i=0; i<len_x_cst; i++){
      for (j=0; j<len_y_cst; j++){
        for (k=0; k<len_z_cst; k++){
          proba = std::modf(profile[i][j][k], &whole);
          for (int n=0; n<int(whole); n++){
            b_p->exosphere[0][idx_tot] = i+2;
            b_p->exosphere[1][idx_tot] = j+2;
            b_p->exosphere[2][idx_tot] = k+2;
            idx_tot++;
          }
          b_p->exo_proba[i*len_y_cst*len_z_cst + j*len_z_cst + k] = proba;
          sP->sum_pla_proba += proba;
        }
      }
    }
    sP->nb_part_add_pla = idx_tot;


  #endif

}

void init_ionosphere(buff_part* b_p, simu_grid* grid, simu_param* sP, MPI_Comm comm){

  #if NB_DIM==2

    int i, j;
    /** profile will first contain a Chapman profile. The sum of this profile (nb_part_tot_physical)
    over the grid is then used together with the number of planetary ions to be
    added (nb_part_add_pla_cst) to normalise this profile (second nested for-loops).
    It then contains floats inidcating the amount of ions to add at each time,
    with the decimals indicating the probability to add (or not) one additional ion.*/
    float profile[len_x_cst][len_y_cst] = { 0. };
    float nb_part_tot_physical = 0.;
    float r, h; // Radius, height above body's surface.
    float xi, y; // Solar Zenith Angle, distance from ionospheric peak, normalised by scale height.
    /** Chapman profile */
    for (i=0; i<len_x_cst; i++){
      for (j=0; j<len_y_cst; j++){

        r = std::sqrt(grid->rsq[i+2][j+2]);
        h = r - sP->r_obs;
        xi = std::acos((grid->xGrid[i+2]-grid->centre_x)/r);
        y = (h-(sP->h0/sP->d_i))/(sP->H/sP->d_i);

        if ((grid->xGrid[i+2]-grid->centre_x)>0.){
          profile[i][j] = std::exp(1-y-std::exp(-y)/std::cos(xi));
        }

        if ((r > sP->r_obs) && (profile[i][j]>1e-10)){
          nb_part_tot_physical += profile[i][j];
        }
        else{
          profile[i][j] = 0.;
        }
      }
    }

    /** Normalisation of the profile ==> sum(profile)=nb_part_add_pla_cst .
    For one MPI process, we need to communicate and use the global MINIMUM value
    calculated locally on each process. */
    float norm_fact;
    if (nb_part_tot_physical>0){
      norm_fact = nb_part_add_pla_cst/nb_part_tot_physical;
    }
    else{
      norm_fact = 999999.; // Necessary to not account for ranks not adding particles.
    }
    //
    struct {
      float nor_fac;
      int rank;
    } send_min, rece_min;
    //
    send_min.nor_fac = norm_fact;
    send_min.rank = sP->mpi_rank_lin;
    //
    MPI_Allreduce(&send_min, &rece_min, 1, MPI_FLOAT_INT, MPI_MINLOC, comm);
    norm_fact = rece_min.nor_fac;
    int rank_to_copy = rece_min.rank;
    if (norm_fact==0){
      std::cout << "| norm_fact=0, not ok." << std::endl;
      sortie();
    }
    //
    float profile_max = 0.;
    int i_max, j_max;
    for (i=0; i<len_x_cst; i++){
      for (j=0; j<len_y_cst; j++){
        profile[i][j] *= norm_fact;
        if (profile[i][j]>profile_max){
          profile_max = profile[i][j];
          i_max = i;
          j_max = j;
        }
      }
    }


    /** The creation rate q_i is computed at the position of highest creation rate in the grid,
    using the (normalised) distance to the nucleus, the neutral density n_n.
    This creation rate and the local amount of macro particles to be added at the same
    single and arbitrary position in the grid provide the weight w_pla of the planetary
    ions. */
    if (profile_max>0.){
      r = std::sqrt(grid->rsq[i_max+2][j_max+2])*sP->d_i;
      h = r - sP->r_obs_SI;
      xi = std::acos((grid->xGrid[i_max+2]-grid->centre_x)/(r/sP->d_i));
      y = (h-sP->h0)/sP->H;
      float q_i = sP->prod0*sP->t0/sP->n0_SI * std::exp( 1 - y - std::exp(-y)/std::cos(xi) );
      sP->w_pla = q_i*sP->dt/profile_max;
    }
    MPI_Bcast( &sP->w_pla, 1, MPI_FLOAT, rank_to_copy, comm);

    /** We now fill the exosphere "model" of the particle buffer, in a 1D array
    indicating for each particle to be added the coordinates (i, j, k) and
    the additional probability to add one more particle there. */
    int idx_tot = 0;
    int idx_lin = 0;
    float whole, proba;

    for (i=0; i<len_x_cst; i++){
      for (j=0; j<len_y_cst; j++){
        proba = std::modf(profile[i][j], &whole);
        for (int n=0; n<int(whole); n++){
          b_p->exosphere[0][idx_tot] = i+2;
          b_p->exosphere[1][idx_tot] = j+2;
          idx_tot++;
        }
        b_p->exo_proba[i*len_y_cst + j] = proba;
        sP->sum_pla_proba += proba;
      }
    }
    sP->nb_part_add_pla = idx_tot;




  #elif NB_DIM==3

    int i, j, k;
    /** profile will first contain a Chapman profile. The sum of this profile (nb_part_tot_physical)
    over the grid is then used together with the number of planetary ions to be
    added (nb_part_add_pla_cst) to normalise this profile (second nested for-loops).
    It then contains floats inidcating the amount of ions to add at each time,
    with the decimals indicating the probability to add (or not) one additional ion.*/
    float profile[len_x_cst][len_y_cst][len_z_cst] = { 0. };
    float nb_part_tot_physical = 0.;
    float r, h; // Radius, height above body's surface.
    float xi, y; // Solar Zenith Angle, distance from ionospheric peak, normalised by scale height.
    /** Chapman profile */
    for (i=0; i<len_x_cst; i++){
      for (j=0; j<len_y_cst; j++){
        for (k=0; k<len_z_cst; k++){

          r = std::sqrt(grid->rsq[i+2][j+2][k+2]);
          h = r - sP->r_obs;
          xi = std::acos((grid->xGrid[i+2]-grid->centre_x)/r);
          y = (h-(sP->h0/sP->d_i))/(sP->H/sP->d_i);

          if ((grid->xGrid[i+2]-grid->centre_x)>0.){
            profile[i][j][k] = std::exp(1-y-std::exp(-y)/std::cos(xi));
          }

          if ((r > sP->r_obs) && (profile[i][j][k]>1e-10)){
            nb_part_tot_physical += profile[i][j][k];
          }
          else{
            profile[i][j][k] = 0.;
          }

        }
      }
    }


    /** Normalisation of the profile ==> sum(profile)=nb_part_add_pla_cst .
    For one MPI process, we need to communicate and use the global MINIMUM value
    calculated locally on each process. */
    float norm_fact;
    if (nb_part_tot_physical>0){
      norm_fact = nb_part_add_pla_cst/nb_part_tot_physical;
    }
    else{
      norm_fact = 999999.; // Necessary to not account for ranks not adding particles.
    }
    //
    struct {
      float nor_fac;
      int rank;
    } send_min, rece_min;
    //
    send_min.nor_fac = norm_fact;
    send_min.rank = sP->mpi_rank_lin;
    //
    MPI_Allreduce(&send_min, &rece_min, 1, MPI_FLOAT_INT, MPI_MINLOC, comm);
    norm_fact = rece_min.nor_fac;
    int rank_to_copy = rece_min.rank;
    if (norm_fact==0){
      std::cout << "| norm_fact=0, not ok." << std::endl;
      sortie();
    }
    //
    float profile_max = 0.;
    int i_max, j_max, k_max;
    for (i=0; i<len_x_cst; i++){
      for (j=0; j<len_y_cst; j++){
        for (k=0; k<len_z_cst; k++){
          profile[i][j][k] *= norm_fact;
          if (profile[i][j][k]>profile_max){
            profile_max = profile[i][j][k];
            i_max = i;
            j_max = j;
            k_max = k;
          }
        }
      }
    }


    /** The creation rate q_i is computed at the position of highest creation rate in the grid,
    using the (normalised) distance to the nucleus, the neutral density n_n.
    This creation rate and the local amount of macro particles to be added at the same
    single and arbitrary position in the grid provide the weight w_pla of the planetary
    ions. */
    if (profile_max>0.){
      r = std::sqrt(grid->rsq[i_max+2][j_max+2][k_max+2])*sP->d_i;
      h = r - sP->r_obs_SI;
      xi = std::acos((grid->xGrid[i_max+2]-grid->centre_x)/(r/sP->d_i));
      y = (h-sP->h0)/sP->H;
      float q_i = sP->prod0*sP->t0/sP->n0_SI * std::exp( 1 - y - std::exp(-y)/std::cos(xi) );
      sP->w_pla = q_i*sP->dt/profile_max;
    }
    MPI_Bcast( &sP->w_pla, 1, MPI_FLOAT, rank_to_copy, comm);

    /** We now fill the exosphere "model" of the particle buffer, in a 1D array
    indicating for each particle to be added the coordinates (i, j, k) and
    the additional probability to add one more particle there. */
    int idx_tot = 0;
    int idx_lin = 0;
    float whole, proba;

    for (i=0; i<len_x_cst; i++){
      for (j=0; j<len_y_cst; j++){
        for (k=0; k<len_z_cst; k++){
          proba = std::modf(profile[i][j][k], &whole);
          for (int n=0; n<int(whole); n++){
            b_p->exosphere[0][idx_tot] = i+2;
            b_p->exosphere[1][idx_tot] = j+2;
            b_p->exosphere[2][idx_tot] = k+2;
            idx_tot++;
          }
          b_p->exo_proba[i*len_y_cst*len_z_cst + j*len_z_cst + k] = proba;
          sP->sum_pla_proba += proba;
        }
      }
    }
    sP->nb_part_add_pla = idx_tot;


  #endif

}





#if NB_DIM==2
  void load_B_field(simu_B_field* B, simu_tank* tank, simu_param* sP,
                    bool fill_tank, std::string path_inputs, int idx_load)
  {

    std::string file_name = path_inputs + "/B_it"
                            + std::to_string(idx_load) + "_rank_"
                            + std::to_string(sP->mpi_rank_y) + "_"
                            + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::NpyArray arr = cnpy::npy_load(file_name);
    float* BFlat = arr.data<float>();
    int h; int i; int j;

    for (int flatInd=0; flatInd<3*(len_x_cst+4)*(len_y_cst+4); flatInd++){
      h = floor(flatInd/((len_x_cst+4)*(len_y_cst+4)));
      i = floor((flatInd - h*(len_x_cst+4)*(len_y_cst+4))/((len_y_cst+4)));
      j = flatInd - h*(len_x_cst+4)*(len_y_cst+4) - i*(len_y_cst+4);
      B->B[h][i][j] = BFlat[flatInd];
      if (fill_tank){
        tank->B[h][i][j] = BFlat[flatInd];
      }
    }

    #if yee_cst
      file_name = path_inputs + "/B_s_it"
                              + std::to_string(idx_load) + "_rank_"
                              + std::to_string(sP->mpi_rank_y) + "_"
                              + std::to_string(sP->mpi_rank_z) + ".npy";
      arr = cnpy::npy_load(file_name);

      for (int flatInd=0; flatInd<3*(len_x_cst+4)*(len_y_cst+4); flatInd++){
        h = floor(flatInd/((len_x_cst+4)*(len_y_cst+4)));
        i = floor((flatInd - h*(len_x_cst+4)*(len_y_cst+4))/((len_y_cst+4)));
        j = flatInd - h*(len_x_cst+4)*(len_y_cst+4) - i*(len_y_cst+4);
        B->B_s[h][i][j] = BFlat[flatInd];
        if (fill_tank){
          tank->B[h][i][j] = BFlat[flatInd];
        }
      }
    #endif

  }

  void load_fields(simu_fields* fields, simu_tank* tank, simu_param* sP,
                    bool fill_tank, std::string path_inputs, int idx_load)
  {

    std::string file_name = path_inputs + "/curr_it"
                            + std::to_string(idx_load) + "_rank_"
                            + std::to_string(sP->mpi_rank_y) + "_"
                            + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::NpyArray arrCurr = cnpy::npy_load(file_name);
    float* currFlat = arrCurr.data<float>();

    file_name = path_inputs + "/dens_it"
                + std::to_string(idx_load) + "_rank_"
                + std::to_string(sP->mpi_rank_y) + "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::NpyArray arrDens = cnpy::npy_load(file_name);
    float* densFlat = arrDens.data<float>();

    file_name = path_inputs + "/E_it"
                            + std::to_string(idx_load) + "_rank_"
                            + std::to_string(sP->mpi_rank_y) + "_"
                            + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::NpyArray arrE = cnpy::npy_load(file_name);
    float* EFlat = arrE.data<float>();
    int h; int i; int j;

    for (int flatInd=0; flatInd<3*(len_x_cst+4)*(len_y_cst+4); flatInd++){
      h = floor(flatInd/((len_x_cst+4)*(len_y_cst+4)));
      i = floor((flatInd - h*(len_x_cst+4)*(len_y_cst+4))/((len_y_cst+4)));
      j = flatInd - h*(len_x_cst+4)*(len_y_cst+4) - i*(len_y_cst+4);
      fields->Ji[h][i][j] = currFlat[flatInd];
      fields->E[h][i][j]  = EFlat[flatInd];
      if (fill_tank){
        tank->Ji[h][i][j]   = currFlat[flatInd];
        tank->E[h][i][j]    = EFlat[flatInd];
      }
    }

    for (int flatInd=0; flatInd<(len_x_cst+4)*(len_y_cst+4); flatInd++){
      i = floor(flatInd/((len_y_cst+4)));
      j = flatInd - i*(len_y_cst+4);
      fields->density[i][j] = densFlat[flatInd];
      if (fill_tank){
        tank->density[i][j] = densFlat[flatInd];
      }
    }

  }

  void load_particles(particles* p, simu_grid* grid, simu_tank* tank, simu_param* sP, MPI_Comm comm,
                      bool fill_tank, std::string path_inputs, int idx_load, bool old_file)
  {

    //__________________________________________________________________________
    // How many files to open?.
    //
    std::string file_name = path_inputs + "/nb_particle_files_it"
                            + std::to_string(idx_load) + "_rank_"
                            + std::to_string(sP->mpi_rank_y) + "_"
                            + std::to_string(sP->mpi_rank_z) + ".txt";
    std::string line;
    int nb_files;
    std::ifstream myfile (file_name);
    if (myfile.is_open()){
      getline (myfile, line);
      nb_files = std::stoi(line);
      myfile.close();
    }
    else {
      std::cout << "Couldn't open file " << file_name << std::endl;
    }
    std::cout << "| nb_files: " << nb_files << std::endl;

    //__________________________________________________________________________
    // How many particles TOTAL?
    //
    sP->nb_part_tank = 0; // If load_particles called for the econd time.
    for (int idx_f=0; idx_f<nb_files; idx_f++){
      file_name = path_inputs + "/nb_particles_" + std::to_string(idx_f) +  "_it"
                  + std::to_string(idx_load) + "_rank_"
                  + std::to_string(sP->mpi_rank_y) + "_"
                  + std::to_string(sP->mpi_rank_z) + ".txt";
      std::ifstream myfile (file_name);
      if (myfile.is_open()){
        getline(myfile, line);
        sP->nb_part_tank += std::stoi(line);
        myfile.close();
      }
      else {
        std::cout << "\n Couldn't open file in load_particle(): " << file_name << std::endl;
        sortie();
      }
    }
    //
    if (fill_tank){
      if (sP->nb_part_tank>tank_size_cst){
        std::cout << "\n| Too many particles for tank! load_particles()\n";
        sortie();
      }
    }
    if (sP->nb_part_tank>pool_size_cst){
      std::cout << "\n| Too many particles for pool! load_particles()\n";
      sortie();
    }


    //__________________________________________________________________________
    // Filling the particles and the tank.
    //
    int* indices = new int [pool_size_cst];
    int idx_shuf;
    for (int i=0; i<pool_size_cst; i++){
      indices[i] = i; // np.arange(i) !
    }
    shuffleOwn(indices, pool_size_cst);


    int idx_part_tot = 0;
    int nb_part_in_file;

    for (int idx_f=0; idx_f<nb_files; idx_f++){

      std::cout << "| Filling from particle file " << idx_f << std::endl;

      file_name = path_inputs + "/particles_" + std::to_string(idx_f) +  "_it" + std::to_string(idx_load) + "_rank_"
                  + std::to_string(sP->mpi_rank_y) + "_"
                  + std::to_string(sP->mpi_rank_z) + ".npy";
      cnpy::NpyArray arrPSW = cnpy::npy_load(file_name);
      float* pSWFlat = arrPSW.data<float>();

      file_name = path_inputs + "/nb_particles_" + std::to_string(idx_f) +  "_it" + std::to_string(idx_load) + "_rank_"
                  + std::to_string(sP->mpi_rank_y) + "_"
                  + std::to_string(sP->mpi_rank_z) + ".txt";
      std::ifstream myfile (file_name);
      if (myfile.is_open()){
        getline (myfile, line);
        nb_part_in_file = std::stoi(line);
        myfile.close();
      }
      else {
        std::cout << "\n| Couldn't open file in load_particle(): " << file_name << std::endl;
        sortie();
      }

      for (int i=0; i<nb_part_in_file; i++){
        idx_shuf = indices[idx_part_tot];
        if (idx_part_tot<sP->nb_part_tank){
          if (old_file){
            p->rx[idx_part_tot] = pSWFlat[i*5  ];
            p->ry[idx_part_tot] = pSWFlat[i*5+1];
            p->vx[idx_part_tot] = pSWFlat[i*5+2];
            p->vy[idx_part_tot] = pSWFlat[i*5+3];
            p->vz[idx_part_tot] = pSWFlat[i*5+4];
            p->ID[idx_part_tot] = 0;
            p->active[idx_part_tot] = true;
          }
          else{
            p->rx[idx_part_tot] = pSWFlat[i*6  ];
            p->ry[idx_part_tot] = pSWFlat[i*6+1];
            p->vx[idx_part_tot] = pSWFlat[i*6+2];
            p->vy[idx_part_tot] = pSWFlat[i*6+3];
            p->vz[idx_part_tot] = pSWFlat[i*6+4];
            p->ID[idx_part_tot] = pSWFlat[i*6+5];
            p->active[idx_part_tot] = true;
          }
        }
        else {
          p->active[idx_part_tot] = false;
        }
        idx_part_tot++;
      }

    }



    //______________________________________________________________________________________
    // Sorting the particles in the tank.
    /** Here a lambda expression is given to std::sort in order to mimicate np.argsort:
    idx_ordered will contain how the ordered indices of pa_h->rx */
    if (fill_tank){
      tank->idx_ordered = std::vector<size_t>(sP->nb_part_tank, 0);
      std::iota(tank->idx_ordered.begin(), tank->idx_ordered.end(), 0);
      std::sort(tank->idx_ordered.begin(), tank->idx_ordered.end(),
                [&](size_t i1, size_t i2) {return p->rx[i1] < p->rx[i2];} );

      int idx_slice = 0; // Within [0, xLen/nb_cell_per_shift_cst-1]
      int nb_part_in_slice = 0;

      for (int i=0; i<sP->nb_part_tank; i++){
        if (p->rx[tank->idx_ordered[i]]>(grid->xGrid[1+(idx_slice+1)*nb_cell_per_shift_cst] + .5*sP->dX)){
          tank->sliceWidth[idx_slice] = nb_part_in_slice;
          if (nb_part_in_slice>injec_size_cst){
            printf("| Slice to wide for buffer! load_particles() %i %i\n", nb_part_in_slice, injec_size_cst);
            sortie();
          }
          tank->indFirstPartSlice[idx_slice] = i-nb_part_in_slice;
          idx_slice++;
          nb_part_in_slice = 0;
        }
        nb_part_in_slice++;
      }
      /// Still need to fill the information for the last column:
      tank->sliceWidth[idx_slice] = nb_part_in_slice;
      tank->indFirstPartSlice[idx_slice] = sP->nb_part_tank-nb_part_in_slice;

      file_name = "products/HK/slice_width_rank"
                  + std::to_string(sP->mpi_rank_y)+ "_"
                  + std::to_string(sP->mpi_rank_z) + ".npy";
      cnpy::npy_save(file_name, tank->sliceWidth, {(long unsigned int) (len_x_cst)}, "w");

      tank->idx_x = 0;
    }

  }

#elif NB_DIM==3
  void load_B_field(simu_B_field* B, simu_tank* tank, simu_param* sP,
                    bool fill_tank, std::string path_inputs, int idx_load)
  {
    std::string file_name = path_inputs + "/B_it"
                            + std::to_string(idx_load) + "_rank_"
                            + std::to_string(sP->mpi_rank_y) + "_"
                            + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::NpyArray arr = cnpy::npy_load(file_name);
    float* BFlat = arr.data<float>();
    int h, i, j, k;

    for (int flatInd=0; flatInd<3*(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4); flatInd++){
      h = int(flatInd/((len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4)));
      i = int( (flatInd - h*(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4))/((len_y_cst+4)*(len_z_cst+4)) );
      j = int( (flatInd - h*(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4) - i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
      k = flatInd - h*(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4) - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);
      B->B[h][i][j][k] = BFlat[flatInd];
      if (fill_tank){
        tank->B[h][i][j][k] = BFlat[flatInd];
      }
    }

    #if yee_cst
      file_name = path_inputs + "/B_s_it"
                              + std::to_string(idx_load) + "_rank_"
                              + std::to_string(sP->mpi_rank_y) + "_"
                              + std::to_string(sP->mpi_rank_z) + ".npy";
      arr = cnpy::npy_load(file_name);

      for (int flatInd=0; flatInd<3*(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4); flatInd++){
        h = int(flatInd/((len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4)));
        i = int( (flatInd - h*(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4))/((len_y_cst+4)*(len_z_cst+4)) );
        j = int( (flatInd - h*(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4) - i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
        k = flatInd - h*(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4) - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);
        B->B_s[h][i][j][k] = BFlat[flatInd];
        if (fill_tank){
          tank->B[h][i][j][k] = BFlat[flatInd];
        }
      }
    #endif

  }

  void load_fields(simu_fields* fields, simu_tank* tank, simu_param* sP,
                   bool fill_tank, std::string path_inputs, int idx_load)
  {

    std::string file_name = path_inputs + "/curr_it" + std::to_string(idx_load) + "_rank_"
                            + std::to_string(sP->mpi_rank_y) + "_"
                            + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::NpyArray arrCurr = cnpy::npy_load(file_name);
    float* currFlat = arrCurr.data<float>();

    file_name = path_inputs + "/dens_it" + std::to_string(idx_load) + "_rank_"
                + std::to_string(sP->mpi_rank_y) + "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::NpyArray arrDens = cnpy::npy_load(file_name);
    float* densFlat = arrDens.data<float>();

    file_name = path_inputs + "/E_it" + std::to_string(idx_load) + "_rank_"
                + std::to_string(sP->mpi_rank_y) + "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::NpyArray arrE = cnpy::npy_load(file_name);
    float* EFlat = arrE.data<float>();
    int h, i, j, k;

    for (int flatInd=0; flatInd<3*(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4); flatInd++){
      h = int(flatInd/((len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4)));
      i = int( (flatInd - h*(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4))/((len_y_cst+4)*(len_z_cst+4)) );
      j = int( (flatInd - h*(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4) - i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
      k = flatInd - h*(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4) - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);
      fields->Ji[h][i][j][k] = currFlat[flatInd];
      fields->E[h][i][j][k]  = EFlat[flatInd];
      if (fill_tank){
        tank->Ji[h][i][j][k]   = currFlat[flatInd];
        tank->E[h][i][j][k]    = EFlat[flatInd];
      }
    }

    for (int flatInd=0; flatInd<(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4); flatInd++){
      i = int( flatInd/((len_y_cst+4)*(len_z_cst+4)) );
      j = int( (flatInd - i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
      k = flatInd - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);
      fields->density[i][j][k] = densFlat[flatInd];
      if (fill_tank){
        tank->density[i][j][k]   = densFlat[flatInd];
      }
    }

  }

  void load_particles(particles* p, simu_grid* grid, simu_tank* tank, simu_param* sP,
                      MPI_Comm comm, bool fill_tank, std::string path_inputs,
                      int idx_load, bool old_file)
  {

    //______________________________________________________________________________________
    // How many files to open?.
    //
    std::string file_name = path_inputs + "/nb_particle_files_it" + std::to_string(idx_load) + "_rank_"
                            + std::to_string(sP->mpi_rank_y) + "_"
                            + std::to_string(sP->mpi_rank_z) + ".txt";
    //
    std::string line;
    int nb_files;
    std::ifstream myfile (file_name);
    if (myfile.is_open()){
      getline (myfile, line);
      nb_files = std::stoi(line);
      myfile.close();
    }
    std::cout << "| nb_files: " << nb_files << std::endl;

    //______________________________________________________________________________________
    // How many particles TOTAL? -> sP->nb_part_tank
    //
    sP->nb_part_tank = 0; // If load_particles called for the second time.
    for (int idx_f=0; idx_f<nb_files; idx_f++){
      //
      file_name = path_inputs + "/nb_particles_" + std::to_string(idx_f) +  "_it" + std::to_string(idx_load) + "_rank_"
                  + std::to_string(sP->mpi_rank_y) + "_"
                  + std::to_string(sP->mpi_rank_z) + ".txt";
      //
      std::ifstream myfile (file_name);
      if (myfile.is_open()){
        getline (myfile, line);
        sP->nb_part_tank += std::stoi(line);
        //
        myfile.close();
      }
      else {
        std::cout << "\n| Couldn't open file in load_particle(): " << file_name << std::endl;
        sortie();
      }
    }
    //
    if (sP->nb_part_tank>tank_size_cst){
      std::cout << sP->nb_part_tank << " " << tank_size_cst << "\n";
      std::cout << "\n| Too many particles for tank! load_particles()\n";
      sortie();
    }
    if (sP->nb_part_tank>pool_size_cst){
      std::cout << sP->nb_part_tank << " " << tank_size_cst << "\n";
      std::cout << "\n| Too many particles for pool! load_particles()\n";
      sortie();
    }


    //______________________________________________________________________________________
    // Filling the particles and the tank.
    //
    int* indices = new int [pool_size_cst];
    int idx_shuf;
    for (int i=0; i<pool_size_cst; i++){
      indices[i] = i; // np.arange(i) !
    }
    shuffleOwn(indices, pool_size_cst);


    int idx_part_tot = 0;
    int nb_part_in_file;

    for (int idx_f=0; idx_f<nb_files; idx_f++){

      file_name = path_inputs + "/nb_particles_" + std::to_string(idx_f) +  "_it" + std::to_string(idx_load) + "_rank_"
                  + std::to_string(sP->mpi_rank_y) + "_"
                  + std::to_string(sP->mpi_rank_z) + ".txt";
      std::ifstream myfile (file_name);
      if (myfile.is_open()){
        getline (myfile, line);
        nb_part_in_file = std::stoi(line);
        myfile.close();
      }
      else {
        std::cout << "\n| Couldn't open file in load_particle(): " << file_name << std::endl;
        sortie();
      }
      //
      file_name = path_inputs + "/particles_" + std::to_string(idx_f) +  "_it" + std::to_string(idx_load) + "_rank_"
                  + std::to_string(sP->mpi_rank_y) + "_"
                  + std::to_string(sP->mpi_rank_z) + ".npy";


      cnpy::NpyArray arrPSW  = cnpy::npy_load(file_name);
      float* pSWFlat = arrPSW.data<float>();

      for (int i=0; i<nb_part_in_file; i++){
        idx_shuf = indices[idx_part_tot];
        if (idx_part_tot<sP->nb_part_tank){
          if (old_file){
            p->rx[idx_part_tot] = pSWFlat[i*6  ];
            p->ry[idx_part_tot] = pSWFlat[i*6+1];
            p->rz[idx_part_tot] = pSWFlat[i*6+2];
            p->vx[idx_part_tot] = pSWFlat[i*6+3];
            p->vy[idx_part_tot] = pSWFlat[i*6+4];
            p->vz[idx_part_tot] = pSWFlat[i*6+5];
            p->active[idx_part_tot] = true;
            p->ID[idx_part_tot] = 0;
          }
          else{
            p->rx[idx_part_tot] = pSWFlat[i*7  ];
            p->ry[idx_part_tot] = pSWFlat[i*7+1];
            p->rz[idx_part_tot] = pSWFlat[i*7+2];
            p->vx[idx_part_tot] = pSWFlat[i*7+3];
            p->vy[idx_part_tot] = pSWFlat[i*7+4];
            p->vz[idx_part_tot] = pSWFlat[i*7+5];
            p->ID[idx_part_tot] = pSWFlat[i*7+6];
            p->active[idx_part_tot] = true;
          }

        }
        else {
          p->active[idx_part_tot] = false;
        }
        idx_part_tot++;
        //
      }


    }


    if (fill_tank){
      //______________________________________________________________________________________
      // Sorting the particles in the tank.
      /** Here a lambda expression is given to std::sort in order to mimicate np.argsort:
      idx_ord will contain how the ordered indices of tank->rx */
      tank->idx_ordered = std::vector<size_t>(sP->nb_part_tank, 0);
      std::iota(tank->idx_ordered.begin(), tank->idx_ordered.end(), 0);
      std::sort(tank->idx_ordered.begin(), tank->idx_ordered.end(),
                [&](size_t i1, size_t i2) { return p->rx[i1] < p->rx[i2]; } );

      int idx_slice = 0; // Within [0, xLen-1]
      int nb_part_in_slice = 0;

      for (int i=0; i<sP->nb_part_tank; i++){
        if (p->rx[tank->idx_ordered[i]]>(grid->xGrid[1+(idx_slice+1)*nb_cell_per_shift_cst] + .5*sP->dX)){
          tank->sliceWidth[idx_slice] = nb_part_in_slice;
          if (nb_part_in_slice>injec_size_cst){
            printf("| Slice to wide for buffer! load_particles() %i %i\n", nb_part_in_slice, injec_size_cst);
            sortie();
          }
          tank->indFirstPartSlice[idx_slice] = i-nb_part_in_slice;
          idx_slice++;
          nb_part_in_slice = 0;
        }

        nb_part_in_slice++;
      }
      /// Still need to fill the information for the last column:
      tank->sliceWidth[idx_slice] = nb_part_in_slice;
      tank->indFirstPartSlice[idx_slice] = sP->nb_part_tank-nb_part_in_slice;

      file_name = "products/HK/slice_width_rank"
                  + std::to_string(sP->mpi_rank_y)+ "_"
                  + std::to_string(sP->mpi_rank_z) + ".npy";
      cnpy::npy_save(file_name, tank->sliceWidth, {(long unsigned int) (len_x_cst)}, "w");

      tank->idx_x = 0;
    }

  }
#endif
