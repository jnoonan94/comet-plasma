#ifndef __KERNELSHK_H_INCLUDED__   // if x.h hasn't been included yet...
#define __KERNELSHK_H_INCLUDED__

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "parameters.h"

__global__ void print_param_k(simu_param* sP)
{
  printf("Print param: %i\n", len_x_cst);
}


__device__ __forceinline__ float atomicMaxfloat (float * addr, float value)
{
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}


//__________________________
__global__ void moments_particles_k(particles* p, moments* mom,  int idx_save,
                                    simu_param* sP, house_keeping* HK)
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x ;


  if ( (idx_save < len_save_t_cst) && (idx < pool_size_cst) && (p->active[idx]) ){

    atomicAdd(&mom->E_kin[idx_save], (p->vx[idx]*p->vx[idx] + p->vy[idx]*p->vy[idx] + p->vz[idx]*p->vz[idx]) ) ;
    atomicAdd(&mom->nb_part_tot[idx_save], 1) ;
    atomicAdd(&mom->vel_part[0][idx_save], p->vx[idx]) ;
    atomicAdd(&mom->vel_part[1][idx_save], p->vy[idx]) ;
    atomicAdd(&mom->vel_part[2][idx_save], p->vz[idx]) ;
    atomicAdd(&mom->vel_th_part[0][idx_save], p->vx[idx]*p->vx[idx]) ;
    atomicAdd(&mom->vel_th_part[1][idx_save], p->vy[idx]*p->vy[idx]) ;
    atomicAdd(&mom->vel_th_part[2][idx_save], p->vz[idx]*p->vz[idx]) ;

    if (p->ID[idx]==0){
      atomicAdd(&HK->nbPartSW[idx_save], 1) ;
    }
    else if (p->ID[idx]==1){
      atomicAdd(&HK->nbPartPla[idx_save], 1) ;
    }
  }
}

__global__ void moments_grids_k(simu_B_field* B, simu_fields* fields, moments* mom,  int idx_save)
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x ;

  if (idx_save<len_save_t_cst){

    if( idx < nb_nodes_cst ){
      #if NB_DIM==2
        int i = floorf(idx/len_y_cst) ;
        int j = idx-i*len_y_cst ;
        i += 2 ;
        j += 2 ;

        atomicAdd(&mom->E_mag[idx_save], (B->B[0][i][j]*B->B[0][i][j] +
                                          B->B[1][i][j]*B->B[1][i][j] +
                                          B->B[2][i][j]*B->B[2][i][j]) ) ; // /nb_nodes_cst ) ;
        atomicAdd(&mom->E_elec[idx_save], (fields->E[0][i][j]*fields->E[0][i][j] +
                                           fields->E[1][i][j]*fields->E[1][i][j] +
                                           fields->E[2][i][j]*fields->E[2][i][j]) ) ; // /nb_nodes_cst ) ;

        atomicAdd(&mom->B_mean[0][idx_save], B->B[0][i][j]/nb_nodes_cst) ;
        atomicAdd(&mom->B_mean[1][idx_save], B->B[1][i][j]/nb_nodes_cst) ;
        atomicAdd(&mom->B_mean[2][idx_save], B->B[2][i][j]/nb_nodes_cst) ;

        atomicAdd(&mom->B_var[0][idx_save], (B->B[0][i][j]*B->B[0][i][j])/nb_nodes_cst) ;
        atomicAdd(&mom->B_var[1][idx_save], (B->B[1][i][j]*B->B[1][i][j])/nb_nodes_cst) ;
        atomicAdd(&mom->B_var[2][idx_save], (B->B[2][i][j]*B->B[2][i][j])/nb_nodes_cst) ;

        atomicAdd(&mom->div_B_mean[idx_save], fields->smooth[i][j]/nb_nodes_cst) ;
        atomicAdd(&mom->div_B_var[idx_save], (fields->smooth[i][j]*fields->smooth[i][j])/nb_nodes_cst) ;
        atomicMaxfloat(&mom->div_B_max[idx_save], fabsf(fields->smooth[i][j])) ;

        atomicAdd(&mom->dens_tot[idx_save], fields->density[i][j]) ;
        atomicAdd(&mom->Ji_mean[0][idx_save], fields->Ji[0][i][j]/(fields->density[i][j]*nb_nodes_cst) ) ;
        atomicAdd(&mom->Ji_mean[1][idx_save], fields->Ji[1][i][j]/(fields->density[i][j]*nb_nodes_cst) ) ;
        atomicAdd(&mom->Ji_mean[2][idx_save], fields->Ji[2][i][j]/(fields->density[i][j]*nb_nodes_cst) ) ;
      #elif NB_DIM==3
        int i = floorf(idx/(len_y_cst*len_z_cst)) ;
        int j = floorf(idx-i*(len_y_cst*len_z_cst))/len_z_cst ;
        int k = idx-i*(len_y_cst*len_z_cst)-j*len_z_cst ;
        i += 2 ;
        j += 2 ;
        k += 2 ;

        atomicAdd(&mom->E_mag[idx_save], (B->B[0][i][j][k]*B->B[0][i][j][k] +
                                          B->B[1][i][j][k]*B->B[1][i][j][k] +
                                          B->B[2][i][j][k]*B->B[2][i][j][k]) ) ; // /nb_nodes_cst) ;
        atomicAdd(&mom->E_elec[idx_save], (fields->E[0][i][j][k]*fields->E[0][i][j][k] +
                                           fields->E[1][i][j][k]*fields->E[1][i][j][k] +
                                           fields->E[2][i][j][k]*fields->E[2][i][j][k]) ) ; // /nb_nodes_cst) ;

        atomicAdd(&mom->B_mean[0][idx_save], B->B[0][i][j][k]/nb_nodes_cst) ;
        atomicAdd(&mom->B_mean[1][idx_save], B->B[1][i][j][k]/nb_nodes_cst) ;
        atomicAdd(&mom->B_mean[2][idx_save], B->B[2][i][j][k]/nb_nodes_cst) ;

        atomicAdd(&mom->B_var[0][idx_save], (B->B[0][i][j][k]*B->B[0][i][j][k])/nb_nodes_cst) ;
        atomicAdd(&mom->B_var[1][idx_save], (B->B[1][i][j][k]*B->B[1][i][j][k])/nb_nodes_cst) ;
        atomicAdd(&mom->B_var[2][idx_save], (B->B[2][i][j][k]*B->B[2][i][j][k])/nb_nodes_cst) ;

        atomicAdd(&mom->div_B_mean[idx_save], fields->smooth[i][j][k]/nb_nodes_cst) ;
        atomicAdd(&mom->div_B_var[idx_save], (fields->smooth[i][j][k]*fields->smooth[i][j][k])/nb_nodes_cst) ;

        atomicAdd(&mom->dens_tot[idx_save], fields->counts[i][j][k]) ; //fields->density[i][j][k]) ;
        // atomicAdd(&mom->vel_tot[ind_it], sqrtf(fields->fluxNum[0][i][j][k]*fields->fluxNum[0][i][j][k]+fields->fluxNum[1][i][j][k]*fields->fluxNum[1][i][j][k]+fields->fluxNum[2][i][j][k]*fields->fluxNum[2][i][j][k]) ) ;
        atomicAdd(&mom->Ji_mean[0][idx_save], fields->Ji[0][i][j][k]/nb_nodes_cst ) ;
        atomicAdd(&mom->Ji_mean[1][idx_save], fields->Ji[1][i][j][k]/nb_nodes_cst ) ;
        atomicAdd(&mom->Ji_mean[2][idx_save], fields->Ji[2][i][j][k]/nb_nodes_cst ) ;
      #endif

    }

  }
}
__global__ void divide_moments_k(moments* mom, int idx_save, simu_param* sP)
{

  if (idx_save < len_save_t_cst){
    // printf("Went too far in divide_moments_k()\n") ;
  // printf("v_x: %.7e, %i particles, indIt: %i.\n", mom->vel_part[0][idx_save], mom->nb_part_tot[idx_save], idx_save) ;
  // printf("W_B: %.7e\n", mom->E_mag[idx_save]) ;
  // mom->E_kin[idx_save] /= mom->nb_part_tot[idx_save] ;
    mom->vel_part[0][idx_save] /= mom->nb_part_tot[idx_save] ;
    mom->vel_part[1][idx_save] /= mom->nb_part_tot[idx_save] ;
    mom->vel_part[2][idx_save] /= mom->nb_part_tot[idx_save] ;
    mom->vel_th_part[0][idx_save] /= mom->nb_part_tot[idx_save] ;
    mom->vel_th_part[1][idx_save] /= mom->nb_part_tot[idx_save] ;
    mom->vel_th_part[2][idx_save] /= mom->nb_part_tot[idx_save] ;
    mom->vel_th_part[0][idx_save] -= mom->vel_part[0][idx_save]*mom->vel_part[0][idx_save] ;
    mom->vel_th_part[1][idx_save] -= mom->vel_part[1][idx_save]*mom->vel_part[1][idx_save] ;
    mom->vel_th_part[2][idx_save] -= mom->vel_part[2][idx_save]*mom->vel_part[2][idx_save] ;

    mom->B_var[0][idx_save] -= mom->B_mean[0][idx_save]*mom->B_mean[0][idx_save] ;
    mom->B_var[1][idx_save] -= mom->B_mean[1][idx_save]*mom->B_mean[1][idx_save] ;
    mom->B_var[2][idx_save] -= mom->B_mean[2][idx_save]*mom->B_mean[2][idx_save] ;

  }

}

__global__ void update_traj_k(particles* p, trajectory* traj, int ind_it, int indPart)
{

  // if (indIt==0){printf("updatetraj %.4e, %.4e, %.4e\n", p->vx[indPart], p->vx[indPart], p->vx[indPart]);}
  traj->rx[ind_it] = p->rx[indPart] ;
  traj->ry[ind_it] = p->ry[indPart] ;
  #if NB_DIM==3
    traj->rz[ind_it] = p->rz[indPart] ;
  #endif
  traj->vx[ind_it] = p->vx[indPart] ;
  traj->vy[ind_it] = p->vy[indPart] ;
  traj->vz[ind_it] = p->vz[indPart] ;
  // if (indIt==0){printf("updatetraj %.4e, %.4e, %.4e\n", traj->vx[indPart], traj->vx[indPart], traj->vx[indPart]);}

}

__global__ void update_fields_time_space_k(simu_fields* fields, simu_B_field* B,
                                           fields_time_space* fields_t_s,
                                           int idx_save, simu_param* sP)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x ;

  // if (idx_save<len_save_t_cst){
  //   printf("Went too far in update_fields_time_space_k()\n") ;
  // }

    if(idx < len_save_x_cst){
      #if NB_DIM==2
        int i = int(idx*rate_save_x_cst + 2) ;
        int j = int((len_y_cst+4)/2.) ;
        #if debug_mode_cst
          if (idx_save >= len_save_t_cst){
            printf("update_fields_time_space_k yo %i\n", idx_save);
          }
          if (i>=len_x_cst){
            printf("update_fields_time_space_k ye\n");
          }
          if (j>=len_y_cst){
            printf("update_fields_time_space_k yu\n");
          }
        #endif
        fields_t_s->density[idx_save][idx] = fields->density[i][j] ;
        fields_t_s->E[0][idx_save][idx] = fields->E[0][i][j] ;
        fields_t_s->E[1][idx_save][idx] = fields->E[1][i][j] ;
        fields_t_s->E[2][idx_save][idx] = fields->E[2][i][j] ;
        fields_t_s->B[0][idx_save][idx] = B->B[0][i][j] ;
        fields_t_s->B[1][idx_save][idx] = B->B[1][i][j] ;
        fields_t_s->B[2][idx_save][idx] = B->B[2][i][j] ;
      #elif NB_DIM==3
        int i = int(idx*rate_save_x_cst + 2) ;
        int j = int((len_y_cst+4)/2.) ;
        int k = int((len_z_cst+4)/2.) ;
        int ind_t = int(idx_save/rate_save_t_cst) ;

        fields_t_s->density[ind_t][idx] = fields->density[i][j][k] ;
        fields_t_s->E[0][ind_t][idx] = fields->E[0][i][j][k] ;
        fields_t_s->E[1][ind_t][idx] = fields->E[1][i][j][k] ;
        fields_t_s->E[2][ind_t][idx] = fields->E[2][i][j][k] ;
        fields_t_s->B[0][ind_t][idx] = B->B[0][i][j][k] ;
        fields_t_s->B[1][ind_t][idx] = B->B[1][i][j][k] ;
        fields_t_s->B[2][ind_t][idx] = B->B[2][i][j][k] ;
      #endif

    }

  // }

}

__global__ void push_probes_k(probes* prob, simu_param* sP, simu_grid* grid, int idx_it)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x ;

  if (idx < nb_probes_cst && (obstacle_cst && !ORF_cst)){
    #if NB_DIM==2

      if ( prob->active[idx] && (idx_it>0) && (idx_it<nb_it_max_cst-1) ){
        /* Probes are shifted correspondingly with the iteration index: */
        prob->rx[idx][idx_it] = prob->rx[idx][idx_it-1] + sP->dX*nb_cell_per_shift_cst/nb_it_per_shift_cst;
        if ((idx_it%nb_it_per_shift_cst)==0){
          prob->rx[idx][idx_it] -= nb_cell_per_shift_cst*sP->dX;
        }
      }
    #elif NB_DIM==3
      // int i = int(idx/(nb_probes_y_cst*nb_probes_z_cst));
      // int j = int( (idx - i*(nb_probes_y_cst*nb_probes_z_cst))/nb_probes_z_cst );
      // int k = idx - i*(nb_probes_y_cst*nb_probes_z_cst) - j*nb_probes_z_cst;
      // /* Probes are shifted correspondingly with the iteration index: */
      // prob->rx[i][j][k] += sP->v_obs*sP->dt;
      // if ((idx_it%nb_it_per_shift_cst)==0){
      //   prob->rx[i][j][k] -= nb_cell_per_shift_cst*sP->dX;
      // }
    #endif
  }
}
__global__ void probe_fields_k(probes* prob, simu_fields* fields, simu_B_field* B,
                                simu_grid* grid, simu_param* sP, int idx_it)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x ;

  if ( (idx < nb_probes_cst) && (prob->active[idx]) ){

    #if NB_DIM==2

      int indX, indY;
      float wx, wy;
      float wx0, wx1, wx2;
      float wy0, wy1, wy2;

      indX = roundf( (prob->rx[idx][idx_it]-grid->xMin)*sP->dX_i + .5) + 1;
      indY = roundf( (prob->ry[idx][idx_it]-grid->yMin)*sP->dX_i + .5) + 1;

      wx = (prob->rx[idx][idx_it]-grid->xGrid[indX])*sP->dX_i;
      wy = (prob->ry[idx][idx_it]-grid->yGrid[indY])*sP->dX_i;

      wx0 = .5*(.5-wx)*(.5-wx);
      wx1 = .75 - wx*wx;
      wx2 = .5*(.5+wx)*(.5+wx);

      wy0 = .5*(.5-wy)*(.5-wy);
      wy1 = .75 - wy*wy;
      wy2 = .5*(.5+wy)*(.5+wy);

      if (indX<1 || indX>len_x_cst+2){
        printf("| Not cool here, x-wise, kernels_HK probe_fields_k it %i, proc %i, i %i, rx %.4e, indX %i\n", idx_it, sP->mpi_rank_lin, idx, prob->rx[idx][idx_it], indX);
      }
      if (indY<1 || indY>len_y_cst+2 ){
        printf("| Not cool here, y-wise, kernels_HK probe_fields_k %i %.4e %i\n", sP->mpi_rank_lin, prob->ry[idx][idx_it], indX);
      }

      prob->E[idx][0][idx_it] = fields->E[0][indX-1][indY-1]* wx0*wy0 +
                              fields->E[0][indX-1][indY  ]* wx0*wy1 +
                              fields->E[0][indX-1][indY+1]* wx0*wy2 +
                              fields->E[0][indX  ][indY-1]* wx1*wy0 +
                              fields->E[0][indX  ][indY  ]* wx1*wy1 +
                              fields->E[0][indX  ][indY+1]* wx1*wy2 +
                              fields->E[0][indX+1][indY-1]* wx2*wy0 +
                              fields->E[0][indX+1][indY  ]* wx2*wy1 +
                              fields->E[0][indX+1][indY+1]* wx2*wy2;
      prob->E[idx][1][idx_it] = fields->E[1][indX-1][indY-1]* wx0*wy0 +
                              fields->E[1][indX-1][indY  ]* wx0*wy1 +
                              fields->E[1][indX-1][indY+1]* wx0*wy2 +
                              fields->E[1][indX  ][indY-1]* wx1*wy0 +
                              fields->E[1][indX  ][indY  ]* wx1*wy1 +
                              fields->E[1][indX  ][indY+1]* wx1*wy2 +
                              fields->E[1][indX+1][indY-1]* wx2*wy0 +
                              fields->E[1][indX+1][indY  ]* wx2*wy1 +
                              fields->E[1][indX+1][indY+1]* wx2*wy2;
      prob->E[idx][2][idx_it] = fields->E[2][indX-1][indY-1]* wx0*wy0 +
                              fields->E[2][indX-1][indY  ]* wx0*wy1 +
                              fields->E[2][indX-1][indY+1]* wx0*wy2 +
                              fields->E[2][indX  ][indY-1]* wx1*wy0 +
                              fields->E[2][indX  ][indY  ]* wx1*wy1 +
                              fields->E[2][indX  ][indY+1]* wx1*wy2 +
                              fields->E[2][indX+1][indY-1]* wx2*wy0 +
                              fields->E[2][indX+1][indY  ]* wx2*wy1 +
                              fields->E[2][indX+1][indY+1]* wx2*wy2;
      prob->B[idx][0][idx_it] = B->B[0][indX-1][indY-1]* wx0*wy0 +
                              B->B[0][indX-1][indY  ]* wx0*wy1 +
                              B->B[0][indX-1][indY+1]* wx0*wy2 +
                              B->B[0][indX  ][indY-1]* wx1*wy0 +
                              B->B[0][indX  ][indY  ]* wx1*wy1 +
                              B->B[0][indX  ][indY+1]* wx1*wy2 +
                              B->B[0][indX+1][indY-1]* wx2*wy0 +
                              B->B[0][indX+1][indY  ]* wx2*wy1 +
                              B->B[0][indX+1][indY+1]* wx2*wy2;
      prob->B[idx][1][idx_it] = B->B[1][indX-1][indY-1]* wx0*wy0 +
                              B->B[1][indX-1][indY  ]* wx0*wy1 +
                              B->B[1][indX-1][indY+1]* wx0*wy2 +
                              B->B[1][indX  ][indY-1]* wx1*wy0 +
                              B->B[1][indX  ][indY  ]* wx1*wy1 +
                              B->B[1][indX  ][indY+1]* wx1*wy2 +
                              B->B[1][indX+1][indY-1]* wx2*wy0 +
                              B->B[1][indX+1][indY  ]* wx2*wy1 +
                              B->B[1][indX+1][indY+1]* wx2*wy2;
      prob->B[idx][2][idx_it] = B->B[2][indX-1][indY-1]* wx0*wy0 +
                              B->B[2][indX-1][indY  ]* wx0*wy1 +
                              B->B[2][indX-1][indY+1]* wx0*wy2 +
                              B->B[2][indX  ][indY-1]* wx1*wy0 +
                              B->B[2][indX  ][indY  ]* wx1*wy1 +
                              B->B[2][indX  ][indY+1]* wx1*wy2 +
                              B->B[2][indX+1][indY-1]* wx2*wy0 +
                              B->B[2][indX+1][indY  ]* wx2*wy1 +
                              B->B[2][indX+1][indY+1]* wx2*wy2;
      prob->Ji[idx][0][idx_it] = fields->Ji[0][indX-1][indY-1]* wx0*wy0 +
                               fields->Ji[0][indX-1][indY  ]* wx0*wy1 +
                               fields->Ji[0][indX-1][indY+1]* wx0*wy2 +
                               fields->Ji[0][indX  ][indY-1]* wx1*wy0 +
                               fields->Ji[0][indX  ][indY  ]* wx1*wy1 +
                               fields->Ji[0][indX  ][indY+1]* wx1*wy2 +
                               fields->Ji[0][indX+1][indY-1]* wx2*wy0 +
                               fields->Ji[0][indX+1][indY  ]* wx2*wy1 +
                               fields->Ji[0][indX+1][indY+1]* wx2*wy2;
      prob->Ji[idx][1][idx_it] = fields->Ji[1][indX-1][indY-1]* wx0*wy0 +
                               fields->Ji[1][indX-1][indY  ]* wx0*wy1 +
                               fields->Ji[1][indX-1][indY+1]* wx0*wy2 +
                               fields->Ji[1][indX  ][indY-1]* wx1*wy0 +
                               fields->Ji[1][indX  ][indY  ]* wx1*wy1 +
                               fields->Ji[1][indX  ][indY+1]* wx1*wy2 +
                               fields->Ji[1][indX+1][indY-1]* wx2*wy0 +
                               fields->Ji[1][indX+1][indY  ]* wx2*wy1 +
                               fields->Ji[1][indX+1][indY+1]* wx2*wy2;
      prob->Ji[idx][2][idx_it] = fields->Ji[2][indX-1][indY-1]* wx0*wy0 +
                               fields->Ji[2][indX-1][indY  ]* wx0*wy1 +
                               fields->Ji[2][indX-1][indY+1]* wx0*wy2 +
                               fields->Ji[2][indX  ][indY-1]* wx1*wy0 +
                               fields->Ji[2][indX  ][indY  ]* wx1*wy1 +
                               fields->Ji[2][indX  ][indY+1]* wx1*wy2 +
                               fields->Ji[2][indX+1][indY-1]* wx2*wy0 +
                               fields->Ji[2][indX+1][indY  ]* wx2*wy1 +
                               fields->Ji[2][indX+1][indY+1]* wx2*wy2;
       prob->J_tot[idx][0][idx_it] = fields->J_tot[0][indX-1][indY-1]* wx0*wy0 +
                                   fields->J_tot[0][indX-1][indY  ]* wx0*wy1 +
                                   fields->J_tot[0][indX-1][indY+1]* wx0*wy2 +
                                   fields->J_tot[0][indX  ][indY-1]* wx1*wy0 +
                                   fields->J_tot[0][indX  ][indY  ]* wx1*wy1 +
                                   fields->J_tot[0][indX  ][indY+1]* wx1*wy2 +
                                   fields->J_tot[0][indX+1][indY-1]* wx2*wy0 +
                                   fields->J_tot[0][indX+1][indY  ]* wx2*wy1 +
                                   fields->J_tot[0][indX+1][indY+1]* wx2*wy2;
       prob->J_tot[idx][1][idx_it] = fields->J_tot[1][indX-1][indY-1]* wx0*wy0 +
                                   fields->J_tot[1][indX-1][indY  ]* wx0*wy1 +
                                   fields->J_tot[1][indX-1][indY+1]* wx0*wy2 +
                                   fields->J_tot[1][indX  ][indY-1]* wx1*wy0 +
                                   fields->J_tot[1][indX  ][indY  ]* wx1*wy1 +
                                   fields->J_tot[1][indX  ][indY+1]* wx1*wy2 +
                                   fields->J_tot[1][indX+1][indY-1]* wx2*wy0 +
                                   fields->J_tot[1][indX+1][indY  ]* wx2*wy1 +
                                   fields->J_tot[1][indX+1][indY+1]* wx2*wy2;
       prob->J_tot[idx][2][idx_it] = fields->J_tot[2][indX-1][indY-1]* wx0*wy0 +
                                   fields->J_tot[2][indX-1][indY  ]* wx0*wy1 +
                                   fields->J_tot[2][indX-1][indY+1]* wx0*wy2 +
                                   fields->J_tot[2][indX  ][indY-1]* wx1*wy0 +
                                   fields->J_tot[2][indX  ][indY  ]* wx1*wy1 +
                                   fields->J_tot[2][indX  ][indY+1]* wx1*wy2 +
                                   fields->J_tot[2][indX+1][indY-1]* wx2*wy0 +
                                   fields->J_tot[2][indX+1][indY  ]* wx2*wy1 +
                                   fields->J_tot[2][indX+1][indY+1]* wx2*wy2;
          prob->density[idx][idx_it] = fields->density[indX-1][indY-1]* wx0*wy0 +
                                     fields->density[indX-1][indY  ]* wx0*wy1 +
                                     fields->density[indX-1][indY+1]* wx0*wy2 +
                                     fields->density[indX  ][indY-1]* wx1*wy0 +
                                     fields->density[indX  ][indY  ]* wx1*wy1 +
                                     fields->density[indX  ][indY+1]* wx1*wy2 +
                                     fields->density[indX+1][indY-1]* wx2*wy0 +
                                     fields->density[indX+1][indY  ]* wx2*wy1 +
                                     fields->density[indX+1][indY+1]* wx2*wy2;
    #elif NB_DIM==3
      // int i = int(idx/(nb_probes_y_cst*nb_probes_z_cst));
      // int j = int( (idx - i*(nb_probes_y_cst*nb_probes_z_cst))/nb_probes_z_cst );
      // int k = idx - i*(nb_probes_y_cst*nb_probes_z_cst) -j*nb_probes_z_cst;

      int indX, indY, indZ;
      float wx, wy, wz;
      float wx0, wx1, wx2;
      float wy0, wy1, wy2;
      float wz0, wz1, wz2;

      indX = roundf( (prob->rx[idx][idx_it]-grid->xMin)*sP->dX_i + .5) + 1;
      indY = roundf( (prob->ry[idx][idx_it]-grid->yMin)*sP->dX_i + .5) + 1;
      indZ = roundf( (prob->rz[idx][idx_it]-grid->zMin)*sP->dX_i + .5) + 1;

      wx = (prob->rx[idx][idx_it]-grid->xGrid[indX])*sP->dX_i;
      wy = (prob->ry[idx][idx_it]-grid->yGrid[indY])*sP->dX_i;
      wz = (prob->rz[idx][idx_it]-grid->zGrid[indZ])*sP->dX_i;

      wx0 = .5*(.5-wx)*(.5-wx);
      wx1 = .75 - wx*wx;
      wx2 = .5*(.5+wx)*(.5+wx);

      wy0 = .5*(.5-wy)*(.5-wy);
      wy1 = .75 - wy*wy;
      wy2 = .5*(.5+wy)*(.5+wy);

      wz0 = .5*(.5-wz)*(.5-wz);
      wz1 = .75 - wz*wz;
      wz2 = .5*(.5+wz)*(.5+wz);


      prob->E[0][idx][idx_it] = fields->E[0][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
                                    fields->E[0][indX-1][indY-1][indZ  ]* wx0*wy0*wz1 +
                                    fields->E[0][indX-1][indY-1][indZ+1]* wx0*wy0*wz2 +
                                    fields->E[0][indX-1][indY  ][indZ-1]* wx0*wy1*wz0 +
                                    fields->E[0][indX-1][indY  ][indZ  ]* wx0*wy1*wz1 +
                                    fields->E[0][indX-1][indY  ][indZ+1]* wx0*wy1*wz2 +
                                    fields->E[0][indX-1][indY+1][indZ-1]* wx0*wy2*wz0 +
                                    fields->E[0][indX-1][indY+1][indZ  ]* wx0*wy2*wz1 +
                                    fields->E[0][indX-1][indY+1][indZ+1]* wx0*wy2*wz2 +
                                    fields->E[0][indX  ][indY-1][indZ-1]* wx1*wy0*wz0 +
                                    fields->E[0][indX  ][indY-1][indZ  ]* wx1*wy0*wz1 +
                                    fields->E[0][indX  ][indY-1][indZ+1]* wx1*wy0*wz2 +
                                    fields->E[0][indX  ][indY  ][indZ-1]* wx1*wy1*wz0 +
                                    fields->E[0][indX  ][indY  ][indZ  ]* wx1*wy1*wz1 +
                                    fields->E[0][indX  ][indY  ][indZ+1]* wx1*wy1*wz2 +
                                    fields->E[0][indX  ][indY+1][indZ-1]* wx1*wy2*wz0 +
                                    fields->E[0][indX  ][indY+1][indZ  ]* wx1*wy2*wz1 +
                                    fields->E[0][indX  ][indY+1][indZ+1]* wx1*wy2*wz2 +
                                    fields->E[0][indX+1][indY-1][indZ-1]* wx2*wy0*wz0 +
                                    fields->E[0][indX+1][indY-1][indZ  ]* wx2*wy0*wz1 +
                                    fields->E[0][indX+1][indY-1][indZ+1]* wx2*wy0*wz2 +
                                    fields->E[0][indX+1][indY  ][indZ-1]* wx2*wy1*wz0 +
                                    fields->E[0][indX+1][indY  ][indZ  ]* wx2*wy1*wz1 +
                                    fields->E[0][indX+1][indY  ][indZ+1]* wx2*wy1*wz2 +
                                    fields->E[0][indX+1][indY+1][indZ-1]* wx2*wy2*wz0 +
                                    fields->E[0][indX+1][indY+1][indZ  ]* wx2*wy2*wz1 +
                                    fields->E[0][indX+1][indY+1][indZ+1]* wx2*wy2*wz2  ;
      prob->E[1][idx][idx_it] = fields->E[1][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
                                    fields->E[1][indX-1][indY-1][indZ  ]* wx0*wy0*wz1 +
                                    fields->E[1][indX-1][indY-1][indZ+1]* wx0*wy0*wz2 +
                                    fields->E[1][indX-1][indY  ][indZ-1]* wx0*wy1*wz0 +
                                    fields->E[1][indX-1][indY  ][indZ  ]* wx0*wy1*wz1 +
                                    fields->E[1][indX-1][indY  ][indZ+1]* wx0*wy1*wz2 +
                                    fields->E[1][indX-1][indY+1][indZ-1]* wx0*wy2*wz0 +
                                    fields->E[1][indX-1][indY+1][indZ  ]* wx0*wy2*wz1 +
                                    fields->E[1][indX-1][indY+1][indZ+1]* wx0*wy2*wz2 +
                                    fields->E[1][indX  ][indY-1][indZ-1]* wx1*wy0*wz0 +
                                    fields->E[1][indX  ][indY-1][indZ  ]* wx1*wy0*wz1 +
                                    fields->E[1][indX  ][indY-1][indZ+1]* wx1*wy0*wz2 +
                                    fields->E[1][indX  ][indY  ][indZ-1]* wx1*wy1*wz0 +
                                    fields->E[1][indX  ][indY  ][indZ  ]* wx1*wy1*wz1 +
                                    fields->E[1][indX  ][indY  ][indZ+1]* wx1*wy1*wz2 +
                                    fields->E[1][indX  ][indY+1][indZ-1]* wx1*wy2*wz0 +
                                    fields->E[1][indX  ][indY+1][indZ  ]* wx1*wy2*wz1 +
                                    fields->E[1][indX  ][indY+1][indZ+1]* wx1*wy2*wz2 +
                                    fields->E[1][indX+1][indY-1][indZ-1]* wx2*wy0*wz0 +
                                    fields->E[1][indX+1][indY-1][indZ  ]* wx2*wy0*wz1 +
                                    fields->E[1][indX+1][indY-1][indZ+1]* wx2*wy0*wz2 +
                                    fields->E[1][indX+1][indY  ][indZ-1]* wx2*wy1*wz0 +
                                    fields->E[1][indX+1][indY  ][indZ  ]* wx2*wy1*wz1 +
                                    fields->E[1][indX+1][indY  ][indZ+1]* wx2*wy1*wz2 +
                                    fields->E[1][indX+1][indY+1][indZ-1]* wx2*wy2*wz0 +
                                    fields->E[1][indX+1][indY+1][indZ  ]* wx2*wy2*wz1 +
                                    fields->E[1][indX+1][indY+1][indZ+1]* wx2*wy2*wz2  ;
      prob->E[2][idx][idx_it] = fields->E[2][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
                                    fields->E[2][indX-1][indY-1][indZ  ]* wx0*wy0*wz1 +
                                    fields->E[2][indX-1][indY-1][indZ+1]* wx0*wy0*wz2 +
                                    fields->E[2][indX-1][indY  ][indZ-1]* wx0*wy1*wz0 +
                                    fields->E[2][indX-1][indY  ][indZ  ]* wx0*wy1*wz1 +
                                    fields->E[2][indX-1][indY  ][indZ+1]* wx0*wy1*wz2 +
                                    fields->E[2][indX-1][indY+1][indZ-1]* wx0*wy2*wz0 +
                                    fields->E[2][indX-1][indY+1][indZ  ]* wx0*wy2*wz1 +
                                    fields->E[2][indX-1][indY+1][indZ+1]* wx0*wy2*wz2 +
                                    fields->E[2][indX  ][indY-1][indZ-1]* wx1*wy0*wz0 +
                                    fields->E[2][indX  ][indY-1][indZ  ]* wx1*wy0*wz1 +
                                    fields->E[2][indX  ][indY-1][indZ+1]* wx1*wy0*wz2 +
                                    fields->E[2][indX  ][indY  ][indZ-1]* wx1*wy1*wz0 +
                                    fields->E[2][indX  ][indY  ][indZ  ]* wx1*wy1*wz1 +
                                    fields->E[2][indX  ][indY  ][indZ+1]* wx1*wy1*wz2 +
                                    fields->E[2][indX  ][indY+1][indZ-1]* wx1*wy2*wz0 +
                                    fields->E[2][indX  ][indY+1][indZ  ]* wx1*wy2*wz1 +
                                    fields->E[2][indX  ][indY+1][indZ+1]* wx1*wy2*wz2 +
                                    fields->E[2][indX+1][indY-1][indZ-1]* wx2*wy0*wz0 +
                                    fields->E[2][indX+1][indY-1][indZ  ]* wx2*wy0*wz1 +
                                    fields->E[2][indX+1][indY-1][indZ+1]* wx2*wy0*wz2 +
                                    fields->E[2][indX+1][indY  ][indZ-1]* wx2*wy1*wz0 +
                                    fields->E[2][indX+1][indY  ][indZ  ]* wx2*wy1*wz1 +
                                    fields->E[2][indX+1][indY  ][indZ+1]* wx2*wy1*wz2 +
                                    fields->E[2][indX+1][indY+1][indZ-1]* wx2*wy2*wz0 +
                                    fields->E[2][indX+1][indY+1][indZ  ]* wx2*wy2*wz1 +
                                    fields->E[2][indX+1][indY+1][indZ+1]* wx2*wy2*wz2  ;
      prob->B[0][idx][idx_it] = B->B[0][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
                                    B->B[0][indX-1][indY-1][indZ  ]* wx0*wy0*wz1 +
                                    B->B[0][indX-1][indY-1][indZ+1]* wx0*wy0*wz2 +
                                    B->B[0][indX-1][indY  ][indZ-1]* wx0*wy1*wz0 +
                                    B->B[0][indX-1][indY  ][indZ  ]* wx0*wy1*wz1 +
                                    B->B[0][indX-1][indY  ][indZ+1]* wx0*wy1*wz2 +
                                    B->B[0][indX-1][indY+1][indZ-1]* wx0*wy2*wz0 +
                                    B->B[0][indX-1][indY+1][indZ  ]* wx0*wy2*wz1 +
                                    B->B[0][indX-1][indY+1][indZ+1]* wx0*wy2*wz2 +
                                    B->B[0][indX  ][indY-1][indZ-1]* wx1*wy0*wz0 +
                                    B->B[0][indX  ][indY-1][indZ  ]* wx1*wy0*wz1 +
                                    B->B[0][indX  ][indY-1][indZ+1]* wx1*wy0*wz2 +
                                    B->B[0][indX  ][indY  ][indZ-1]* wx1*wy1*wz0 +
                                    B->B[0][indX  ][indY  ][indZ  ]* wx1*wy1*wz1 +
                                    B->B[0][indX  ][indY  ][indZ+1]* wx1*wy1*wz2 +
                                    B->B[0][indX  ][indY+1][indZ-1]* wx1*wy2*wz0 +
                                    B->B[0][indX  ][indY+1][indZ  ]* wx1*wy2*wz1 +
                                    B->B[0][indX  ][indY+1][indZ+1]* wx1*wy2*wz2 +
                                    B->B[0][indX+1][indY-1][indZ-1]* wx2*wy0*wz0 +
                                    B->B[0][indX+1][indY-1][indZ  ]* wx2*wy0*wz1 +
                                    B->B[0][indX+1][indY-1][indZ+1]* wx2*wy0*wz2 +
                                    B->B[0][indX+1][indY  ][indZ-1]* wx2*wy1*wz0 +
                                    B->B[0][indX+1][indY  ][indZ  ]* wx2*wy1*wz1 +
                                    B->B[0][indX+1][indY  ][indZ+1]* wx2*wy1*wz2 +
                                    B->B[0][indX+1][indY+1][indZ-1]* wx2*wy2*wz0 +
                                    B->B[0][indX+1][indY+1][indZ  ]* wx2*wy2*wz1 +
                                    B->B[0][indX+1][indY+1][indZ+1]* wx2*wy2*wz2  ;
      prob->B[1][idx][idx_it] = B->B[1][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
                                    B->B[1][indX-1][indY-1][indZ  ]* wx0*wy0*wz1 +
                                    B->B[1][indX-1][indY-1][indZ+1]* wx0*wy0*wz2 +
                                    B->B[1][indX-1][indY  ][indZ-1]* wx0*wy1*wz0 +
                                    B->B[1][indX-1][indY  ][indZ  ]* wx0*wy1*wz1 +
                                    B->B[1][indX-1][indY  ][indZ+1]* wx0*wy1*wz2 +
                                    B->B[1][indX-1][indY+1][indZ-1]* wx0*wy2*wz0 +
                                    B->B[1][indX-1][indY+1][indZ  ]* wx0*wy2*wz1 +
                                    B->B[1][indX-1][indY+1][indZ+1]* wx0*wy2*wz2 +
                                    B->B[1][indX  ][indY-1][indZ-1]* wx1*wy0*wz0 +
                                    B->B[1][indX  ][indY-1][indZ  ]* wx1*wy0*wz1 +
                                    B->B[1][indX  ][indY-1][indZ+1]* wx1*wy0*wz2 +
                                    B->B[1][indX  ][indY  ][indZ-1]* wx1*wy1*wz0 +
                                    B->B[1][indX  ][indY  ][indZ  ]* wx1*wy1*wz1 +
                                    B->B[1][indX  ][indY  ][indZ+1]* wx1*wy1*wz2 +
                                    B->B[1][indX  ][indY+1][indZ-1]* wx1*wy2*wz0 +
                                    B->B[1][indX  ][indY+1][indZ  ]* wx1*wy2*wz1 +
                                    B->B[1][indX  ][indY+1][indZ+1]* wx1*wy2*wz2 +
                                    B->B[1][indX+1][indY-1][indZ-1]* wx2*wy0*wz0 +
                                    B->B[1][indX+1][indY-1][indZ  ]* wx2*wy0*wz1 +
                                    B->B[1][indX+1][indY-1][indZ+1]* wx2*wy0*wz2 +
                                    B->B[1][indX+1][indY  ][indZ-1]* wx2*wy1*wz0 +
                                    B->B[1][indX+1][indY  ][indZ  ]* wx2*wy1*wz1 +
                                    B->B[1][indX+1][indY  ][indZ+1]* wx2*wy1*wz2 +
                                    B->B[1][indX+1][indY+1][indZ-1]* wx2*wy2*wz0 +
                                    B->B[1][indX+1][indY+1][indZ  ]* wx2*wy2*wz1 +
                                    B->B[1][indX+1][indY+1][indZ+1]* wx2*wy2*wz2;
      prob->B[2][idx][idx_it] = B->B[2][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
                                    B->B[2][indX-1][indY-1][indZ  ]* wx0*wy0*wz1 +
                                    B->B[2][indX-1][indY-1][indZ+1]* wx0*wy0*wz2 +
                                    B->B[2][indX-1][indY  ][indZ-1]* wx0*wy1*wz0 +
                                    B->B[2][indX-1][indY  ][indZ  ]* wx0*wy1*wz1 +
                                    B->B[2][indX-1][indY  ][indZ+1]* wx0*wy1*wz2 +
                                    B->B[2][indX-1][indY+1][indZ-1]* wx0*wy2*wz0 +
                                    B->B[2][indX-1][indY+1][indZ  ]* wx0*wy2*wz1 +
                                    B->B[2][indX-1][indY+1][indZ+1]* wx0*wy2*wz2 +
                                    B->B[2][indX  ][indY-1][indZ-1]* wx1*wy0*wz0 +
                                    B->B[2][indX  ][indY-1][indZ  ]* wx1*wy0*wz1 +
                                    B->B[2][indX  ][indY-1][indZ+1]* wx1*wy0*wz2 +
                                    B->B[2][indX  ][indY  ][indZ-1]* wx1*wy1*wz0 +
                                    B->B[2][indX  ][indY  ][indZ  ]* wx1*wy1*wz1 +
                                    B->B[2][indX  ][indY  ][indZ+1]* wx1*wy1*wz2 +
                                    B->B[2][indX  ][indY+1][indZ-1]* wx1*wy2*wz0 +
                                    B->B[2][indX  ][indY+1][indZ  ]* wx1*wy2*wz1 +
                                    B->B[2][indX  ][indY+1][indZ+1]* wx1*wy2*wz2 +
                                    B->B[2][indX+1][indY-1][indZ-1]* wx2*wy0*wz0 +
                                    B->B[2][indX+1][indY-1][indZ  ]* wx2*wy0*wz1 +
                                    B->B[2][indX+1][indY-1][indZ+1]* wx2*wy0*wz2 +
                                    B->B[2][indX+1][indY  ][indZ-1]* wx2*wy1*wz0 +
                                    B->B[2][indX+1][indY  ][indZ  ]* wx2*wy1*wz1 +
                                    B->B[2][indX+1][indY  ][indZ+1]* wx2*wy1*wz2 +
                                    B->B[2][indX+1][indY+1][indZ-1]* wx2*wy2*wz0 +
                                    B->B[2][indX+1][indY+1][indZ  ]* wx2*wy2*wz1 +
                                    B->B[2][indX+1][indY+1][indZ+1]* wx2*wy2*wz2;

    #endif
  }
}


__global__ void reset_rms_k(root_mean_sqr* rms){
  rms->meansqr = 0. ;
  rms->mean = 0. ;
}
#if NB_DIM==2
  __global__ void rms_k(root_mean_sqr* rms, simu_param* sP, float G[len_x_cst+4][len_y_cst+4]){

    int idx = threadIdx.x + blockIdx.x*blockDim.x ;

    if (idx < nb_nodes_cst){

      int i = int(idx/len_y_cst) ;
      int j = idx-i*len_y_cst ;
      i += 2 ;
      j += 2 ;

      atomicAdd(&rms->meansqr, G[i][j]*G[i][j]/nb_nodes_cst) ;
      atomicAdd(&rms->mean, G[i][j]/nb_nodes_cst) ;
    }
  }
#elif NB_DIM==3
    __global__ void rms_k(root_mean_sqr* rms, simu_param* sP, float G[len_x_cst+4][len_y_cst+4][len_z_cst+4]){

      int idx = threadIdx.x + blockIdx.x*blockDim.x ;

      if (idx < nb_nodes_cst){

        int i = floorf(idx/(len_y_cst*len_z_cst)) ;
        int j = floorf(idx-i*(len_y_cst*len_z_cst))/len_z_cst ;
        int k = idx-i*(len_y_cst*len_z_cst)-j*len_z_cst ;
        i += 2 ;
        j += 2 ;
        k += 2 ;

        atomicAdd(&rms->meansqr, G[i][j][k]*G[i][j][k]/nb_nodes_cst) ;
        atomicAdd(&rms->mean, G[i][j][k]/nb_nodes_cst) ;
      }
    }
#endif
__global__ void sum_rms_k(root_mean_sqr* rms, simu_param* sP, float G[nb_it_max_cst], int idx_save)
{
  G[idx_save] = sqrt(rms->meansqr + rms->mean*rms->mean) ;
  rms->meansqr = 0. ;
  rms->mean = 0. ;
}



__global__ void reset_state_solver_k(state_solver* sta_sol)
{
  sta_sol->break_solver          = false ;
  // sta_sol->invalidB             = false ;
  // sta_sol->invalidE             = false ;
  // sta_sol->invalidDens          = false ;
  // sta_sol->invalidCurr          = false ;
  // sta_sol->invalidCurrTot       = false ;
  // sta_sol->invalid_new_part_idx = false ;
  sta_sol->error_ID = 0;
}
__global__ void check_grids_k(simu_B_field* B, simu_fields* fields, state_solver* staSol,
                              int indIt, int nbNode, int tag, simu_param* sP)
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x ;

  if(idx < nbNode){
    #if NB_DIM==2
      int i = floorf(idx/(len_y_cst+4)) ;
      int j = idx-i*(len_y_cst+4) ;
      // printf("%i, %i, %i, %.4e\n", idx, i, j, fields->densNum[i][j]) ;

      if ( (B->B[0][i][j]!=B->B[0][i][j]) ||
           (B->B[1][i][j]!=B->B[1][i][j]) ||
           (B->B[2][i][j]!=B->B[2][i][j]) ||
           (B->B[0][i][j] > 1e3) ||
           (B->B[1][i][j] > 1e3) ||
           (B->B[2][i][j] > 1e3)){
        // printf("| it %i, tag %i, B contains a nan value at (%i, %i, %i) (checkGrids_k)\n",
        //        indIt, tag, i, j) ;
        // staSol->invalidB = true ;
        staSol->error_ID = 3;
        staSol->break_solver = true ;
      }
      if ( (fields->E[0][i][j]!=fields->E[0][i][j]) ||
           (fields->E[1][i][j]!=fields->E[1][i][j]) ||
           (fields->E[2][i][j]!=fields->E[2][i][j]) ||
           (fields->E[0][i][j] > 1e3) ||
           (fields->E[1][i][j] > 1e3) ||
           (fields->E[2][i][j] > 1e3)){
        // printf("| it %i, tag %i, E contains a nan value at (%i, %i) (checkGrids_k)\n",
        //       indIt, tag, i, j) ;
        // staSol->invalidE = true ;
        staSol->error_ID = 4;
        staSol->break_solver = true ;
      }
      if ( (fields->J_tot[0][i][j]!=fields->J_tot[0][i][j]) ||
           (fields->J_tot[1][i][j]!=fields->J_tot[1][i][j]) ||
           (fields->J_tot[2][i][j]!=fields->J_tot[2][i][j])){
        // printf("| it %i, tag %i, currTot contains a nan value at (%i, %i, %i) (checkGrids_k)\n",
        //        indIt, tag, i, j, k) ;
        // staSol->invalidCurrTot = true ;
        staSol->error_ID = 5;
        staSol->break_solver = true ;
      }
      if ( (fields->fluxNum[0][i][j]!=fields->fluxNum[0][i][j]) ||
           (fields->fluxNum[1][i][j]!=fields->fluxNum[1][i][j]) ||
           (fields->fluxNum[2][i][j]!=fields->fluxNum[2][i][j])){
        // printf("| it %i, tag %i, curr contains a nan value at (%i, %i, %i) (checkGrids_k)\n",
        //        indIt, tag, i, j, k) ;
        // staSol->invalidCurr = true ;
        staSol->error_ID = 6;
        staSol->break_solver = true ;
      }

      if ((fields->density[i][j]<0) || (fields->density[i][j] != fields->density[i][j])){
        // printf("| it %i, tag %i, densNum contains a value <= 0 or a nan value at (%i, %i) %.2e. (checkGrids_k)\n",
        // indIt, tag, i, j, fields->density[i][j]) ;
        // staSol->invalidDens = true ;
        staSol->error_ID = 7;
        staSol->break_solver = true ;
      }


    #elif NB_DIM==3
      int i = int( idx/((len_y_cst+4)*(len_z_cst+4)) ) ;
      int j = int( (idx - i*(len_y_cst+4)*(len_z_cst+4)) / (len_z_cst+4) ) ;
      int k = idx - i*(len_y_cst+4)*(len_z_cst+4)-j*(len_z_cst+4) ;
      // printf("%i, %i, %i, %.4e\n", idx, i, j, fields->densNum[i][j][k]) ;

      if ( (B->B[0][i][j][k]!=B->B[0][i][j][k]) ||
           (B->B[1][i][j][k]!=B->B[1][i][j][k]) ||
           (B->B[2][i][j][k]!=B->B[2][i][j][k])){
        // printf("| it %i, tag %i, B contains a nan value at (%i, %i, %i) (checkGrids_k)\n",
        //        indIt, tag, i, j, k) ;
        // staSol->invalidB = true ;
        staSol->error_ID = 3;
        staSol->break_solver = true ;
      }
      if ( (fields->E[0][i][j][k]!=fields->E[0][i][j][k]) ||
           (fields->E[1][i][j][k]!=fields->E[1][i][j][k]) ||
           (fields->E[2][i][j][k]!=fields->E[2][i][j][k])){
        // printf("| it %i, tag %i, E contains a nan value at (%i, %i) (checkGrids_k)\n",
              // indIt, tag, i, j) ;
        // staSol->invalidE = true ;
        staSol->error_ID = 4;
        staSol->break_solver = true ;
      }
      if ( (fields->J_tot[0][i][j][k]!=fields->J_tot[0][i][j][k]) ||
           (fields->J_tot[1][i][j][k]!=fields->J_tot[1][i][j][k]) ||
           (fields->J_tot[2][i][j][k]!=fields->J_tot[2][i][j][k])){
        // printf("| it %i, tag %i, currTot contains a nan value at (%i, %i, %i) (checkGrids_k)\n",
        //        indIt, tag, i, j, k) ;
        // staSol->invalidCurrTot = true ;
        staSol->error_ID = 5;
        staSol->break_solver = true ;
      }
      if ( (fields->fluxNum[0][i][j][k]!=fields->fluxNum[0][i][j][k]) ||
           (fields->fluxNum[1][i][j][k]!=fields->fluxNum[1][i][j][k]) ||
           (fields->fluxNum[2][i][j][k]!=fields->fluxNum[2][i][j][k])){
        // printf("| it %i, tag %i, curr contains a nan value at (%i, %i, %i) (checkGrids_k)\n",
        //        indIt, tag, i, j, k) ;
        // staSol->invalidCurr = true ;
        staSol->error_ID = 6;
        staSol->break_solver = true ;
      }

      if ((fields->density[i][j][k]<0) || (fields->density[i][j][k] != fields->density[i][j][k])){
        // printf("| it %i, tag %i, densNum contains a value <= 0 or a nan value at (%i, %i) %.2e. (checkGrids_k)\n",
        // indIt, tag, i, j, fields->densNum[i][j][k]) ;
        // staSol->invalidDens = true ;
        staSol->error_ID = 7;
        staSol->break_solver = true ;
      }
    #endif
  }
}
__global__ void print_state_solver_k(state_solver* sta_sol, int indIt, int tag)
{

  // if (staSol->invalidB){             printf("| it %i, tag %i, invalid B-field.\n", indIt, tag) ; }
  // if (staSol->invalidE){             printf("| it %i, tag %i, invalid E-field.\n", indIt, tag) ; }
  // if (staSol->invalidDens){          printf("| it %i, tag %i, invalid density.\n", indIt, tag) ; }
  // if (staSol->invalidCurr){          printf("| it %i, tag %i, invalid current.\n", indIt, tag) ; }
  // if (staSol->invalidCurrTot){       printf("| it %i, tag %i, invalid total current.\n", indIt, tag) ; }
  // if (staSol->invalid_new_part_idx){ printf("| it %i, tag %i, invalid index for new particle(s).\n", indIt, tag) ; }
  // if (staSol->break_solver){ printf("| it %i, tag %i, I'm sorry what?\n", indIt, tag) ; }
  // if (sta_sol->error_ID != 0){
    // printf("| %s.\n", sta_sol->error_str2[sta_sol->error_ID-1]);
  // }

}


__global__ void HK_run_time_k(house_keeping* HK, int idx_save, float t, int run_time)
{

  HK->simuTime[idx_save] = t ;
  HK->runTime[idx_save] = run_time ;

}

__global__ void HK_nb_part_send_down_k(house_keeping* h_k, int nb_part_send, int idx_save)
{

  h_k->nb_part_comm_up[idx_save] = nb_part_send ;
}
__global__ void HK_nb_part_send_up_k(house_keeping* h_k, int nb_part_send, int idx_save)
{

  h_k->nb_part_comm_down[idx_save] = nb_part_send ;
}

//__________________________________________________________________________________________________________________

#endif
