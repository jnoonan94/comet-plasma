#ifndef __KERNELSPARTICLES_CUH_INCLUDED__
#define __KERNELSPARTICLES_CUH_INCLUDED__

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "parameters.h"


//_________________________________________________________________________________________
//
// Particle kernels dealing with random numbers
//
__global__ void setup_rand_k(curandState * state, unsigned long seed, int n)
{
    int idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (idx < n){
        curand_init(seed, idx, 0, &state[idx]);
    }
}


__device__ float generate(curandState* globalState, int ind)
{
    // Copy state to local memory:
    curandState localState = globalState[ind];
    // Apply uniform distribution with calculated random:
    float rndval = curand_uniform( &localState );
    // Update state:
    globalState[ind] = localState;
    // Return value:
    return rndval;
}

__global__ void rand_uniform_k(float* randomFloatUni, curandState* globalState, int n){

    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    // Only call gen on the kernels we have inited
    // (one per device container element)
    if (idx < n){
        randomFloatUni[idx] = generate(globalState, idx);
    }
}

__device__ float normal_sample_k(float mean = 0.0, float standardDeviation = 1.0, float u1 = 1., float u2 = 1.){
  /* Box-Muller algorithm for sampling a normal distributions with given mean and std.
  **/
  float r = sqrtf( -2.0*logf(u1) );
  float theta = 2.0*PI*u2;
  return mean + standardDeviation*r*sinf(theta);
}



//_________________________________________________________________________________________
//
// Fundamental physical particles kernels.
//
/**
*  Advancing position (Boris scheme), with velocity at interleaved time.
* Part of the boundary treatment is also done here to decrease the amount of pass
* in the particles. Not ideal naming, to be fixed.
*/
__global__ void boris_pos_k(particles* p, simu_grid* grid, simu_param* sP)
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if ( (idx<pool_size_cst) && (p->active[idx]) ){

    p->rx[idx] += sP->dt*p->vx[idx];
    p->ry[idx] += sP->dt*p->vy[idx];
    #if NB_DIM==3
      p->rz[idx] += sP->dt*p->vz[idx];
    #endif




    #if obstacle_cst

        // Rule along x.
        if (p->rx[idx]<grid->xMin){
          p->active[idx] = false;
        }
        ///// ?????
        else if (p->rx[idx]>grid->xMax){
          p->active[idx] = false;
        }
        /////// ?????
        #if ORF_cst
          if (p->rx[idx]>grid->xMax-sP->dX){
            p->active[idx] = false;
          }
        #endif


        // // Rule along y.
        // if (p->ry[idx]<grid->yMin){
        //   #if (!periodic_yz_cst)
        //     if ( (sP->mpi_nb_proc_y==1) || (sP->mpi_rank_y==0) ){
        //       if (p->ID[idx]==0){
        //         p->active[idx] = false;
        //       }
        //       else if (p->ID[idx]==1){
        //         p->active[idx] = false;
        //       }
        //     }
        //   #else
        //     if (p->ID[idx]==0){
        //       if (sP->mpi_nb_proc_y==1){
        //         p->ry[idx] += len_y_cst*sP->dX;
        //       }
        //     }
        //     else {
        //       if ( (sP->mpi_nb_proc_y==1) || (sP->mpi_rank_y==0) ){
        //         if (p->ID[idx]==0){
        //           p->active[idx] = false;
        //         }
        //         else if (p->ID[idx]==1){
        //           p->active[idx] = false;
        //         }
        //       }
        //     }
        //   #endif
        // }
        // //
        // else if (p->ry[idx]>grid->yMax){
        //   #if (!periodic_yz_cst)
        //     if ( (sP->mpi_nb_proc_y==1) || (sP->mpi_rank_y==sP->mpi_nb_proc_y-1) ){
        //       if (p->ID[idx]==0){
        //         p->active[idx] = false;
        //       }
        //       else if (p->ID[idx]==1){
        //         p->active[idx] = false;
        //       }
        //     }
        //   #else
        //     if (p->ID[idx]==0){
        //       if (sP->mpi_nb_proc_y==1){
        //         p->ry[idx] -= len_y_cst*sP->dX;
        //       }
        //     }
        //     else{
        //       if ( (sP->mpi_nb_proc_y==1) || (sP->mpi_rank_y==sP->mpi_nb_proc_y-1) ){
        //         if (p->ID[idx]==0){
        //           p->active[idx] = false;
        //         }
        //         else if (p->ID[idx]==1){
        //           p->active[idx] = false;
        //         }
        //       }
        //     }
        //   #endif
        // }

        // Rule along y.
        if (p->ry[idx]<grid->yMin){
          #if (!periodic_yz_cst)
            if ( (sP->mpi_nb_proc_y==1) || (sP->mpi_rank_y==0) ){
              p->active[idx] = false;
            }
          #else
            if (p->ID[idx]==0){
              if (sP->mpi_nb_proc_y==1){
                p->ry[idx] += len_y_cst*sP->dX;
              }
            }
            else if (p->ID[idx]==1){
              if ( (sP->mpi_nb_proc_y==1) || (sP->mpi_rank_y==0) ){
                p->active[idx] = false;
              }
            }
          #endif
        }
        //
        else if (p->ry[idx]>grid->yMax){
          #if (!periodic_yz_cst)
            if ( (sP->mpi_nb_proc_y==1) || (sP->mpi_rank_y==sP->mpi_nb_proc_y-1) ){
              p->active[idx] = false;
            }
          #else
            if (p->ID[idx]==0){
              if (sP->mpi_nb_proc_y==1){
                p->ry[idx] -= len_y_cst*sP->dX;
              }
            }
            else if (p->ID[idx]==1){
              if ( (sP->mpi_nb_proc_y==1) || (sP->mpi_rank_y==sP->mpi_nb_proc_y-1) ){
                p->active[idx] = false;
              }
            }
          #endif
        }

        // Rule along z.
        #if NB_DIM==3
          if (p->rz[idx]<grid->zMin){
            #if (!periodic_yz_cst)
              if ( (sP->mpi_nb_proc_z==1) || (sP->mpi_rank_z==0) ){
                if (p->ID[idx]==0){
                  p->active[idx] = false;
                }
                else if (p->ID[idx]==1){
                  p->active[idx] = false;
                }
              }
            #else
              if (p->ID[idx]==0){
                if (sP->mpi_nb_proc_z==1){
                  p->rz[idx] += len_z_cst*sP->dX;
                }
              }
              else {
                if ( (sP->mpi_nb_proc_z==1) || (sP->mpi_rank_z==0) ){
                  if (p->ID[idx]==0){
                    p->active[idx] = false;
                  }
                  else if (p->ID[idx]==1){
                    p->active[idx] = false;
                  }
                }
              }
            #endif
          }
          //
          else if (p->rz[idx]>grid->zMax){
            #if (!periodic_yz_cst)
              if ( (sP->mpi_nb_proc_z==1) || (sP->mpi_rank_z==sP->mpi_nb_proc_z-1) ){
                if (p->ID[idx]==0){
                  p->active[idx] = false;
                }
                else if (p->ID[idx]==1){
                  p->active[idx] = false;
                }
              }
            #else
            if (p->ID[idx]==0){
              if (sP->mpi_nb_proc_z==1){
                p->rz[idx] -= len_z_cst*sP->dX;
              }
            }
            else{
              if ( (sP->mpi_nb_proc_z==1) || (sP->mpi_rank_z==sP->mpi_nb_proc_z-1) ){
                if (p->ID[idx]==0){
                  p->active[idx] = false;
                }
                else if (p->ID[idx]==1){
                  p->active[idx] = false;
                }
              }
            }
            #endif
          }
        #endif

      #if solid_body_cst
        #if NB_DIM==2
          if ( ((p->rx[idx]-grid->centre_x)*(p->rx[idx]-grid->centre_x) +
                (p->ry[idx]-grid->centre_y)*(p->ry[idx]-grid->centre_y)) < sP->r_obs_sqr ){
            p->active[idx] = false;
          }
        #elif NB_DIM==3
          // if ( ((p->rx[idx]-grid->centre_x)*(p->rx[idx]-grid->centre_x) +
          //       (p->ry[idx]-grid->centre_y)*(p->ry[idx]-grid->centre_y) +
          //       (p->rz[idx]-grid->centre_z)*(p->rz[idx]-grid->centre_z)) < sP->r_obs_sqr ){
          if ( ((p->rx[idx]-sP->centre_x*grid->xMax)*(p->rx[idx]-sP->centre_x*grid->xMax) +
                (p->ry[idx]-grid->centre_y)*(p->ry[idx]-grid->centre_y) +
                (p->rz[idx]-grid->centre_z)*(p->rz[idx]-grid->centre_z)) < sP->r_obs_sqr ){
            p->active[idx] = false;
          }
        #endif
      #endif

    #else // Periodic along all directions!
      if (p->rx[idx]<grid->xMin){
        p->rx[idx] += len_x_cst*sP->dX;
      }
      else if (p->rx[idx]>grid->xMax){
        p->rx[idx] -= len_x_cst*sP->dX;
      }
      if (sP->mpi_nb_proc_y==1){
        if (p->ry[idx]<grid->yMin){
          p->ry[idx] += len_y_cst*sP->dX;
        }
        else if (p->ry[idx]>grid->yMax){
          p->ry[idx] -= len_y_cst*sP->dX;
        }
      }
      #if NB_DIM==3
        if (sP->mpi_nb_proc_z==1){
          if (p->rz[idx]<grid->zMin){
            p->rz[idx] += len_z_cst*sP->dX;
          }
          else if (p->rz[idx]>grid->zMax){
            p->rz[idx] -= len_z_cst*sP->dX;
          }
        }
      #endif

    #endif



    #if debug_mode_cst
      if (p->ry[idx]>(2.*grid->yMaz)){
        printf("| A particle was pushed twice the size of the box! In boris_pos_k() 1\n");
        p->active[idx] = false;
      }
      else if (p->ry[idx]<(2.*grid->yMin)){
        printf("| A particle was pushed twice the size of the box! In boris_pos_k() 2\n");
        p->active[idx] = false;
      }
    #endif


  }
}

/**
 Advancing velocity according to the Boris scheme, with fields known at
interleaved time.
See great page https://www.particleincell.com/2011/vxb-rotation/ .
Interpolation of the fields at the particles' position using order 2 (5 point stencil)
scheme.
*/
__global__ void boris_vel_k(particles* p, simu_fields* fields, simu_B_field* B, simu_grid* grid, simu_param* sP){


  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if ((idx<pool_size_cst)&&(p->active[idx])){

    #if NB_DIM==2
      float v_minx; float v_miny; float v_minz;
    	float v_primx; float v_primy; float v_primz;
    	float v_plusx; float v_plusy; float v_plusz;

    	float tx; float ty; float tz;
    	float tt;

      float EI[3];
      float BI[3];

      int indX, indY;
      float wx, wy;
      float wx0, wx1, wx2;
      float wy0, wy1, wy2;

      float Z;
      if (p->ID[idx]==0){ Z = 1.; }
      else { Z = float(sP->Z_pla); }

      indX = roundf( (p->rx[idx]-grid->xMin)*sP->dX_i + .5) + 1;
      indY = roundf( (p->ry[idx]-grid->yMin)*sP->dX_i + .5) + 1;

      wx = (p->rx[idx]-grid->xGrid[indX])*sP->dX_i;
      wy = (p->ry[idx]-grid->yGrid[indY])*sP->dX_i;

      wx0 = .5*(.5-wx)*(.5-wx);
      wx1 = .75 - wx*wx;
      wx2 = .5*(.5+wx)*(.5+wx);

      wy0 = .5*(.5-wy)*(.5-wy);
      wy1 = .75 - wy*wy;
      wy2 = .5*(.5+wy)*(.5+wy);


    	EI[0] = fields->E[0][indX-1][indY-1]* wx0*wy0 +
              fields->E[0][indX-1][indY  ]* wx0*wy1 +
              fields->E[0][indX-1][indY+1]* wx0*wy2 +
              fields->E[0][indX  ][indY-1]* wx1*wy0 +
              fields->E[0][indX  ][indY  ]* wx1*wy1 +
              fields->E[0][indX  ][indY+1]* wx1*wy2 +
    	        fields->E[0][indX+1][indY-1]* wx2*wy0 +
    	        fields->E[0][indX+1][indY  ]* wx2*wy1 +
    	        fields->E[0][indX+1][indY+1]* wx2*wy2;
    	EI[1] = fields->E[1][indX-1][indY-1]* wx0*wy0 +
                fields->E[1][indX-1][indY  ]* wx0*wy1 +
                fields->E[1][indX-1][indY+1]* wx0*wy2 +
                fields->E[1][indX  ][indY-1]* wx1*wy0 +
                fields->E[1][indX  ][indY  ]* wx1*wy1 +
                fields->E[1][indX  ][indY+1]* wx1*wy2 +
    	        fields->E[1][indX+1][indY-1]* wx2*wy0 +
    	        fields->E[1][indX+1][indY  ]* wx2*wy1 +
    	        fields->E[1][indX+1][indY+1]* wx2*wy2;
    	EI[2] = fields->E[2][indX-1][indY-1]* wx0*wy0 +
                fields->E[2][indX-1][indY  ]* wx0*wy1 +
                fields->E[2][indX-1][indY+1]* wx0*wy2 +
                fields->E[2][indX  ][indY-1]* wx1*wy0 +
                fields->E[2][indX  ][indY  ]* wx1*wy1 +
                fields->E[2][indX  ][indY+1]* wx1*wy2 +
    	        fields->E[2][indX+1][indY-1]* wx2*wy0 +
    	        fields->E[2][indX+1][indY  ]* wx2*wy1 +
    	        fields->E[2][indX+1][indY+1]* wx2*wy2;
      BI[0] = B->B[0][indX-1][indY-1]* wx0*wy0 +
              B->B[0][indX-1][indY  ]* wx0*wy1 +
              B->B[0][indX-1][indY+1]* wx0*wy2 +
              B->B[0][indX  ][indY-1]* wx1*wy0 +
              B->B[0][indX  ][indY  ]* wx1*wy1 +
              B->B[0][indX  ][indY+1]* wx1*wy2 +
              B->B[0][indX+1][indY-1]* wx2*wy0 +
              B->B[0][indX+1][indY  ]* wx2*wy1 +
              B->B[0][indX+1][indY+1]* wx2*wy2;
      BI[1] = B->B[1][indX-1][indY-1]* wx0*wy0 +
              B->B[1][indX-1][indY  ]* wx0*wy1 +
              B->B[1][indX-1][indY+1]* wx0*wy2 +
              B->B[1][indX  ][indY-1]* wx1*wy0 +
              B->B[1][indX  ][indY  ]* wx1*wy1 +
              B->B[1][indX  ][indY+1]* wx1*wy2 +
              B->B[1][indX+1][indY-1]* wx2*wy0 +
              B->B[1][indX+1][indY  ]* wx2*wy1 +
              B->B[1][indX+1][indY+1]* wx2*wy2;
      BI[2] = B->B[2][indX-1][indY-1]* wx0*wy0 +
              B->B[2][indX-1][indY  ]* wx0*wy1 +
              B->B[2][indX-1][indY+1]* wx0*wy2 +
              B->B[2][indX  ][indY-1]* wx1*wy0 +
              B->B[2][indX  ][indY  ]* wx1*wy1 +
              B->B[2][indX  ][indY+1]* wx1*wy2 +
              B->B[2][indX+1][indY-1]* wx2*wy0 +
              B->B[2][indX+1][indY  ]* wx2*wy1 +
              B->B[2][indX+1][indY+1]* wx2*wy2;
      #if dipole_cst
        BI[0] += B->B_dip[0][indX-1][indY-1]* wx0*wy0 +
                 B->B_dip[0][indX-1][indY  ]* wx0*wy1 +
                 B->B_dip[0][indX-1][indY+1]* wx0*wy2 +
                 B->B_dip[0][indX  ][indY-1]* wx1*wy0 +
                 B->B_dip[0][indX  ][indY  ]* wx1*wy1 +
                 B->B_dip[0][indX  ][indY+1]* wx1*wy2 +
                 B->B_dip[0][indX+1][indY-1]* wx2*wy0 +
                 B->B_dip[0][indX+1][indY  ]* wx2*wy1 +
                 B->B_dip[0][indX+1][indY+1]* wx2*wy2;
        BI[1] += B->B_dip[1][indX-1][indY-1]* wx0*wy0 +
                 B->B_dip[1][indX-1][indY  ]* wx0*wy1 +
                 B->B_dip[1][indX-1][indY+1]* wx0*wy2 +
                 B->B_dip[1][indX  ][indY-1]* wx1*wy0 +
                 B->B_dip[1][indX  ][indY  ]* wx1*wy1 +
                 B->B_dip[1][indX  ][indY+1]* wx1*wy2 +
                 B->B_dip[1][indX+1][indY-1]* wx2*wy0 +
                 B->B_dip[1][indX+1][indY  ]* wx2*wy1 +
                 B->B_dip[1][indX+1][indY+1]* wx2*wy2;
        BI[2] += B->B_dip[2][indX-1][indY-1]* wx0*wy0 +
                 B->B_dip[2][indX-1][indY  ]* wx0*wy1 +
                 B->B_dip[2][indX-1][indY+1]* wx0*wy2 +
                 B->B_dip[2][indX  ][indY-1]* wx1*wy0 +
                 B->B_dip[2][indX  ][indY  ]* wx1*wy1 +
                 B->B_dip[2][indX  ][indY+1]* wx1*wy2 +
                 B->B_dip[2][indX+1][indY-1]* wx2*wy0 +
                 B->B_dip[2][indX+1][indY  ]* wx2*wy1 +
                 B->B_dip[2][indX+1][indY+1]* wx2*wy2;
      #endif

      tx = 1./Z * BI[0] * 0.5 * sP->dt;
      ty = 1./Z * BI[1] * 0.5 * sP->dt;
      tz = 1./Z * BI[2] * 0.5 * sP->dt;

      tt = tx*tx + ty*ty + tz*tz;

      v_minx = p->vx[idx] + 1./Z * EI[0] * 0.5 * sP->dt;
      v_miny = p->vy[idx] + 1./Z * EI[1] * 0.5 * sP->dt;
      v_minz = p->vz[idx] + 1./Z * EI[2] * 0.5 * sP->dt;

      v_primx = v_minx + v_miny*tz - v_minz*ty;
      v_primy = v_miny + v_minz*tx - v_minx*tz;
      v_primz = v_minz + v_minx*ty - v_miny*tx;

      v_plusx = v_minx + v_primy*(2.*tz/(1+tt)) - v_primz*(2.*ty/(1+tt));
      v_plusy = v_miny + v_primz*(2.*tx/(1+tt)) - v_primx*(2.*tz/(1+tt));
      v_plusz = v_minz + v_primx*(2.*ty/(1+tt)) - v_primy*(2.*tx/(1+tt));

      p->vx[idx] = v_plusx + 1./Z * EI[0] * 0.5 * sP->dt;
      p->vy[idx] = v_plusy + 1./Z * EI[1] * 0.5 * sP->dt;
      p->vz[idx] = v_plusz + 1./Z * EI[2] * 0.5 * sP->dt;

    #elif NB_DIM==3

      float v_minx; float v_miny; float v_minz;
    	float v_primx; float v_primy; float v_primz;
    	float v_plusx; float v_plusy; float v_plusz;

    	float tx; float ty; float tz;
    	float tt;

      float EI[3];
      float BI[3];

      int indX, indY, indZ;
      float wx, wy, wz;
      float wx0, wx1, wx2;
      float wy0, wy1, wy2;
      float wz0, wz1, wz2;

      float Z;
      if (p->ID[idx]==0){ Z = 1.; }
      else { Z = float(sP->Z_pla); }

      indX = roundf( (p->rx[idx]-grid->xMin)*sP->dX_i + .5) + 1;
      indY = roundf( (p->ry[idx]-grid->yMin)*sP->dX_i + .5) + 1;
      indZ = roundf( (p->rz[idx]-grid->zMin)*sP->dX_i + .5) + 1;

      wx = (p->rx[idx]-grid->xGrid[indX])*sP->dX_i;
      wy = (p->ry[idx]-grid->yGrid[indY])*sP->dX_i;
      wz = (p->rz[idx]-grid->zGrid[indZ])*sP->dX_i;

      wx0 = .5*(.5-wx)*(.5-wx);
      wx1 = .75 - wx*wx;
      wx2 = .5*(.5+wx)*(.5+wx);

      wy0 = .5*(.5-wy)*(.5-wy);
      wy1 = .75 - wy*wy;
      wy2 = .5*(.5+wy)*(.5+wy);

      wz0 = .5*(.5-wz)*(.5-wz);
      wz1 = .75 - wz*wz;
      wz2 = .5*(.5+wz)*(.5+wz);


    	EI[0] = fields->E[0][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
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
    	EI[1] = fields->E[1][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
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
    	EI[2] = fields->E[2][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
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
      BI[0] = B->B[0][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
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
      BI[1] = B->B[1][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
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
      BI[2] = B->B[2][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
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
      #if dipole_cst
      BI[0] += B->B_dip[0][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
               B->B_dip[0][indX-1][indY-1][indZ  ]* wx0*wy0*wz1 +
               B->B_dip[0][indX-1][indY-1][indZ+1]* wx0*wy0*wz2 +
               B->B_dip[0][indX-1][indY  ][indZ-1]* wx0*wy1*wz0 +
               B->B_dip[0][indX-1][indY  ][indZ  ]* wx0*wy1*wz1 +
               B->B_dip[0][indX-1][indY  ][indZ+1]* wx0*wy1*wz2 +
               B->B_dip[0][indX-1][indY+1][indZ-1]* wx0*wy2*wz0 +
               B->B_dip[0][indX-1][indY+1][indZ  ]* wx0*wy2*wz1 +
               B->B_dip[0][indX-1][indY+1][indZ+1]* wx0*wy2*wz2 +
               B->B_dip[0][indX  ][indY-1][indZ-1]* wx1*wy0*wz0 +
               B->B_dip[0][indX  ][indY-1][indZ  ]* wx1*wy0*wz1 +
               B->B_dip[0][indX  ][indY-1][indZ+1]* wx1*wy0*wz2 +
               B->B_dip[0][indX  ][indY  ][indZ-1]* wx1*wy1*wz0 +
               B->B_dip[0][indX  ][indY  ][indZ  ]* wx1*wy1*wz1 +
               B->B_dip[0][indX  ][indY  ][indZ+1]* wx1*wy1*wz2 +
               B->B_dip[0][indX  ][indY+1][indZ-1]* wx1*wy2*wz0 +
               B->B_dip[0][indX  ][indY+1][indZ  ]* wx1*wy2*wz1 +
               B->B_dip[0][indX  ][indY+1][indZ+1]* wx1*wy2*wz2 +
    	         B->B_dip[0][indX+1][indY-1][indZ-1]* wx2*wy0*wz0 +
    	         B->B_dip[0][indX+1][indY-1][indZ  ]* wx2*wy0*wz1 +
    	         B->B_dip[0][indX+1][indY-1][indZ+1]* wx2*wy0*wz2 +
    	         B->B_dip[0][indX+1][indY  ][indZ-1]* wx2*wy1*wz0 +
    	         B->B_dip[0][indX+1][indY  ][indZ  ]* wx2*wy1*wz1 +
    	         B->B_dip[0][indX+1][indY  ][indZ+1]* wx2*wy1*wz2 +
    	         B->B_dip[0][indX+1][indY+1][indZ-1]* wx2*wy2*wz0 +
    	         B->B_dip[0][indX+1][indY+1][indZ  ]* wx2*wy2*wz1 +
    	         B->B_dip[0][indX+1][indY+1][indZ+1]* wx2*wy2*wz2  ;
      BI[1] += B->B_dip[1][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
               B->B_dip[1][indX-1][indY-1][indZ  ]* wx0*wy0*wz1 +
               B->B_dip[1][indX-1][indY-1][indZ+1]* wx0*wy0*wz2 +
               B->B_dip[1][indX-1][indY  ][indZ-1]* wx0*wy1*wz0 +
               B->B_dip[1][indX-1][indY  ][indZ  ]* wx0*wy1*wz1 +
               B->B_dip[1][indX-1][indY  ][indZ+1]* wx0*wy1*wz2 +
               B->B_dip[1][indX-1][indY+1][indZ-1]* wx0*wy2*wz0 +
               B->B_dip[1][indX-1][indY+1][indZ  ]* wx0*wy2*wz1 +
               B->B_dip[1][indX-1][indY+1][indZ+1]* wx0*wy2*wz2 +
               B->B_dip[1][indX  ][indY-1][indZ-1]* wx1*wy0*wz0 +
               B->B_dip[1][indX  ][indY-1][indZ  ]* wx1*wy0*wz1 +
               B->B_dip[1][indX  ][indY-1][indZ+1]* wx1*wy0*wz2 +
               B->B_dip[1][indX  ][indY  ][indZ-1]* wx1*wy1*wz0 +
               B->B_dip[1][indX  ][indY  ][indZ  ]* wx1*wy1*wz1 +
               B->B_dip[1][indX  ][indY  ][indZ+1]* wx1*wy1*wz2 +
               B->B_dip[1][indX  ][indY+1][indZ-1]* wx1*wy2*wz0 +
               B->B_dip[1][indX  ][indY+1][indZ  ]* wx1*wy2*wz1 +
               B->B_dip[1][indX  ][indY+1][indZ+1]* wx1*wy2*wz2 +
    	         B->B_dip[1][indX+1][indY-1][indZ-1]* wx2*wy0*wz0 +
    	         B->B_dip[1][indX+1][indY-1][indZ  ]* wx2*wy0*wz1 +
    	         B->B_dip[1][indX+1][indY-1][indZ+1]* wx2*wy0*wz2 +
    	         B->B_dip[1][indX+1][indY  ][indZ-1]* wx2*wy1*wz0 +
    	         B->B_dip[1][indX+1][indY  ][indZ  ]* wx2*wy1*wz1 +
    	         B->B_dip[1][indX+1][indY  ][indZ+1]* wx2*wy1*wz2 +
    	         B->B_dip[1][indX+1][indY+1][indZ-1]* wx2*wy2*wz0 +
    	         B->B_dip[1][indX+1][indY+1][indZ  ]* wx2*wy2*wz1 +
    	         B->B_dip[1][indX+1][indY+1][indZ+1]* wx2*wy2*wz2;
      BI[2] += B->B_dip[2][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
               B->B_dip[2][indX-1][indY-1][indZ  ]* wx0*wy0*wz1 +
               B->B_dip[2][indX-1][indY-1][indZ+1]* wx0*wy0*wz2 +
               B->B_dip[2][indX-1][indY  ][indZ-1]* wx0*wy1*wz0 +
               B->B_dip[2][indX-1][indY  ][indZ  ]* wx0*wy1*wz1 +
               B->B_dip[2][indX-1][indY  ][indZ+1]* wx0*wy1*wz2 +
               B->B_dip[2][indX-1][indY+1][indZ-1]* wx0*wy2*wz0 +
               B->B_dip[2][indX-1][indY+1][indZ  ]* wx0*wy2*wz1 +
               B->B_dip[2][indX-1][indY+1][indZ+1]* wx0*wy2*wz2 +
               B->B_dip[2][indX  ][indY-1][indZ-1]* wx1*wy0*wz0 +
               B->B_dip[2][indX  ][indY-1][indZ  ]* wx1*wy0*wz1 +
               B->B_dip[2][indX  ][indY-1][indZ+1]* wx1*wy0*wz2 +
               B->B_dip[2][indX  ][indY  ][indZ-1]* wx1*wy1*wz0 +
               B->B_dip[2][indX  ][indY  ][indZ  ]* wx1*wy1*wz1 +
               B->B_dip[2][indX  ][indY  ][indZ+1]* wx1*wy1*wz2 +
               B->B_dip[2][indX  ][indY+1][indZ-1]* wx1*wy2*wz0 +
               B->B_dip[2][indX  ][indY+1][indZ  ]* wx1*wy2*wz1 +
               B->B_dip[2][indX  ][indY+1][indZ+1]* wx1*wy2*wz2 +
    	         B->B_dip[2][indX+1][indY-1][indZ-1]* wx2*wy0*wz0 +
    	         B->B_dip[2][indX+1][indY-1][indZ  ]* wx2*wy0*wz1 +
    	         B->B_dip[2][indX+1][indY-1][indZ+1]* wx2*wy0*wz2 +
    	         B->B_dip[2][indX+1][indY  ][indZ-1]* wx2*wy1*wz0 +
    	         B->B_dip[2][indX+1][indY  ][indZ  ]* wx2*wy1*wz1 +
    	         B->B_dip[2][indX+1][indY  ][indZ+1]* wx2*wy1*wz2 +
    	         B->B_dip[2][indX+1][indY+1][indZ-1]* wx2*wy2*wz0 +
    	         B->B_dip[2][indX+1][indY+1][indZ  ]* wx2*wy2*wz1 +
    	         B->B_dip[2][indX+1][indY+1][indZ+1]* wx2*wy2*wz2;
      #endif

      // #if (dipole_cst && !ORF_cst)
      //   p->vx[idx] -= sP->v_obs;
      // #endif

      tx = 1./Z * BI[0] * 0.5 * sP->dt;
      ty = 1./Z * BI[1] * 0.5 * sP->dt;
      tz = 1./Z * BI[2] * 0.5 * sP->dt;

      tt = tx*tx + ty*ty + tz*tz;
      v_minx = p->vx[idx] + 1./Z * EI[0] * 0.5 * sP->dt;
      v_miny = p->vy[idx] + 1./Z * EI[1] * 0.5 * sP->dt;
      v_minz = p->vz[idx] + 1./Z * EI[2] * 0.5 * sP->dt;

      v_primx = v_minx + v_miny*tz - v_minz*ty;
      v_primy = v_miny + v_minz*tx - v_minx*tz;
      v_primz = v_minz + v_minx*ty - v_miny*tx;

      v_plusx = v_minx + v_primy*(2.*tz/(1+tt)) - v_primz*(2.*ty/(1+tt));
      v_plusy = v_miny + v_primz*(2.*tx/(1+tt)) - v_primx*(2.*tz/(1+tt));
      v_plusz = v_minz + v_primx*(2.*ty/(1+tt)) - v_primy*(2.*tx/(1+tt));

      // printf("%.2e %.2e %.2e %.2e %.2e\n", p->vx[idx], p->vy[idx], p->vz[idx], EI[0], v_plusx);
      p->vx[idx] = v_plusx + 1./Z * EI[0] * 0.5 * sP->dt;
      // if (p->vx[idx]!=0.){
      //   printf("%.2e %.2e %.2e %.2e %.2e\n", p->vx[idx], p->vy[idx], p->vz[idx], EI[0], v_plusx);
      // }
      p->vy[idx] = v_plusy + 1./Z * EI[1] * 0.5 * sP->dt;
      p->vz[idx] = v_plusz + 1./Z * EI[2] * 0.5 * sP->dt;
      //
      // #if (dipole_cst && !ORF_cst)
      //   p->vx[idx] += sP->v_obs;
      // #endif

    #endif

  }
}




/**
 Advancing velocity according to the Boris scheme, with fields known at
interleaved time.
See great page https://www.particleincell.com/2011/vxb-rotation/ .
Interpolation of the fields at the particles' position using order 2 (5 point stencil)
scheme.
*/
__global__ void boris_vel_test_part_k(particles* p, simu_fields* fields,
                                      simu_B_field* B, simu_grid* grid,
                                      simu_param* sP, int idx_it)
{


  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if ((idx<pool_size_cst)&&(p->active[idx])){

    float v_minx; float v_miny; float v_minz;
    float v_primx; float v_primy; float v_primz;
    float v_plusx; float v_plusy; float v_plusz;

    float tx; float ty; float tz;
    float tt;

    float EI[3];
    float BI[3];

    float Z = 1.;
    //__________________________________________________________________________
    // Oblique Whistler from Hsieh 2017.
    // float Psi = sP->omega*simuTime - sP->kvec_para*p->rz[idx] - sP->kvec_perp*p->rx[idx];
    // float den = ( cos(Psi)*cos(Psi) + sP->As*sP->As*(1-sP->Ap*tan(sP->theta))*(1-sP->Ap*tan(sP->theta))*sin(Psi)*sin(Psi) + tan(sP->theta)*tan(sP->theta)*cos(Psi)*cos(Psi) );
    // den = pow(den, .5);
    // BI[0] = sP->deltaB/den;
    // BI[1] = BI[0]*sP->As*(1-sP->Ap*tan(sP->theta));
    // BI[2] = BI[0]*tan(sP->theta);
    // //
    // EI[0] = BI[0]*sP->vPhase_para*sP->As;
    // EI[1] = BI[0]*sP->vPhase_para;
    // EI[2] = BI[0]*sP->vPhase_para*sP->As*sP->Ap;
    // //
    // BI[0] =   cos(Psi)*BI[0];
    // BI[1] =   sin(Psi)*BI[1];
    // BI[2] =  -cos(Psi)*BI[2] + sP->B0;
    // EI[0] =   sin(Psi)*EI[0];
    // EI[1] =  -cos(Psi)*EI[1];
    // EI[2] =   sin(Psi)*EI[2];


    //__________________________________________________________________________
    // Whistler by hand.
    // BI[0] = sP->deltaB*( cos(-sP->kvec*p->rz[idx] + sP->omega*simuTime) ); //+ cos(kvec*p->rz[idx] + .8*omega*simuTime) + cos(kvec*p->rz[idx] + 1.3*omega*simuTime) );
    // BI[1] = sP->deltaB*( sin(-sP->kvec*p->rz[idx] + sP->omega*simuTime) ); //+ sin(kvec*p->rz[idx] + .8*omega*simuTime) + sin(kvec*p->rz[idx] + 1.3*omega*simuTime) );
    // BI[2] = sP->B0;
    // //
    // EI[0] = sP->deltaE*( cos(-sP->kvec*p->rz[idx] + sP->omega*simuTime - PI/2.) ); //+ cos(kvec*p->rz[idx] + .8*omega*simuTime + PI/2.) + cos(kvec*p->rz[idx] + 1.3*omega*simuTime + PI/2.) );
    // EI[1] = sP->deltaE*( sin(-sP->kvec*p->rz[idx] + sP->omega*simuTime - PI/2.) ); //+ sin(kvec*p->rz[idx] + .8*omega*simuTime + PI/2.) + sin(kvec*p->rz[idx] + 1.3*omega*simuTime + PI/2.) );
    // EI[2] = 0.;
    //__________________________________________________________________________
    // Oblic Whistler.
    // BI[0] =                      sP->deltaB*cos(-sP->kvec*p->rz[idx] + sP->omega*simuTime);
    // BI[1] = -sin(sP->psi)*sP->B0*sP->deltaB*sin(-sP->kvec*p->rz[idx] + sP->omega*simuTime);
    // BI[2] =  cos(sP->psi)*sP->B0 + .5*sP->deltaB*sin(-sP->kvec*p->rz[idx] + sP->omega*simuTime);
    // //
    // EI[0] = sP->deltaE*cos(-sP->kvec*p->rz[idx] + sP->omega*simuTime - PI/2.);
    // EI[1] = sP->deltaE*sin(-sP->kvec*p->rz[idx] + sP->omega*simuTime - PI/2.);
    // EI[2] = .5*sP->deltaE*cos(-sP->kvec*p->rz[idx] + sP->omega*simuTime);
    //__________________________________________________________________________
    // Plane, oblic.
    BI[0] = 0.;
    BI[1] = 0.;//-sin(sP->psi)*B0;
    BI[2] = 0.;// cos(sP->psi)*B0 + sP->deltaB*( sin(sP->kvec*p->rz[idx] + sP->omega*simuTime) );

    EI[0] = sP->delta_E*( sin(sP->k_wave*p->rx[idx] - sP->omega_wave*idx_it*sP->dt) );
    // EI[0] += sP->delta_E*( sin(-sP->k_wave*p->rx[idx] - sP->omega_wave*idx_it*sP->dt) );
    EI[1] = 0.;
    EI[2] = 0.;
    //__________________________________________________________________________
    // Electrostatic, B0 out of plane (y), Ew perpendicular.
    // BI[0] = 0.;
    // BI[1] = 0.;
    // BI[2] = sP->B0;
    // //
    // EI[0] = sP->deltaE*( cos(-sP->kvec*p->rx[idx] + sP->omega*simuTime) );
    // EI[1] = sP->deltaE*( sin(-sP->kvec*p->rz[idx] + sP->omega*simuTime) );
    // EI[2] = 0.;

    tx = 1./Z * BI[0] * 0.5 * sP->dt;
    ty = 1./Z * BI[1] * 0.5 * sP->dt;
    tz = 1./Z * BI[2] * 0.5 * sP->dt;

    tt = tx*tx + ty*ty + tz*tz;

    v_minx = p->vx[idx] + 1./Z * EI[0] * 0.5 * sP->dt;
    v_miny = p->vy[idx] + 1./Z * EI[1] * 0.5 * sP->dt;
    v_minz = p->vz[idx] + 1./Z * EI[2] * 0.5 * sP->dt;

    v_primx = v_minx + v_miny*tz - v_minz*ty;
    v_primy = v_miny + v_minz*tx - v_minx*tz;
    v_primz = v_minz + v_minx*ty - v_miny*tx;

    v_plusx = v_minx + v_primy*(2.*tz/(1+tt)) - v_primz*(2.*ty/(1+tt));
    v_plusy = v_miny + v_primz*(2.*tx/(1+tt)) - v_primx*(2.*tz/(1+tt));
    v_plusz = v_minz + v_primx*(2.*ty/(1+tt)) - v_primy*(2.*tx/(1+tt));

    p->vx[idx] = v_plusx + 1./Z * EI[0] * 0.5 * sP->dt;
    p->vy[idx] = v_plusy + 1./Z * EI[1] * 0.5 * sP->dt;
    p->vz[idx] = v_plusz + 1./Z * EI[2] * 0.5 * sP->dt;
  }
}







__global__ void boundaries_per_spec_k(particles* p, simu_grid* grid, simu_param* sP)
{
  /* Dealing with particles' position at the boundaries, taking their species
  into account.
  **/

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if ((idx<pool_size_cst)&&(p->active[idx])){
    // if (p->ID[idx]==0){  // If the particle is a proton:
      if (p->rx[idx]<grid->xMin){
        p->active[idx] = false;
      }
      // else
      // if (p->rx[idx]>grid->xMax){
      //   p->active[idx] = false;
      // }
      // if (p->ry[idx]<grid->yMin){
      //   p->ry[idx]     += len_y_cst*sP->dX;
      // }
      // else if (p->ry[idx]>grid->yMax){
      //   p->ry[idx]     -= len_y_cst*sP->dX;
      // }
    // }
    // else{  // Else, the particle is a cometary one, it is freed:
    //   if (p->rx[idx]<grid->xMin){
    //     p->active[idx] = false;
    //   }
    //   else if (p->rx[idx]>grid->xMax){
    //     p->active[idx] = false;
    //   }
    //   if (p->ry[idx]<grid->yMin){
    //     p->active[idx] = false;
    //   }
    //   else if (p->ry[idx]>grid->yMax){
    //     p->active[idx] = false;
    //   }
    // }
  }
}



// __global__ void upstream_downstream_boundary_k(particles* p, simu_grid* grid, simu_param* sP, int idx_it)
// {
//   /* Dealing with particles' position at the upstream and downstrem boundaries.
//   **/
//
//   int idx = threadIdx.x + blockIdx.x*blockDim.x;
//
//   if ( (idx<pool_size_cst) && (p->active[idx]) ){
//
//       if (p->rx[idx]<grid->xMin){
//         p->active[idx] = false;
//       }
//       // else if ( idx_it>0 && (p->rx[idx]>(grid->xMax-2*sP->dX*nb_cell_per_shift_cst)) ){
//       else if ( !(inject_turb_cst && idx_it==0) && (p->rx[idx]>(grid->xMax-2*sP->dX*nb_cell_per_shift_cst)) ){
//         p->active[idx] = false;
//       }
//
//   }
// }



//_________________________________________________________________________________________
//
// Particle kernels for more complex setups (adding an obstacle, changing
// reference frame, etc).
//
__global__ void add_part_SW_node_k(particles* p, buff_part* b_p, simu_grid* grid, float* randomFloatUni, simu_param* sP, float dx=0.)
{

  int idx = blockIdx.x*blockDim.x+threadIdx.x;

  if (idx<(2*nb_cell_per_shift_cst*sP->nb_part_add_sw)){
    float sigma = 1./sqrtf(2) * sP->v_thi/sP->v_A ;
    int idx_new = b_p->idx_free[idx];
    #if NB_DIM==2
      int idx_lin = int(idx/nb_part_node_cst);
      int i = int(idx_lin/len_y_cst);
      int j = idx_lin - i*len_y_cst;
    #elif NB_DIM==3
      int idx_lin = int(idx/nb_part_node_cst);
      int i = int( idx_lin/(len_y_cst*len_z_cst) );
      int j = int( (idx_lin - i*(len_y_cst*len_z_cst))/len_z_cst );
      int k = idx_lin - i*(len_y_cst*len_z_cst) - j*len_z_cst;
    #endif
    #if debug_mode_cst
      if (p->active[idx_new]){
        printf("That particle was not available!\n");
      }
    #endif
    p->rx[idx_new] = grid->xMax - randomFloatUni[idx*9]*sP->dX - sP->dX*i + dx;
    // printf("%.4e\n", p->rx[idx_new]);
    p->ry[idx_new] = grid->yGrid[j+2] + (randomFloatUni[idx*9+1]-.5)*sP->dX;
    #if NB_DIM==3
      p->rz[idx_new] = grid->zGrid[k+2] + (randomFloatUni[idx*9+2]-.5)*sP->dX;
    #endif
    p->vx[idx_new] = normal_sample_k(0., sigma, randomFloatUni[idx*9+3], randomFloatUni[idx*9+4]);
    p->vy[idx_new] = normal_sample_k(0., sigma, randomFloatUni[idx*9+5], randomFloatUni[idx*9+6]);
    p->vz[idx_new] = normal_sample_k(0., sigma, randomFloatUni[idx*9+7], randomFloatUni[idx*9+8]);
    p->active[idx_new] = true;
    p->ID[idx_new] = 0;
    // #if (dipole_cst && !ORF_cst)
    //   p->vx[idx_new] -= sP->v_obs;
    // #endif
  }
}

__global__ void inject_SW_ORF_k(particles* p, buff_part* b_p, simu_grid* grid, float* randomFloatUni, simu_param* sP)
{
  int idx = blockIdx.x*blockDim.x+threadIdx.x;

  if (idx<sP->nb_part_add_sw){
    float sigma = 1./sqrtf(2) * sP->v_thi/sP->v_A ;
    int idx_new = b_p->idx_free[idx];
    #if NB_DIM==2
      // int nb_part_node_ORF = sP->nb_part_add_ORF/len_y_cst;
      int j = idx/nb_part_node_cst;
    #elif NB_DIM==3
      // int nb_part_node_ORF = sP->nb_part_add_ORF/(len_y_cst*len_z_cst);
      int idx_lin = idx/nb_part_node_cst;
      int j = idx_lin/len_z_cst;
      int k = idx_lin - j*len_z_cst;
    #endif
    p->rx[idx_new] = grid->xMax - randomFloatUni[idx*9  ]*sP->dX;//sP->delta_x_inj_ORF;
    // p->ry[idx_new] = grid->yMin + randomFloatUni[idx*9+1]*sP->dX*len_y_cst;
    p->ry[idx_new] = grid->yGrid[j+2] + (randomFloatUni[idx*9+1]-.5)*sP->dX;
    #if NB_DIM==3
      // p->rz[idx_new] = grid->zMin + randomFloatUni[idx*9+2]*sP->dX*len_z_cst;
      p->rz[idx_new] = grid->zGrid[k+2] + (randomFloatUni[idx*9+2]-.5)*sP->dX;
    #endif
    p->vx[idx_new] = normal_sample_k(-sP->v_obs, sigma, randomFloatUni[idx*9+3], randomFloatUni[idx*9+4]);
    p->vy[idx_new] = normal_sample_k(        0., sigma, randomFloatUni[idx*9+5], randomFloatUni[idx*9+6]);
    p->vz[idx_new] = normal_sample_k(        0., sigma, randomFloatUni[idx*9+7], randomFloatUni[idx*9+8]);
    p->active[idx_new] = true;
    p->ID[idx_new] = 0  ;
  }
}

__global__ void print_tmp_k( buff_part* b_p, int tag)
{
  for (int j=0; j<100; j++){
    printf("yo ye %i %i %i\n", tag, j, b_p->exosphere[1][j]);
  }
}
__global__ void add_part_exo_node_k(particles* p, buff_part* b_p, simu_grid* grid,
                                    float* randomFloatUni, int indIt, simu_param* sP)
{

  int idx    = blockIdx.x*blockDim.x+threadIdx.x;

  if (idx<sP->nb_part_add_pla){

    int newInd = b_p->idx_free[idx];
    int i = b_p->exosphere[0][idx];
    int j = b_p->exosphere[1][idx];
    #if NB_DIM==3
      int k = b_p->exosphere[2][idx];
    #endif

    p->rx[newInd] = grid->xGrid[i] + (randomFloatUni[idx*3  ]-.5)*sP->dX;
    p->ry[newInd] = grid->yGrid[j] + (randomFloatUni[idx*3+1]-.5)*sP->dX;
    #if NB_DIM==3
      p->rz[newInd] = grid->zGrid[k] + (randomFloatUni[idx*3+2]-.5)*sP->dX;
    #endif
    p->vx[newInd] = sP->u0/sP->v_A * p->rx[newInd]/std::sqrt(p->rx[newInd]*p->rx[newInd] + p->ry[newInd]*p->ry[newInd]);
    p->vy[newInd] = sP->u0/sP->v_A * p->ry[newInd]/std::sqrt(p->rx[newInd]*p->rx[newInd] + p->ry[newInd]*p->ry[newInd]);
    p->vz[newInd] = 0.;
    p->active[newInd] = true;
    p->ID[newInd] = 1;
    #if !ORF_cst
      p->rx[newInd] += indIt%nb_it_per_shift_cst*sP->v_obs*sP->dt;
      p->vx[newInd] += sP->v_obs;
    #endif


    #if debug_mode_cst
      if (newInd>=pool_size_cst){
        printf("| Out of particle pool, add_part_exo_node_k.\n");
      }
      if (j>len_y_cst+3){
        printf("| j>len_y_cst+3, add_part_exo_node_k.\n");
      }
      if (  p->active[newInd] ){
        printf("| This particle was not available! add_part_exo_k\n");
      }
      if (p->rx[newInd]>grid->xMax || p->rx[newInd]<grid->xMin){
        printf("| Issue here x. %.4e %.4e %.4e %.4e\n", grid->xMin, grid->xGrid[i], (randomFloatUni[idx*3 ]-.5)*sP->dX, grid->xMax );
      }
      if (p->ry[newInd]>grid->yMax || p->ry[newInd]<grid->yMin){
        printf("| Issue here y. %i %.4e %.4e %.4e %.4e\n", idx, j, grid->yGrid[j], (randomFloatUni[idx*3+1]-.5)*sP->dX, grid->yMax );
      }
    #endif
  }
}
__global__ void add_proba_exo_k(particles* p, buff_part* b_p, simu_grid* grid,
                                float* randomFloatUni, int indIt, simu_param* sP,
                                state_solver* sta_sol, house_keeping* h_k)
{
  int idx = blockIdx.x*blockDim.x+threadIdx.x;

  int idx_save = int(indIt/rate_save_t_cst);

  if (idx<nb_nodes_cst){
    #if NB_DIM==2
      int i = int(idx/len_y_cst);
      int j = idx - i*len_y_cst;
      i += 2;
      j += 2;
    #elif NB_DIM==3
      int i = int(idx/(len_y_cst*len_z_cst));
      int j = int( (idx-i*len_y_cst*len_z_cst)/len_z_cst );
      int k = idx - i*len_y_cst*len_z_cst - j*len_z_cst;
      i += 2;
      j += 2;
      k += 2;
    #endif

    if (randomFloatUni[idx*3+2]<b_p->exo_proba[idx]){
      //
      int my_idx = atomicAdd(&b_p->next_idx, 1);
      int newInd = b_p->idx_free[my_idx];
      //
      p->rx[newInd] = grid->xGrid[i] + (randomFloatUni[idx*3  ]-.5)*sP->dX;
      p->ry[newInd] = grid->yGrid[j] + (randomFloatUni[idx*3+1]-.5)*sP->dX;
      #if NB_DIM==3
        p->rz[newInd] = grid->zGrid[k] + (randomFloatUni[idx*3+2]-.5)*sP->dX;
      #endif
      p->vx[newInd] = sP->u0/sP->v_A * p->rx[newInd]/std::sqrt(p->rx[newInd]*p->rx[newInd] + p->ry[newInd]*p->ry[newInd]);
      p->vy[newInd] = sP->u0/sP->v_A * p->ry[newInd]/std::sqrt(p->rx[newInd]*p->rx[newInd] + p->ry[newInd]*p->ry[newInd]);
      p->vz[newInd] = 0.;
      #if !ORF_cst
        p->rx[newInd] += indIt%nb_it_per_shift_cst*sP->v_obs*sP->dt;
        p->vx[newInd] += sP->v_obs;
      #endif
      p->active[newInd] = true;
      p->ID[newInd] = 1;
      if (idx_save<len_save_t_cst){
        atomicAdd(&h_k->nb_pla_proba[idx_save], 1);
      }


      #if debug_mode_cst
        if (my_idx>=buff_size_cst){
          printf("| Issue here in add_proba_exo_k.\n");
        }
        if (newInd>=pool_size_cst){
          printf("| Out of particle pool, add_proba_exo_k.\n");
        }
        if (p->active[newInd]){
          printf("| This particle was not available! add_proba_exo_k\n");
        }
        if (j>len_y_cst+3){
          printf("| j>len_y_cst+3, add_proba_exo_node_k.\n");
        }
        if (p->rx[newInd]>grid->xMax || p->rx[newInd]<grid->xMin){
          printf("| Issue here yeet x. %.4e %.4e %.4e %.4e\n", grid->xMin, grid->xGrid[i], (randomFloatUni[idx*3  ]-.5)*sP->dX, grid->xMax );
        }
        if (p->ry[newInd]>grid->yMax || p->ry[newInd]<grid->yMin){
          printf("| Issue here yeet y. %i %.4e %.4e %.4e %.4e\n", idx, j, grid->yGrid[j], (randomFloatUni[idx*3+1]-.5)*sP->dX, grid->yMax );
        }
      #endif
    }
  }
}

__global__ void add_part_ionosphere_k(particles* p, buff_part* b_p, simu_grid* grid, float* randomFloatUni, int indIt, simu_param* sP)
{

  int idx    = blockIdx.x*blockDim.x+threadIdx.x;

  if (idx<sP->nb_part_add_pla){

    int newInd = b_p->idx_free[idx];
    float theta = idx*2*PI/sP->nb_part_add_pla;
    p->rx[newInd] = grid->centre_x + sP->r_obs*cos(theta);
    p->ry[newInd] = grid->centre_y + sP->r_obs*sin(theta);
    // #if NB_DIM==3
    //   p->rz[newInd] = grid->zGrid[k] + (randomFloatUni[idx*3+2]-.5)*sP->dX;
    // #endif
    // if (p->rx[newInd]>grid->xMax || p->rx[newInd]<grid->xMin){
    //   printf("Issue here. %.4e %.4e %.4e %.4e\n", grid->xMin, grid->xGrid[j], (randomFloatUni[idx*2+1]-.5)*sP->dX, grid->xMax );
    // }
    // if (p->ry[newInd]>grid->yMax || p->ry[newInd]<grid->yMin){
    //   printf("Issue here y. %.4e %.4e %.4e %.4e\n", grid->yMin, grid->yGrid[j], (randomFloatUni[idx*2+1]-.5)*sP->dX, grid->yMax );
    // }
    p->vx[newInd] = 0.;//sP->u0/sP->v_A*p->rx[newInd]/std::sqrt(p->rx[newInd]*p->rx[newInd] + p->ry[newInd]*p->ry[newInd]);
    p->vy[newInd] = 0.;//sP->u0/sP->v_A*p->ry[newInd]/std::sqrt(p->rx[newInd]*p->rx[newInd] + p->ry[newInd]*p->ry[newInd]);
    p->vz[newInd] = 0.;
    p->active[newInd] = true;
    p->ID[newInd] = 1;
  }
}

__global__ void reset_idx_free_k(buff_part* b_p)
{
  b_p->next_idx = 0;
}
__global__ void update_idx_free_k(particles* p, buff_part* b_p)
{
  /** Kernel searching all free particles' spots in device memory and updating the
  buffer accordingly.
        google: cuda make operation atomic ==>
        https://forums.developer.nvidia.com/t/can-one-force-two-operations-to-occur-atomically-together/38563 **/
  // All particles in memory are visited, with index idx:
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  // If the particle idx actually exists && is free:
  if ( (idx<pool_size_cst) && (p->active[idx]==false) ){
    // We will add its index to the buffer of free particles:
    //   -> We save the previous index of the buffer in my_idx, indicating where to write:
    //       no other thread can possibly write there as well (atomic operation)!
    //   -> If this index doesn't exceed the buffer's size, we write the index
    //       of the particle in the global memory, i.e. idx.
    int my_idx = atomicAdd(&b_p->next_idx, 1);
    //
    if (my_idx < buff_size_cst){
      b_p->idx_free[my_idx] = idx;
    }
  }
}





__global__ void reset_idx_comm_k(buff_part* b_p)
{
  b_p->next_idx_comm = 0;
}
__global__ void print_idx_comm_k(buff_part* b_p, int mpi_rank)
{
  printf("  %i particles to communicate on proc %i.\n", b_p->next_idx_comm, mpi_rank);
}
__global__ void search_idx_down_y_k(particles* p, simu_grid* grid, buff_part* b_p, simu_param* sP, int idx_it)
{

  int idx = blockIdx.x*blockDim.x+threadIdx.x;

  if ( (idx<pool_size_cst) && (p->ry[idx]<grid->yMin) && (p->active[idx]) ){

    int my_idx = atomicAdd(&b_p->next_idx_comm, 1);
    if (my_idx < buff_size_cst){
        #if NB_DIM==2
          b_p->buff_send[my_idx*6  ] = p->rx[idx];
          b_p->buff_send[my_idx*6+1] = p->ry[idx];
          b_p->buff_send[my_idx*6+2] = p->vx[idx];
          b_p->buff_send[my_idx*6+3] = p->vy[idx];
          b_p->buff_send[my_idx*6+4] = p->vz[idx];
          b_p->buff_send[my_idx*6+5] = float(p->ID[idx]);
          // The particle is freed on this process:
          p->active[idx] = false;
          // Periodicity!:
          if (sP->mpi_rank_y==0){
            b_p->buff_send[my_idx*6+1] += sP->mpi_nb_proc_y*len_y_cst*sP->dX;
          }
        #elif NB_DIM==3
          b_p->buff_send[my_idx*7  ] = p->rx[idx];
          b_p->buff_send[my_idx*7+1] = p->ry[idx];
          b_p->buff_send[my_idx*7+2] = p->rz[idx];
          b_p->buff_send[my_idx*7+3] = p->vx[idx];
          b_p->buff_send[my_idx*7+4] = p->vy[idx];
          b_p->buff_send[my_idx*7+5] = p->vz[idx];
          b_p->buff_send[my_idx*7+6] = float(p->ID[idx]);
          p->active[idx] = false; // The particle is freed on this process.

          if (sP->mpi_rank_y==0){ // Periodicity!
            b_p->buff_send[my_idx*7+1] += sP->mpi_nb_proc_y*len_y_cst*sP->dX;
          }
        #endif

    }
    else{
      printf("PAS BON, search_ind_down_y, more particles to communicate than slots in the buffer, %i.\n", my_idx);
    }
  }
}
__global__ void search_idx_up_y_k(particles* p, simu_grid* grid, buff_part* b_p, simu_param* sP)
{

  int idx = blockIdx.x*blockDim.x+threadIdx.x;

  if ( (idx<pool_size_cst) && (p->ry[idx]>grid->yMax) && (p->active[idx]) ){

    int my_idx = atomicAdd(&b_p->next_idx_comm, 1);
    if (my_idx < buff_size_cst){
      #if NB_DIM==2
        b_p->buff_send[my_idx*6  ] = p->rx[idx];
        b_p->buff_send[my_idx*6+1] = p->ry[idx];
        b_p->buff_send[my_idx*6+2] = p->vx[idx];
        b_p->buff_send[my_idx*6+3] = p->vy[idx];
        b_p->buff_send[my_idx*6+4] = p->vz[idx];
        b_p->buff_send[my_idx*6+5] = float(p->ID[idx]);
        // The particle is freed on this process:
        p->active[idx] = false;
        // Periodicity! :
        if ( sP->mpi_rank_y==(sP->mpi_nb_proc_y-1) ){
          b_p->buff_send[my_idx*6+1] -= sP->mpi_nb_proc_y*len_y_cst*sP->dX;
        }
      #elif NB_DIM==3
        b_p->buff_send[my_idx*7  ] = p->rx[idx];
        b_p->buff_send[my_idx*7+1] = p->ry[idx];
        b_p->buff_send[my_idx*7+2] = p->rz[idx];
        b_p->buff_send[my_idx*7+3] = p->vx[idx];
        b_p->buff_send[my_idx*7+4] = p->vy[idx];
        b_p->buff_send[my_idx*7+5] = p->vz[idx];
        b_p->buff_send[my_idx*7+6] = float(p->ID[idx]);
        // The particle is freed on this process:
        p->active[idx] = false;
        // Periodicity! :
        if ( sP->mpi_rank_y==(sP->mpi_nb_proc_y-1) ){
          b_p->buff_send[my_idx*7+1] -= sP->mpi_nb_proc_y*len_y_cst*sP->dX;
        }
      #endif

    }
    else{
      printf("PAS BON, rank y %i rank z %i, search_ind_up_y, more part to comm than slots in the buffer, %i.\n", sP->mpi_rank_y, sP->mpi_rank_z, my_idx);
      // printf("idx: %i, min: %.2e, max: %.2e, rx: %.2e\n", idx, grid->xMin, grid->xMax, p->rx[idx]);
    }
  }
}
#if NB_DIM==3
__global__ void search_idx_down_z_k(particles* p, simu_grid* grid, buff_part* b_p, simu_param* sP, int idx_it)
{

  int idx = blockIdx.x*blockDim.x+threadIdx.x;

  if ( (idx<pool_size_cst) && (p->rz[idx]<grid->zMin) && (p->active[idx]) ){

    int my_idx = atomicAdd(&b_p->next_idx_comm, 1);
    if (my_idx < buff_size_cst){
        #if NB_DIM==2
          printf("| IT DOES NOT MAKE SENSE.\n")
        #elif NB_DIM==3
          b_p->buff_send[my_idx*7  ] = p->rx[idx];
          b_p->buff_send[my_idx*7+1] = p->ry[idx];
          b_p->buff_send[my_idx*7+2] = p->rz[idx];
          b_p->buff_send[my_idx*7+3] = p->vx[idx];
          b_p->buff_send[my_idx*7+4] = p->vy[idx];
          b_p->buff_send[my_idx*7+5] = p->vz[idx];
          b_p->buff_send[my_idx*7+6] = float(p->ID[idx]);
          p->active[idx] = false; // The particle is freed on this process.

          if (sP->mpi_rank_z==0){ // Periodicity!
            b_p->buff_send[my_idx*7+2] += sP->mpi_nb_proc_z*len_z_cst*sP->dX;
          }
        #endif

    }
    else{
      printf("PAS BON, search_ind_down_z, more particles to communicate than slots in the buffer, %i.\n", my_idx);
    }
  }
}
__global__ void search_idx_up_z_k(particles* p, simu_grid* grid, buff_part* b_p, simu_param* sP)
{

  int idx = blockIdx.x*blockDim.x+threadIdx.x;

  if ( (idx<pool_size_cst) && (p->rz[idx]>grid->zMax) && (p->active[idx]) ){

    int my_idx = atomicAdd(&b_p->next_idx_comm, 1);
    if (my_idx < buff_size_cst){
      #if NB_DIM==2
        printf("| IT DOES NOT MAKE SENSE.\n")
      #elif NB_DIM==3
        b_p->buff_send[my_idx*7  ] = p->rx[idx];
        b_p->buff_send[my_idx*7+1] = p->ry[idx];
        b_p->buff_send[my_idx*7+2] = p->rz[idx];
        b_p->buff_send[my_idx*7+3] = p->vx[idx];
        b_p->buff_send[my_idx*7+4] = p->vy[idx];
        b_p->buff_send[my_idx*7+5] = p->vz[idx];
        b_p->buff_send[my_idx*7+6] = float(p->ID[idx]);
        // The particle is freed on this process:
        p->active[idx] = false;
        // Periodicity! :
        if ( sP->mpi_rank_z==(sP->mpi_nb_proc_z-1) ){
          b_p->buff_send[my_idx*7+2] -= sP->mpi_nb_proc_z*len_z_cst*sP->dX;
        }
      #endif

    }
    else{
      printf("PAS BON, search_ind_up_z, more particles to communicate than slots in the buffer, %i.\n", my_idx);
      // printf("idx: %i, min: %.2e, max: %.2e, rx: %.2e\n", idx, grid->xMin, grid->xMax, p->rx[idx]);
    }
  }
}
#endif
__global__ void update_nb_part_rece_k(buff_part* b_p, int nb_part_rece)
{
  b_p->nb_part_rece = nb_part_rece;
}
__global__ void copy_buff_part_k(particles* p, buff_part* b_p)
{

  int idx = blockIdx.x*blockDim.x+threadIdx.x;

  if (idx<b_p->nb_part_rece){
    int newInd    = b_p->idx_free[idx];
    #if NB_DIM==2
      p->rx[newInd] = b_p->buff_rece[idx*6  ];
      p->ry[newInd] = b_p->buff_rece[idx*6+1];
      p->vx[newInd] = b_p->buff_rece[idx*6+2];
      p->vy[newInd] = b_p->buff_rece[idx*6+3];
      p->vz[newInd] = b_p->buff_rece[idx*6+4];
      p->active[newInd] = true;
      p->ID[newInd] = int(b_p->buff_rece[idx*6+5]);
    #elif NB_DIM==3
      p->rx[newInd] = b_p->buff_rece[idx*7  ];
      p->ry[newInd] = b_p->buff_rece[idx*7+1];
      p->rz[newInd] = b_p->buff_rece[idx*7+2];
      p->vx[newInd] = b_p->buff_rece[idx*7+3];
      p->vy[newInd] = b_p->buff_rece[idx*7+4];
      p->vz[newInd] = b_p->buff_rece[idx*7+5];
      p->ID[newInd] = int(b_p->buff_rece[idx*7+6]);
      p->active[newInd] = true;

    #endif
  }

}

__global__ void shift_particles_k(particles* p, simu_param* sP, simu_grid* grid, int idx_it)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if ( (idx<pool_size_cst) && (p->active[idx]) ){

    p->rx[idx] -= nb_cell_per_shift_cst*sP->dX;

    /* We immediately deal with particles' position at the upstream and downstrem boundaries.*/
    if (p->rx[idx]<grid->xMin){
      /* In any case, particles dowstream of the box are de-activated. */
      p->active[idx] = false;
    }
    else if ( !(inject_turb_cst && idx_it==0) && (p->rx[idx]>(grid->xMax-2*sP->dX*nb_cell_per_shift_cst)) ){
      /* Since we work in the solar wind reference frame, particles at upstream
      * boundary can actually significantly escape the domain there as well. To
      * mitigate this loss, in laminar or turbulent runs, we remove all particles
      * in a slice with width twice the number of cell-per-shift, and re-populate
      * the first with either the previous turbulent slice or nominal laminar
      * parameters. */
      p->active[idx] = false;
    }

  }
}


__global__ void inject_part_k(injector* injec, particles* p, buff_part* b_p, simu_grid* grid,
                              simu_param* sP, int nb_part_add, bool reinjection)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;  // The index of the particle to add.

  if ( idx<nb_part_add ){
    float dx = 0.;
    if (reinjection){
      dx = -nb_cell_per_shift_cst*sP->dX;
    }
    int newInd  = b_p->idx_free[idx];  // This is a free index within p.
    if (p->active[newInd]){
      printf("| This particle was not available!  inject_part_k\n");
    }
    p->rx[newInd] = injec->rx[idx] + dx;
    p->ry[newInd] = injec->ry[idx];
    #if NB_DIM==3
      p->rz[newInd] = injec->rz[idx];
    #endif
    p->vx[newInd] = injec->vx[idx];
    p->vy[newInd] = injec->vy[idx];
    p->vz[newInd] = injec->vz[idx];
    p->active[newInd] = true;
    p->ID[newInd] = 0;

    if (p->rx[newInd]<0 || p->rx[newInd]>grid->xMax){
      printf("Particle injected outside the domain in x. inject_part_k\n");
    }
    if (p->ry[newInd]<0 || p->ry[newInd]>grid->yMax){
      printf("Particle injected outside the domain in y. inject_part_k\n");
    }
  }
}




#endif
