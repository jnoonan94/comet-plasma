#ifndef __KERNELSPARTICLESTOGRID_CUH_INCLUDED__   // if x.h hasn't been included yet...
#define __KERNELSPARTICLESTOGRID_CUH_INCLUDED__

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "parameters.h"


__global__ void part2grid_counts_flux_num_k(particles* p, simu_grid* grid, simu_fields* fields, simu_param* sP, int ind_it, int flag){

  int idx = threadIdx.x + blockIdx.x*blockDim.x ;

  if ( idx<pool_size_cst && p->active[idx] ){

    #if NB_DIM==2
      int indX, indY ;
      float wx, wy ;
      float wx0, wx1, wx2 ;
      float wy0, wy1, wy2 ;

      indX = roundf( (p->rx[idx]-grid->xMin)*sP->dX_i + .5) + 1 ;
      indY = roundf( (p->ry[idx]-grid->yMin)*sP->dX_i + .5) + 1 ;

      wx = (p->rx[idx]-grid->xGrid[indX])*sP->dX_i ;
      wy = (p->ry[idx]-grid->yGrid[indY])*sP->dX_i ;

      wx0 = .5*(.5-wx)*(.5-wx) ;
      wx1 = .75 - wx*wx ;
      wx2 = .5*(.5+wx)*(.5+wx) ;

      wy0 = .5*(.5-wy)*(.5-wy) ;
      wy1 = .75 - wy*wy ;
      wy2 = .5*(.5+wy)*(.5+wy) ;


      #if debug_mode_cst
        if ( (indX<1) || (indX>len_x_cst+2) ){
          printf("| !! It %i.%i, particle in part2grid_counts_flux_num_k, indX %i, rx %.4e, rank %i, ID %i, idx %i\n", ind_it, flag, indX, p->rx[idx], sP->mpi_rank, p->ID[idx], idx) ;
          p->active[idx] = false ;
        }
        else if ( (indY<1) || (indY>len_y_cst+2) ){
          printf("| !! It %i.%i, particle in part2grid_counts_flux_num_k, indY %i, ry %.4e, rank %i, ID %i, idx %i\n", ind_it, flag, indY, p->ry[idx], sP->mpi_rank, p->ID[idx], idx) ;
          p->active[idx] = false ;
        }
        if ( (indX<1) || (indX>len_x_cst+2) ){
          printf("| !! It %i.%i, particle in part2grid_counts_flux_num_k, indX %i, rx %.4e, rank %i, ID %i, idx %i\n", ind_it, flag, indX, p->rx[idx], sP->mpi_rank, p->ID[idx], idx) ;
          p->active[idx] = false ;
        }
        else if ( (indY<1) || (indY>len_y_cst+2) ){
          printf("| !! It %i.%i, particle in part2grid_counts_flux_num_k, indY %i, ry %.4e, rank %i, ID %i, idx %i\n", ind_it, flag, indY, p->ry[idx], sP->mpi_rank, p->ID[idx], idx) ;
          p->active[idx] = false ;
        }
        else if (wx0>1 || wx0<0 || wx1>1 || wx1<0 || wx2>1 || wx2<0 ){
          printf("| !! no no no.\n") ;
        }
        else if (wy0>1 || wy0<0 || wy1>1 || wy1<0 || wy2>1 || wy2<0 ){
          printf("| !! ne ne ne.\n") ;
        }
      #endif


      /// Here below, the clean logic of the algorithm would be indX<2 andor indX>len_x_cst+1,
      ///  because otherwise it would mean a particle has passed the xMin xMax and is still
      ///  valid on this process. It happens/appears that a few particles end up with
      ///  EXACTLY the position xMin or xMax (same for y). This is not repeatable because
      ///  of ... what the memory was conainting before runtime? + rounding error?

      if (p->ID[idx]==0){
    		atomicAdd(&fields->counts[indX-1][indY-1], wx0*wy0 ) ;
        atomicAdd(&fields->counts[indX-1][indY  ], wx0*wy1 ) ;
        atomicAdd(&fields->counts[indX-1][indY+1], wx0*wy2 ) ;
        atomicAdd(&fields->counts[indX  ][indY-1], wx1*wy0 ) ;
        atomicAdd(&fields->counts[indX  ][indY  ], wx1*wy1 ) ;
        atomicAdd(&fields->counts[indX  ][indY+1], wx1*wy2 ) ;
    		atomicAdd(&fields->counts[indX+1][indY-1], wx2*wy0 ) ;
    		atomicAdd(&fields->counts[indX+1][indY  ], wx2*wy1 ) ;
    		atomicAdd(&fields->counts[indX+1][indY+1], wx2*wy2 ) ;

        atomicAdd(&fields->fluxNum[0][indX-1][indY-1] , p->vx[idx] * wx0*wy0 ) ;
        atomicAdd(&fields->fluxNum[0][indX-1][indY  ] , p->vx[idx] * wx0*wy1 ) ;
        atomicAdd(&fields->fluxNum[0][indX-1][indY+1] , p->vx[idx] * wx0*wy2 ) ;
        atomicAdd(&fields->fluxNum[0][indX  ][indY-1] , p->vx[idx] * wx1*wy0 ) ;
        atomicAdd(&fields->fluxNum[0][indX  ][indY  ] , p->vx[idx] * wx1*wy1 ) ;
        atomicAdd(&fields->fluxNum[0][indX  ][indY+1] , p->vx[idx] * wx1*wy2 ) ;
        atomicAdd(&fields->fluxNum[0][indX+1][indY-1] , p->vx[idx] * wx2*wy0 ) ;
        atomicAdd(&fields->fluxNum[0][indX+1][indY  ] , p->vx[idx] * wx2*wy1 ) ;
        atomicAdd(&fields->fluxNum[0][indX+1][indY+1] , p->vx[idx] * wx2*wy2 ) ;
        //______________
        atomicAdd(&fields->fluxNum[1][indX-1][indY-1] , p->vy[idx] * wx0*wy0 ) ;
        atomicAdd(&fields->fluxNum[1][indX-1][indY  ] , p->vy[idx] * wx0*wy1 ) ;
        atomicAdd(&fields->fluxNum[1][indX-1][indY+1] , p->vy[idx] * wx0*wy2 ) ;
        atomicAdd(&fields->fluxNum[1][indX  ][indY-1] , p->vy[idx] * wx1*wy0 ) ;
        atomicAdd(&fields->fluxNum[1][indX  ][indY  ] , p->vy[idx] * wx1*wy1 ) ;
        atomicAdd(&fields->fluxNum[1][indX  ][indY+1] , p->vy[idx] * wx1*wy2 ) ;
        atomicAdd(&fields->fluxNum[1][indX+1][indY-1] , p->vy[idx] * wx2*wy0 ) ;
        atomicAdd(&fields->fluxNum[1][indX+1][indY  ] , p->vy[idx] * wx2*wy1 ) ;
        atomicAdd(&fields->fluxNum[1][indX+1][indY+1] , p->vy[idx] * wx2*wy2 ) ;
        //______________
        atomicAdd(&fields->fluxNum[2][indX-1][indY-1] , p->vz[idx] * wx0*wy0 ) ;
        atomicAdd(&fields->fluxNum[2][indX-1][indY  ] , p->vz[idx] * wx0*wy1 ) ;
        atomicAdd(&fields->fluxNum[2][indX-1][indY+1] , p->vz[idx] * wx0*wy2 ) ;
        atomicAdd(&fields->fluxNum[2][indX  ][indY-1] , p->vz[idx] * wx1*wy0 ) ;
        atomicAdd(&fields->fluxNum[2][indX  ][indY  ] , p->vz[idx] * wx1*wy1 ) ;
        atomicAdd(&fields->fluxNum[2][indX  ][indY+1] , p->vz[idx] * wx1*wy2 ) ;
        atomicAdd(&fields->fluxNum[2][indX+1][indY-1] , p->vz[idx] * wx2*wy0 ) ;
        atomicAdd(&fields->fluxNum[2][indX+1][indY  ] , p->vz[idx] * wx2*wy1 ) ;
        atomicAdd(&fields->fluxNum[2][indX+1][indY+1] , p->vz[idx] * wx2*wy2 ) ;
      }
      else {
    		atomicAdd(&fields->counts_pla[indX-1][indY-1], wx0*wy0 ) ;
        atomicAdd(&fields->counts_pla[indX-1][indY  ], wx0*wy1 ) ;
        atomicAdd(&fields->counts_pla[indX-1][indY+1], wx0*wy2 ) ;
        atomicAdd(&fields->counts_pla[indX  ][indY-1], wx1*wy0 ) ;
        atomicAdd(&fields->counts_pla[indX  ][indY  ], wx1*wy1 ) ;
        atomicAdd(&fields->counts_pla[indX  ][indY+1], wx1*wy2 ) ;
    		atomicAdd(&fields->counts_pla[indX+1][indY-1], wx2*wy0 ) ;
    		atomicAdd(&fields->counts_pla[indX+1][indY  ], wx2*wy1 ) ;
    		atomicAdd(&fields->counts_pla[indX+1][indY+1], wx2*wy2 ) ;

        atomicAdd(&fields->fluxNum_pla[0][indX-1][indY-1] , p->vx[idx] * wx0*wy0 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX-1][indY  ] , p->vx[idx] * wx0*wy1 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX-1][indY+1] , p->vx[idx] * wx0*wy2 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX  ][indY-1] , p->vx[idx] * wx1*wy0 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX  ][indY  ] , p->vx[idx] * wx1*wy1 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX  ][indY+1] , p->vx[idx] * wx1*wy2 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX+1][indY-1] , p->vx[idx] * wx2*wy0 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX+1][indY  ] , p->vx[idx] * wx2*wy1 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX+1][indY+1] , p->vx[idx] * wx2*wy2 ) ;
        //______________
        atomicAdd(&fields->fluxNum_pla[1][indX-1][indY-1] , p->vy[idx] * wx0*wy0 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX-1][indY  ] , p->vy[idx] * wx0*wy1 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX-1][indY+1] , p->vy[idx] * wx0*wy2 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX  ][indY-1] , p->vy[idx] * wx1*wy0 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX  ][indY  ] , p->vy[idx] * wx1*wy1 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX  ][indY+1] , p->vy[idx] * wx1*wy2 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX+1][indY-1] , p->vy[idx] * wx2*wy0 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX+1][indY  ] , p->vy[idx] * wx2*wy1 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX+1][indY+1] , p->vy[idx] * wx2*wy2 ) ;
        //______________
        atomicAdd(&fields->fluxNum_pla[2][indX-1][indY-1] , p->vz[idx] * wx0*wy0 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX-1][indY  ] , p->vz[idx] * wx0*wy1 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX-1][indY+1] , p->vz[idx] * wx0*wy2 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX  ][indY-1] , p->vz[idx] * wx1*wy0 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX  ][indY  ] , p->vz[idx] * wx1*wy1 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX  ][indY+1] , p->vz[idx] * wx1*wy2 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX+1][indY-1] , p->vz[idx] * wx2*wy0 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX+1][indY  ] , p->vz[idx] * wx2*wy1 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX+1][indY+1] , p->vz[idx] * wx2*wy2 ) ;
      }



    #elif NB_DIM==3
      int indX, indY, indZ ;
      float wx, wy, wz ;
      float wx0, wx1, wx2 ;
      float wy0, wy1, wy2 ;
      float wz0, wz1, wz2 ;

      indX = roundf( (p->rx[idx]-grid->xMin)*sP->dX_i + .5) + 1 ;
      indY = roundf( (p->ry[idx]-grid->yMin)*sP->dX_i + .5) + 1 ;
      indZ = roundf( (p->rz[idx]-grid->zMin)*sP->dX_i + .5) + 1 ;

      wx = (p->rx[idx]-grid->xGrid[indX])*sP->dX_i ;
      wy = (p->ry[idx]-grid->yGrid[indY])*sP->dX_i ;
      wz = (p->rz[idx]-grid->zGrid[indZ])*sP->dX_i ;

      wx0 = .5*(.5-wx)*(.5-wx) ;
      wx1 = .75 - wx*wx ;
      wx2 = .5*(.5+wx)*(.5+wx) ;

      wy0 = .5*(.5-wy)*(.5-wy) ;
      wy1 = .75 - wy*wy ;
      wy2 = .5*(.5+wy)*(.5+wy) ;

      wz0 = .5*(.5-wz)*(.5-wz) ;
      wz1 = .75 - wz*wz ;
      wz2 = .5*(.5+wz)*(.5+wz) ;

      /// Here below, the clean logic of the algorithm would be indX<2 andor indX>len_x_cst+1,
      ///  because otherwise it would mean a particle has passed the xMin xMax and is still
      ///  valid on this process. It happens/appears that a few particles end up with
      ///  EXACTLY the position xMin or xMax (same for y). This is not repeatable because
      ///  of ... what the memory was conainting before runtime? + rounding error?
      #if debug_mode_cst
        if ( (indX<1) || (indX>len_x_cst+2) ){
          printf("| !! It %i.%i, particle in part2grid_counts_flux_num_k, indX %i, rx %.4e, rank %i, ID %i, idx %i\n", ind_it, flag, indX, p->rx[idx], sP->mpi_rank, p->ID[idx], idx) ;
          p->active[idx] = false ;
        }
        else if ( (indY<1) || (indY>len_y_cst+2) ){
          printf("| !! It %i.%i, particle in part2grid_counts_flux_num_k, indY %i, ry %.4e, rank %i, ID %i, idx %i\n", ind_it, flag, indY, p->ry[idx], sP->mpi_rank, p->ID[idx], idx) ;
          p->active[idx] = false ;
        }
        else if ( (indZ<1) || (indZ>len_z_cst+2) ){
          printf("| !! It %i.%i, particle in part2grid_counts_flux_num_k, indY %i, ry %.4e, rank %i, ID %i, idx %i\n", ind_it, flag, indY, p->ry[idx], sP->mpi_rank, p->ID[idx], idx) ;
          p->active[idx] = false ;
        }
      #endif

      if (p->ID[idx]==0){
    		atomicAdd(&fields->counts[indX-1][indY-1][indZ-1], wx0*wy0*wz0 ) ;
        atomicAdd(&fields->counts[indX-1][indY-1][indZ  ], wx0*wy0*wz1 ) ;
        atomicAdd(&fields->counts[indX-1][indY-1][indZ+1], wx0*wy0*wz2 ) ;
        atomicAdd(&fields->counts[indX-1][indY  ][indZ-1], wx0*wy1*wz0 ) ;
        atomicAdd(&fields->counts[indX-1][indY  ][indZ  ], wx0*wy1*wz1 ) ;
        atomicAdd(&fields->counts[indX-1][indY  ][indZ+1], wx0*wy1*wz2 ) ;
        atomicAdd(&fields->counts[indX-1][indY+1][indZ-1], wx0*wy2*wz0 ) ;
        atomicAdd(&fields->counts[indX-1][indY+1][indZ  ], wx0*wy2*wz1 ) ;
        atomicAdd(&fields->counts[indX-1][indY+1][indZ+1], wx0*wy2*wz2 ) ;
        atomicAdd(&fields->counts[indX  ][indY-1][indZ-1], wx1*wy0*wz0 ) ;
        atomicAdd(&fields->counts[indX  ][indY-1][indZ  ], wx1*wy0*wz1 ) ;
        atomicAdd(&fields->counts[indX  ][indY-1][indZ+1], wx1*wy0*wz2 ) ;
        atomicAdd(&fields->counts[indX  ][indY  ][indZ-1], wx1*wy1*wz0 ) ;
        atomicAdd(&fields->counts[indX  ][indY  ][indZ  ], wx1*wy1*wz1 ) ;
        atomicAdd(&fields->counts[indX  ][indY  ][indZ+1], wx1*wy1*wz2 ) ;
        atomicAdd(&fields->counts[indX  ][indY+1][indZ-1], wx1*wy2*wz0 ) ;
        atomicAdd(&fields->counts[indX  ][indY+1][indZ  ], wx1*wy2*wz1 ) ;
        atomicAdd(&fields->counts[indX  ][indY+1][indZ+1], wx1*wy2*wz2 ) ;
    		atomicAdd(&fields->counts[indX+1][indY-1][indZ-1], wx2*wy0*wz0 ) ;
    		atomicAdd(&fields->counts[indX+1][indY-1][indZ  ], wx2*wy0*wz1 ) ;
    		atomicAdd(&fields->counts[indX+1][indY-1][indZ+1], wx2*wy0*wz2 ) ;
    		atomicAdd(&fields->counts[indX+1][indY  ][indZ-1], wx2*wy1*wz0 ) ;
    		atomicAdd(&fields->counts[indX+1][indY  ][indZ  ], wx2*wy1*wz1 ) ;
    		atomicAdd(&fields->counts[indX+1][indY  ][indZ+1], wx2*wy1*wz2 ) ;
    		atomicAdd(&fields->counts[indX+1][indY+1][indZ-1], wx2*wy2*wz0 ) ;
    		atomicAdd(&fields->counts[indX+1][indY+1][indZ  ], wx2*wy2*wz1 ) ;
    		atomicAdd(&fields->counts[indX+1][indY+1][indZ+1], wx2*wy2*wz2 ) ;

        atomicAdd(&fields->fluxNum[0][indX-1][indY-1][indZ-1] , p->vx[idx] * wx0*wy0*wz0 ) ;
        atomicAdd(&fields->fluxNum[0][indX-1][indY-1][indZ  ] , p->vx[idx] * wx0*wy0*wz1 ) ;
        atomicAdd(&fields->fluxNum[0][indX-1][indY-1][indZ+1] , p->vx[idx] * wx0*wy0*wz2 ) ;
        atomicAdd(&fields->fluxNum[0][indX-1][indY  ][indZ-1] , p->vx[idx] * wx0*wy1*wz0 ) ;
        atomicAdd(&fields->fluxNum[0][indX-1][indY  ][indZ  ] , p->vx[idx] * wx0*wy1*wz1 ) ;
        atomicAdd(&fields->fluxNum[0][indX-1][indY  ][indZ+1] , p->vx[idx] * wx0*wy1*wz2 ) ;
        atomicAdd(&fields->fluxNum[0][indX-1][indY+1][indZ-1] , p->vx[idx] * wx0*wy2*wz0 ) ;
        atomicAdd(&fields->fluxNum[0][indX-1][indY+1][indZ  ] , p->vx[idx] * wx0*wy2*wz1 ) ;
        atomicAdd(&fields->fluxNum[0][indX-1][indY+1][indZ+1] , p->vx[idx] * wx0*wy2*wz2 ) ;
        atomicAdd(&fields->fluxNum[0][indX  ][indY-1][indZ-1] , p->vx[idx] * wx1*wy0*wz0 ) ;
        atomicAdd(&fields->fluxNum[0][indX  ][indY-1][indZ  ] , p->vx[idx] * wx1*wy0*wz1 ) ;
        atomicAdd(&fields->fluxNum[0][indX  ][indY-1][indZ+1] , p->vx[idx] * wx1*wy0*wz2 ) ;
        atomicAdd(&fields->fluxNum[0][indX  ][indY  ][indZ-1] , p->vx[idx] * wx1*wy1*wz0 ) ;
        atomicAdd(&fields->fluxNum[0][indX  ][indY  ][indZ  ] , p->vx[idx] * wx1*wy1*wz1 ) ;
        atomicAdd(&fields->fluxNum[0][indX  ][indY  ][indZ+1] , p->vx[idx] * wx1*wy1*wz2 ) ;
        atomicAdd(&fields->fluxNum[0][indX  ][indY+1][indZ-1] , p->vx[idx] * wx1*wy2*wz0 ) ;
        atomicAdd(&fields->fluxNum[0][indX  ][indY+1][indZ  ] , p->vx[idx] * wx1*wy2*wz1 ) ;
        atomicAdd(&fields->fluxNum[0][indX  ][indY+1][indZ+1] , p->vx[idx] * wx1*wy2*wz2 ) ;
        atomicAdd(&fields->fluxNum[0][indX+1][indY-1][indZ-1] , p->vx[idx] * wx2*wy0*wz0 ) ;
        atomicAdd(&fields->fluxNum[0][indX+1][indY-1][indZ  ] , p->vx[idx] * wx2*wy0*wz1 ) ;
        atomicAdd(&fields->fluxNum[0][indX+1][indY-1][indZ+1] , p->vx[idx] * wx2*wy0*wz2 ) ;
        atomicAdd(&fields->fluxNum[0][indX+1][indY  ][indZ-1] , p->vx[idx] * wx2*wy1*wz0 ) ;
        atomicAdd(&fields->fluxNum[0][indX+1][indY  ][indZ  ] , p->vx[idx] * wx2*wy1*wz1 ) ;
        atomicAdd(&fields->fluxNum[0][indX+1][indY  ][indZ+1] , p->vx[idx] * wx2*wy1*wz2 ) ;
        atomicAdd(&fields->fluxNum[0][indX+1][indY+1][indZ-1] , p->vx[idx] * wx2*wy2*wz0 ) ;
        atomicAdd(&fields->fluxNum[0][indX+1][indY+1][indZ  ] , p->vx[idx] * wx2*wy2*wz1 ) ;
        atomicAdd(&fields->fluxNum[0][indX+1][indY+1][indZ+1] , p->vx[idx] * wx2*wy2*wz2 ) ;
        //______________
        atomicAdd(&fields->fluxNum[1][indX-1][indY-1][indZ-1] , p->vy[idx] * wx0*wy0*wz0 ) ;
        atomicAdd(&fields->fluxNum[1][indX-1][indY-1][indZ  ] , p->vy[idx] * wx0*wy0*wz1 ) ;
        atomicAdd(&fields->fluxNum[1][indX-1][indY-1][indZ+1] , p->vy[idx] * wx0*wy0*wz2 ) ;
        atomicAdd(&fields->fluxNum[1][indX-1][indY  ][indZ-1] , p->vy[idx] * wx0*wy1*wz0 ) ;
        atomicAdd(&fields->fluxNum[1][indX-1][indY  ][indZ  ] , p->vy[idx] * wx0*wy1*wz1 ) ;
        atomicAdd(&fields->fluxNum[1][indX-1][indY  ][indZ+1] , p->vy[idx] * wx0*wy1*wz2 ) ;
        atomicAdd(&fields->fluxNum[1][indX-1][indY+1][indZ-1] , p->vy[idx] * wx0*wy2*wz0 ) ;
        atomicAdd(&fields->fluxNum[1][indX-1][indY+1][indZ  ] , p->vy[idx] * wx0*wy2*wz1 ) ;
        atomicAdd(&fields->fluxNum[1][indX-1][indY+1][indZ+1] , p->vy[idx] * wx0*wy2*wz2 ) ;
        atomicAdd(&fields->fluxNum[1][indX  ][indY-1][indZ-1] , p->vy[idx] * wx1*wy0*wz0 ) ;
        atomicAdd(&fields->fluxNum[1][indX  ][indY-1][indZ  ] , p->vy[idx] * wx1*wy0*wz1 ) ;
        atomicAdd(&fields->fluxNum[1][indX  ][indY-1][indZ+1] , p->vy[idx] * wx1*wy0*wz2 ) ;
        atomicAdd(&fields->fluxNum[1][indX  ][indY  ][indZ-1] , p->vy[idx] * wx1*wy1*wz0 ) ;
        atomicAdd(&fields->fluxNum[1][indX  ][indY  ][indZ  ] , p->vy[idx] * wx1*wy1*wz1 ) ;
        atomicAdd(&fields->fluxNum[1][indX  ][indY  ][indZ+1] , p->vy[idx] * wx1*wy1*wz2 ) ;
        atomicAdd(&fields->fluxNum[1][indX  ][indY+1][indZ-1] , p->vy[idx] * wx1*wy2*wz0 ) ;
        atomicAdd(&fields->fluxNum[1][indX  ][indY+1][indZ  ] , p->vy[idx] * wx1*wy2*wz1 ) ;
        atomicAdd(&fields->fluxNum[1][indX  ][indY+1][indZ+1] , p->vy[idx] * wx1*wy2*wz2 ) ;
        atomicAdd(&fields->fluxNum[1][indX+1][indY-1][indZ-1] , p->vy[idx] * wx2*wy0*wz0 ) ;
        atomicAdd(&fields->fluxNum[1][indX+1][indY-1][indZ  ] , p->vy[idx] * wx2*wy0*wz1 ) ;
        atomicAdd(&fields->fluxNum[1][indX+1][indY-1][indZ+1] , p->vy[idx] * wx2*wy0*wz2 ) ;
        atomicAdd(&fields->fluxNum[1][indX+1][indY  ][indZ-1] , p->vy[idx] * wx2*wy1*wz0 ) ;
        atomicAdd(&fields->fluxNum[1][indX+1][indY  ][indZ  ] , p->vy[idx] * wx2*wy1*wz1 ) ;
        atomicAdd(&fields->fluxNum[1][indX+1][indY  ][indZ+1] , p->vy[idx] * wx2*wy1*wz2 ) ;
        atomicAdd(&fields->fluxNum[1][indX+1][indY+1][indZ-1] , p->vy[idx] * wx2*wy2*wz0 ) ;
        atomicAdd(&fields->fluxNum[1][indX+1][indY+1][indZ  ] , p->vy[idx] * wx2*wy2*wz1 ) ;
        atomicAdd(&fields->fluxNum[1][indX+1][indY+1][indZ+1] , p->vy[idx] * wx2*wy2*wz2 ) ;
        //______________
        atomicAdd(&fields->fluxNum[2][indX-1][indY-1][indZ-1] , p->vz[idx] * wx0*wy0*wz0 ) ;
        atomicAdd(&fields->fluxNum[2][indX-1][indY-1][indZ  ] , p->vz[idx] * wx0*wy0*wz1 ) ;
        atomicAdd(&fields->fluxNum[2][indX-1][indY-1][indZ+1] , p->vz[idx] * wx0*wy0*wz2 ) ;
        atomicAdd(&fields->fluxNum[2][indX-1][indY  ][indZ-1] , p->vz[idx] * wx0*wy1*wz0 ) ;
        atomicAdd(&fields->fluxNum[2][indX-1][indY  ][indZ  ] , p->vz[idx] * wx0*wy1*wz1 ) ;
        atomicAdd(&fields->fluxNum[2][indX-1][indY  ][indZ+1] , p->vz[idx] * wx0*wy1*wz2 ) ;
        atomicAdd(&fields->fluxNum[2][indX-1][indY+1][indZ-1] , p->vz[idx] * wx0*wy2*wz0 ) ;
        atomicAdd(&fields->fluxNum[2][indX-1][indY+1][indZ  ] , p->vz[idx] * wx0*wy2*wz1 ) ;
        atomicAdd(&fields->fluxNum[2][indX-1][indY+1][indZ+1] , p->vz[idx] * wx0*wy2*wz2 ) ;
        atomicAdd(&fields->fluxNum[2][indX  ][indY-1][indZ-1] , p->vz[idx] * wx1*wy0*wz0 ) ;
        atomicAdd(&fields->fluxNum[2][indX  ][indY-1][indZ  ] , p->vz[idx] * wx1*wy0*wz1 ) ;
        atomicAdd(&fields->fluxNum[2][indX  ][indY-1][indZ+1] , p->vz[idx] * wx1*wy0*wz2 ) ;
        atomicAdd(&fields->fluxNum[2][indX  ][indY  ][indZ-1] , p->vz[idx] * wx1*wy1*wz0 ) ;
        atomicAdd(&fields->fluxNum[2][indX  ][indY  ][indZ  ] , p->vz[idx] * wx1*wy1*wz1 ) ;
        atomicAdd(&fields->fluxNum[2][indX  ][indY  ][indZ+1] , p->vz[idx] * wx1*wy1*wz2 ) ;
        atomicAdd(&fields->fluxNum[2][indX  ][indY+1][indZ-1] , p->vz[idx] * wx1*wy2*wz0 ) ;
        atomicAdd(&fields->fluxNum[2][indX  ][indY+1][indZ  ] , p->vz[idx] * wx1*wy2*wz1 ) ;
        atomicAdd(&fields->fluxNum[2][indX  ][indY+1][indZ+1] , p->vz[idx] * wx1*wy2*wz2 ) ;
        atomicAdd(&fields->fluxNum[2][indX+1][indY-1][indZ-1] , p->vz[idx] * wx2*wy0*wz0 ) ;
        atomicAdd(&fields->fluxNum[2][indX+1][indY-1][indZ  ] , p->vz[idx] * wx2*wy0*wz1 ) ;
        atomicAdd(&fields->fluxNum[2][indX+1][indY-1][indZ+1] , p->vz[idx] * wx2*wy0*wz2 ) ;
        atomicAdd(&fields->fluxNum[2][indX+1][indY  ][indZ-1] , p->vz[idx] * wx2*wy1*wz0 ) ;
        atomicAdd(&fields->fluxNum[2][indX+1][indY  ][indZ  ] , p->vz[idx] * wx2*wy1*wz1 ) ;
        atomicAdd(&fields->fluxNum[2][indX+1][indY  ][indZ+1] , p->vz[idx] * wx2*wy1*wz2 ) ;
        atomicAdd(&fields->fluxNum[2][indX+1][indY+1][indZ-1] , p->vz[idx] * wx2*wy2*wz0 ) ;
        atomicAdd(&fields->fluxNum[2][indX+1][indY+1][indZ  ] , p->vz[idx] * wx2*wy2*wz1 ) ;
        atomicAdd(&fields->fluxNum[2][indX+1][indY+1][indZ+1] , p->vz[idx] * wx2*wy2*wz2 ) ;
      }
      else {
    		atomicAdd(&fields->counts_pla[indX-1][indY-1][indZ-1], wx0*wy0*wz0 ) ;
        atomicAdd(&fields->counts_pla[indX-1][indY-1][indZ  ], wx0*wy0*wz1 ) ;
        atomicAdd(&fields->counts_pla[indX-1][indY-1][indZ+1], wx0*wy0*wz2 ) ;
        atomicAdd(&fields->counts_pla[indX-1][indY  ][indZ-1], wx0*wy1*wz0 ) ;
        atomicAdd(&fields->counts_pla[indX-1][indY  ][indZ  ], wx0*wy1*wz1 ) ;
        atomicAdd(&fields->counts_pla[indX-1][indY  ][indZ+1], wx0*wy1*wz2 ) ;
        atomicAdd(&fields->counts_pla[indX-1][indY+1][indZ-1], wx0*wy2*wz0 ) ;
        atomicAdd(&fields->counts_pla[indX-1][indY+1][indZ  ], wx0*wy2*wz1 ) ;
        atomicAdd(&fields->counts_pla[indX-1][indY+1][indZ+1], wx0*wy2*wz2 ) ;
        atomicAdd(&fields->counts_pla[indX  ][indY-1][indZ-1], wx1*wy0*wz0 ) ;
        atomicAdd(&fields->counts_pla[indX  ][indY-1][indZ  ], wx1*wy0*wz1 ) ;
        atomicAdd(&fields->counts_pla[indX  ][indY-1][indZ+1], wx1*wy0*wz2 ) ;
        atomicAdd(&fields->counts_pla[indX  ][indY  ][indZ-1], wx1*wy1*wz0 ) ;
        atomicAdd(&fields->counts_pla[indX  ][indY  ][indZ  ], wx1*wy1*wz1 ) ;
        atomicAdd(&fields->counts_pla[indX  ][indY  ][indZ+1], wx1*wy1*wz2 ) ;
        atomicAdd(&fields->counts_pla[indX  ][indY+1][indZ-1], wx1*wy2*wz0 ) ;
        atomicAdd(&fields->counts_pla[indX  ][indY+1][indZ  ], wx1*wy2*wz1 ) ;
        atomicAdd(&fields->counts_pla[indX  ][indY+1][indZ+1], wx1*wy2*wz2 ) ;
    		atomicAdd(&fields->counts_pla[indX+1][indY-1][indZ-1], wx2*wy0*wz0 ) ;
    		atomicAdd(&fields->counts_pla[indX+1][indY-1][indZ  ], wx2*wy0*wz1 ) ;
    		atomicAdd(&fields->counts_pla[indX+1][indY-1][indZ+1], wx2*wy0*wz2 ) ;
    		atomicAdd(&fields->counts_pla[indX+1][indY  ][indZ-1], wx2*wy1*wz0 ) ;
    		atomicAdd(&fields->counts_pla[indX+1][indY  ][indZ  ], wx2*wy1*wz1 ) ;
    		atomicAdd(&fields->counts_pla[indX+1][indY  ][indZ+1], wx2*wy1*wz2 ) ;
    		atomicAdd(&fields->counts_pla[indX+1][indY+1][indZ-1], wx2*wy2*wz0 ) ;
    		atomicAdd(&fields->counts_pla[indX+1][indY+1][indZ  ], wx2*wy2*wz1 ) ;
    		atomicAdd(&fields->counts_pla[indX+1][indY+1][indZ+1], wx2*wy2*wz2 ) ;

        atomicAdd(&fields->fluxNum_pla[0][indX-1][indY-1][indZ-1], p->vx[idx] * wx0*wy0*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX-1][indY-1][indZ  ], p->vx[idx] * wx0*wy0*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX-1][indY-1][indZ+1], p->vx[idx] * wx0*wy0*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX-1][indY  ][indZ-1], p->vx[idx] * wx0*wy1*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX-1][indY  ][indZ  ], p->vx[idx] * wx0*wy1*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX-1][indY  ][indZ+1], p->vx[idx] * wx0*wy1*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX-1][indY+1][indZ-1], p->vx[idx] * wx0*wy2*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX-1][indY+1][indZ  ], p->vx[idx] * wx0*wy2*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX-1][indY+1][indZ+1], p->vx[idx] * wx0*wy2*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX  ][indY-1][indZ-1], p->vx[idx] * wx1*wy0*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX  ][indY-1][indZ  ], p->vx[idx] * wx1*wy0*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX  ][indY-1][indZ+1], p->vx[idx] * wx1*wy0*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX  ][indY  ][indZ-1], p->vx[idx] * wx1*wy1*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX  ][indY  ][indZ  ], p->vx[idx] * wx1*wy1*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX  ][indY  ][indZ+1], p->vx[idx] * wx1*wy1*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX  ][indY+1][indZ-1], p->vx[idx] * wx1*wy2*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX  ][indY+1][indZ  ], p->vx[idx] * wx1*wy2*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX  ][indY+1][indZ+1], p->vx[idx] * wx1*wy2*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX+1][indY-1][indZ-1], p->vx[idx] * wx2*wy0*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX+1][indY-1][indZ  ], p->vx[idx] * wx2*wy0*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX+1][indY-1][indZ+1], p->vx[idx] * wx2*wy0*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX+1][indY  ][indZ-1], p->vx[idx] * wx2*wy1*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX+1][indY  ][indZ  ], p->vx[idx] * wx2*wy1*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX+1][indY  ][indZ+1], p->vx[idx] * wx2*wy1*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX+1][indY+1][indZ-1], p->vx[idx] * wx2*wy2*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX+1][indY+1][indZ  ], p->vx[idx] * wx2*wy2*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[0][indX+1][indY+1][indZ+1], p->vx[idx] * wx2*wy2*wz2 ) ;
        //______________
        atomicAdd(&fields->fluxNum_pla[1][indX-1][indY-1][indZ-1], p->vy[idx] * wx0*wy0*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX-1][indY-1][indZ  ], p->vy[idx] * wx0*wy0*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX-1][indY-1][indZ+1], p->vy[idx] * wx0*wy0*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX-1][indY  ][indZ-1], p->vy[idx] * wx0*wy1*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX-1][indY  ][indZ  ], p->vy[idx] * wx0*wy1*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX-1][indY  ][indZ+1], p->vy[idx] * wx0*wy1*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX-1][indY+1][indZ-1], p->vy[idx] * wx0*wy2*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX-1][indY+1][indZ  ], p->vy[idx] * wx0*wy2*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX-1][indY+1][indZ+1], p->vy[idx] * wx0*wy2*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX  ][indY-1][indZ-1], p->vy[idx] * wx1*wy0*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX  ][indY-1][indZ  ], p->vy[idx] * wx1*wy0*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX  ][indY-1][indZ+1], p->vy[idx] * wx1*wy0*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX  ][indY  ][indZ-1], p->vy[idx] * wx1*wy1*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX  ][indY  ][indZ  ], p->vy[idx] * wx1*wy1*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX  ][indY  ][indZ+1], p->vy[idx] * wx1*wy1*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX  ][indY+1][indZ-1], p->vy[idx] * wx1*wy2*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX  ][indY+1][indZ  ], p->vy[idx] * wx1*wy2*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX  ][indY+1][indZ+1], p->vy[idx] * wx1*wy2*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX+1][indY-1][indZ-1], p->vy[idx] * wx2*wy0*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX+1][indY-1][indZ  ], p->vy[idx] * wx2*wy0*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX+1][indY-1][indZ+1], p->vy[idx] * wx2*wy0*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX+1][indY  ][indZ-1], p->vy[idx] * wx2*wy1*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX+1][indY  ][indZ  ], p->vy[idx] * wx2*wy1*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX+1][indY  ][indZ+1], p->vy[idx] * wx2*wy1*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX+1][indY+1][indZ-1], p->vy[idx] * wx2*wy2*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX+1][indY+1][indZ  ], p->vy[idx] * wx2*wy2*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[1][indX+1][indY+1][indZ+1], p->vy[idx] * wx2*wy2*wz2 ) ;
        //______________
        atomicAdd(&fields->fluxNum_pla[2][indX-1][indY-1][indZ-1], p->vz[idx] * wx0*wy0*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX-1][indY-1][indZ  ], p->vz[idx] * wx0*wy0*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX-1][indY-1][indZ+1], p->vz[idx] * wx0*wy0*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX-1][indY  ][indZ-1], p->vz[idx] * wx0*wy1*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX-1][indY  ][indZ  ], p->vz[idx] * wx0*wy1*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX-1][indY  ][indZ+1], p->vz[idx] * wx0*wy1*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX-1][indY+1][indZ-1], p->vz[idx] * wx0*wy2*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX-1][indY+1][indZ  ], p->vz[idx] * wx0*wy2*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX-1][indY+1][indZ+1], p->vz[idx] * wx0*wy2*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX  ][indY-1][indZ-1], p->vz[idx] * wx1*wy0*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX  ][indY-1][indZ  ], p->vz[idx] * wx1*wy0*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX  ][indY-1][indZ+1], p->vz[idx] * wx1*wy0*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX  ][indY  ][indZ-1], p->vz[idx] * wx1*wy1*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX  ][indY  ][indZ  ], p->vz[idx] * wx1*wy1*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX  ][indY  ][indZ+1], p->vz[idx] * wx1*wy1*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX  ][indY+1][indZ-1], p->vz[idx] * wx1*wy2*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX  ][indY+1][indZ  ], p->vz[idx] * wx1*wy2*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX  ][indY+1][indZ+1], p->vz[idx] * wx1*wy2*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX+1][indY-1][indZ-1], p->vz[idx] * wx2*wy0*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX+1][indY-1][indZ  ], p->vz[idx] * wx2*wy0*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX+1][indY-1][indZ+1], p->vz[idx] * wx2*wy0*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX+1][indY  ][indZ-1], p->vz[idx] * wx2*wy1*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX+1][indY  ][indZ  ], p->vz[idx] * wx2*wy1*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX+1][indY  ][indZ+1], p->vz[idx] * wx2*wy1*wz2 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX+1][indY+1][indZ-1], p->vz[idx] * wx2*wy2*wz0 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX+1][indY+1][indZ  ], p->vz[idx] * wx2*wy2*wz1 ) ;
        atomicAdd(&fields->fluxNum_pla[2][indX+1][indY+1][indZ+1], p->vz[idx] * wx2*wy2*wz2 ) ;
      }
    #endif

	}
}

#endif
