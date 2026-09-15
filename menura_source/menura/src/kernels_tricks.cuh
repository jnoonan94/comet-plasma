#ifndef __KERNELSTRICKS_CUH_INCLUDED__   // if x.h hasn't been included yet...
#define __KERNELSTRICKS_CUH_INCLUDED__

#include <cuda_runtime.h>

#include "parameters.h"



__global__ void search_smooth_k(simu_fields* fields, simu_param* sP)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < nb_nodes_cst){
    #if NB_DIM==2
      int i = int( idx/(len_y_cst) );
      int j = idx - i*(len_y_cst);
      i += 4;
      j += 4;

      if (abs(fields->E[0][i][j])>sP->E_max_smooth){
        fields->smooth_field = true;
        fields->smooth_idx[0] = i;
        fields->smooth_idx[1] = j;
      }
      if (abs(fields->E[1][i][j])>sP->E_max_smooth){
        fields->smooth_field = true;
        fields->smooth_idx[0] = i;
        fields->smooth_idx[1] = j;
      }
      if (abs(fields->E[2][i][j])>sP->E_max_smooth){
        fields->smooth_field = true;
        fields->smooth_idx[0] = i;
        fields->smooth_idx[1] = j;
      }
      // if ( (fields->E_hal[0][i][j]*fields->E_hal[1][i][j]) > 200 ){
      //   fields->smooth_field = true;
      //   fields->smooth_idx[0] = i;
      //   fields->smooth_idx[1] = j;
      // }
  #elif NB_DIM==3

      int i = int( idx/((len_y_cst)*(len_z_cst)) );
      int j = int( (idx - i*(len_y_cst)*(len_z_cst))/(len_z_cst) );
      int k = idx - i*(len_y_cst)*(len_z_cst) - j*(len_z_cst);
      i += 4;  // Order 0 smooth.
      j += 4;
      k += 4;

      float E_max = sP->E_max_smooth;
      if (fields->region_ID[i][j][k]==2){
        // The node is in the loosely defined "inner coma", cf. identify_region_k.
        E_max = 100;
      }

      if (abs(fields->E[0][i][j][k])>E_max){
        fields->smooth_field = true;
        fields->smooth_idx[0] = i;
        fields->smooth_idx[1] = j;
        fields->smooth_idx[2] = k;
      }
      if (abs(fields->E[1][i][j][k])>E_max){
        fields->smooth_field = true;
        fields->smooth_idx[0] = i;
        fields->smooth_idx[1] = j;
        fields->smooth_idx[2] = k;
      }
      if (abs(fields->E[2][i][j][k])>E_max){
        fields->smooth_field = true;
        fields->smooth_idx[0] = i;
        fields->smooth_idx[1] = j;
        fields->smooth_idx[2] = k;
      }
    #endif
  }
}


// __global__ void deriv_order_0_k(simu_fields* fields)
// {
//   int idx = threadIdx.x + blockIdx.x*blockDim.x;
//
//   if (idx<nb_nodes_cst){
//     int i = floorf(idx/len_y_cst);
//     int j = idx - i*len_y_cst;
//     i += 2;
//     j += 2;
//
//     fields->E_hal[0][i][j] = fabsf(fields->E[2][i+1][j  ] - fields->E[2][i][j]);
//     fields->E_hal[1][i][j] = fabsf(fields->E[2][i  ][j+1] - fields->E[2][i][j]);
//
//   }
// }

__global__ void smooth_field_all_k
#if NB_DIM==2
(float G[len_x_cst+4][len_y_cst+4], simu_fields* fields)
#elif NB_DIM==3
(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4], simu_fields* fields)
#endif
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < nb_nodes_tot_cst){
    #if NB_DIM==2
      int i = floorf(idx/(len_y_cst+4));
      int j = idx - i*(len_y_cst+4);

      fields->smooth[i][j] = 4*G[i][j] +
                              2*(G[i-1][j] + G[i+1][j] + G[i][j+1] + G[i][j-1]) +
                              G[i-1][j-1] + G[i-1][j+1] + G[i+1][j-1] + G[i+1][j+1];
      fields->smooth[i][j] /= 16.;


    #elif NB_DIM==3
      int i = int( idx/((len_y_cst+4)*(len_z_cst+4)) );
      int j = int( (idx - i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);

      fields->smooth[i][j][k] = 8*G[i][j][k] +
                                4*(G[i-1][j][k] + G[i+1][j][k] + G[i][j+1][k] + G[i][j-1][k] + G[i][j][k-1] + G[i][j][k+1]) +
                                2*G[i][j+1][k-1] +
                                2*G[i][j+1][k+1] +
                                2*G[i][j-1][k-1] +
                                2*G[i][j-1][k+1] +
                                2*G[i+1][j][k-1] +
                                2*G[i+1][j][k+1] +
                                2*G[i-1][j][k-1] +
                                2*G[i-1][j][k+1] +
                                2*G[i+1][j+1][k] +
                                2*G[i+1][j-1][k] +
                                2*G[i-1][j+1][k] +
                                2*G[i-1][j-1][k] +
                                G[i+1][j+1][k+1] +
                                G[i+1][j+1][k-1] +
                                G[i+1][j-1][k+1] +
                                G[i+1][j-1][k-1] +
                                G[i-1][j+1][k+1] +
                                G[i-1][j+1][k-1] +
                                G[i-1][j-1][k+1] +
                                G[i-1][j-1][k-1];
      fields->smooth[i][j][k] /= 64.;
    #endif
  }
}

__global__ void smooth_patch_k
#if NB_DIM==2
(float G[len_x_cst+4][len_y_cst+4], simu_fields* fields)
#elif NB_DIM==3
(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4], simu_fields* fields)
#endif
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  #if NB_DIM==2
    if (idx < len_patch_cst*len_patch_cst){
      /* di and dj are the indices within the sub-domain to be smoothed. */
      int di = floorf(idx/len_patch_cst);
      int dj = idx - di*len_patch_cst;
      int half_len_patch = int(len_patch_cst/2);
      di -= int(half_len_patch/2);
      dj -= int(half_len_patch/2);
      /* i and j are the indices within the main simulation domain. */
      int i = fields->smooth_idx[0] + di;
      int j = fields->smooth_idx[1] + dj;

      if ((i>=2) && (i<=len_x_cst+1) && (j>=2) && (j<=len_y_cst+1)){
        fields->smooth[i][j] = 4*G[i][j] +
                                2*(G[i-1][j] + G[i+1][j] + G[i][j+1] + G[i][j-1]) +
                                G[i-1][j-1] + G[i-1][j+1] + G[i+1][j-1] + G[i+1][j+1];
        fields->smooth[i][j] /= 16.;
      }
    }

  #elif NB_DIM==3
    if (idx < len_patch_cst*len_patch_cst*len_patch_cst){
      /* di, dj and dk are the indices within the sub-domain to be smoothed. */
      int di = int(idx/(len_patch_cst*len_patch_cst));
      int dj = int( (idx - di*len_patch_cst*len_patch_cst)/len_patch_cst );
      int dk = idx - di*len_patch_cst*len_patch_cst - dj*len_patch_cst;
      int half_len_patch = int(len_patch_cst/2);
      di -= int(half_len_patch/2);
      dj -= int(half_len_patch/2);
      dk -= int(half_len_patch/2);
      /* i, j and k are the indices within the main simulation domain. */
      int i = fields->smooth_idx[0] + di;
      int j = fields->smooth_idx[1] + dj;
      int k = fields->smooth_idx[2] + dk;

      fields->smooth[i][j][k] = 8*G[i][j][k] +
                                4*(G[i-1][j][k] + G[i+1][j][k] + G[i][j+1][k] + G[i][j-1][k] + G[i][j][k-1] + G[i][j][k+1]) +
                                2*G[i][j+1][k-1] +
                                2*G[i][j+1][k+1] +
                                2*G[i][j-1][k-1] +
                                2*G[i][j-1][k+1] +
                                2*G[i+1][j][k-1] +
                                2*G[i+1][j][k+1] +
                                2*G[i-1][j][k-1] +
                                2*G[i-1][j][k+1] +
                                2*G[i+1][j+1][k] +
                                2*G[i+1][j-1][k] +
                                2*G[i-1][j+1][k] +
                                2*G[i-1][j-1][k] +
                                G[i+1][j+1][k+1] +
                                G[i+1][j+1][k-1] +
                                G[i+1][j-1][k+1] +
                                G[i+1][j-1][k-1] +
                                G[i-1][j+1][k+1] +
                                G[i-1][j+1][k-1] +
                                G[i-1][j-1][k+1] +
                                G[i-1][j-1][k-1];
      fields->smooth[i][j][k] /= 64.;
    }
  #endif
}

__global__ void copy_smooth_patch_k
#if NB_DIM==2
(float G[len_x_cst+4][len_y_cst+4], simu_fields* fields, int idx_it)
#elif NB_DIM==3
(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4], simu_fields* fields, int idx_it)
#endif
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  #if NB_DIM==2
    if (idx < len_patch_cst*len_patch_cst){
      /* di and dj are the indices within the sub-domain to be smoothed. */
      int di = floorf(idx/len_patch_cst);
      int dj = idx - di*len_patch_cst;
      int half_len_patch = int(len_patch_cst/2);
      di -= int(half_len_patch/2);
      dj -= int(half_len_patch/2);
      /* i and j are the indices within the main simulation domain. */
      int i = fields->smooth_idx[0] + di;
      int j = fields->smooth_idx[1] + dj;
      /* eps_i and eps_j are the weighting of the sommothing, maximum (i.e. 1)
      at the centre of the window, i.e. at the one node that triggered the smoothing,
      and going linearly towards 0 at the boarders of the window.*/
      float eps_i = (half_len_patch-float(fabsf(di)))/half_len_patch;
      /* eps_ij may possibly be negative on the edge depending on the dimension/parity
       of the sub-domain. Undesired. */
      if (eps_i<0){
        eps_i = 0.;
      }
      float eps_j = (half_len_patch-float(fabsf(dj)))/half_len_patch;
      if (eps_j<0){
        eps_j = 0.;
      }

      /* Simply replace the field value by the weighted sum of the old value and the smoothed value. */
      if ((i>=2) && (i<=len_x_cst+1) && (j>=2) && (j<=len_y_cst+1)){
        G[i][j] = eps_i*eps_j*fields->smooth[i][j] + (1 - eps_i*eps_j)*G[i][j];
      }
    }
  #elif NB_DIM==3
    if (idx < len_patch_cst*len_patch_cst*len_patch_cst){
      /* di, dj and dk are the indices within the sub-domain to be smoothed. */
      int di = int(idx/(len_patch_cst*len_patch_cst));
      int dj = int( (idx - di*len_patch_cst*len_patch_cst)/len_patch_cst );
      int dk = idx - di*len_patch_cst*len_patch_cst - dj*len_patch_cst;
      int half_len_patch = int(len_patch_cst/2);
      di -= int(half_len_patch/2);
      dj -= int(half_len_patch/2);
      dk -= int(half_len_patch/2);
      /* i, j and k are the indices within the main simulation domain. */
      int i = fields->smooth_idx[0] + di;
      int j = fields->smooth_idx[1] + dj;
      int k = fields->smooth_idx[2] + dk;
      /* eps_i, eps_j and eps_k are the weighting of the sommothing, maximum (1)
      at the centre of the window, i.e. at the one node that triggered the smoothing,
      and going linearly towards 0 at the boarders of the window.*/
      float eps_i = (half_len_patch-float(fabsf(di)))/half_len_patch;
      /* eps_ij may possibly be negative on the edge depending on the dimension/parity
       of the sub-domain. Undesired. */
      if (eps_i<0){
        eps_i = 0.;
      }
      float eps_j = (half_len_patch-float(fabsf(dj)))/half_len_patch;
      if (eps_j<0){
        eps_j = 0.;
      }
      float eps_k = (half_len_patch-float(fabsf(dj)))/half_len_patch;
      if (eps_k<0){
        eps_k = 0.;
      }

      /* Simply replace the field value by the weighted sum of the old value and the smoothed value. */
      if ((i>=0) && (i<=len_x_cst+3) && (j>=0) && (j<=len_y_cst+3) && (k>=0) && (k<=len_z_cst+3)){
        G[i][j][k] = eps_i*eps_j*eps_k*fields->smooth[i][j][k] + (1 - eps_i*eps_j*eps_k)*G[i][j][k];
      }
    }
  #endif
}




//__________________________________________________________________________________________________________________
//
// Tricks kernels.
//
__global__ void smooth_downstream_k
#if NB_DIM==2
(float G[len_x_cst+4][len_y_cst+4], simu_fields* fields, simu_param* sP)
#elif NB_DIM==3
(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4], simu_fields* fields, simu_param* sP)
#endif
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  #if NB_DIM==2
    if(idx < sP->width_smooth_downstream*(len_y_cst+2)){
      int i = floorf(idx/(len_y_cst+2));
      int j = idx - i*(len_y_cst+2);
      i++;
      j++;

      fields->smooth[i][j] = 4*G[i][j] +
                              2*(G[i-1][j] + G[i+1][j] + G[i][j+1] + G[i][j-1]) +
                              G[i-1][j-1] + G[i-1][j+1] + G[i+1][j-1] + G[i+1][j+1];
      fields->smooth[i][j] /= 16.;

    }


  #elif NB_DIM==3
    if(idx < sP->width_smooth_downstream*(len_y_cst+2)*(len_z_cst+2)){
      int i = int( idx/((len_y_cst+2)*(len_z_cst+2)) );
      int j = int( (idx - i*(len_y_cst+2)*(len_z_cst+2))/(len_z_cst+2) );
      int k = idx - i*(len_y_cst+2)*(len_z_cst+2) - j*(len_z_cst+2);
      i++;
      j++;
      k++;

      fields->smooth[i][j][k] = 4*G[i][j][k] +
                                2*(G[i-1][j][k] + G[i+1][j][k] + G[i][j+1][k] + G[i][j-1][k] + G[i][j][k-1] + G[i][j][k+1]) +
                                G[i][j+1][k-1] +
                                G[i][j+1][k+1] +
                                G[i][j-1][k-1] +
                                G[i][j-1][k+1] +
                                G[i+1][j][k-1] +
                                G[i+1][j][k+1] +
                                G[i-1][j][k-1] +
                                G[i-1][j][k+1] +
                                G[i+1][j+1][k] +
                                G[i+1][j-1][k] +
                                G[i-1][j+1][k] +
                                G[i-1][j-1][k];
                                // G[i-1][j+1][k-1] +
                                // G[i-1][j+1][k+1]
                                // G[i+1][j+1][k-1] +
                                // G[i+1][j+1][k+1];
      fields->smooth[i][j][k] /= 28.;
    }

  #endif
}

__global__ void copy_smooth_downstream_k
#if NB_DIM==2
(float G[len_x_cst+4][len_y_cst+4], simu_fields* fields, simu_param* sP)
#elif NB_DIM==3
(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4], simu_fields* fields, simu_param* sP)
#endif
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  #if NB_DIM==2

    if(idx < sP->width_smooth_downstream*(len_y_cst+2)){
      int i = floorf(idx/(len_y_cst+2));
      int j = idx - i*(len_y_cst+2);
      i++;
      j++;

      if (i<int(sP->width_smooth_downstream/2.)+1){ // +1 otherwise a negative at the transition.
        G[i][j] = fields->smooth[i][j];
      }
      else{
        float a = 1. - 2.*(1-float(i)/sP->width_smooth_downstream);
        if (a<0. || a>1.){
          printf("Not cool copy_smooth_downstream. %i %.4e \n", i, a);
        }
        G[i][j] = (1-a)*fields->smooth[i][j] + a*G[i][j];
      }
    }


  #elif NB_DIM==3

    if(idx < sP->width_smooth_downstream*(len_y_cst+2)*(len_z_cst+2)){
      int i = int( idx/((len_y_cst+2)*(len_z_cst+2)) );
      int j = int( (idx - i*(len_y_cst+2)*(len_z_cst+2))/(len_z_cst+2) );
      int k = idx - i*(len_y_cst+2)*(len_z_cst+2) - j*(len_z_cst+2);
      i++;
      j++;
      k++;

      if (i<int(sP->width_smooth_downstream/2.)+1){
        G[i][j][k] = fields->smooth[i][j][k];
      }
      else{
        float a = 1. - 2.*(1-float(i)/sP->width_smooth_downstream);
        if (a<0. || a>1.){
          printf("Not cool copy_smooth_downstream.");
        }
        G[i][j][k] = (1-a)*fields->smooth[i][j][k] + a*G[i][j][k];
      }
    }

  #endif
}





// __global__ void stag_scalar_k(float G[len_x_cst+4][len_y_cst+4], simu_fields* fields)
// {
//   int idx = threadIdx.x + blockIdx.x*blockDim.x;
//
//   if (idx<nb_nodes_cst){
//     int i = int(idx/len_y_cst);
//     int j = idx - i*len_y_cst;
//     i += 2;
//     j += 2;
//
//     fields->E_stag[0][i][j] = G[i  ][j  ] * .25 +
//                               G[i  ][j+1] * .25 +
//                               G[i+1][j  ] * .25 +
//                               G[i+1][j+1] * .25;
//   }
// }
// __global__ void destag_scalar_k(float G[len_x_cst+4][len_y_cst+4], simu_fields* fields, simu_grid* grid)
// {
//   int idx = threadIdx.x + blockIdx.x*blockDim.x;
//
//   if (idx<nb_nodes_cst){
//     int i = int(idx/len_y_cst);
//     int j = idx - i*len_y_cst;
//     i += 2;
//     j += 2;
//
//     G[i][j] =
//             fields->E_stag[0][i  ][j  ] * .25 +
//             fields->E_stag[0][i  ][j-1] * .25 +
//             fields->E_stag[0][i-1][j  ] * .25 +
//             fields->E_stag[0][i-1][j-1] * .25;
//
//     // int indX = i-1;//roundf( (p->rx[idx]-grid->xMin)*sP->dX_i + .5) + 1;
//     // int indY = j-1;//roundf( (p->ry[idx]-grid->yMin)*sP->dX_i + .5) + 1;
//     //
//     // float wx = .5;// (p->rx[idx]-grid->xGrid[indX])*sP->dX_i;
//     // float wy = .5;// (p->ry[idx]-grid->yGrid[indY])*sP->dX_i;
//     //
//     // float wx0 = .5*(.5-wx)*(.5-wx);
//     // float wx1 = .75 - wx*wx;
//     // float wx2 = .5*(.5+wx)*(.5+wx);
//     //
//     // float wy0 = .5*(.5-wy)*(.5-wy);
//     // float wy1 = .75 - wy*wy;
//     // float wy2 = .5*(.5+wy)*(.5+wy);
//     //
//     // G[i][j] =
//     //         fields->E_stag[0][indX-1][indY-1]* wx0*wy0 +
//     //         fields->E_stag[0][indX-1][indY  ]* wx0*wy1 +
//     //         fields->E_stag[0][indX-1][indY+1]* wx0*wy2 +
//     //         fields->E_stag[0][indX  ][indY-1]* wx1*wy0 +
//     //         fields->E_stag[0][indX  ][indY  ]* wx1*wy1 +
//     //         fields->E_stag[0][indX  ][indY+1]* wx1*wy2 +
//     //         fields->E_stag[0][indX+1][indY-1]* wx2*wy0 +
//     //         fields->E_stag[0][indX+1][indY  ]* wx2*wy1 +
//     //         fields->E_stag[0][indX+1][indY+1]* wx2*wy2 ;
//
//
//   }
// }
// __global__ void destag_scalar_region_k(float G[len_x_cst+4][len_y_cst+4], simu_fields* fields, simu_grid* grid)
// {
//   int idx = threadIdx.x + blockIdx.x*blockDim.x;
//
//   if (idx<nb_nodes_cst){
//     int i = int(idx/len_y_cst);
//     int j = idx - i*len_y_cst;
//     i += 2;
//     j += 2;
//
//     // float E_max = 100;
//     if ( (grid->xGrid[i]>0. && grid->xGrid[i]<322. && grid->yGrid[j]>205. && grid->yGrid[j]<295.)
//         || grid->yGrid[j]>496.
//         || grid->yGrid[j]<4.
//         || grid->xGrid[i]<4.){
//
//       G[i][j] =
//               fields->E_stag[0][i  ][j  ] * .25 +
//               fields->E_stag[0][i  ][j-1] * .25 +
//               fields->E_stag[0][i-1][j  ] * .25 +
//               fields->E_stag[0][i-1][j-1] * .25;
//     // if (fields->E[0][i][j]>E_max || fields->E[1][i][j]>E_max || fields->E[2][i][j]>E_max){
//       // int indX = i-1;//roundf( (p->rx[idx]-grid->xMin)*sP->dX_i + .5) + 1;
//       // int indY = j-1;//roundf( (p->ry[idx]-grid->yMin)*sP->dX_i + .5) + 1;
//       //
//       // float wx = .5;// (p->rx[idx]-grid->xGrid[indX])*sP->dX_i;
//       // float wy = .5;// (p->ry[idx]-grid->yGrid[indY])*sP->dX_i;
//       //
//       // float wx0 = .5*(.5-wx)*(.5-wx);
//       // float wx1 = .75 - wx*wx;
//       // float wx2 = .5*(.5+wx)*(.5+wx);
//       //
//       // float wy0 = .5*(.5-wy)*(.5-wy);
//       // float wy1 = .75 - wy*wy;
//       // float wy2 = .5*(.5+wy)*(.5+wy);
//       //
//       //
//       // G[i][j] =
//       //         fields->E_stag[0][indX-1][indY-1]* wx0*wy0 +
//       //         fields->E_stag[0][indX-1][indY  ]* wx0*wy1 +
//       //         fields->E_stag[0][indX-1][indY+1]* wx0*wy2 +
//       //         fields->E_stag[0][indX  ][indY-1]* wx1*wy0 +
//       //         fields->E_stag[0][indX  ][indY  ]* wx1*wy1 +
//       //         fields->E_stag[0][indX  ][indY+1]* wx1*wy2 +
//       //         fields->E_stag[0][indX+1][indY-1]* wx2*wy0 +
//       //         fields->E_stag[0][indX+1][indY  ]* wx2*wy1 +
//       //         fields->E_stag[0][indX+1][indY+1]* wx2*wy2 ;
//     }
//   }
// }
// __global__ void copy_stag_scalar_k(float G[len_x_cst+4][len_y_cst+4], simu_fields* fields)
// {
//   int idx = threadIdx.x + blockIdx.x*blockDim.x;
//
//   if (idx<nb_nodes_tot_cst){
//     int i = int(idx/(len_y_cst+4));
//     int j = idx - i*(len_y_cst+4);
//
//       fields->E_stag[0][i][j] = G[i][j];
//   }
// }
//
//
#if NB_DIM==2
__global__ void stag_k(float G[3][len_x_cst+4][len_y_cst+4], simu_fields* fields)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx<nb_nodes_cst){
    int i = int(idx/len_y_cst);
    int j = idx - i*len_y_cst;
    i += 2;
    j += 2;

    fields->E_stag[0][i][j] = G[0][i  ][j  ] * .25 +
                              G[0][i  ][j+1] * .25 +
                              G[0][i+1][j  ] * .25 +
                              G[0][i+1][j+1] * .25;
    fields->E_stag[1][i][j] = G[1][i  ][j  ] * .25 +
                              G[1][i  ][j+1] * .25 +
                              G[1][i+1][j  ] * .25 +
                              G[1][i+1][j+1] * .25;
    fields->E_stag[2][i][j] = G[2][i  ][j  ] * .25 +
                              G[2][i  ][j+1] * .25 +
                              G[2][i+1][j  ] * .25 +
                              G[2][i+1][j+1] * .25;
  }
}
__global__ void destag_k(float G[3][len_x_cst+4][len_y_cst+4], simu_fields* fields, simu_grid* grid)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx<nb_nodes_cst){
    int i = int(idx/len_y_cst);
    int j = idx - i*len_y_cst;
    i += 2;
    j += 2;

    // G[0][i][j] =
    //         fields->E_stag[0][i  ][j  ] * .25 +
    //         fields->E_stag[0][i  ][j-1] * .25 +
    //         fields->E_stag[0][i-1][j  ] * .25 +
    //         fields->E_stag[0][i-1][j-1] * .25;
    // G[1][i][j] =
    //         fields->E_stag[0][i  ][j  ] * .25 +
    //         fields->E_stag[0][i  ][j-1] * .25 +
    //         fields->E_stag[0][i-1][j  ] * .25 +
    //         fields->E_stag[0][i-1][j-1] * .25;
    // G[2][i][j] =
    //         fields->E_stag[0][i  ][j  ] * .25 +
    //         fields->E_stag[0][i  ][j-1] * .25 +
    //         fields->E_stag[0][i-1][j  ] * .25 +
    //         fields->E_stag[0][i-1][j-1] * .25;

    // float E_max = 100;

      int indX = i-1;//roundf( (p->rx[idx]-grid->xMin)*sP->dX_i + .5) + 1;
      int indY = j-1;//roundf( (p->ry[idx]-grid->yMin)*sP->dX_i + .5) + 1;

      float wx = .5;// (p->rx[idx]-grid->xGrid[indX])*sP->dX_i;
      float wy = .5;// (p->ry[idx]-grid->yGrid[indY])*sP->dX_i;

      float wx0 = .5*(.5-wx)*(.5-wx);
      float wx1 = .75 - wx*wx;
      float wx2 = .5*(.5+wx)*(.5+wx);

      float wy0 = .5*(.5-wy)*(.5-wy);
      float wy1 = .75 - wy*wy;
      float wy2 = .5*(.5+wy)*(.5+wy);


      G[0][i][j] =
              fields->E_stag[0][indX-1][indY-1]* wx0*wy0 +
              fields->E_stag[0][indX-1][indY  ]* wx0*wy1 +
              fields->E_stag[0][indX-1][indY+1]* wx0*wy2 +
              fields->E_stag[0][indX  ][indY-1]* wx1*wy0 +
              fields->E_stag[0][indX  ][indY  ]* wx1*wy1 +
              fields->E_stag[0][indX  ][indY+1]* wx1*wy2 +
              fields->E_stag[0][indX+1][indY-1]* wx2*wy0 +
              fields->E_stag[0][indX+1][indY  ]* wx2*wy1 +
              fields->E_stag[0][indX+1][indY+1]* wx2*wy2;
      G[1][i][j] =
              fields->E_stag[1][indX-1][indY-1]* wx0*wy0 +
              fields->E_stag[1][indX-1][indY  ]* wx0*wy1 +
              fields->E_stag[1][indX-1][indY+1]* wx0*wy2 +
              fields->E_stag[1][indX  ][indY-1]* wx1*wy0 +
              fields->E_stag[1][indX  ][indY  ]* wx1*wy1 +
              fields->E_stag[1][indX  ][indY+1]* wx1*wy2 +
              fields->E_stag[1][indX+1][indY-1]* wx2*wy0 +
              fields->E_stag[1][indX+1][indY  ]* wx2*wy1 +
              fields->E_stag[1][indX+1][indY+1]* wx2*wy2 ;
      G[2][i][j] =
              fields->E_stag[2][indX-1][indY-1]* wx0*wy0 +
              fields->E_stag[2][indX-1][indY  ]* wx0*wy1 +
              fields->E_stag[2][indX-1][indY+1]* wx0*wy2 +
              fields->E_stag[2][indX  ][indY-1]* wx1*wy0 +
              fields->E_stag[2][indX  ][indY  ]* wx1*wy1 +
              fields->E_stag[2][indX  ][indY+1]* wx1*wy2 +
              fields->E_stag[2][indX+1][indY-1]* wx2*wy0 +
              fields->E_stag[2][indX+1][indY  ]* wx2*wy1 +
              fields->E_stag[2][indX+1][indY+1]* wx2*wy2 ;
      // G[0][i][j][k] = fields->E_stag[0][i][j][k];
      // G[1][i][j][k] = fields->E_stag[1][i][j][k];
      // G[2][i][j][k] = fields->E_stag[2][i][j][k];

  }
}
__global__ void destag_region_k(float G[3][len_x_cst+4][len_y_cst+4], simu_fields* fields, simu_grid* grid)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx<nb_nodes_cst){
    int i = int(idx/len_y_cst);
    int j = idx - i*len_y_cst;
    i += 2;
    j += 2;

    // float E_max = 100;

    // if (fields->E[0][i][j]>E_max || fields->E[1][i][j]>E_max || fields->E[2][i][j]>E_max){
    if ( (grid->xGrid[i]>0. && grid->xGrid[i]<322. && grid->yGrid[j]>205. && grid->yGrid[j]<295.)
        || grid->yGrid[j]>490.
        || grid->yGrid[j]<10.
        || grid->xGrid[i]<10.){

      G[0][i][j] =
              fields->E_stag[0][i  ][j  ] * .25 +
              fields->E_stag[0][i  ][j-1] * .25 +
              fields->E_stag[0][i-1][j  ] * .25 +
              fields->E_stag[0][i-1][j-1] * .25;
      G[1][i][j] =
              fields->E_stag[0][i  ][j  ] * .25 +
              fields->E_stag[0][i  ][j-1] * .25 +
              fields->E_stag[0][i-1][j  ] * .25 +
              fields->E_stag[0][i-1][j-1] * .25;
      G[2][i][j] =
              fields->E_stag[0][i  ][j  ] * .25 +
              fields->E_stag[0][i  ][j-1] * .25 +
              fields->E_stag[0][i-1][j  ] * .25 +
              fields->E_stag[0][i-1][j-1] * .25;
      // int indX = i-1;//roundf( (p->rx[idx]-grid->xMin)*sP->dX_i + .5) + 1;
      // int indY = j-1;//roundf( (p->ry[idx]-grid->yMin)*sP->dX_i + .5) + 1;
      //
      // float wx = .5;// (p->rx[idx]-grid->xGrid[indX])*sP->dX_i;
      // float wy = .5;// (p->ry[idx]-grid->yGrid[indY])*sP->dX_i;
      //
      // float wx0 = .5*(.5-wx)*(.5-wx);
      // float wx1 = .75 - wx*wx;
      // float wx2 = .5*(.5+wx)*(.5+wx);
      //
      // float wy0 = .5*(.5-wy)*(.5-wy);
      // float wy1 = .75 - wy*wy;
      // float wy2 = .5*(.5+wy)*(.5+wy);
      //
      //
      // G[0][i][j] =
      //         fields->E_stag[0][indX-1][indY-1]* wx0*wy0 +
      //         fields->E_stag[0][indX-1][indY  ]* wx0*wy1 +
      //         fields->E_stag[0][indX-1][indY+1]* wx0*wy2 +
      //         fields->E_stag[0][indX  ][indY-1]* wx1*wy0 +
      //         fields->E_stag[0][indX  ][indY  ]* wx1*wy1 +
      //         fields->E_stag[0][indX  ][indY+1]* wx1*wy2 +
      //         fields->E_stag[0][indX+1][indY-1]* wx2*wy0 +
      //         fields->E_stag[0][indX+1][indY  ]* wx2*wy1 +
      //         fields->E_stag[0][indX+1][indY+1]* wx2*wy2;
      // G[1][i][j] =
      //         fields->E_stag[1][indX-1][indY-1]* wx0*wy0 +
      //         fields->E_stag[1][indX-1][indY  ]* wx0*wy1 +
      //         fields->E_stag[1][indX-1][indY+1]* wx0*wy2 +
      //         fields->E_stag[1][indX  ][indY-1]* wx1*wy0 +
      //         fields->E_stag[1][indX  ][indY  ]* wx1*wy1 +
      //         fields->E_stag[1][indX  ][indY+1]* wx1*wy2 +
      //         fields->E_stag[1][indX+1][indY-1]* wx2*wy0 +
      //         fields->E_stag[1][indX+1][indY  ]* wx2*wy1 +
      //         fields->E_stag[1][indX+1][indY+1]* wx2*wy2 ;
      // G[2][i][j] =
      //         fields->E_stag[2][indX-1][indY-1]* wx0*wy0 +
      //         fields->E_stag[2][indX-1][indY  ]* wx0*wy1 +
      //         fields->E_stag[2][indX-1][indY+1]* wx0*wy2 +
      //         fields->E_stag[2][indX  ][indY-1]* wx1*wy0 +
      //         fields->E_stag[2][indX  ][indY  ]* wx1*wy1 +
      //         fields->E_stag[2][indX  ][indY+1]* wx1*wy2 +
      //         fields->E_stag[2][indX+1][indY-1]* wx2*wy0 +
      //         fields->E_stag[2][indX+1][indY  ]* wx2*wy1 +
      //         fields->E_stag[2][indX+1][indY+1]* wx2*wy2 ;
    }
  }
}
__global__ void copy_stag_k(float G[3][len_x_cst+4][len_y_cst+4], simu_fields* fields)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx<nb_nodes_tot_cst){
    int i = int(idx/(len_y_cst+4));
    int j = idx-i*(len_y_cst+4);

    fields->E_stag[0][i][j] = G[0][i][j];
    fields->E_stag[1][i][j] = G[1][i][j];
    fields->E_stag[2][i][j] = G[2][i][j];
  }
}
#endif
//
//
//
//
//
//
//
//
//
//
// __global__ void anti_alias_k(simu_B_field* B, simu_fields* fields)
// {
//
//   int idx = threadIdx.x + blockIdx.x*blockDim.x;
//
//   if (idx<nb_nodes_cst){
//     int i = int(idx/len_y_cst);
//     int j = idx-i*len_y_cst;
//     i += 2;
//     j += 2;
//
//     B->B[0][i][j] = .5 * (B->B[0][i][j] + fields->E_stag[0][i][j]);
//     B->B[1][i][j] = .5 * (B->B[1][i][j] + fields->E_stag[1][i][j]);
//     B->B[2][i][j] = .5 * (B->B[2][i][j] + fields->E_stag[2][i][j]);
//   }
//
// }
//
// __global__ void pimp_density_k(simu_fields* fields)
// {
//
//   int idx = threadIdx.x + blockIdx.x*blockDim.x;
//
//   if (idx<(len_y_cst+4)){
//
//     fields->density[0][idx] = 10.;
//     fields->density[1][idx] = 10.;
//     fields->density[2][idx] = 10.;
//
//   }
// }
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
// #elif NB_DIM==3
// __global__ void positive_poynting_k(simu_fields* fields, simu_B_field* B)
// {
//   int idx = threadIdx.x + blockIdx.x*blockDim.x;
//
//   if (idx < (len_y_cst+4)*(len_z_cst+4)){
//     int j = int(idx/(len_z_cst+4));
//     int k = idx - (j*len_z_cst+4);
//     float poynting_x = fields->E[1][2][j][k]*B->B[2][2][j][k] - fields->E[2][2][j][k]*B->B[1][2][j][k];
//     if (poynting_x>0){
//       B->B[2][2][j][k] *= -1;
//       B->B[1][2][j][k] *= -1;
//     }
//   }
// }
//
// __global__ void stag_scalar_k(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4], simu_fields* fields)
// {
//   int idx = threadIdx.x + blockIdx.x*blockDim.x;
//
//   if (idx<nb_nodes_cst){
//     int i = int( idx/(len_y_cst*len_z_cst) );
//     int j = int( (idx-i*len_y_cst*len_z_cst)/len_z_cst );
//     int k = idx - i*len_y_cst*len_z_cst - j*len_z_cst;
//     i += 2;
//     j += 2;
//     k += 2;
//
//     int indX = i;//roundf( (p->rx[idx]-grid->xMin)*sP->dX_i + .5) + 1;
//     int indY = j;//roundf( (p->ry[idx]-grid->yMin)*sP->dX_i + .5) + 1;
//     int indZ = k;//roundf( (p->rz[idx]-grid->zMin)*sP->dX_i + .5) + 1;
//
//     float wx = .5;//(p->rx[idx]-grid->xGrid[indX])*sP->dX_i;
//     float wy = .5;//(p->ry[idx]-grid->yGrid[indY])*sP->dX_i;
//     float wz = .5;//(p->rz[idx]-grid->zGrid[indZ])*sP->dX_i;
//
//     float wx0 = .5*(.5-wx)*(.5-wx);
//     float wx1 = .75 - wx*wx;
//     float wx2 = .5*(.5+wx)*(.5+wx);
//
//     float wy0 = .5*(.5-wy)*(.5-wy);
//     float wy1 = .75 - wy*wy;
//     float wy2 = .5*(.5+wy)*(.5+wy);
//
//     float wz0 = .5*(.5-wz)*(.5-wz);
//     float wz1 = .75 - wz*wz;
//     float wz2 = .5*(.5+wz)*(.5+wz);
//
//
//     fields->E_stag[0][i][j][k] =
//             G[indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
//             G[indX-1][indY-1][indZ  ]* wx0*wy0*wz1 +
//             G[indX-1][indY-1][indZ+1]* wx0*wy0*wz2 +
//             G[indX-1][indY  ][indZ-1]* wx0*wy1*wz0 +
//             G[indX-1][indY  ][indZ  ]* wx0*wy1*wz1 +
//             G[indX-1][indY  ][indZ+1]* wx0*wy1*wz2 +
//             G[indX-1][indY+1][indZ-1]* wx0*wy2*wz0 +
//             G[indX-1][indY+1][indZ  ]* wx0*wy2*wz1 +
//             G[indX-1][indY+1][indZ+1]* wx0*wy2*wz2 +
//             G[indX  ][indY-1][indZ-1]* wx1*wy0*wz0 +
//             G[indX  ][indY-1][indZ  ]* wx1*wy0*wz1 +
//             G[indX  ][indY-1][indZ+1]* wx1*wy0*wz2 +
//             G[indX  ][indY  ][indZ-1]* wx1*wy1*wz0 +
//             G[indX  ][indY  ][indZ  ]* wx1*wy1*wz1 +
//             G[indX  ][indY  ][indZ+1]* wx1*wy1*wz2 +
//             G[indX  ][indY+1][indZ-1]* wx1*wy2*wz0 +
//             G[indX  ][indY+1][indZ  ]* wx1*wy2*wz1 +
//             G[indX  ][indY+1][indZ+1]* wx1*wy2*wz2 +
//             G[indX+1][indY-1][indZ-1]* wx2*wy0*wz0 +
//             G[indX+1][indY-1][indZ  ]* wx2*wy0*wz1 +
//             G[indX+1][indY-1][indZ+1]* wx2*wy0*wz2 +
//             G[indX+1][indY  ][indZ-1]* wx2*wy1*wz0 +
//             G[indX+1][indY  ][indZ  ]* wx2*wy1*wz1 +
//             G[indX+1][indY  ][indZ+1]* wx2*wy1*wz2 +
//             G[indX+1][indY+1][indZ-1]* wx2*wy2*wz0 +
//             G[indX+1][indY+1][indZ  ]* wx2*wy2*wz1 +
//             G[indX+1][indY+1][indZ+1]* wx2*wy2*wz2  ;
//   }
// }
// __global__ void destag_scalar_k(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4], simu_fields* fields)
// {
//   int idx = threadIdx.x + blockIdx.x*blockDim.x;
//
//   if (idx<nb_nodes_cst){
//     int i = int( idx/(len_y_cst*len_z_cst) );
//     int j = int( (idx-i*len_y_cst*len_z_cst)/len_z_cst );
//     int k = idx - i*len_y_cst*len_z_cst - j*len_z_cst;
//     i += 2;
//     j += 2;
//     k += 2;
//
//     int indX = i-1;//roundf( (p->rx[idx]-grid->xMin)*sP->dX_i + .5) + 1;
//     int indY = j-1;//roundf( (p->ry[idx]-grid->yMin)*sP->dX_i + .5) + 1;
//     int indZ = k-1;//roundf( (p->rz[idx]-grid->zMin)*sP->dX_i + .5) + 1;
//
//     float wx = .5;// (p->rx[idx]-grid->xGrid[indX])*sP->dX_i;
//     float wy = .5;// (p->ry[idx]-grid->yGrid[indY])*sP->dX_i;
//     float wz = .5;// (p->rz[idx]-grid->zGrid[indZ])*sP->dX_i;
//
//     float wx0 = .5*(.5-wx)*(.5-wx);
//     float wx1 = .75 - wx*wx;
//     float wx2 = .5*(.5+wx)*(.5+wx);
//
//     float wy0 = .5*(.5-wy)*(.5-wy);
//     float wy1 = .75 - wy*wy;
//     float wy2 = .5*(.5+wy)*(.5+wy);
//
//     float wz0 = .5*(.5-wz)*(.5-wz);
//     float wz1 = .75 - wz*wz;
//     float wz2 = .5*(.5+wz)*(.5+wz);
//
//
//     G[i][j][k] =
//             fields->E_stag[0][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
//             fields->E_stag[0][indX-1][indY-1][indZ  ]* wx0*wy0*wz1 +
//             fields->E_stag[0][indX-1][indY-1][indZ+1]* wx0*wy0*wz2 +
//             fields->E_stag[0][indX-1][indY  ][indZ-1]* wx0*wy1*wz0 +
//             fields->E_stag[0][indX-1][indY  ][indZ  ]* wx0*wy1*wz1 +
//             fields->E_stag[0][indX-1][indY  ][indZ+1]* wx0*wy1*wz2 +
//             fields->E_stag[0][indX-1][indY+1][indZ-1]* wx0*wy2*wz0 +
//             fields->E_stag[0][indX-1][indY+1][indZ  ]* wx0*wy2*wz1 +
//             fields->E_stag[0][indX-1][indY+1][indZ+1]* wx0*wy2*wz2 +
//             fields->E_stag[0][indX  ][indY-1][indZ-1]* wx1*wy0*wz0 +
//             fields->E_stag[0][indX  ][indY-1][indZ  ]* wx1*wy0*wz1 +
//             fields->E_stag[0][indX  ][indY-1][indZ+1]* wx1*wy0*wz2 +
//             fields->E_stag[0][indX  ][indY  ][indZ-1]* wx1*wy1*wz0 +
//             fields->E_stag[0][indX  ][indY  ][indZ  ]* wx1*wy1*wz1 +
//             fields->E_stag[0][indX  ][indY  ][indZ+1]* wx1*wy1*wz2 +
//             fields->E_stag[0][indX  ][indY+1][indZ-1]* wx1*wy2*wz0 +
//             fields->E_stag[0][indX  ][indY+1][indZ  ]* wx1*wy2*wz1 +
//             fields->E_stag[0][indX  ][indY+1][indZ+1]* wx1*wy2*wz2 +
//             fields->E_stag[0][indX+1][indY-1][indZ-1]* wx2*wy0*wz0 +
//             fields->E_stag[0][indX+1][indY-1][indZ  ]* wx2*wy0*wz1 +
//             fields->E_stag[0][indX+1][indY-1][indZ+1]* wx2*wy0*wz2 +
//             fields->E_stag[0][indX+1][indY  ][indZ-1]* wx2*wy1*wz0 +
//             fields->E_stag[0][indX+1][indY  ][indZ  ]* wx2*wy1*wz1 +
//             fields->E_stag[0][indX+1][indY  ][indZ+1]* wx2*wy1*wz2 +
//             fields->E_stag[0][indX+1][indY+1][indZ-1]* wx2*wy2*wz0 +
//             fields->E_stag[0][indX+1][indY+1][indZ  ]* wx2*wy2*wz1 +
//             fields->E_stag[0][indX+1][indY+1][indZ+1]* wx2*wy2*wz2  ;
//     // G[0][i][j][k] = fields->E_stag[0][i][j][k];
//     // G[1][i][j][k] = fields->E_stag[1][i][j][k];
//     // G[2][i][j][k] = fields->E_stag[2][i][j][k];
//   }
// }
// __global__ void copy_stag_scalar_k(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4], simu_fields* fields)
// {
//   int idx = threadIdx.x + blockIdx.x*blockDim.x;
//
//   if (idx<nb_nodes_tot_cst){
//     int i = int( idx/((len_y_cst+4)*(len_z_cst+4)) );
//     int j = int( (idx - i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
//     int k = idx - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);
//
//     fields->E_stag[0][i][j][k] = G[i][j][k];
//   }
// }
//
//
#if NB_DIM==3
__global__ void stag_k(float G[3][len_x_cst+4][len_y_cst+4][len_z_cst+4], simu_fields* fields)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx<nb_nodes_cst){
    int i = int( idx/(len_y_cst*len_z_cst) );
    int j = int( (idx-i*len_y_cst*len_z_cst)/len_z_cst );
    int k = idx - i*len_y_cst*len_z_cst - j*len_z_cst;
    i += 2;
    j += 2;
    k += 2;

    int indX = i;//roundf( (p->rx[idx]-grid->xMin)*sP->dX_i + .5) + 1;
    int indY = j;//roundf( (p->ry[idx]-grid->yMin)*sP->dX_i + .5) + 1;
    int indZ = k;//roundf( (p->rz[idx]-grid->zMin)*sP->dX_i + .5) + 1;

    float wx = .5;//(p->rx[idx]-grid->xGrid[indX])*sP->dX_i;
    float wy = .5;//(p->ry[idx]-grid->yGrid[indY])*sP->dX_i;
    float wz = .5;//(p->rz[idx]-grid->zGrid[indZ])*sP->dX_i;

    float wx0 = .5*(.5-wx)*(.5-wx);
    float wx1 = .75 - wx*wx;
    float wx2 = .5*(.5+wx)*(.5+wx);

    float wy0 = .5*(.5-wy)*(.5-wy);
    float wy1 = .75 - wy*wy;
    float wy2 = .5*(.5+wy)*(.5+wy);

    float wz0 = .5*(.5-wz)*(.5-wz);
    float wz1 = .75 - wz*wz;
    float wz2 = .5*(.5+wz)*(.5+wz);


    fields->E_stag[0][i][j][k] =
            G[0][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
            G[0][indX-1][indY-1][indZ  ]* wx0*wy0*wz1 +
            G[0][indX-1][indY-1][indZ+1]* wx0*wy0*wz2 +
            G[0][indX-1][indY  ][indZ-1]* wx0*wy1*wz0 +
            G[0][indX-1][indY  ][indZ  ]* wx0*wy1*wz1 +
            G[0][indX-1][indY  ][indZ+1]* wx0*wy1*wz2 +
            G[0][indX-1][indY+1][indZ-1]* wx0*wy2*wz0 +
            G[0][indX-1][indY+1][indZ  ]* wx0*wy2*wz1 +
            G[0][indX-1][indY+1][indZ+1]* wx0*wy2*wz2 +
            G[0][indX  ][indY-1][indZ-1]* wx1*wy0*wz0 +
            G[0][indX  ][indY-1][indZ  ]* wx1*wy0*wz1 +
            G[0][indX  ][indY-1][indZ+1]* wx1*wy0*wz2 +
            G[0][indX  ][indY  ][indZ-1]* wx1*wy1*wz0 +
            G[0][indX  ][indY  ][indZ  ]* wx1*wy1*wz1 +
            G[0][indX  ][indY  ][indZ+1]* wx1*wy1*wz2 +
            G[0][indX  ][indY+1][indZ-1]* wx1*wy2*wz0 +
            G[0][indX  ][indY+1][indZ  ]* wx1*wy2*wz1 +
            G[0][indX  ][indY+1][indZ+1]* wx1*wy2*wz2 +
            G[0][indX+1][indY-1][indZ-1]* wx2*wy0*wz0 +
            G[0][indX+1][indY-1][indZ  ]* wx2*wy0*wz1 +
            G[0][indX+1][indY-1][indZ+1]* wx2*wy0*wz2 +
            G[0][indX+1][indY  ][indZ-1]* wx2*wy1*wz0 +
            G[0][indX+1][indY  ][indZ  ]* wx2*wy1*wz1 +
            G[0][indX+1][indY  ][indZ+1]* wx2*wy1*wz2 +
            G[0][indX+1][indY+1][indZ-1]* wx2*wy2*wz0 +
            G[0][indX+1][indY+1][indZ  ]* wx2*wy2*wz1 +
            G[0][indX+1][indY+1][indZ+1]* wx2*wy2*wz2  ;
    fields->E_stag[1][i][j][k] =
            G[1][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
            G[1][indX-1][indY-1][indZ  ]* wx0*wy0*wz1 +
            G[1][indX-1][indY-1][indZ+1]* wx0*wy0*wz2 +
            G[1][indX-1][indY  ][indZ-1]* wx0*wy1*wz0 +
            G[1][indX-1][indY  ][indZ  ]* wx0*wy1*wz1 +
            G[1][indX-1][indY  ][indZ+1]* wx0*wy1*wz2 +
            G[1][indX-1][indY+1][indZ-1]* wx0*wy2*wz0 +
            G[1][indX-1][indY+1][indZ  ]* wx0*wy2*wz1 +
            G[1][indX-1][indY+1][indZ+1]* wx0*wy2*wz2 +
            G[1][indX  ][indY-1][indZ-1]* wx1*wy0*wz0 +
            G[1][indX  ][indY-1][indZ  ]* wx1*wy0*wz1 +
            G[1][indX  ][indY-1][indZ+1]* wx1*wy0*wz2 +
            G[1][indX  ][indY  ][indZ-1]* wx1*wy1*wz0 +
            G[1][indX  ][indY  ][indZ  ]* wx1*wy1*wz1 +
            G[1][indX  ][indY  ][indZ+1]* wx1*wy1*wz2 +
            G[1][indX  ][indY+1][indZ-1]* wx1*wy2*wz0 +
            G[1][indX  ][indY+1][indZ  ]* wx1*wy2*wz1 +
            G[1][indX  ][indY+1][indZ+1]* wx1*wy2*wz2 +
            G[1][indX+1][indY-1][indZ-1]* wx2*wy0*wz0 +
            G[1][indX+1][indY-1][indZ  ]* wx2*wy0*wz1 +
            G[1][indX+1][indY-1][indZ+1]* wx2*wy0*wz2 +
            G[1][indX+1][indY  ][indZ-1]* wx2*wy1*wz0 +
            G[1][indX+1][indY  ][indZ  ]* wx2*wy1*wz1 +
            G[1][indX+1][indY  ][indZ+1]* wx2*wy1*wz2 +
            G[1][indX+1][indY+1][indZ-1]* wx2*wy2*wz0 +
            G[1][indX+1][indY+1][indZ  ]* wx2*wy2*wz1 +
            G[1][indX+1][indY+1][indZ+1]* wx2*wy2*wz2  ;
    fields->E_stag[2][i][j][k]
          = G[2][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
            G[2][indX-1][indY-1][indZ  ]* wx0*wy0*wz1 +
            G[2][indX-1][indY-1][indZ+1]* wx0*wy0*wz2 +
            G[2][indX-1][indY  ][indZ-1]* wx0*wy1*wz0 +
            G[2][indX-1][indY  ][indZ  ]* wx0*wy1*wz1 +
            G[2][indX-1][indY  ][indZ+1]* wx0*wy1*wz2 +
            G[2][indX-1][indY+1][indZ-1]* wx0*wy2*wz0 +
            G[2][indX-1][indY+1][indZ  ]* wx0*wy2*wz1 +
            G[2][indX-1][indY+1][indZ+1]* wx0*wy2*wz2 +
            G[2][indX  ][indY-1][indZ-1]* wx1*wy0*wz0 +
            G[2][indX  ][indY-1][indZ  ]* wx1*wy0*wz1 +
            G[2][indX  ][indY-1][indZ+1]* wx1*wy0*wz2 +
            G[2][indX  ][indY  ][indZ-1]* wx1*wy1*wz0 +
            G[2][indX  ][indY  ][indZ  ]* wx1*wy1*wz1 +
            G[2][indX  ][indY  ][indZ+1]* wx1*wy1*wz2 +
            G[2][indX  ][indY+1][indZ-1]* wx1*wy2*wz0 +
            G[2][indX  ][indY+1][indZ  ]* wx1*wy2*wz1 +
            G[2][indX  ][indY+1][indZ+1]* wx1*wy2*wz2 +
            G[2][indX+1][indY-1][indZ-1]* wx2*wy0*wz0 +
            G[2][indX+1][indY-1][indZ  ]* wx2*wy0*wz1 +
            G[2][indX+1][indY-1][indZ+1]* wx2*wy0*wz2 +
            G[2][indX+1][indY  ][indZ-1]* wx2*wy1*wz0 +
            G[2][indX+1][indY  ][indZ  ]* wx2*wy1*wz1 +
            G[2][indX+1][indY  ][indZ+1]* wx2*wy1*wz2 +
            G[2][indX+1][indY+1][indZ-1]* wx2*wy2*wz0 +
            G[2][indX+1][indY+1][indZ  ]* wx2*wy2*wz1 +
            G[2][indX+1][indY+1][indZ+1]* wx2*wy2*wz2  ;
  }
}
__global__ void destag_k(float G[3][len_x_cst+4][len_y_cst+4][len_z_cst+4], simu_fields* fields, simu_grid* grid)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx<nb_nodes_cst){
    int i = int( idx/(len_y_cst*len_z_cst) );
    int j = int( (idx-i*len_y_cst*len_z_cst)/len_z_cst );
    int k = idx - i*len_y_cst*len_z_cst - j*len_z_cst;
    i += 2;
    j += 2;
    k += 2;

    int indX = i-1;//roundf( (p->rx[idx]-grid->xMin)*sP->dX_i + .5) + 1;
    int indY = j-1;//roundf( (p->ry[idx]-grid->yMin)*sP->dX_i + .5) + 1;
    int indZ = k-1;//roundf( (p->rz[idx]-grid->zMin)*sP->dX_i + .5) + 1;

    float wx = .5;// (p->rx[idx]-grid->xGrid[indX])*sP->dX_i;
    float wy = .5;// (p->ry[idx]-grid->yGrid[indY])*sP->dX_i;
    float wz = .5;// (p->rz[idx]-grid->zGrid[indZ])*sP->dX_i;

    float wx0 = .5*(.5-wx)*(.5-wx);
    float wx1 = .75 - wx*wx;
    float wx2 = .5*(.5+wx)*(.5+wx);

    float wy0 = .5*(.5-wy)*(.5-wy);
    float wy1 = .75 - wy*wy;
    float wy2 = .5*(.5+wy)*(.5+wy);

    float wz0 = .5*(.5-wz)*(.5-wz);
    float wz1 = .75 - wz*wz;
    float wz2 = .5*(.5+wz)*(.5+wz);


    G[0][i][j][k] =
            fields->E_stag[0][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
            fields->E_stag[0][indX-1][indY-1][indZ  ]* wx0*wy0*wz1 +
            fields->E_stag[0][indX-1][indY-1][indZ+1]* wx0*wy0*wz2 +
            fields->E_stag[0][indX-1][indY  ][indZ-1]* wx0*wy1*wz0 +
            fields->E_stag[0][indX-1][indY  ][indZ  ]* wx0*wy1*wz1 +
            fields->E_stag[0][indX-1][indY  ][indZ+1]* wx0*wy1*wz2 +
            fields->E_stag[0][indX-1][indY+1][indZ-1]* wx0*wy2*wz0 +
            fields->E_stag[0][indX-1][indY+1][indZ  ]* wx0*wy2*wz1 +
            fields->E_stag[0][indX-1][indY+1][indZ+1]* wx0*wy2*wz2 +
            fields->E_stag[0][indX  ][indY-1][indZ-1]* wx1*wy0*wz0 +
            fields->E_stag[0][indX  ][indY-1][indZ  ]* wx1*wy0*wz1 +
            fields->E_stag[0][indX  ][indY-1][indZ+1]* wx1*wy0*wz2 +
            fields->E_stag[0][indX  ][indY  ][indZ-1]* wx1*wy1*wz0 +
            fields->E_stag[0][indX  ][indY  ][indZ  ]* wx1*wy1*wz1 +
            fields->E_stag[0][indX  ][indY  ][indZ+1]* wx1*wy1*wz2 +
            fields->E_stag[0][indX  ][indY+1][indZ-1]* wx1*wy2*wz0 +
            fields->E_stag[0][indX  ][indY+1][indZ  ]* wx1*wy2*wz1 +
            fields->E_stag[0][indX  ][indY+1][indZ+1]* wx1*wy2*wz2 +
            fields->E_stag[0][indX+1][indY-1][indZ-1]* wx2*wy0*wz0 +
            fields->E_stag[0][indX+1][indY-1][indZ  ]* wx2*wy0*wz1 +
            fields->E_stag[0][indX+1][indY-1][indZ+1]* wx2*wy0*wz2 +
            fields->E_stag[0][indX+1][indY  ][indZ-1]* wx2*wy1*wz0 +
            fields->E_stag[0][indX+1][indY  ][indZ  ]* wx2*wy1*wz1 +
            fields->E_stag[0][indX+1][indY  ][indZ+1]* wx2*wy1*wz2 +
            fields->E_stag[0][indX+1][indY+1][indZ-1]* wx2*wy2*wz0 +
            fields->E_stag[0][indX+1][indY+1][indZ  ]* wx2*wy2*wz1 +
            fields->E_stag[0][indX+1][indY+1][indZ+1]* wx2*wy2*wz2  ;
    G[1][i][j][k] =
            fields->E_stag[1][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
            fields->E_stag[1][indX-1][indY-1][indZ  ]* wx0*wy0*wz1 +
            fields->E_stag[1][indX-1][indY-1][indZ+1]* wx0*wy0*wz2 +
            fields->E_stag[1][indX-1][indY  ][indZ-1]* wx0*wy1*wz0 +
            fields->E_stag[1][indX-1][indY  ][indZ  ]* wx0*wy1*wz1 +
            fields->E_stag[1][indX-1][indY  ][indZ+1]* wx0*wy1*wz2 +
            fields->E_stag[1][indX-1][indY+1][indZ-1]* wx0*wy2*wz0 +
            fields->E_stag[1][indX-1][indY+1][indZ  ]* wx0*wy2*wz1 +
            fields->E_stag[1][indX-1][indY+1][indZ+1]* wx0*wy2*wz2 +
            fields->E_stag[1][indX  ][indY-1][indZ-1]* wx1*wy0*wz0 +
            fields->E_stag[1][indX  ][indY-1][indZ  ]* wx1*wy0*wz1 +
            fields->E_stag[1][indX  ][indY-1][indZ+1]* wx1*wy0*wz2 +
            fields->E_stag[1][indX  ][indY  ][indZ-1]* wx1*wy1*wz0 +
            fields->E_stag[1][indX  ][indY  ][indZ  ]* wx1*wy1*wz1 +
            fields->E_stag[1][indX  ][indY  ][indZ+1]* wx1*wy1*wz2 +
            fields->E_stag[1][indX  ][indY+1][indZ-1]* wx1*wy2*wz0 +
            fields->E_stag[1][indX  ][indY+1][indZ  ]* wx1*wy2*wz1 +
            fields->E_stag[1][indX  ][indY+1][indZ+1]* wx1*wy2*wz2 +
            fields->E_stag[1][indX+1][indY-1][indZ-1]* wx2*wy0*wz0 +
            fields->E_stag[1][indX+1][indY-1][indZ  ]* wx2*wy0*wz1 +
            fields->E_stag[1][indX+1][indY-1][indZ+1]* wx2*wy0*wz2 +
            fields->E_stag[1][indX+1][indY  ][indZ-1]* wx2*wy1*wz0 +
            fields->E_stag[1][indX+1][indY  ][indZ  ]* wx2*wy1*wz1 +
            fields->E_stag[1][indX+1][indY  ][indZ+1]* wx2*wy1*wz2 +
            fields->E_stag[1][indX+1][indY+1][indZ-1]* wx2*wy2*wz0 +
            fields->E_stag[1][indX+1][indY+1][indZ  ]* wx2*wy2*wz1 +
            fields->E_stag[1][indX+1][indY+1][indZ+1]* wx2*wy2*wz2  ;
    G[2][i][j][k] =
            fields->E_stag[2][indX-1][indY-1][indZ-1]* wx0*wy0*wz0 +
            fields->E_stag[2][indX-1][indY-1][indZ  ]* wx0*wy0*wz1 +
            fields->E_stag[2][indX-1][indY-1][indZ+1]* wx0*wy0*wz2 +
            fields->E_stag[2][indX-1][indY  ][indZ-1]* wx0*wy1*wz0 +
            fields->E_stag[2][indX-1][indY  ][indZ  ]* wx0*wy1*wz1 +
            fields->E_stag[2][indX-1][indY  ][indZ+1]* wx0*wy1*wz2 +
            fields->E_stag[2][indX-1][indY+1][indZ-1]* wx0*wy2*wz0 +
            fields->E_stag[2][indX-1][indY+1][indZ  ]* wx0*wy2*wz1 +
            fields->E_stag[2][indX-1][indY+1][indZ+1]* wx0*wy2*wz2 +
            fields->E_stag[2][indX  ][indY-1][indZ-1]* wx1*wy0*wz0 +
            fields->E_stag[2][indX  ][indY-1][indZ  ]* wx1*wy0*wz1 +
            fields->E_stag[2][indX  ][indY-1][indZ+1]* wx1*wy0*wz2 +
            fields->E_stag[2][indX  ][indY  ][indZ-1]* wx1*wy1*wz0 +
            fields->E_stag[2][indX  ][indY  ][indZ  ]* wx1*wy1*wz1 +
            fields->E_stag[2][indX  ][indY  ][indZ+1]* wx1*wy1*wz2 +
            fields->E_stag[2][indX  ][indY+1][indZ-1]* wx1*wy2*wz0 +
            fields->E_stag[2][indX  ][indY+1][indZ  ]* wx1*wy2*wz1 +
            fields->E_stag[2][indX  ][indY+1][indZ+1]* wx1*wy2*wz2 +
            fields->E_stag[2][indX+1][indY-1][indZ-1]* wx2*wy0*wz0 +
            fields->E_stag[2][indX+1][indY-1][indZ  ]* wx2*wy0*wz1 +
            fields->E_stag[2][indX+1][indY-1][indZ+1]* wx2*wy0*wz2 +
            fields->E_stag[2][indX+1][indY  ][indZ-1]* wx2*wy1*wz0 +
            fields->E_stag[2][indX+1][indY  ][indZ  ]* wx2*wy1*wz1 +
            fields->E_stag[2][indX+1][indY  ][indZ+1]* wx2*wy1*wz2 +
            fields->E_stag[2][indX+1][indY+1][indZ-1]* wx2*wy2*wz0 +
            fields->E_stag[2][indX+1][indY+1][indZ  ]* wx2*wy2*wz1 +
            fields->E_stag[2][indX+1][indY+1][indZ+1]* wx2*wy2*wz2  ;
    // G[0][i][j][k] = fields->E_stag[0][i][j][k];
    // G[1][i][j][k] = fields->E_stag[1][i][j][k];
    // G[2][i][j][k] = fields->E_stag[2][i][j][k];
  }
}
__global__ void copy_stag_k(float G[3][len_x_cst+4][len_y_cst+4][len_z_cst+4], simu_fields* fields)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx<nb_nodes_tot_cst){
    int i = int( idx/((len_y_cst+4)*(len_z_cst+4)) );
    int j = int( (idx - i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
    int k = idx - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);

    fields->E_stag[0][i][j][k] = G[0][i][j][k];
    fields->E_stag[1][i][j][k] = G[1][i][j][k];
    fields->E_stag[2][i][j][k] = G[2][i][j][k];
  }
}
#endif


#endif


__global__ void save_smooth_patch_k(simu_fields* fields, simu_grid* grid, int idx_it)
{
  /* Saving the patches location and time. */
  int idx_save = atomicAdd(&fields->idx_save_smooth, 1);

  if (idx_save < smooth_patch_save_len_cst){
    #if NB_DIM==2
      fields->save_smooth[0][idx_save] = grid->xGrid[fields->smooth_idx[0]];
      fields->save_smooth[1][idx_save] = grid->yGrid[fields->smooth_idx[1]];
      fields->save_smooth[2][idx_save] = idx_it;
    #elif NB_DIM==3
      fields->save_smooth[0][idx_save] = grid->xGrid[fields->smooth_idx[0]];
      fields->save_smooth[1][idx_save] = grid->yGrid[fields->smooth_idx[1]];
      fields->save_smooth[2][idx_save] = grid->zGrid[fields->smooth_idx[2]];
      fields->save_smooth[3][idx_save] = idx_it;
    #endif
  }
}

// #endif
