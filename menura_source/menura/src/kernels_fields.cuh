#ifndef __KERNELSFIELDS_CUH_INCLUDED__   // if x.h hasn't been included yet...
#define __KERNELSFIELDS_CUH_INCLUDED__

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "parameters.h"


//_________________________________________________________________________________________
//
// Fundamental physical field kernels: solving Maxwell.
//
__global__ void faraday_k(simu_B_field* B_in, simu_B_field* B_out,
                          simu_fields* fields, float deltaT, simu_param* sP,
                          simu_grid* grid)
{
  /**
  YO Ye ha!
  */
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if(idx < nb_nodes_cst){

    #if NB_DIM==2
      int i = int(idx/len_y_cst);
      int j = idx-i*len_y_cst;
      i += 2;
      j += 2;
      //
      float dyEz = 1/(12.*sP->dX)*( fields->E[2][i  ][j-2]-8*fields->E[2][i  ][j-1]+8*fields->E[2][i  ][j+1]-fields->E[2][i  ][j+2] );
      float dxEz = 1/(12.*sP->dX)*( fields->E[2][i-2][j  ]-8*fields->E[2][i-1][j  ]+8*fields->E[2][i+1][j  ]-fields->E[2][i+2][j  ] );
      float dxEy = 1/(12.*sP->dX)*( fields->E[1][i-2][j  ]-8*fields->E[1][i-1][j  ]+8*fields->E[1][i+1][j  ]-fields->E[1][i+2][j  ] );
      float dyEx = 1/(12.*sP->dX)*( fields->E[0][i  ][j-2]-8*fields->E[0][i  ][j-1]+8*fields->E[0][i  ][j+1]-fields->E[0][i  ][j+2] );
      //
      B_out->B[0][i][j] = B_in->B[0][i][j] - deltaT * dyEz ;
      B_out->B[1][i][j] = B_in->B[1][i][j] - deltaT * -dxEz ;
      B_out->B[2][i][j] = B_in->B[2][i][j] - deltaT * (dxEy - dyEx);
      //
      fields->curl_E[0][i][j] = deltaT * dyEz ;
      fields->curl_E[1][i][j] = deltaT * -dxEz ;
      fields->curl_E[2][i][j] = deltaT*(dxEy - dyEx);
    //
    #elif NB_DIM==3
      int i = int( idx/(len_y_cst*len_z_cst) );
      int j = int( (idx-i*(len_y_cst*len_z_cst))/len_z_cst );
      int k = idx - i*(len_y_cst*len_z_cst) - j*len_z_cst;
      i += 2;
      j += 2;
      k += 2;
      //
      float dyEz = 1./(12.*sP->dX) * ( fields->E[2][i  ][j-2][k  ]-8*fields->E[2][i  ][j-1][k  ]+8*fields->E[2][i  ][j+1][k  ]-fields->E[2][i  ][j+2][k  ] );
      float dzEy = 1./(12.*sP->dX) * ( fields->E[1][i  ][j  ][k-2]-8*fields->E[1][i  ][j  ][k-1]+8*fields->E[1][i  ][j  ][k+1]-fields->E[1][i  ][j  ][k+2] );
      float dzEx = 1./(12.*sP->dX) * ( fields->E[0][i  ][j  ][k-2]-8*fields->E[0][i  ][j  ][k-1]+8*fields->E[0][i  ][j  ][k+1]-fields->E[0][i  ][j  ][k+2] );
      float dxEz = 1./(12.*sP->dX) * ( fields->E[2][i-2][j  ][k  ]-8*fields->E[2][i-1][j  ][k  ]+8*fields->E[2][i+1][j  ][k  ]-fields->E[2][i+2][j  ][k  ] );
      float dxEy = 1./(12.*sP->dX) * ( fields->E[1][i-2][j  ][k  ]-8*fields->E[1][i-1][j  ][k  ]+8*fields->E[1][i+1][j  ][k  ]-fields->E[1][i+2][j  ][k  ] );
      float dyEx = 1./(12.*sP->dX) * ( fields->E[0][i  ][j-2][k  ]-8*fields->E[0][i  ][j-1][k  ]+8*fields->E[0][i  ][j+1][k  ]-fields->E[0][i  ][j+2][k  ] );

      // #if (dipole_cst && !ORF_cst)
      //   B_out->B[0][i][j][k] = B_in->B[0][i][j][k] - deltaT*( (dyEz - dzEy) );// - B_out->dtdBdip[0][i][j][k] );
      //   B_out->B[1][i][j][k] = B_in->B[1][i][j][k] - deltaT*( (dzEx - dxEz) );// - B_out->dtdBdip[1][i][j][k] );
      //   B_out->B[2][i][j][k] = B_in->B[2][i][j][k] - deltaT*( (dxEy - dyEx) );// - B_out->dtdBdip[2][i][j][k] );
      // #else
      B_out->B[0][i][j][k] = B_in->B[0][i][j][k] - deltaT*( dyEz - dzEy );
      B_out->B[1][i][j][k] = B_in->B[1][i][j][k] - deltaT*( dzEx - dxEz );
      B_out->B[2][i][j][k] = B_in->B[2][i][j][k] - deltaT*( dxEy - dyEx );
      // #endif
      //
      fields->curl_E[0][i][j][k] = deltaT*( dyEz - dzEy );
      fields->curl_E[1][i][j][k] = deltaT*( dzEx - dxEz );
      fields->curl_E[2][i][j][k] = deltaT*( dxEy - dyEx );
      //
    #endif

  }
}


#if yee_cst
  __global__ void faraday_yee_k(simu_B_field* B_in, simu_B_field* B_out,
                            simu_fields* fields, float deltaT, simu_param* sP,
                            simu_grid* grid)
  {
    /**
    YO Ye ha!
    */
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    #if NB_DIM==2
      if(idx < (len_x_cst+2)*(len_y_cst+2)){
        int i = int(idx/(len_y_cst+2));
        int j = idx-i*(len_y_cst+2);
        i += 1;
        j += 1;
        //
        float dyEz = sP->dX_i * (fields->E_s[2][i  ][j+1] - fields->E_s[2][i][j]);
        float dxEz = sP->dX_i * (fields->E_s[2][i+1][j  ] - fields->E_s[2][i][j]);
        float dxEy = sP->dX_i * (fields->E_s[1][i+1][j  ] - fields->E_s[1][i][j]);
        float dyEx = sP->dX_i * (fields->E_s[0][i  ][j+1] - fields->E_s[0][i][j]);
        //
        B_out->B_s[0][i][j] = B_in->B_s[0][i][j] - deltaT * dyEz ;
        B_out->B_s[1][i][j] = B_in->B_s[1][i][j] - deltaT * -dxEz ;
        B_out->B_s[2][i][j] = B_in->B_s[2][i][j] - deltaT*(dxEy - dyEx);
        //
        fields->curl_E[0][i][j] = deltaT * dyEz ;
        fields->curl_E[1][i][j] = deltaT * -dxEz ;
        fields->curl_E[2][i][j] = deltaT*(dxEy - dyEx);

      }
    #elif NB_DIM==3
      if(idx < (len_x_cst+2)*(len_y_cst+2)*(len_z_cst+2)){
        int i = int( idx/((len_y_cst+2)*(len_z_cst+2)) );
        int j = int( (idx - i*(len_y_cst+2)*(len_z_cst+2))/(len_z_cst+2) );
        int k = idx - i*(len_y_cst+2)*(len_z_cst+2) - j*(len_z_cst+2);
        i += 1;
        j += 1;
        k += 1;
        //
        float dyEz = sP->dX_i * (fields->E_s[2][i  ][j+1][k  ] - fields->E_s[2][i][j][k]);
        float dzEy = sP->dX_i * (fields->E_s[1][i  ][j  ][k+1] - fields->E_s[1][i][j][k]);
        float dzEx = sP->dX_i * (fields->E_s[0][i  ][j  ][k+1] - fields->E_s[0][i][j][k]);
        float dxEz = sP->dX_i * (fields->E_s[2][i+1][j  ][k  ] - fields->E_s[2][i][j][k]);
        float dxEy = sP->dX_i * (fields->E_s[1][i+1][j  ][k  ] - fields->E_s[1][i][j][k]);
        float dyEx = sP->dX_i * (fields->E_s[0][i  ][j+1][k  ] - fields->E_s[0][i][j][k]);
        //
        B_out->B_s[0][i][j][k] = B_in->B_s[0][i][j][k] - deltaT*( dyEz - dzEy );
        B_out->B_s[1][i][j][k] = B_in->B_s[1][i][j][k] - deltaT*( dzEx - dxEz );
        B_out->B_s[2][i][j][k] = B_in->B_s[2][i][j][k] - deltaT*( dxEy - dyEx );
      }
    #endif
  }


  __global__ void stag_E_k(simu_fields* fields)
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    #if NB_DIM==2
      if (idx<(len_x_cst+3)*(len_y_cst+3)){
        int i = int(idx/(len_y_cst+3));
        int j = idx - i*(len_y_cst+3);
        i += 1;
        j += 1;

        fields->E_s[0][i][j] = fields->E[0][i  ][j  ] * .5 +
                               fields->E[0][i  ][j-1] * .5;
        fields->E_s[1][i][j] = fields->E[1][i  ][j  ] * .5 +
                               fields->E[1][i-1][j  ] * .5;
        fields->E_s[2][i][j] = fields->E[2][i  ][j  ] * .25 +
                               fields->E[2][i  ][j-1] * .25 +
                               fields->E[2][i-1][j  ] * .25 +
                               fields->E[2][i-1][j-1] * .25;
        // fields->E_s[0][i][j] = 0;
        // fields->E_s[1][i][j] = 0;
        // fields->E_s[2][i][j] = 0;
      }
    #elif NB_DIM==3
      if(idx < (len_x_cst+3)*(len_y_cst+3)*(len_z_cst+3)){
        int i = int( idx/((len_y_cst+3)*(len_z_cst+3)) );
        int j = int( (idx - i*(len_y_cst+3)*(len_z_cst+3))/(len_z_cst+3) );
        int k = idx - i*(len_y_cst+3)*(len_z_cst+3) - j*(len_z_cst+3);
        i += 1;
        j += 1;
        k += 1;
        //
        fields->E_s[0][i][j][k] = fields->E[0][i  ][j  ][k  ] * .25 +
                                  fields->E[0][i  ][j  ][k-1] * .25 +
                                  fields->E[0][i  ][j-1][k  ] * .25 +
                                  fields->E[0][i  ][j-1][k-1] * .25;
        fields->E_s[1][i][j][k] = fields->E[1][i  ][j  ][k  ] * .25 +
                                  fields->E[1][i  ][j  ][k-1] * .25 +
                                  fields->E[1][i-1][j  ][k  ] * .25 +
                                  fields->E[1][i-1][j  ][k-1] * .25;
        fields->E_s[2][i][j][k] = fields->E[2][i  ][j  ][k  ] * .25 +
                                  fields->E[2][i  ][j-1][k  ] * .25 +
                                  fields->E[2][i-1][j  ][k  ] * .25 +
                                  fields->E[2][i-1][j-1][k  ] * .25;
      }
    #endif
  }
  __global__ void unstag_J_tot_k(simu_fields* fields)
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    #if NB_DIM==2
      if (idx<(len_x_cst+3)*(len_y_cst+3)){
        int i = int(idx/(len_y_cst+3));
        int j = idx - i*(len_y_cst+3);
        // i += 1;
        // j += 1;
        //
        fields->J_tot[0][i][j] = fields->J_tot_s[0][i  ][j  ] * .5 +
                                 fields->J_tot_s[0][i  ][j+1] * .5;
        fields->J_tot[1][i][j] = fields->J_tot_s[1][i  ][j  ] * .5 +
                                 fields->J_tot_s[1][i+1][j  ] * .5;
        fields->J_tot[2][i][j] = fields->J_tot_s[2][i  ][j  ] * .25 +
                                 fields->J_tot_s[2][i  ][j+1] * .25 +
                                 fields->J_tot_s[2][i+1][j  ] * .25 +
                                 fields->J_tot_s[2][i+1][j+1] * .25;
      }
    #elif NB_DIM==3
      if(idx < (len_x_cst+3)*(len_y_cst+3)*(len_z_cst+3)){
        int i = int( idx/((len_y_cst+3)*(len_z_cst+3)) );
        int j = int( (idx - i*(len_y_cst+3)*(len_z_cst+3))/(len_z_cst+3) );
        int k = idx - i*(len_y_cst+3)*(len_z_cst+3) - j*(len_z_cst+3);
        //
        fields->J_tot[0][i][j] = fields->J_tot_s[0][i  ][j  ][k  ] * .25 +
                                 fields->J_tot_s[0][i  ][j  ][k+1] * .25 +
                                 fields->J_tot_s[0][i  ][j+1][k  ] * .25 +
                                 fields->J_tot_s[0][i  ][j+1][k+1] * .25;
        fields->J_tot[1][i][j] = fields->J_tot_s[1][i  ][j  ][k  ] * .25 +
                                 fields->J_tot_s[1][i  ][j  ][k+1] * .25 +
                                 fields->J_tot_s[1][i+1][j  ][k  ] * .25 +
                                 fields->J_tot_s[1][i+1][j  ][k+1] * .25;
        fields->J_tot[2][i][j] = fields->J_tot_s[2][i  ][j  ][k  ] * .25 +
                                 fields->J_tot_s[2][i  ][j+1][k  ] * .25 +
                                 fields->J_tot_s[2][i+1][j  ][k  ] * .25 +
                                 fields->J_tot_s[2][i+1][j+1][k  ] * .25;
      }
    #endif
  }
  __global__ void unstag_B_k(simu_B_field* B)
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    #if NB_DIM==2
      if (idx<(len_x_cst+3)*(len_y_cst+3)){
        int i = int(idx/(len_y_cst+3));
        int j = idx - i*(len_y_cst+3);
        // i += 1;
        // j += 1;
        //
        B->B[0][i][j] = B->B_s[0][i  ][j  ] * .5 +
                        B->B_s[0][i+1][j  ] * .5;
        B->B[1][i][j] = B->B_s[1][i  ][j  ] * .5 +
                        B->B_s[1][i  ][j+1] * .5;
        B->B[2][i][j] = B->B_s[2][i  ][j  ];
      }
    #elif NB_DIM==3
      if(idx < (len_x_cst+3)*(len_y_cst+3)*(len_z_cst+3)){
        int i = int( idx/((len_y_cst+3)*(len_z_cst+3)) );
        int j = int( (idx - i*(len_y_cst+3)*(len_z_cst+3))/(len_z_cst+3) );
        int k = idx - i*(len_y_cst+3)*(len_z_cst+3) - j*(len_z_cst+3);
        //
        B->B[0][i][j][k] = B->B_s[0][i  ][j  ][k  ] * .5 +
                           B->B_s[0][i+1][j  ][k  ] * .5;
        B->B[1][i][j][k] = B->B_s[1][i  ][j  ][k  ] * .5 +
                           B->B_s[1][i  ][j+1][k  ] * .5;
        B->B[2][i][j][k] = B->B_s[2][i  ][j  ][k  ] * .5 +
                           B->B_s[2][i  ][j  ][k+1] * .5;
      }
    #endif
  }

  __global__ void curl_B_yee_k(simu_B_field* B, simu_fields* fields, simu_param* sP)
  {
    /**
    YO Ye ha!
    */
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    #if NB_DIM==2
      if(idx < (len_x_cst+2)*(len_y_cst+2)){
        int i = int(idx/(len_y_cst+2));
        int j = idx-i*(len_y_cst+2);
        i += 1;
        j += 1;
        //
        float dyBz = sP->dX_i * (B->B_s[2][i  ][j  ] - B->B_s[2][i  ][j-1]);
        float dxBz = sP->dX_i * (B->B_s[2][i  ][j  ] - B->B_s[2][i-1][j  ]);
        float dxBy = sP->dX_i * (B->B_s[1][i  ][j  ] - B->B_s[1][i-1][j  ]);
        float dyBx = sP->dX_i * (B->B_s[0][i  ][j  ] - B->B_s[0][i  ][j-1]);
        //
        fields->J_tot_s[0][i][j] = dyBz ;
        fields->J_tot_s[1][i][j] = -dxBz ;
        fields->J_tot_s[2][i][j] = dxBy - dyBx;
      }
    #elif NB_DIM==3
      if(idx < (len_x_cst+2)*(len_y_cst+2)*(len_z_cst+2)){
        int i = int( idx/((len_y_cst+2)*(len_z_cst+2)) );
        int j = int( (idx - i*(len_y_cst+2)*(len_z_cst+2))/(len_z_cst+2) );
        int k = idx - i*(len_y_cst+2)*(len_z_cst+2) - j*(len_z_cst+2);
        i += 1;
        j += 1;
        k += 1;
        //
        printf("ihihbvievbökiajrvn ");
        float dyBz = sP->dX_i * (B->B_s[2][i  ][j+1][k  ] - B->B_s[2][i][j][k]);
        float dzBy = sP->dX_i * (B->B_s[1][i  ][j  ][k+1] - B->B_s[1][i][j][k]);
        float dzBx = sP->dX_i * (B->B_s[0][i  ][j  ][k+1] - B->B_s[0][i][j][k]);
        float dxBz = sP->dX_i * (B->B_s[2][i+1][j  ][k  ] - B->B_s[2][i][j][k]);
        float dxBy = sP->dX_i * (B->B_s[1][i+1][j  ][k  ] - B->B_s[1][i][j][k]);
        float dyBx = sP->dX_i * (B->B_s[0][i  ][j+1][k  ] - B->B_s[0][i][j][k]);
        //
        fields->J_tot_s[0][i][j][k] = dyBz - dzBy;
        fields->J_tot_s[1][i][j][k] = dzBx - dxBz;
        fields->J_tot_s[2][i][j][k] = dxBy - dyBx;
      }
    #endif
  }

  __global__ void div_B_yee_k(simu_B_field* B, simu_fields* fields, simu_param* sP)
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    #if NB_DIM==2
      if(idx < (len_x_cst+3)*(len_y_cst+3)){
        int i = int(idx/(len_y_cst+3));
        int j = idx-i*(len_y_cst+3);
        //
        float dxBx = sP->dX_i * ( B->B_s[0][i+1][j  ]-B->B_s[0][i  ][j  ] );
        float dyBy = sP->dX_i * ( B->B_s[1][i  ][j+1]-B->B_s[1][i  ][j  ] );
        //
        fields->smooth[i][j] = dxBx + dyBy;
        // std::cout << dxBx << " " << dyBy << " " << fields->smooth[i][j] << std::endl;
      //
    }
    #elif NB_DIM==3
      if(idx < (len_x_cst+3)*(len_y_cst+3)*(len_z_cst+3)){
        int i = int(idx/((len_y_cst+3)*(len_z_cst+3)));
        int j = int((idx-i*((len_y_cst+3)*(len_z_cst+3)))/(len_z_cst+3));
        int k = idx-i*((len_y_cst+3)*(len_z_cst+3))-j*(len_z_cst+3);
        i += 2;
        j += 2;
        k += 2;
        //
        float dxBx = sP->dX_i * ( B->B_s[0][i+1][j  ][k  ]-B->B_s[0][i][j][k] );
        float dyBy = sP->dX_i * ( B->B_s[1][i  ][j+1][k  ]-B->B_s[1][i][j][k] );
        float dyBy = sP->dX_i * ( B->B_s[1][i  ][j  ][k+1]-B->B_s[1][i][j][k] );
        //
        fields->smooth[i][j][k] = dxBx + dyBy + dzBz;
      }
    #endif
  }

#endif



__global__ void curl_B_k(simu_B_field* B, simu_fields* fields, simu_param* sP)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if(idx < nb_nodes_cst){

    #if NB_DIM==2
      int i = int(idx/len_y_cst);
      int j = idx-i*len_y_cst;
      i += 2;
      j += 2;
      // fields->J_tot[0][i][j] = 1./(12.*sP->dX*mu0)*( B->B[2][i  ][j-2]-8*B->B[2][i  ][j-1]+8*B->B[2][i  ][j+1]-B->B[2][i  ][j+2] );
      fields->J_tot[0][i][j] = -8*B->B[2][i  ][j-1]+8*B->B[2][i  ][j+1];
      fields->J_tot[0][i][j] += B->B[2][i  ][j-2]-B->B[2][i  ][j+2];
      fields->J_tot[0][i][j] *= 1./(12.*sP->dX);
      // fields->J_tot[1][i][j] = 1./(12.*sP->dX*mu0)*(-B->B[2][i-2][j  ]+8*B->B[2][i-1][j  ]-8*B->B[2][i+1][j  ]+B->B[2][i+2][j  ] );
      fields->J_tot[1][i][j] = 8*B->B[2][i-1][j  ]-8*B->B[2][i+1][j  ];
      fields->J_tot[1][i][j] += -B->B[2][i-2][j  ]+B->B[2][i+2][j  ];
      fields->J_tot[1][i][j] *= 1./(12.*sP->dX);
      // fields->J_tot[2][i][j] = 1./(12.*sP->dX*mu0)*( B->B[1][i-2][j  ]-8*B->B[1][i-1][j  ]+8*B->B[1][i+1][j  ]-B->B[1][i+2][j  ] - B->B[0][i  ][j-2]+8*B->B[0][i  ][j-1]-8*B->B[0][i  ][j+1]+B->B[0][i  ][j+2] );
      fields->J_tot[2][i][j] = -8*B->B[1][i-1][j  ]+8*B->B[1][i+1][j  ];
      fields->J_tot[2][i][j] += B->B[1][i-2][j  ]-B->B[1][i+2][j  ];
      fields->J_tot[2][i][j] += 8*B->B[0][i  ][j-1]-8*B->B[0][i  ][j+1];
      fields->J_tot[2][i][j] += -B->B[0][i  ][j-2]+B->B[0][i  ][j+2];
      fields->J_tot[2][i][j] *= 1./(12.*sP->dX);
    //
    #elif NB_DIM==3
      int i = int(idx/(len_y_cst*len_z_cst));
      int j = int((idx-i*(len_y_cst*len_z_cst))/len_z_cst);
      int k = idx-i*(len_y_cst*len_z_cst)-j*len_z_cst;
      i += 2;
      j += 2;
      k += 2;
      //
      float dyBz = 1./(12.*sP->dX) * ( B->B[2][i  ][j-2][k  ]-8*B->B[2][i  ][j-1][k  ]+8*B->B[2][i  ][j+1][k  ]-B->B[2][i  ][j+2][k  ] );
      float dzBy = 1./(12.*sP->dX) * ( B->B[1][i  ][j  ][k-2]-8*B->B[1][i  ][j  ][k-1]+8*B->B[1][i  ][j  ][k+1]-B->B[1][i  ][j  ][k+2] );
      float dzBx = 1./(12.*sP->dX) * ( B->B[0][i  ][j  ][k-2]-8*B->B[0][i  ][j  ][k-1]+8*B->B[0][i  ][j  ][k+1]-B->B[0][i  ][j  ][k+2] );
      float dxBz = 1./(12.*sP->dX) * ( B->B[2][i-2][j  ][k  ]-8*B->B[2][i-1][j  ][k  ]+8*B->B[2][i+1][j  ][k  ]-B->B[2][i+2][j  ][k  ] );
      float dxBy = 1./(12.*sP->dX) * ( B->B[1][i-2][j  ][k  ]-8*B->B[1][i-1][j  ][k  ]+8*B->B[1][i+1][j  ][k  ]-B->B[1][i+2][j  ][k  ] );
      float dyBx = 1./(12.*sP->dX) * ( B->B[0][i  ][j-2][k  ]-8*B->B[0][i  ][j-1][k  ]+8*B->B[0][i  ][j+1][k  ]-B->B[0][i  ][j+2][k  ] );
      //
      fields->J_tot[0][i][j][k] = dyBz - dzBy;
      fields->J_tot[1][i][j][k] = dzBx - dxBz;
      fields->J_tot[2][i][j][k] = dxBy - dyBx;
    #endif

  }
}


__global__ void div_B_k(simu_B_field* B, simu_fields* fields, simu_param* sP)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if(idx < nb_nodes_cst){

    #if NB_DIM==2
      int i = int(idx/len_y_cst);
      int j = idx-i*len_y_cst;
      i += 2;
      j += 2;
      //
      float dxBx = 1./(12.*sP->dX) * ( B->B[0][i-2][j  ]-8*B->B[0][i-1][j  ]+8*B->B[0][i+1][j  ]-B->B[0][i+2][j  ] );
      float dyBy = 1./(12.*sP->dX) * ( B->B[1][i  ][j-2]-8*B->B[1][i  ][j-1]+8*B->B[1][i  ][j+1]-B->B[1][i  ][j+2] );
      //
      fields->smooth[i][j] = dxBx + dyBy;
    //
    #elif NB_DIM==3
      int i = int(idx/(len_y_cst*len_z_cst));
      int j = int((idx-i*(len_y_cst*len_z_cst))/len_z_cst);
      int k = idx-i*(len_y_cst*len_z_cst)-j*len_z_cst;
      i += 2;
      j += 2;
      k += 2;
      //
      float dxBx = 1./(12.*sP->dX) * ( B->B[0][i-2][j  ][k  ]-8*B->B[0][i-1][j  ][k  ]+8*B->B[0][i+1][j  ][k  ]-B->B[0][i+2][j  ][k  ] );
      float dyBy = 1./(12.*sP->dX) * ( B->B[1][i  ][j-2][k  ]-8*B->B[1][i  ][j-1][k  ]+8*B->B[1][i  ][j+1][k  ]-B->B[1][i  ][j+2][k  ] );
      float dzBz = 1./(12.*sP->dX) * ( B->B[2][i  ][j  ][k-2]-8*B->B[2][i  ][j  ][k-1]+8*B->B[2][i  ][j  ][k+1]-B->B[2][i  ][j  ][k+2] );
      //
      fields->smooth[i][j][k] = dxBx + dyBy + dzBz;
    #endif

  }
}




#if dipole_cst
__global__ void compute_potential_k(simu_B_field* B, simu_grid* grid, simu_param* sP, float delta_x)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  #if NB_DIM==2
    if (idx < (len_x_cst+4)*(len_y_cst+4)*5){
      int i = int( idx/((len_y_cst+4)*5) );
      int j = int( (idx-i*(len_y_cst+4)*5)/5 );
      int k = idx - i*(len_y_cst+4)*5 - j*5;
      //
      float rx = grid->xGrid[i] - (grid->centre_x + delta_x);
      float ry = grid->yGrid[j] - grid->centre_y;
      float rz = (k-2)*sP->dX;
      // float r3_SI = std::pow( (rx*rx + ry*ry + rz*rz)*sP->d_i*sP->d_i, 3./2.);
      float r3_SI = (rx*rx + ry*ry + rz*rz)*sP->d_i*sP->d_i;
      // The magnetic scalar potential from the magnetic pole limit:
      B->phi[i][j][k] = (sP->dip_mom_SI[0]*rx*sP->d_i
                       + sP->dip_mom_SI[1]*ry*sP->d_i
                       + sP->dip_mom_SI[2]*rz*sP->d_i)/r3_SI*1e-8;
    }
  #elif NB_DIM==3
    if (idx < nb_nodes_tot_cst){
      int i = int( idx/((len_y_cst+4)*(len_z_cst+4)) );
      int j = int( (idx-i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);
      //
      float rx = grid->xGrid[i] - (grid->centre_x + delta_x);
      float ry = grid->yGrid[j] - grid->centre_y;
      float rz = grid->zGrid[k] - grid->centre_z;
      float r3_SI = std::pow( (rx*rx + ry*ry + rz*rz)*sP->d_i*sP->d_i, 3./2.);
      // The magnetic scalar potential from the magnetic pole limit:
      B->phi[i][j][k] = (sP->dip_mom_SI[0]*rx*sP->d_i
                       + sP->dip_mom_SI[1]*ry*sP->d_i
                       + sP->dip_mom_SI[2]*rz*sP->d_i)/r3_SI;
    }
  #endif

}
__global__ void compute_dipole_k//(simu_B_field* B, simu_fields* fields, simu_grid* grid, simu_param* sP)
#if NB_DIM==2
(float B_dip[3][len_x_cst+4][len_y_cst+4], simu_B_field* B, simu_fields* fields, simu_grid* grid, simu_param* sP)
#elif NB_DIM==3
(float B_dip[3][len_x_cst+4][len_y_cst+4][len_z_cst+4], simu_B_field* B, simu_fields* fields, simu_grid* grid, simu_param* sP)
#endif
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < nb_nodes_cst){
    #if NB_DIM==2
      int i = int( idx/len_y_cst );
      int j = idx - i*len_y_cst;
      i += 2;
      j += 2;
      //
      float dxdphi = 1./(12.*sP->dX*sP->d_i) * ( B->phi[i-2][j  ][2  ]-8*B->phi[i-1][j  ][2  ]+8*B->phi[i+1][j  ][2  ]-B->phi[i+2][j  ][2  ] );
      float dydphi = 1./(12.*sP->dX*sP->d_i) * ( B->phi[i  ][j-2][2  ]-8*B->phi[i  ][j-1][2  ]+8*B->phi[i  ][j+1][2  ]-B->phi[i  ][j+2][2  ] );
      float dzdphi = 1./(12.*sP->dX*sP->d_i) * ( B->phi[i  ][j  ][0  ]-8*B->phi[i  ][j  ][1  ]+8*B->phi[i  ][j  ][3  ]-B->phi[i  ][j  ][4  ] );
      B->B_dip[0][i][j] = dxdphi/sP->B0_SI;
      B->B_dip[1][i][j] = dydphi/sP->B0_SI;
      B->B_dip[2][i][j] = dzdphi/sP->B0_SI;
    #elif NB_DIM==3
      int i = int( idx/(len_y_cst*len_z_cst) );
      int j = int( (idx-i*len_y_cst*len_z_cst)/len_z_cst );
      int k = idx - i*len_y_cst*len_z_cst - j*len_z_cst;
      i += 2;
      j += 2;
      k += 2;
      //
      float dxdphi = 1./(12.*sP->dX*sP->d_i) * ( B->phi[i-2][j  ][k  ]-8*B->phi[i-1][j  ][k  ]+8*B->phi[i+1][j  ][k  ]-B->phi[i+2][j  ][k  ] );
      float dydphi = 1./(12.*sP->dX*sP->d_i) * ( B->phi[i  ][j-2][k  ]-8*B->phi[i  ][j-1][k  ]+8*B->phi[i  ][j+1][k  ]-B->phi[i  ][j+2][k  ] );
      float dzdphi = 1./(12.*sP->dX*sP->d_i) * ( B->phi[i  ][j  ][k-2]-8*B->phi[i  ][j  ][k-1]+8*B->phi[i  ][j  ][k+1]-B->phi[i  ][j  ][k+2] );
      B->B_dip[0][i][j][k] = dxdphi/sP->B0_SI;
      B->B_dip[1][i][j][k] = dydphi/sP->B0_SI;
      B->B_dip[2][i][j][k] = dzdphi/sP->B0_SI;
    #endif
  }

}
// __global__ void compute_deriv_dipole_k(simu_B_field* B_0, simu_B_field* B_1, simu_fields* fields,
//                                        simu_grid* grid, simu_param* sP, float delta_x)
// {
//   int idx = threadIdx.x + blockIdx.x*blockDim.x;
//
//   if (idx < nb_nodes_cst){
//     #if NB_DIM==2
//       printf("Not implemented calculate_deriv_dipole_k 2D\n");
//     #elif NB_DIM==3
//       int i = int( idx/(len_y_cst*len_z_cst) );
//       int j = int( (idx-i*len_y_cst*len_z_cst)/len_z_cst );
//       int k = idx - i*len_y_cst*len_z_cst - j*len_z_cst;
//       i += 2;
//       j += 2;
//       k += 2;
//       // float rx = grid->xGrid[i]-grid->centre_x-delta_x;
//       // float ry = grid->yGrid[j]-grid->centre_y;
//       // float rz = grid->zGrid[k]-grid->centre_z;
//       // float rsq = rx*rx + ry*ry + rz*rz;
//       // Central finite difference used for the dipole's time derivative.
//       if (fields->region_ID[i][j][k]!=2){
//         B_0->dtdBdip[0][i][j][k] = (B_1->B_dip[0][i][j][k] - B_0->B_dip[0][i][j][k]) / sP->dt;
//         B_0->dtdBdip[1][i][j][k] = (B_1->B_dip[1][i][j][k] - B_0->B_dip[1][i][j][k]) / sP->dt;
//         B_0->dtdBdip[2][i][j][k] = (B_1->B_dip[2][i][j][k] - B_0->B_dip[2][i][j][k]) / sP->dt;
//       }
//       else{
//         B_0->dtdBdip[0][i][j][k] = 0.;
//         B_0->dtdBdip[1][i][j][k] = 0.;
//         B_0->dtdBdip[2][i][j][k] = 0.;
//       }
//     #endif
//   }
//
// }
__global__ void copy_dipole_k(simu_B_field* B_in, simu_B_field* B_out)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < nb_nodes_cst){
    #if NB_DIM==2
      int i = int( idx/len_y_cst );
      int j = idx - i*len_y_cst ;
      i += 2;
      j += 2;
      //
      B_out->B_dip[0][i][j] = B_in->B_dip[0][i][j];
      B_out->B_dip[1][i][j] = B_in->B_dip[1][i][j];
      B_out->B_dip[2][i][j] = B_in->B_dip[2][i][j];
    #elif NB_DIM==3
      int i = int( idx/(len_y_cst*len_z_cst) );
      int j = int( (idx-i*len_y_cst*len_z_cst)/len_z_cst );
      int k = idx - i*len_y_cst*len_z_cst - j*len_z_cst;
      i += 2;
      j += 2;
      k += 2;
      //
      B_out->B_dip[0][i][j][k] = B_in->B_dip[0][i][j][k];
      B_out->B_dip[1][i][j][k] = B_in->B_dip[1][i][j][k];
      B_out->B_dip[2][i][j][k] = B_in->B_dip[2][i][j][k];
      // B_1->dtdBdip[0][i][j][k] = B_0->dtdBdip[0][i][j][k];
      // B_1->dtdBdip[1][i][j][k] = B_0->dtdBdip[1][i][j][k];
      // B_1->dtdBdip[2][i][j][k] = B_0->dtdBdip[2][i][j][k];
    #endif
  }

}
#endif

#if NB_DIM==2

  __global__ void pressure_k(simu_fields* fields, float dens[len_x_cst+4][len_y_cst+4], simu_param* sP){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if(idx < nb_nodes_tot_cst){
      int i = int( idx/(len_y_cst+4) );
      int j = idx - i*(len_y_cst+4);
      //
      fields->pres[i][j] = sP->Beta_e*pow(dens[i][j], sP->poly_ind);  // Normalised pressure.
    }
  }

  __global__ void ohm_k(simu_fields* fields, simu_B_field* B, simu_grid* grid,
                        float density[len_x_cst+4][len_y_cst+4], float Ji[3][len_x_cst+4][len_y_cst+4],
                        int indIt, simu_param* sP, int ohm_ID){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if(idx < nb_nodes_cst){
      int i = int(idx/len_y_cst);
      int j = idx-i*len_y_cst;
      i += 2;
      j += 2;
      //
      float E_mot [3];
      float E_hal [3];
      float E_amb [3];
      float E_res [3];
      float E_hyp_res [3];
      //
      if (fields->region_ID[i][j]==0){
        /* The node is in nominal plasma */
        #if dipole_cst
          E_mot[0] = -1./(density[i][j]) * (Ji[1][i][j]*(B->B[2][i][j]+B->B_dip[2][i][j]) - Ji[2][i][j]*(B->B[1][i][j]+B->B_dip[1][i][j]));
          E_mot[1] = -1./(density[i][j]) * (Ji[2][i][j]*(B->B[0][i][j]+B->B_dip[0][i][j]) - Ji[0][i][j]*(B->B[2][i][j]+B->B_dip[2][i][j]));
          E_mot[2] = -1./(density[i][j]) * (Ji[0][i][j]*(B->B[1][i][j]+B->B_dip[1][i][j]) - Ji[1][i][j]*(B->B[0][i][j]+B->B_dip[0][i][j]));
          //
          #if !ORF_cst
            if (ohm_ID==0){
              E_mot[0] -= 0.;
              E_mot[1] -= ( sP->v_obs*(B->B_dip[2][i][j]+sP->B0_z) );
              E_mot[2] -= (-sP->v_obs*(B->B_dip[1][i][j]+sP->B0_y) );
            }
          #endif
          //
          E_hal[0] = 1./(density[i][j]) * (fields->J_tot[1][i][j]*(B->B[2][i][j]+B->B_dip[2][i][j]) - fields->J_tot[2][i][j]*(B->B[1][i][j]+B->B_dip[1][i][j]));
          E_hal[1] = 1./(density[i][j]) * (fields->J_tot[2][i][j]*(B->B[0][i][j]+B->B_dip[0][i][j]) - fields->J_tot[0][i][j]*(B->B[2][i][j]+B->B_dip[2][i][j]));
          E_hal[2] = 1./(density[i][j]) * (fields->J_tot[0][i][j]*(B->B[1][i][j]+B->B_dip[1][i][j]) - fields->J_tot[1][i][j]*(B->B[0][i][j]+B->B_dip[0][i][j]));
        #else
          E_mot[0] = -1./(density[i][j]) * (Ji[1][i][j]*B->B[2][i][j] - Ji[2][i][j]*B->B[1][i][j]);
          E_mot[1] = -1./(density[i][j]) * (Ji[2][i][j]*B->B[0][i][j] - Ji[0][i][j]*B->B[2][i][j]);
          E_mot[2] = -1./(density[i][j]) * (Ji[0][i][j]*B->B[1][i][j] - Ji[1][i][j]*B->B[0][i][j]);
          //
          E_hal[0] = 1./(density[i][j]) * (fields->J_tot[1][i][j]*B->B[2][i][j] - fields->J_tot[2][i][j]*B->B[1][i][j]);
          E_hal[1] = 1./(density[i][j]) * (fields->J_tot[2][i][j]*B->B[0][i][j] - fields->J_tot[0][i][j]*B->B[2][i][j]);
          E_hal[2] = 1./(density[i][j]) * (fields->J_tot[0][i][j]*B->B[1][i][j] - fields->J_tot[1][i][j]*B->B[0][i][j]);
        #endif
        //
        if (ohm_ID==0){
          // curl of grad of scalar is zero!
          E_amb[0] = 0.;
          E_amb[1] = 0.;
          E_amb[2] = 0.;
        }
        else{
          E_amb[0] = - 1./(2.*density[i][j]) * 1./(12.*sP->dX) * ( -fields->pres[i+2][j]+8*fields->pres[i+1][j]-8*fields->pres[i-1][j]+fields->pres[i-2][j] );
          E_amb[1] = - 1./(2.*density[i][j]) * 1./(12.*sP->dX) * ( -fields->pres[i][j+2]+8*fields->pres[i][j+1]-8*fields->pres[i][j-1]+fields->pres[i][j-2] );
          E_amb[2] = 0.;
        }
        //
        // if (ohm_ID==0){
          //
          E_res[0] = sP->eta_res_norm*fields->J_tot[0][i][j];
          E_res[1] = sP->eta_res_norm*fields->J_tot[1][i][j];
          E_res[2] = sP->eta_res_norm*fields->J_tot[2][i][j];
          //
          float d2xBx2 = 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[0][i+2][j  ]+16*fields->J_tot[0][i+1][j  ]-30*fields->J_tot[0][i][j]+16*fields->J_tot[0][i-1][j  ]-fields->J_tot[0][i-2][j  ]);
          float d2yBx2 = 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[0][i  ][j+2]+16*fields->J_tot[0][i  ][j+1]-30*fields->J_tot[0][i][j]+16*fields->J_tot[0][i  ][j-1]-fields->J_tot[0][i  ][j-2]);
          float d2xBy2 = 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[1][i+2][j  ]+16*fields->J_tot[1][i+1][j  ]-30*fields->J_tot[1][i][j]+16*fields->J_tot[1][i-1][j  ]-fields->J_tot[1][i-2][j  ]);
          float d2yBy2 = 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[1][i  ][j+2]+16*fields->J_tot[1][i  ][j+1]-30*fields->J_tot[1][i][j]+16*fields->J_tot[1][i  ][j-1]-fields->J_tot[1][i  ][j-2]);
          float d2xBz2 = 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[2][i+2][j  ]+16*fields->J_tot[2][i+1][j  ]-30*fields->J_tot[2][i][j]+16*fields->J_tot[2][i-1][j  ]-fields->J_tot[2][i-2][j  ]);
          float d2yBz2 = 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[2][i  ][j+2]+16*fields->J_tot[2][i  ][j+1]-30*fields->J_tot[2][i][j]+16*fields->J_tot[2][i  ][j-1]-fields->J_tot[2][i  ][j-2]);

          E_hyp_res[0] = -sP->eta_hyp_res * (d2xBx2 + d2yBx2);
          E_hyp_res[1] = -sP->eta_hyp_res * (d2xBy2 + d2yBy2);
          E_hyp_res[2] = -sP->eta_hyp_res * (d2xBz2 + d2yBz2);
          // E_hyp_res[0] = -1. * sP->eta_hyp_res * 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[0][i+2][j]+16*fields->J_tot[0][i+1][j]-30*fields->J_tot[0][i][j]-16*fields->J_tot[0][i-1][j]+fields->J_tot[0][i-2][j]);
          // E_hyp_res[1] = -1. * sP->eta_hyp_res * 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[1][i][j+2]+16*fields->J_tot[1][i][j+1]-30*fields->J_tot[1][i][j]-16*fields->J_tot[1][i][j-1]+fields->J_tot[1][i][j-2]);
          // E_hyp_res[2] = 0.;

        // }
        // else{
        //   // Conservation of momentum!
        //   E_res[0] = 0.;
        //   E_res[1] = 0.;
        //   E_res[2] = 0.;
        //   //
        //   E_hyp_res[0] = 0.;
        //   E_hyp_res[1] = 0.;
        //   E_hyp_res[2] = 0.;
        // }

      }
      else if (fields->region_ID[i][j]==1){
        /* The node is in a vacuum region, outside the body */
        #if (ORF_cst && !dipole_cst)
          // Necessary to not get ring currents on the surface of the body. But what's the physics?
          E_mot[0] = 0.;
          E_mot[1] = -sP->v_obs*sP->B0_z;
          E_mot[2] =  sP->v_obs*sP->B0_y;
        // #elif (!ORF_cst && dipole_cst)
        //   if ( ohm_ID==2){// ohm_ID==1 ||
        //     E_mot[0] = 0.;
        //     E_mot[1] = -1*(-sP->v_obs*(sP->B0_z));
        //     E_mot[2] = -1*( sP->v_obs*(sP->B0_y));
        //   }
        //   else{
        //     E_mot[0] = 0.;
        //     E_mot[1] = 0.;
        //     E_mot[2] = 0.;
        //   }
        #else
          E_mot[0] = 0.;
          E_mot[1] = 0.;
          E_mot[2] = 0.;
        #endif
        //
        E_hal[0] = 0.;
        E_hal[1] = 0.;
        E_hal[2] = 0.;
        //
        E_amb[0] = 0.;
        E_amb[1] = 0.;
        E_amb[2] = 0.;
        //
        // if (ohm_ID==0){
          E_res[0] = sP->eta_res_vac*fields->J_tot[0][i][j];
          E_res[1] = sP->eta_res_vac*fields->J_tot[1][i][j];
          E_res[2] = sP->eta_res_vac*fields->J_tot[2][i][j];
        // }
        // else{
        //   // Conservation of momentum!
        //   E_res[0] = 0.;
        //   E_res[1] = 0.;
        //   E_res[2] = 0.;
        // }
        //
        E_hyp_res[0] = 0.;
        E_hyp_res[1] = 0.;
        E_hyp_res[2] = 0.;
      }
      else if (fields->region_ID[i][j]==2){
        /* The node is within the body.*/
        #if (ORF_cst && !dipole_cst)
          // Necessary to not get ring currents on the surface of the body. But what's the physics?
          E_mot[0] = 0.;
          E_mot[1] = -sP->v_obs*sP->B0_z;
          E_mot[2] =  sP->v_obs*sP->B0_y;
        #elif (!ORF_cst && dipole_cst)
          if ( ohm_ID==2 || ohm_ID==1){
            E_mot[0] = 0.;
            E_mot[1] = -1*(-sP->v_obs*(B->B_dip[2][i][j])); //+sP->B0_z
            E_mot[2] = -1*( sP->v_obs*(B->B_dip[1][i][j])); //+sP->B0_y
          }
          else{
            E_mot[0] = 0.;
            E_mot[1] = 0.;
            E_mot[2] = 0.;
          }
        #else
          E_mot[0] = 0.;
          E_mot[1] = 0.;
          E_mot[2] = 0.;
        #endif
        //
        E_hal[0] = 0.;
        E_hal[1] = 0.;
        E_hal[2] = 0.;
        //
        E_amb[0] = 0.;
        E_amb[1] = 0.;
        E_amb[2] = 0.;
        //
        // if (ohm_ID==0){
          E_res[0] = sP->eta_res_obs*fields->J_tot[0][i][j];
          E_res[1] = sP->eta_res_obs*fields->J_tot[1][i][j];
          E_res[2] = sP->eta_res_obs*fields->J_tot[2][i][j];
        // }
        // else{
        //   // Conservation of momentum!
        //   E_res[0] = 0.;
        //   E_res[1] = 0.;
        //   E_res[2] = 0.;
        // }
        //
        E_hyp_res[0] = 0.;
        E_hyp_res[1] = 0.;
        E_hyp_res[2] = 0.;
      }



      // #if (ionosphere_cst)
      //   float E_max_hall = 40;
      //   if (E_hal[0]>E_max_hall){
      //     E_hal[0] = E_max_hall;
      //   }
      //   if (E_hal[0]<-E_max_hall){
      //     E_hal[0] = -E_max_hall;
      //   }
      //   if (E_hal[1]>E_max_hall){
      //     E_hal[1] = E_max_hall;
      //   }
      //   if (E_hal[1]<-E_max_hall){
      //     E_hal[1] = -E_max_hall;
      //   }
      //   if (E_hal[2]>E_max_hall){
      //     E_hal[2] = E_max_hall;
      //   }
      //   if (E_hal[2]<-E_max_hall){
      //     E_hal[2] = -E_max_hall;
      //   }
      // #endif


      //
      fields->E[0][i][j] = sP->e_mot*E_mot[0] + sP->e_hal*E_hal[0] + sP->e_amb*E_amb[0] + E_res[0] + E_hyp_res[0];
      fields->E[1][i][j] = sP->e_mot*E_mot[1] + sP->e_hal*E_hal[1] + sP->e_amb*E_amb[1] + E_res[1] + E_hyp_res[1];
      fields->E[2][i][j] = sP->e_mot*E_mot[2] + sP->e_hal*E_hal[2] + sP->e_amb*E_amb[2] + E_res[2] + E_hyp_res[2];




      #if obstacle_cst
        if (i<sP->width_smooth_downstream){
          if (fields->E[0][i][j]>50){
            fields->E[0][i][j] = 50;
          }
          if (fields->E[0][i][j]<-50){
            fields->E[0][i][j] = -50;
          }
          if (fields->E[1][i][j]>50){
            fields->E[1][i][j] = 50;
          }
          if (fields->E[1][i][j]<-50){
            fields->E[1][i][j] = -50;
          }
          if (fields->E[2][i][j]>50){
            fields->E[2][i][j] = 50;
          }
          if (fields->E[2][i][j]<-50){
            fields->E[2][i][j] = -50;
          }
        }

        // float E_maxx = 110;
        // if (fields->E[0][i][j]>E_maxx){
        //   fields->E[0][i][j] = E_maxx;
        // }
        // if (fields->E[0][i][j]<-E_maxx){
        //   fields->E[0][i][j] = -E_maxx;
        // }
        // if (fields->E[1][i][j]>E_maxx){
        //   fields->E[1][i][j] = E_maxx;
        // }
        // if (fields->E[1][i][j]<-E_maxx){
        //   fields->E[1][i][j] = -E_maxx;
        // }
        // if (fields->E[2][i][j]>E_maxx){
        //   fields->E[2][i][j] = E_maxx;
        // }
        // if (fields->E[2][i][j]<-E_maxx){
        //   fields->E[2][i][j] = -E_maxx;
        // }
      #endif

      #if (dipole_cst)
        float E_max2 = 100;
        // if (grid->rsq[i][j] < pow(100.*sP->r_obs, 2)){
          if (fields->E[0][i][j]>E_max2){
            fields->E[0][i][j] = E_max2;
          }
          if (fields->E[0][i][j]<-E_max2){
            fields->E[0][i][j] = -E_max2;
          }
          if (fields->E[1][i][j]>E_max2){
            fields->E[1][i][j] = E_max2;
          }
          if (fields->E[1][i][j]<-E_max2){
            fields->E[1][i][j] = -E_max2;
          }
          if (fields->E[2][i][j]>E_max2){
            fields->E[2][i][j] = E_max2;
          }
          if (fields->E[2][i][j]<-E_max2){
            fields->E[2][i][j] = -E_max2;
          }
        // }
      #endif



    }
  }

#elif NB_DIM==3
  __global__ void pressure_k(simu_fields* fields, float dens[len_x_cst+4][len_y_cst+4][len_z_cst+4], simu_param* sP)
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if(idx < nb_nodes_tot_cst){
      int i = int( idx/((len_y_cst+4)*(len_z_cst+4)) );
      int j = int( (idx - i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);
      //
      fields->pres[i][j][k] = sP->Beta_e*pow(dens[i][j][k], sP->poly_ind);  // Normalised pressure.
    }
  }

  __global__ void ohm_k(simu_fields* fields, simu_B_field* B, simu_grid* grid,
                        float density[len_x_cst+4][len_y_cst+4][len_z_cst+4],
                        float Ji[3][len_x_cst+4][len_y_cst+4][len_z_cst+4],
                        int indIt, simu_param* sP, int ohm_ID)
  {

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if(idx < nb_nodes_cst){
      int i = int( idx/(len_y_cst*len_z_cst) );
      int j = int( (idx-i*len_y_cst*len_z_cst)/len_z_cst );
      int k = idx - i*len_y_cst*len_z_cst - j*len_z_cst;
      i += 2;
      j += 2;
      k += 2;
      //
      float E_mot [3];
      float E_hal [3];
      float E_amb [3];
      float E_res [3];
      float E_hyp_res [3];
      //

      if (fields->region_ID[i][j][k]==0){
        /* The node is in nominal plasma */
        #if dipole_cst
          E_mot[0] = -1/(density[i][j][k]) * (Ji[1][i][j][k]*(B->B[2][i][j][k]+B->B_dip[2][i][j][k]) - Ji[2][i][j][k]*(B->B[1][i][j][k]+B->B_dip[1][i][j][k]));
          E_mot[1] = -1/(density[i][j][k]) * (Ji[2][i][j][k]*(B->B[0][i][j][k]+B->B_dip[0][i][j][k]) - Ji[0][i][j][k]*(B->B[2][i][j][k]+B->B_dip[2][i][j][k]));
          E_mot[2] = -1/(density[i][j][k]) * (Ji[0][i][j][k]*(B->B[1][i][j][k]+B->B_dip[1][i][j][k]) - Ji[1][i][j][k]*(B->B[0][i][j][k]+B->B_dip[0][i][j][k]));
          //
          #if (!ORF_cst && dipole_cst)
            if (ohm_ID==0){
              /* This is done only while pushing B (ohm_ID==0),
                  equivalent to accounting for dB_dip/dt in Faraday.*/
              E_mot[0] -= 0.;
              E_mot[1] -= ( sP->v_obs*(B->B_dip[2][i][j][k]+sP->B0_z) );
              E_mot[2] -= (-sP->v_obs*(B->B_dip[1][i][j][k]+sP->B0_y) );
            }
          #endif
          //
          E_hal[0] = 1/(density[i][j][k]) * (fields->J_tot[1][i][j][k]*(B->B[2][i][j][k]+B->B_dip[2][i][j][k]) - fields->J_tot[2][i][j][k]*(B->B[1][i][j][k]+B->B_dip[1][i][j][k]));
          E_hal[1] = 1/(density[i][j][k]) * (fields->J_tot[2][i][j][k]*(B->B[0][i][j][k]+B->B_dip[0][i][j][k]) - fields->J_tot[0][i][j][k]*(B->B[2][i][j][k]+B->B_dip[2][i][j][k]));
          E_hal[2] = 1/(density[i][j][k]) * (fields->J_tot[0][i][j][k]*(B->B[1][i][j][k]+B->B_dip[1][i][j][k]) - fields->J_tot[1][i][j][k]*(B->B[0][i][j][k]+B->B_dip[0][i][j][k]));
        #else
          E_mot[0] = -1/(density[i][j][k]) * (Ji[1][i][j][k]*B->B[2][i][j][k] - Ji[2][i][j][k]*B->B[1][i][j][k]);
          E_mot[1] = -1/(density[i][j][k]) * (Ji[2][i][j][k]*B->B[0][i][j][k] - Ji[0][i][j][k]*B->B[2][i][j][k]);
          E_mot[2] = -1/(density[i][j][k]) * (Ji[0][i][j][k]*B->B[1][i][j][k] - Ji[1][i][j][k]*B->B[0][i][j][k]);
          //
          E_hal[0] = 1/(density[i][j][k]) * (fields->J_tot[1][i][j][k]*B->B[2][i][j][k] - fields->J_tot[2][i][j][k]*B->B[1][i][j][k]);
          E_hal[1] = 1/(density[i][j][k]) * (fields->J_tot[2][i][j][k]*B->B[0][i][j][k] - fields->J_tot[0][i][j][k]*B->B[2][i][j][k]);
          E_hal[2] = 1/(density[i][j][k]) * (fields->J_tot[0][i][j][k]*B->B[1][i][j][k] - fields->J_tot[1][i][j][k]*B->B[0][i][j][k]);
        #endif
        //
        if (ohm_ID==0){
          // curl of grad of scalar is zero!
          E_amb[0] = 0.;
          E_amb[1] = 0.;
          E_amb[2] = 0.;
        }
        else{
          E_amb[0] = - 1./(2.*density[i][j][k]) * 1./(12.*sP->dX) * ( -fields->pres[i+2][j][k]+8*fields->pres[i+1][j][k]-8*fields->pres[i-1][j][k]+fields->pres[i-2][j][k] );
          E_amb[1] = - 1./(2.*density[i][j][k]) * 1./(12.*sP->dX) * ( -fields->pres[i][j+2][k]+8*fields->pres[i][j+1][k]-8*fields->pres[i][j-1][k]+fields->pres[i][j-2][k] );
          E_amb[2] = - 1./(2.*density[i][j][k]) * 1./(12.*sP->dX) * ( -fields->pres[i][j][k+2]+8*fields->pres[i][j][k+1]-8*fields->pres[i][j][k-1]+fields->pres[i][j][k-2] );
        }
        //
        // if (ohm_ID==0){
          E_res[0] = sP->eta_res_norm*fields->J_tot[0][i][j][k];
          E_res[1] = sP->eta_res_norm*fields->J_tot[1][i][j][k];
          E_res[2] = sP->eta_res_norm*fields->J_tot[2][i][j][k];
          //
          float d2xBx2 = 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[0][i+2][j  ][k  ]+16*fields->J_tot[0][i+1][j  ][k  ]-30*fields->J_tot[0][i][j][k]+16*fields->J_tot[0][i-1][j  ][k  ]-fields->J_tot[0][i-2][j  ][k  ]);
          float d2yBx2 = 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[0][i  ][j+2][k  ]+16*fields->J_tot[0][i  ][j+1][k  ]-30*fields->J_tot[0][i][j][k]+16*fields->J_tot[0][i  ][j-1][k  ]-fields->J_tot[0][i  ][j-2][k  ]);
          float d2zBx2 = 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[0][i  ][j  ][k+2]+16*fields->J_tot[0][i  ][j  ][k+1]-30*fields->J_tot[0][i][j][k]+16*fields->J_tot[0][i  ][j  ][k-1]-fields->J_tot[0][i  ][j  ][k-2]);
          float d2xBy2 = 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[1][i+2][j  ][k  ]+16*fields->J_tot[1][i+1][j  ][k  ]-30*fields->J_tot[1][i][j][k]+16*fields->J_tot[1][i-1][j  ][k  ]-fields->J_tot[1][i-2][j  ][k  ]);
          float d2yBy2 = 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[1][i  ][j+2][k  ]+16*fields->J_tot[1][i  ][j+1][k  ]-30*fields->J_tot[1][i][j][k]+16*fields->J_tot[1][i  ][j-1][k  ]-fields->J_tot[1][i  ][j-2][k  ]);
          float d2zBy2 = 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[1][i  ][j  ][k+2]+16*fields->J_tot[1][i  ][j  ][k+1]-30*fields->J_tot[1][i][j][k]+16*fields->J_tot[1][i  ][j  ][k-1]-fields->J_tot[1][i  ][j  ][k-2]);
          float d2xBz2 = 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[2][i+2][j  ][k  ]+16*fields->J_tot[2][i+1][j  ][k  ]-30*fields->J_tot[2][i][j][k]+16*fields->J_tot[2][i-1][j  ][k  ]-fields->J_tot[2][i-2][j  ][k  ]);
          float d2yBz2 = 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[2][i  ][j+2][k  ]+16*fields->J_tot[2][i  ][j+1][k  ]-30*fields->J_tot[2][i][j][k]+16*fields->J_tot[2][i  ][j-1][k  ]-fields->J_tot[2][i  ][j-2][k  ]);
          float d2zBz2 = 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[2][i  ][j  ][k+2]+16*fields->J_tot[2][i  ][j  ][k+1]-30*fields->J_tot[2][i][j][k]+16*fields->J_tot[2][i  ][j  ][k-1]-fields->J_tot[2][i  ][j  ][k-2]);

          E_hyp_res[0] = -sP->eta_hyp_res * (d2xBx2 + d2yBx2 + d2zBx2);
          E_hyp_res[1] = -sP->eta_hyp_res * (d2xBy2 + d2yBy2 + d2zBy2);
          E_hyp_res[2] = -sP->eta_hyp_res * (d2xBz2 + d2yBz2 + d2zBz2);
        // }
        // else{
        //   // Conservation of momentum!
        //   E_res[0] = 0.;
        //   E_res[1] = 0.;
        //   E_res[2] = 0.;
        //   //
        //   E_hyp_res[0] = 0.;
        //   E_hyp_res[1] = 0.;
        //   E_hyp_res[2] = 0.;
        // }
      }
      else if (fields->region_ID[i][j][k]==1){
        /* The node is in a vacuum region, outside the body */
        #if (ORF_cst && !dipole_cst)
          // Necessary to not get ring currents on the surface of the body. But what's the physics?
          E_mot[0] = 0.;
          E_mot[1] = -sP->v_obs*sP->B0_z;
          E_mot[2] =  sP->v_obs*sP->B0_y;
        // #elif (!ORF_cst && dipole_cst)
        //   if ( ohm_ID==1 || ohm_ID==2){// ohm_ID==1 ||
        //     E_mot[0] = 0.;
        //     E_mot[1] = -1*(-sP->v_obs*(sP->B0_z));// - 1*(-sP->v_obs*(B->B_dip[2][i][j][k]));//B->B_dip[2][i][j][k]+
        //     E_mot[2] = -1*( sP->v_obs*(sP->B0_y));// - 1*( sP->v_obs*(B->B_dip[1][i][j][k]));//B->B_dip[1][i][j][k]+
        //   }
        //   else{
        //     E_mot[0] = 0.;
        //     E_mot[1] = 0.;
        //     E_mot[2] = 0.;
        //   }
        #else
          E_mot[0] = 0.;
          E_mot[1] = 0.;
          E_mot[2] = 0.;
        #endif
        //
        E_hal[0] = 0.;
        E_hal[1] = 0.;
        E_hal[2] = 0.;
        //
        E_amb[0] = 0.;
        E_amb[1] = 0.;
        E_amb[2] = 0.;
        //
        if (ohm_ID==0){
          E_res[0] = sP->eta_res_vac*fields->J_tot[0][i][j][k];
          E_res[1] = sP->eta_res_vac*fields->J_tot[1][i][j][k];
          E_res[2] = sP->eta_res_vac*fields->J_tot[2][i][j][k];
        }
        else{
          E_res[0] = 0.;
          E_res[1] = 0.;
          E_res[2] = 0.;
        }
        //
        E_hyp_res[0] = 0.;
        E_hyp_res[1] = 0.;
        E_hyp_res[2] = 0.;
      }
      else if (fields->region_ID[i][j][k]==2){
        /* The node is within the body.*/
        #if (ORF_cst && !dipole_cst)
          // Necessary to not get ring currents on the surface of the body. But what's the physics?
          E_mot[0] = 0.;
          E_mot[1] = -sP->v_obs*sP->B0_z;
          E_mot[2] =  sP->v_obs*sP->B0_y;
        #elif (!ORF_cst && dipole_cst)
          // if ( ohm_ID==1){// ohm_ID==1 ||
          //   E_mot[0] = 0.;
          //   E_mot[1] = -1*(-sP->v_obs*(B->B_dip[2][i][j][k]+sP->B0_z));//
          //   E_mot[2] = -1*( sP->v_obs*(B->B_dip[1][i][j][k]+sP->B0_y));//
          // }
          if ( ohm_ID==2 || ohm_ID==1){
            E_mot[0] = 0.;
            E_mot[1] = -1*(-sP->v_obs*(B->B_dip[2][i][j][k]));// - 1*(-sP->v_obs*(B->B_dip[2][i][j][k]));//B->B_dip[2][i][j][k]+
            E_mot[2] = -1*( sP->v_obs*(B->B_dip[1][i][j][k]));// - 1*( sP->v_obs*(B->B_dip[1][i][j][k]));//B->B_dip[1][i][j][k]+
          }
          else{
            E_mot[0] = 0.;
            E_mot[1] = 0.;
            E_mot[2] = 0.;
          }
        #else
          E_mot[0] = 0.;
          E_mot[1] = 0.;
          E_mot[2] = 0.;
        #endif
        //
        E_hal[0] = 0.;
        E_hal[1] = 0.;
        E_hal[2] = 0.;
        //
        E_amb[0] = 0.;
        E_amb[1] = 0.;
        E_amb[2] = 0.;
        //
        if (ohm_ID==0){
          E_res[0] = sP->eta_res_obs*fields->J_tot[0][i][j][k];
          E_res[1] = sP->eta_res_obs*fields->J_tot[1][i][j][k];
          E_res[2] = sP->eta_res_obs*fields->J_tot[2][i][j][k];
        }
        else{
          E_res[0] = 0.;
          E_res[1] = 0.;
          E_res[2] = 0.;
        }
        //
        E_hyp_res[0] = 0.;
        E_hyp_res[1] = 0.;
        E_hyp_res[2] = 0.;
      }


      fields->E[0][i][j][k] = sP->e_mot*E_mot[0] + sP->e_hal*E_hal[0] + sP->e_amb*E_amb[0] + E_res[0] + E_hyp_res[0];
      fields->E[1][i][j][k] = sP->e_mot*E_mot[1] + sP->e_hal*E_hal[1] + sP->e_amb*E_amb[1] + E_res[1] + E_hyp_res[1];
      fields->E[2][i][j][k] = sP->e_mot*E_mot[2] + sP->e_hal*E_hal[2] + sP->e_amb*E_amb[2] + E_res[2] + E_hyp_res[2];



      #if inject_pla_cst
        float E_max = 50;
        if (i<10){
          if (fields->E[0][i][j][k]>E_max){
             fields->E[0][i][j][k] /= 10;
          }
          if (fields->E[0][i][j][k]<-E_max){
             fields->E[0][i][j][k] /= 10;
          }
          if (fields->E[1][i][j][k]>E_max){
             fields->E[1][i][j][k] /= 10;
          }
          if (fields->E[1][i][j][k]<-E_max){
             fields->E[1][i][j][k] /= 10;
          }
          if (fields->E[2][i][j][k]>E_max){
             fields->E[2][i][j][k] /= 10;
          }
          if (fields->E[2][i][j][k]<-E_max){
             fields->E[2][i][j][k] /= 10;
          }
        }
      #endif

      #if (dipole_cst || ionosphere_cst)
        float E_max2 = 100;
        // if (grid->rsq[i][j] < pow(100.*sP->r_obs, 2)){
          if (fields->E[0][i][j][k]>E_max2){
            fields->E[0][i][j][k] = E_max2;
          }
          if (fields->E[0][i][j][k]<-E_max2){
            fields->E[0][i][j][k] = -E_max2;
          }
          if (fields->E[1][i][j][k]>E_max2){
            fields->E[1][i][j][k] = E_max2;
          }
          if (fields->E[1][i][j][k]<-E_max2){
            fields->E[1][i][j][k] = -E_max2;
          }
          if (fields->E[2][i][j][k]>E_max2){
            fields->E[2][i][j][k] = E_max2;
          }
          if (fields->E[2][i][j][k]<-E_max2){
            fields->E[2][i][j][k] = -E_max2;
          }
        // }
      #endif
    }
  }
#endif


__global__ void ohm_components_k(simu_fields* fields, simu_B_field* B, int indIt, simu_param* sP)
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if(idx < nb_nodes_cst){
    #if NB_DIM==2
      int i = int(idx/len_y_cst);
      int j = idx-i*len_y_cst;
      i += 2;
      j += 2;

      fields->E_mot[0][i][j] = -1/(fields->density[i][j]) * (fields->Ji[1][i][j]*B->B[2][i][j] - fields->Ji[2][i][j]*B->B[1][i][j]);
      fields->E_mot[1][i][j] = -1/(fields->density[i][j]) * (fields->Ji[2][i][j]*B->B[0][i][j] - fields->Ji[0][i][j]*B->B[2][i][j]);
      fields->E_mot[2][i][j] = -1/(fields->density[i][j]) * (fields->Ji[0][i][j]*B->B[1][i][j] - fields->Ji[1][i][j]*B->B[0][i][j]);

      fields->E_hal[0][i][j] = 1/(fields->density[i][j]) * (fields->J_tot[1][i][j]*B->B[2][i][j] - fields->J_tot[2][i][j]*B->B[1][i][j]);
      fields->E_hal[1][i][j] = 1/(fields->density[i][j]) * (fields->J_tot[2][i][j]*B->B[0][i][j] - fields->J_tot[0][i][j]*B->B[2][i][j]);
      fields->E_hal[2][i][j] = 1/(fields->density[i][j]) * (fields->J_tot[0][i][j]*B->B[1][i][j] - fields->J_tot[1][i][j]*B->B[0][i][j]);

      fields->E_amb[0][i][j] = - 1./(2.*fields->density[i][j]) * 1./(12.*sP->dX) * ( -fields->pres[i+2][j]+8*fields->pres[i+1][j]-8*fields->pres[i-1][j]+fields->pres[i-2][j] );
      fields->E_amb[1][i][j] = - 1./(2.*fields->density[i][j]) * 1./(12.*sP->dX) * ( -fields->pres[i][j+2]+8*fields->pres[i][j+1]-8*fields->pres[i][j-1]+fields->pres[i][j-2] );
      fields->E_amb[2][i][j] = 0;

      fields->E_res[0][i][j] = sP->eta_res_norm*fields->J_tot[0][i][j];
      fields->E_res[1][i][j] = sP->eta_res_norm*fields->J_tot[1][i][j];
      fields->E_res[2][i][j] = sP->eta_res_norm*fields->J_tot[2][i][j];
    #elif NB_DIM==3
      int i = int(idx/(len_y_cst*len_z_cst));
      int j = int((idx-i*(len_y_cst*len_z_cst))/len_z_cst);
      int k = idx-i*(len_y_cst*len_z_cst)-j*len_z_cst;
      i += 2;
      j += 2;
      k += 2;

      float E_mot [3];
      float E_hal [3];
      float E_amb [3];
      // float E_res [3];
      // float E_hyp_res [3];

      if (fields->region_ID[i][j][k]==0){
        /* The node is in nominal plasma */
        #if dipole_cst
          E_mot[0] = -1/(fields->density[i][j][k]) * (fields->Ji[1][i][j][k]*(B->B[2][i][j][k]+B->B_dip[2][i][j][k]) - fields->Ji[2][i][j][k]*(B->B[1][i][j][k]+B->B_dip[1][i][j][k]));
          E_mot[1] = -1/(fields->density[i][j][k]) * (fields->Ji[2][i][j][k]*(B->B[0][i][j][k]+B->B_dip[0][i][j][k]) - fields->Ji[0][i][j][k]*(B->B[2][i][j][k]+B->B_dip[2][i][j][k]));
          E_mot[2] = -1/(fields->density[i][j][k]) * (fields->Ji[0][i][j][k]*(B->B[1][i][j][k]+B->B_dip[1][i][j][k]) - fields->Ji[1][i][j][k]*(B->B[0][i][j][k]+B->B_dip[0][i][j][k]));
          //
          // #if !ORF_cst && add_ohm_term_cst
          //   // Lorentz transform:
          //   E_mot[0] += 0.;
          //   E_mot[1] += -( sP->v_obs*B->B_dip[2][i][j][k] );
          //   E_mot[2] += -(-sP->v_obs*B->B_dip[1][i][j][k] );
          // #endif
          //
          E_hal[0] = 1/(fields->density[i][j][k]) * (fields->J_tot[1][i][j][k]*(B->B[2][i][j][k]+B->B_dip[2][i][j][k]) - fields->J_tot[2][i][j][k]*(B->B[1][i][j][k]+B->B_dip[1][i][j][k]));
          E_hal[1] = 1/(fields->density[i][j][k]) * (fields->J_tot[2][i][j][k]*(B->B[0][i][j][k]+B->B_dip[0][i][j][k]) - fields->J_tot[0][i][j][k]*(B->B[2][i][j][k]+B->B_dip[2][i][j][k]));
          E_hal[2] = 1/(fields->density[i][j][k]) * (fields->J_tot[0][i][j][k]*(B->B[1][i][j][k]+B->B_dip[1][i][j][k]) - fields->J_tot[1][i][j][k]*(B->B[0][i][j][k]+B->B_dip[0][i][j][k]));
        #else
          E_mot[0] = -1/(fields->density[i][j][k]) * (fields->Ji[1][i][j][k]*B->B[2][i][j][k] - fields->Ji[2][i][j][k]*B->B[1][i][j][k]);
          E_mot[1] = -1/(fields->density[i][j][k]) * (fields->Ji[2][i][j][k]*B->B[0][i][j][k] - fields->Ji[0][i][j][k]*B->B[2][i][j][k]);
          E_mot[2] = -1/(fields->density[i][j][k]) * (fields->Ji[0][i][j][k]*B->B[1][i][j][k] - fields->Ji[1][i][j][k]*B->B[0][i][j][k]);
          //
          E_hal[0] = 1/(fields->density[i][j][k]) * (fields->J_tot[1][i][j][k]*B->B[2][i][j][k] - fields->J_tot[2][i][j][k]*B->B[1][i][j][k]);
          E_hal[1] = 1/(fields->density[i][j][k]) * (fields->J_tot[2][i][j][k]*B->B[0][i][j][k] - fields->J_tot[0][i][j][k]*B->B[2][i][j][k]);
          E_hal[2] = 1/(fields->density[i][j][k]) * (fields->J_tot[0][i][j][k]*B->B[1][i][j][k] - fields->J_tot[1][i][j][k]*B->B[0][i][j][k]);
        #endif
        //
        E_amb[0] = - 1./(2.*fields->density[i][j][k]) * 1./(12.*sP->dX) * ( -fields->pres[i+2][j][k]+8*fields->pres[i+1][j][k]-8*fields->pres[i-1][j][k]+fields->pres[i-2][j][k] );
        E_amb[1] = - 1./(2.*fields->density[i][j][k]) * 1./(12.*sP->dX) * ( -fields->pres[i][j+2][k]+8*fields->pres[i][j+1][k]-8*fields->pres[i][j-1][k]+fields->pres[i][j-2][k] );
        E_amb[2] = - 1./(2.*fields->density[i][j][k]) * 1./(12.*sP->dX) * ( -fields->pres[i][j][k+2]+8*fields->pres[i][j][k+1]-8*fields->pres[i][j][k-1]+fields->pres[i][j][k-2] );
        //
        // E_res[0] = sP->eta_res_norm*fields->J_tot[0][i][j][k];
        // E_res[1] = sP->eta_res_norm*fields->J_tot[1][i][j][k];
        // E_res[2] = sP->eta_res_norm*fields->J_tot[2][i][j][k];
        // //
        // E_hyp_res[0] = -1. * sP->eta_hyp_res * 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[0][i+2][j][k]+16*fields->J_tot[0][i+1][j][k]-30*fields->J_tot[0][i][j][k]-16*fields->J_tot[0][i-1][j][k]+fields->J_tot[0][i-2][j][k]);
        // E_hyp_res[1] = -1. * sP->eta_hyp_res * 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[1][i][j+2][k]+16*fields->J_tot[1][i][j+1][k]-30*fields->J_tot[1][i][j][k]-16*fields->J_tot[1][i][j-1][k]+fields->J_tot[1][i][j-2][k]);
        // E_hyp_res[2] = -1. * sP->eta_hyp_res * 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[2][i][j][k+2]+16*fields->J_tot[2][i][j][k+1]-30*fields->J_tot[2][i][j][k]-16*fields->J_tot[2][i][j][k-1]+fields->J_tot[2][i][j][k-2]);
      }
      else if (fields->region_ID[i][j][k]==1){
        /* The node is in a vacuum region, outside the body */
        #if (ORF_cst && !dipole_cst)
          // Necessary to not get ring currents on the surface of the body. But what's the physics?
          E_mot[0] = 0.;
          E_mot[1] = -sP->v_obs*sP->B0_z;
          E_mot[2] =  sP->v_obs*sP->B0_y;
        #else
          E_mot[0] = 0.;
          E_mot[1] = 0.;
          E_mot[2] = 0.;
        #endif
        //
        // #if (dipole_cst && !ORF_cst && add_ohm_term_cst)
        //     // Lorentz transform:
        //     E_mot[0] += 0.;
        //     E_mot[1] += -( sP->v_obs*B->B_dip[2][i][j][k] );
        //     E_mot[2] += -(-sP->v_obs*B->B_dip[1][i][j][k] );
        //     // E_mot[0] += (Ji[1][i][j][k]/density[i][j][k]*B->B_dip[2][i][j][k] - Ji[2][i][j][k]/density[i][j][k]*B->B_dip[1][i][j][k]);
        //     // E_mot[1] += (Ji[2][i][j][k]/density[i][j][k]*B->B_dip[0][i][j][k] - (Ji[0][i][j][k]/density[i][j][k]+sP->v_obs)*B->B_dip[2][i][j][k]);
        //     // E_mot[2] += ((Ji[0][i][j][k]/density[i][j][k]+sP->v_obs)*B->B_dip[1][i][j][k] - Ji[1][i][j][k]/density[i][j][k]*B->B_dip[0][i][j][k]);
        // #endif
        //
        E_hal[0] = 0.;
        E_hal[1] = 0.;
        E_hal[2] = 0.;
        //
        E_amb[0] = 0.;
        E_amb[1] = 0.;
        E_amb[2] = 0.;
        //
        // E_res[0] = sP->eta_res_vac*fields->J_tot[0][i][j][k];
        // E_res[1] = sP->eta_res_vac*fields->J_tot[1][i][j][k];
        // E_res[2] = sP->eta_res_vac*fields->J_tot[2][i][j][k];
        // //
        // E_hyp_res[0] = 0.;
        // E_hyp_res[1] = 0.;
        // E_hyp_res[2] = 0.;
      }
      else if (fields->region_ID[i][j][k]==2){
        /* The node is within the body.*/
        #if (ORF_cst && !dipole_cst)
          // Necessary to not get ring currents on the surface of the body. But what's the physics?
          E_mot[0] = 0.;
          E_mot[1] = -sP->v_obs*sP->B0_z;
          E_mot[2] =  sP->v_obs*sP->B0_y;
        #else
          E_mot[0] = 0.;
          E_mot[1] = 0.;
          E_mot[2] = 0.;
        #endif
        //
        E_hal[0] = 0.;
        E_hal[1] = 0.;
        E_hal[2] = 0.;
        //
        E_amb[0] = 0.;
        E_amb[1] = 0.;
        E_amb[2] = 0.;
        //
        // E_res[0] = sP->eta_res_obs*fields->J_tot[0][i][j][k];
        // E_res[1] = sP->eta_res_obs*fields->J_tot[1][i][j][k];
        // E_res[2] = sP->eta_res_obs*fields->J_tot[2][i][j][k];
        // //
        // E_hyp_res[0] = 0.;
        // E_hyp_res[1] = 0.;
        // E_hyp_res[2] = 0.;
      }
      //
      else if (fields->region_ID[i][j][k]==3){
        /* Resistive downstream layer. */
        #if dipole_cst
          E_mot[0] = -1/(fields->density[i][j][k]) * (fields->Ji[1][i][j][k]*(B->B[2][i][j][k]+B->B_dip[2][i][j][k]) - fields->Ji[2][i][j][k]*(B->B[1][i][j][k]+B->B_dip[1][i][j][k]));
          E_mot[1] = -1/(fields->density[i][j][k]) * (fields->Ji[2][i][j][k]*(B->B[0][i][j][k]+B->B_dip[0][i][j][k]) - fields->Ji[0][i][j][k]*(B->B[2][i][j][k]+B->B_dip[2][i][j][k]));
          E_mot[2] = -1/(fields->density[i][j][k]) * (fields->Ji[0][i][j][k]*(B->B[1][i][j][k]+B->B_dip[1][i][j][k]) - fields->Ji[1][i][j][k]*(B->B[0][i][j][k]+B->B_dip[0][i][j][k]));
          //
          E_hal[0] = 1/(fields->density[i][j][k]) * (fields->J_tot[1][i][j][k]*(B->B[2][i][j][k]+B->B_dip[2][i][j][k]) - fields->J_tot[2][i][j][k]*(B->B[1][i][j][k]+B->B_dip[1][i][j][k]));
          E_hal[1] = 1/(fields->density[i][j][k]) * (fields->J_tot[2][i][j][k]*(B->B[0][i][j][k]+B->B_dip[0][i][j][k]) - fields->J_tot[0][i][j][k]*(B->B[2][i][j][k]+B->B_dip[2][i][j][k]));
          E_hal[2] = 1/(fields->density[i][j][k]) * (fields->J_tot[0][i][j][k]*(B->B[1][i][j][k]+B->B_dip[1][i][j][k]) - fields->J_tot[1][i][j][k]*(B->B[0][i][j][k]+B->B_dip[0][i][j][k]));
        #else
          E_mot[0] = -1/(fields->density[i][j][k]) * (fields->Ji[1][i][j][k]*B->B[2][i][j][k] - fields->Ji[2][i][j][k]*B->B[1][i][j][k]);
          E_mot[1] = -1/(fields->density[i][j][k]) * (fields->Ji[2][i][j][k]*B->B[0][i][j][k] - fields->Ji[0][i][j][k]*B->B[2][i][j][k]);
          E_mot[2] = -1/(fields->density[i][j][k]) * (fields->Ji[0][i][j][k]*B->B[1][i][j][k] - fields->Ji[1][i][j][k]*B->B[0][i][j][k]);
          //
          E_hal[0] = 1/(fields->density[i][j][k]) * (fields->J_tot[1][i][j][k]*B->B[2][i][j][k] - fields->J_tot[2][i][j][k]*B->B[1][i][j][k]);
          E_hal[1] = 1/(fields->density[i][j][k]) * (fields->J_tot[2][i][j][k]*B->B[0][i][j][k] - fields->J_tot[0][i][j][k]*B->B[2][i][j][k]);
          E_hal[2] = 1/(fields->density[i][j][k]) * (fields->J_tot[0][i][j][k]*B->B[1][i][j][k] - fields->J_tot[1][i][j][k]*B->B[0][i][j][k]);
        #endif
        //
        E_amb[0] = - 1./(2.*fields->density[i][j][k]) * 1./(12.*sP->dX) * ( -fields->pres[i+2][j][k]+8*fields->pres[i+1][j][k]-8*fields->pres[i-1][j][k]+fields->pres[i-2][j][k] );
        E_amb[1] = - 1./(2.*fields->density[i][j][k]) * 1./(12.*sP->dX) * ( -fields->pres[i][j+2][k]+8*fields->pres[i][j+1][k]-8*fields->pres[i][j-1][k]+fields->pres[i][j-2][k] );
        E_amb[2] = - 1./(2.*fields->density[i][j][k]) * 1./(12.*sP->dX) * ( -fields->pres[i][j][k+2]+8*fields->pres[i][j][k+1]-8*fields->pres[i][j][k-1]+fields->pres[i][j][k-2] );
        //
        // E_res[0] = sP->eta_res_obs*fields->J_tot[0][i][j][k];
        // E_res[1] = sP->eta_res_obs*fields->J_tot[1][i][j][k];
        // E_res[2] = sP->eta_res_obs*fields->J_tot[2][i][j][k];
        // //
        // E_hyp_res[0] = -1. * sP->eta_hyp_res * 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[0][i+2][j][k]+16*fields->J_tot[0][i+1][j][k]-30*fields->J_tot[0][i][j][k]-16*fields->J_tot[0][i-1][j][k]+fields->J_tot[0][i-2][j][k]);
        // E_hyp_res[1] = -1. * sP->eta_hyp_res * 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[1][i][j+2][k]+16*fields->J_tot[1][i][j+1][k]-30*fields->J_tot[1][i][j][k]-16*fields->J_tot[1][i][j-1][k]+fields->J_tot[1][i][j-2][k]);
        // E_hyp_res[2] = -1. * sP->eta_hyp_res * 1./(12.*sP->dX*sP->dX) * (-fields->J_tot[2][i][j][k+2]+16*fields->J_tot[2][i][j][k+1]-30*fields->J_tot[2][i][j][k]-16*fields->J_tot[2][i][j][k-1]+fields->J_tot[2][i][j][k-2]);
      }

      fields->E_mot[0][i][j][k] = E_mot[0];
      fields->E_mot[1][i][j][k] = E_mot[1];
      fields->E_mot[2][i][j][k] = E_mot[2];

      fields->E_hal[0][i][j][k] = E_hal[0];
      fields->E_hal[1][i][j][k] = E_hal[1];
      fields->E_hal[2][i][j][k] = E_hal[2];

      fields->E_amb[0][i][j][k] = E_amb[0];
      fields->E_amb[1][i][j][k] = E_amb[1];
      fields->E_amb[2][i][j][k] = E_amb[2];
    #endif

  }
}

__global__ void predict_correct_k(simu_B_field* Bout, simu_B_field* Bin)
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if(idx < nb_nodes_cst){
    #if NB_DIM==2
      int i = int(idx/len_y_cst);
      int j = idx - i*len_y_cst;
      i += 2;
      j += 2;
      //
      Bout->B[0][i][j] = .5 * (Bin->B[0][i][j] + Bout->B[0][i][j]);
      Bout->B[1][i][j] = .5 * (Bin->B[1][i][j] + Bout->B[1][i][j]);
      Bout->B[2][i][j] = .5 * (Bin->B[2][i][j] + Bout->B[2][i][j]);
    #elif NB_DIM==3
      int i = int( idx/(len_y_cst*len_z_cst));
      int j = int( (idx - i*(len_y_cst*len_z_cst)) /len_z_cst );
      int k = idx - i*(len_y_cst*len_z_cst) - j*len_z_cst;
      i += 2;
      j += 2;
      k += 2;
      //
      Bout->B[0][i][j][k] = .5 * (Bin->B[0][i][j][k] + Bout->B[0][i][j][k]);
      Bout->B[1][i][j][k] = .5 * (Bin->B[1][i][j][k] + Bout->B[1][i][j][k]);
      Bout->B[2][i][j][k] = .5 * (Bin->B[2][i][j][k] + Bout->B[2][i][j][k]);
    #endif
  }
}





__global__ void analytical_EB_k(simu_B_field* B, simu_fields* fields,
                                  simu_param* sP, simu_grid* grid, int idx_it)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx<nb_nodes_tot_cst){

    #if NB_DIM==2
      int i = int( idx/(len_y_cst+4) );
      int j = idx - i*(len_y_cst+4);
      //
      //__________________________________________________________________________
      // Plane, oblic.
      B->B[0][i][j] = 1.;
      B->B[1][i][j] = 0.;
      B->B[2][i][j] = 0.;
      //
      fields->E[0][i][j] = sP->delta_E*( sin(sP->k_wave*grid->xGrid[i] - sP->omega_wave*idx_it*sP->dt) );
      fields->E[0][i][j] += sP->delta_E*( sin(-sP->k_wave*grid->xGrid[i] - sP->omega_wave*idx_it*sP->dt) );
      fields->E[1][i][j] = 0.;
      fields->E[2][i][j] = 0.;
    #endif
  }

}




__global__ void add_counts_k(simu_fields* fields, simu_param* sP)
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if(idx < nb_nodes_tot_cst){
    #if NB_DIM==2
      int i = int(idx/(len_y_cst+4));
      int j = idx - i*(len_y_cst+4);
      //
      fields->density[i][j] = sP->w_sw*fields->counts[i][j] + sP->w_pla*fields->counts_pla[i][j];

      /** Below, we do not want to apply dens_min to the outermost guard nodes,
      to not add this constant values when communicating/adding the buffers. */
      // if (fields->density[i][j]<sP->dens_min
      //     && i>1 && i<len_x_cst+2
      //     && j>1 && j<len_y_cst+2){
      //   fields->density[i][j] = sP->dens_min;
      // }
    #elif NB_DIM==3
      int i = int( idx/((len_y_cst+4)*(len_z_cst+4)) );
      int j = int( (idx - i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);
      //
      fields->density[i][j][k] = sP->w_sw*fields->counts[i][j][k] + sP->w_pla*fields->counts_pla[i][j][k];
      // if (fields->density[i][j][k]<sP->dens_min
      //     && i>1 && i<len_x_cst+2
      //     && j>1 && j<len_y_cst+2
      //     && k>1 && k<len_z_cst+2){
      //   fields->density[i][j][k] = sP->dens_min;
      // }
    #endif
  }

}

__global__ void add_curr_k(simu_fields* fields, simu_param* sP)
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if(idx < nb_nodes_tot_cst){
    #if NB_DIM==2
      int i = int(idx/(len_y_cst+4));
      int j = idx - i*(len_y_cst+4);
      //
      fields->Ji[0][i][j] = sP->w_sw*fields->fluxNum[0][i][j] + sP->w_pla*fields->fluxNum_pla[0][i][j];
      fields->Ji[1][i][j] = sP->w_sw*fields->fluxNum[1][i][j] + sP->w_pla*fields->fluxNum_pla[1][i][j];
      fields->Ji[2][i][j] = sP->w_sw*fields->fluxNum[2][i][j] + sP->w_pla*fields->fluxNum_pla[2][i][j];
      //
    #elif NB_DIM==3
      int i = int( idx/((len_y_cst+4)*(len_z_cst+4)) );
      int j = int( (idx-i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);
      //
      fields->Ji[0][i][j][k] = sP->w_sw*fields->fluxNum[0][i][j][k] + sP->w_pla*fields->fluxNum_pla[0][i][j][k];
      fields->Ji[1][i][j][k] = sP->w_sw*fields->fluxNum[1][i][j][k] + sP->w_pla*fields->fluxNum_pla[1][i][j][k];
      fields->Ji[2][i][j][k] = sP->w_sw*fields->fluxNum[2][i][j][k] + sP->w_pla*fields->fluxNum_pla[2][i][j][k];
      //
    #endif
  }

}

__global__ void add_counts_b_k(simu_fields* fields, simu_param* sP)
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if(idx < nb_nodes_tot_cst){
    #if NB_DIM==2
      int i = int(idx/(len_y_cst+4));
      int j = idx - i*(len_y_cst+4);
      //
      fields->density_b[i][j] = sP->w_sw*fields->counts[i][j] + sP->w_pla*fields->counts_pla[i][j];
      fields->Lambda[i][j]    = sP->w_sw*fields->counts[i][j] + sP->w_pla*fields->counts_pla[i][j]/(sP->Z_pla); //*q*q/m;

      /** Below, we do not want to apply dens_min to the outermost guard nodes,
      to not add this constant values when communicating/adding the buffers. */
      // if (fields->density_b[i][j]<sP->dens_min
      //     && i>1 && i<len_x_cst+2
      //     && j>1 && j<len_y_cst+2){
      //   fields->density_b[i][j] = sP->dens_min;
      // }
    #elif NB_DIM==3
      int i = int( idx/((len_y_cst+4)*(len_z_cst+4)) );
      int j = int( (idx-i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);

      fields->density_b[i][j][k] = sP->w_sw*fields->counts[i][j][k] + sP->w_pla*fields->counts_pla[i][j][k];
      fields->Lambda[i][j][k]    = sP->w_sw*fields->counts[i][j][k] + sP->w_pla*fields->counts_pla[i][j][k]/(sP->Z_pla); //*q*q/m;
      // if (fields->density_b[i][j][k]<sP->dens_min
      //     && i>1 && i<len_x_cst+2
      //     && j>1 && j<len_y_cst+2
      //     && k>1 && k<len_z_cst+2){
      //   fields->density_b[i][j][k] = sP->dens_min;
      // }
    #endif
  }

}

__global__ void add_curr_b_k(simu_fields* fields, simu_param* sP, int ind_it)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if(idx < nb_nodes_tot_cst){
    #if NB_DIM==2
      int i = int(idx/(len_y_cst+4));
      int j = idx - i*(len_y_cst+4);
      //
      fields->Ji_b[0][i][j] = (sP->w_sw*fields->fluxNum[0][i][j] + sP->w_pla*fields->fluxNum_pla[0][i][j]);/// fields->density[i][j];
      fields->Ji_b[1][i][j] = (sP->w_sw*fields->fluxNum[1][i][j] + sP->w_pla*fields->fluxNum_pla[1][i][j]);/// fields->density[i][j];
      fields->Ji_b[2][i][j] = (sP->w_sw*fields->fluxNum[2][i][j] + sP->w_pla*fields->fluxNum_pla[2][i][j]);/// fields->density[i][j];
      //
      fields->Gamma[0][i][j] = (sP->w_sw*fields->fluxNum[0][i][j] + sP->w_pla*fields->fluxNum_pla[0][i][j]/(sP->Z_pla)); //*(q/sP->q0)*(q/sP->q0)/(m/sP->m0);
      fields->Gamma[1][i][j] = (sP->w_sw*fields->fluxNum[1][i][j] + sP->w_pla*fields->fluxNum_pla[1][i][j]/(sP->Z_pla)); //*(q/sP->q0)*(q/sP->q0)/(m/sP->m0);
      fields->Gamma[2][i][j] = (sP->w_sw*fields->fluxNum[2][i][j] + sP->w_pla*fields->fluxNum_pla[2][i][j]/(sP->Z_pla)); //*(q/sP->q0)*(q/sP->q0)/(m/sP->m0);
    #elif NB_DIM==3
      int i = int( idx/((len_y_cst+4)*(len_z_cst+4)) );
      int j = int( (idx-i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);
      //
      fields->Ji_b[0][i][j][k] = (sP->w_sw*fields->fluxNum[0][i][j][k] + sP->w_pla*fields->fluxNum_pla[0][i][j][k]);/// fields->density[i][j][k];
      fields->Ji_b[1][i][j][k] = (sP->w_sw*fields->fluxNum[1][i][j][k] + sP->w_pla*fields->fluxNum_pla[1][i][j][k]);/// fields->density[i][j][k];
      fields->Ji_b[2][i][j][k] = (sP->w_sw*fields->fluxNum[2][i][j][k] + sP->w_pla*fields->fluxNum_pla[2][i][j][k]);/// fields->density[i][j][k];
      //
      fields->Gamma[0][i][j][k] = (sP->w_sw*fields->fluxNum[0][i][j][k] + sP->w_pla*fields->fluxNum_pla[0][i][j][k]/(sP->Z_pla)); //*(q/sP->q0)*(q/sP->q0)/(m/sP->m0);
      fields->Gamma[1][i][j][k] = (sP->w_sw*fields->fluxNum[1][i][j][k] + sP->w_pla*fields->fluxNum_pla[1][i][j][k]/(sP->Z_pla)); //*(q/sP->q0)*(q/sP->q0)/(m/sP->m0);
      fields->Gamma[2][i][j][k] = (sP->w_sw*fields->fluxNum[2][i][j][k] + sP->w_pla*fields->fluxNum_pla[2][i][j][k]/(sP->Z_pla)); //*(q/sP->q0)*(q/sP->q0)/(m/sP->m0);
    #endif
  }

}

__global__ void current_advance_k(simu_fields* fields, simu_B_field* B, simu_param* sP, int indIt)
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if(idx < nb_nodes_tot_cst){
    #if NB_DIM==2
      int i = int(idx/(len_y_cst+4));
      int j = idx - i*(len_y_cst+4);
      //
      fields->Ji[0][i][j] = fields->Ji_b[0][i][j] + .5*sP->dt * (fields->Lambda[i][j]*fields->E[0][i][j] + fields->Gamma[1][i][j]*B->B[2][i][j] - fields->Gamma[2][i][j]*B->B[1][i][j]);
      fields->Ji[1][i][j] = fields->Ji_b[1][i][j] + .5*sP->dt * (fields->Lambda[i][j]*fields->E[1][i][j] + fields->Gamma[2][i][j]*B->B[0][i][j] - fields->Gamma[0][i][j]*B->B[2][i][j]);
      fields->Ji[2][i][j] = fields->Ji_b[2][i][j] + .5*sP->dt * (fields->Lambda[i][j]*fields->E[2][i][j] + fields->Gamma[0][i][j]*B->B[1][i][j] - fields->Gamma[1][i][j]*B->B[0][i][j]);
    #elif NB_DIM==3
      int i = int( idx/((len_y_cst+4)*(len_z_cst+4)) );
      int j = int( (idx-i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);
      //
      #if !dipole_cst
        fields->Ji[0][i][j][k] = fields->Ji_b[0][i][j][k] + .5*sP->dt * (fields->Lambda[i][j][k]*fields->E[0][i][j][k] + fields->Gamma[1][i][j][k]*B->B[2][i][j][k] - fields->Gamma[2][i][j][k]*B->B[1][i][j][k]);
        fields->Ji[1][i][j][k] = fields->Ji_b[1][i][j][k] + .5*sP->dt * (fields->Lambda[i][j][k]*fields->E[1][i][j][k] + fields->Gamma[2][i][j][k]*B->B[0][i][j][k] - fields->Gamma[0][i][j][k]*B->B[2][i][j][k]);
        fields->Ji[2][i][j][k] = fields->Ji_b[2][i][j][k] + .5*sP->dt * (fields->Lambda[i][j][k]*fields->E[2][i][j][k] + fields->Gamma[0][i][j][k]*B->B[1][i][j][k] - fields->Gamma[1][i][j][k]*B->B[0][i][j][k]);
      #else
        fields->Ji[0][i][j][k] = fields->Ji_b[0][i][j][k] + .5*sP->dt * (fields->Lambda[i][j][k]*fields->E[0][i][j][k] + fields->Gamma[1][i][j][k]*(B->B[2][i][j][k]+B->B_dip[2][i][j][k]) - fields->Gamma[2][i][j][k]*(B->B[1][i][j][k]+B->B_dip[1][i][j][k]) );
        fields->Ji[1][i][j][k] = fields->Ji_b[1][i][j][k] + .5*sP->dt * (fields->Lambda[i][j][k]*fields->E[1][i][j][k] + fields->Gamma[2][i][j][k]*(B->B[0][i][j][k]+B->B_dip[0][i][j][k]) - fields->Gamma[0][i][j][k]*(B->B[2][i][j][k]+B->B_dip[2][i][j][k]) );
        fields->Ji[2][i][j][k] = fields->Ji_b[2][i][j][k] + .5*sP->dt * (fields->Lambda[i][j][k]*fields->E[2][i][j][k] + fields->Gamma[0][i][j][k]*(B->B[1][i][j][k]+B->B_dip[1][i][j][k]) - fields->Gamma[1][i][j][k]*(B->B[0][i][j][k]+B->B_dip[0][i][j][k]) );
        // fields->Ji[0][i][j][k] = (fields->Lambda[i][j][k]*fields->E[0][i][j][k] + fields->Gamma[1][i][j][k]*(B->B[2][i][j][k]+B->B_dip[2][i][j][k]) - fields->Gamma[2][i][j][k]*(B->B[1][i][j][k]+B->B_dip[1][i][j][k]) );
        // fields->Ji[1][i][j][k] = (fields->Lambda[i][j][k]*fields->E[1][i][j][k] + fields->Gamma[2][i][j][k]*(B->B[0][i][j][k]+B->B_dip[0][i][j][k]) - fields->Gamma[0][i][j][k]*(B->B[2][i][j][k]+B->B_dip[2][i][j][k]) );
        // fields->Ji[2][i][j][k] = (fields->Lambda[i][j][k]*fields->E[2][i][j][k] + fields->Gamma[0][i][j][k]*(B->B[1][i][j][k]+B->B_dip[1][i][j][k]) - fields->Gamma[1][i][j][k]*(B->B[0][i][j][k]+B->B_dip[0][i][j][k]) );
        // fields->Ji[0][i][j][k] = fields->Gamma[1][i][j][k]*(B->B[2][i][j][k]+B->B_dip[2][i][j][k]) - fields->Gamma[2][i][j][k]*(B->B[1][i][j][k]+B->B_dip[1][i][j][k]) ;
        // fields->Ji[1][i][j][k] = fields->Gamma[2][i][j][k]*(B->B[0][i][j][k]+B->B_dip[0][i][j][k]) - fields->Gamma[0][i][j][k]*(B->B[2][i][j][k]+B->B_dip[2][i][j][k]) ;
        // fields->Ji[2][i][j][k] = fields->Gamma[0][i][j][k]*(B->B[1][i][j][k]+B->B_dip[1][i][j][k]) - fields->Gamma[1][i][j][k]*(B->B[0][i][j][k]+B->B_dip[0][i][j][k]) ;
      #endif
    #endif
  }
}

__global__ void average_moments_k(simu_fields* fields)
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if(idx < nb_nodes_tot_cst){
    #if NB_DIM==2
      int i = int(idx/(len_y_cst+4));
      int j = idx - i*(len_y_cst+4);

      fields->density[i][j]  = .5*(fields->density[i][j]  + fields->density_b[i][j]);
      fields->Ji[0][i][j]    = .5*(fields->Ji[0][i][j]    + fields->Ji_b[0][i][j]);
      fields->Ji[1][i][j]    = .5*(fields->Ji[1][i][j]    + fields->Ji_b[1][i][j]);
      fields->Ji[2][i][j]    = .5*(fields->Ji[2][i][j]    + fields->Ji_b[2][i][j]);
    #elif NB_DIM==3
      int i = int( idx/((len_y_cst+4)*(len_z_cst+4)) );
      int j = int( (idx-i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);

      fields->density[i][j][k]  = .5*(fields->density[i][j][k]  + fields->density_b[i][j][k]);
      fields->Ji[0][i][j][k]    = .5*(fields->Ji[0][i][j][k]    + fields->Ji_b[0][i][j][k]);
      fields->Ji[1][i][j][k]    = .5*(fields->Ji[1][i][j][k]    + fields->Ji_b[1][i][j][k]);
      fields->Ji[2][i][j][k]    = .5*(fields->Ji[2][i][j][k]    + fields->Ji_b[2][i][j][k]);
    #endif
  }
}

__global__ void identify_region_k(simu_fields* fields, simu_grid* grid, simu_param* sP, float delta_x)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < nb_nodes_tot_cst){
    #if NB_DIM==2
      int i = int(idx/(len_y_cst+4));
      int j = idx - i*(len_y_cst+4);
      //
      float rx = grid->xGrid[i] - grid->centre_x - delta_x;
      float ry = grid->yGrid[j] - grid->centre_y;
      float rsq = (rx*rx + ry*ry);
      //
      if (solid_body_cst && rsq < sP->r_obs_sqr){
        // The node is within the solid body.
        fields->region_ID[i][j] = 2;
      }
      // else if (inject_pla_cst && rsq < 400){
      //   // This is used as a tag of the "inner coma", for smoothing purpose.
      //   fields->region_ID[i][j] = 2;
      // }
      else if (fields->density[i][j] < sP->dens_min){//
        // The node is in a vacuum region.
        fields->region_ID[i][j] = 1;
      }
      else {
        // The node is in a plasma dense enough.
        fields->region_ID[i][j] = 0;
      }

      // if (i<4 && (fields->density[i][j][k] > sP->dens_min)){
      //   // Resistive downstream layer.
      //   fields->region_ID[i][j][k] = 3;
      // }
    #elif NB_DIM==3
      int i = int( idx/((len_y_cst+4)*(len_z_cst+4)) );
      int j = int( (idx-i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);
      //
      float rx = grid->xGrid[i] - (grid->centre_x + delta_x);
      float ry = grid->yGrid[j] - grid->centre_y;
      float rz = grid->zGrid[k] - grid->centre_z;
      float rsq = (rx*rx + ry*ry + rz*rz);
      //
      if (solid_body_cst && rsq < sP->r_obs_sqr){
        // The node is within the solid body.
        fields->region_ID[i][j][k] = 2;
      }
      else if (fields->density[i][j][k] <= sP->dens_min){
        // The node is in a vacuum region.
        fields->region_ID[i][j][k] = 1;
      }
      else {
        // The node is in a plasma dense enough.
        fields->region_ID[i][j][k] = 0;
      }

      // if (i<4 && (fields->density[i][j][k] > sP->dens_min)){
      //   // Resistive downstream layer.
      //   fields->region_ID[i][j][k] = 3;
      // }
    #endif
  }
}


__global__ void set_centre_k(simu_grid* grid, simu_param* sP, int idx_it)
{
    /* The centre of the origin is put at its default location: */
    grid->centre_x = sP->centre_x*grid->xMax;
    /* It is then shifted correspondingly with the iteration index: */
    grid->centre_x += sP->v_obs*sP->dt * (idx_it%nb_it_per_shift_cst);
}








//_________________________________________________________________________________________
//
// Field handling kernels: no physics, dealing with injector, copies of guard nodes, etc.
//
__global__ void copy_fields_k(simu_B_field* B_in, simu_B_field* B_out, simu_fields* fields)
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if(idx < nb_nodes_tot_cst){
    #if NB_DIM==2
      int i = int(idx/(len_y_cst+4));
      int j = idx - i*(len_y_cst+4);
      //
      B_out->B[0][i][j] = B_in->B[0][i][j];
      B_out->B[1][i][j] = B_in->B[1][i][j];
      B_out->B[2][i][j] = B_in->B[2][i][j];
      //
      fields->E_mot[0][i][j] = fields->E[0][i][j];
      fields->E_mot[1][i][j] = fields->E[1][i][j];
      fields->E_mot[2][i][j] = fields->E[2][i][j];
      //
      #if yee_cst
        B_out->B_s[0][i][j] = B_in->B_s[0][i][j];
        B_out->B_s[1][i][j] = B_in->B_s[1][i][j];
        B_out->B_s[2][i][j] = B_in->B_s[2][i][j];
      #endif

    #elif NB_DIM==3
      int i = int( idx/((len_y_cst+4)*(len_z_cst+4)) );
      int j = int( (idx-i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);
      //
      B_out->B[0][i][j][k] = B_in->B[0][i][j][k];
      B_out->B[1][i][j][k] = B_in->B[1][i][j][k];
      B_out->B[2][i][j][k] = B_in->B[2][i][j][k];
      //
      fields->E_mot[0][i][j][k] = fields->E[0][i][j][k];
      fields->E_mot[1][i][j][k] = fields->E[1][i][j][k];
      fields->E_mot[2][i][j][k] = fields->E[2][i][j][k];
      //
      #if yee_cst
        B_out->B_s[0][i][j][k] = B_in->B_s[0][i][j][k];
        B_out->B_s[1][i][j][k] = B_in->B_s[1][i][j][k];
        B_out->B_s[2][i][j][k] = B_in->B_s[2][i][j][k];
      #endif
    #endif
  }

}
__global__ void shift_field_k(simu_B_field* B_in, simu_B_field* B_out, simu_fields* fields)
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  #if NB_DIM==2
    if (idx < (len_x_cst+4-nb_cell_per_shift_cst)*(len_y_cst+4)){
        int i = int(idx/(len_y_cst+4));
        int j = idx - i*(len_y_cst+4);
        //
        B_out->B[0][i][j] = B_in->B[0][i+nb_cell_per_shift_cst][j];
        B_out->B[1][i][j] = B_in->B[1][i+nb_cell_per_shift_cst][j];
        B_out->B[2][i][j] = B_in->B[2][i+nb_cell_per_shift_cst][j];
        //
        fields->E[0][i][j] = fields->E_mot[0][i+nb_cell_per_shift_cst][j];
        fields->E[1][i][j] = fields->E_mot[1][i+nb_cell_per_shift_cst][j];
        fields->E[2][i][j] = fields->E_mot[2][i+nb_cell_per_shift_cst][j];
        //
        #if yee_cst
          B_out->B_s[0][i][j] = B_in->B_s[0][i+nb_cell_per_shift_cst][j];
          B_out->B_s[1][i][j] = B_in->B_s[1][i+nb_cell_per_shift_cst][j];
          B_out->B_s[2][i][j] = B_in->B_s[2][i+nb_cell_per_shift_cst][j];
        #endif
      }
  #elif NB_DIM==3
    if (idx < (len_x_cst+4-nb_cell_per_shift_cst)*(len_y_cst+4)*(len_z_cst+4)){
      int i = int( idx/((len_y_cst+4)*(len_z_cst+4)) );
      int j = int( (idx-i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);

      // if (fields->region_ID[i][j][k]!=2){
        B_out->B[0][i][j][k] = B_in->B[0][i+nb_cell_per_shift_cst][j][k];
        B_out->B[1][i][j][k] = B_in->B[1][i+nb_cell_per_shift_cst][j][k];
        B_out->B[2][i][j][k] = B_in->B[2][i+nb_cell_per_shift_cst][j][k];

        fields->E[0][i][j][k] = fields->E_mot[0][i+nb_cell_per_shift_cst][j][k];
        fields->E[1][i][j][k] = fields->E_mot[1][i+nb_cell_per_shift_cst][j][k];
        fields->E[2][i][j][k] = fields->E_mot[2][i+nb_cell_per_shift_cst][j][k];

        #if yee_cst
          B_out->B_s[0][i][j][k] = B_in->B_s[0][i+nb_cell_per_shift_cst][j][k];
          B_out->B_s[1][i][j][k] = B_in->B_s[1][i+nb_cell_per_shift_cst][j][k];
          B_out->B_s[2][i][j][k] = B_in->B_s[2][i+nb_cell_per_shift_cst][j][k];
        #endif
      // }
    }
  #endif


}

__global__ void inject_E_B_k(injector* injec, simu_B_field* Ba, simu_B_field* Bb, simu_fields* fields)
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  #if NB_DIM==2
    if (idx<(2+nb_cell_per_shift_cst)*(len_y_cst+4)){
      int i = int(idx/(len_y_cst+4));
      int j = idx - i*(len_y_cst+4);
      #if !yee_cst
        Ba->B[0][len_x_cst+2-nb_cell_per_shift_cst+i][j] = injec->B[0][i][j];
        Ba->B[1][len_x_cst+2-nb_cell_per_shift_cst+i][j] = injec->B[1][i][j];
        Ba->B[2][len_x_cst+2-nb_cell_per_shift_cst+i][j] = injec->B[2][i][j];
        Bb->B[0][len_x_cst+2-nb_cell_per_shift_cst+i][j] = injec->B[0][i][j];
        Bb->B[1][len_x_cst+2-nb_cell_per_shift_cst+i][j] = injec->B[1][i][j];
        Bb->B[2][len_x_cst+2-nb_cell_per_shift_cst+i][j] = injec->B[2][i][j];
      #else
        Ba->B_s[0][len_x_cst+2-nb_cell_per_shift_cst+i][j] = injec->B[0][i][j];
        Ba->B_s[1][len_x_cst+2-nb_cell_per_shift_cst+i][j] = injec->B[1][i][j];
        Ba->B_s[2][len_x_cst+2-nb_cell_per_shift_cst+i][j] = injec->B[2][i][j];
        Bb->B_s[0][len_x_cst+2-nb_cell_per_shift_cst+i][j] = injec->B[0][i][j];
        Bb->B_s[1][len_x_cst+2-nb_cell_per_shift_cst+i][j] = injec->B[1][i][j];
        Bb->B_s[2][len_x_cst+2-nb_cell_per_shift_cst+i][j] = injec->B[2][i][j];
      #endif
      fields->E[0][len_x_cst+2-nb_cell_per_shift_cst+i][j] = injec->E[0][i][j];
      fields->E[1][len_x_cst+2-nb_cell_per_shift_cst+i][j] = injec->E[1][i][j];
      fields->E[2][len_x_cst+2-nb_cell_per_shift_cst+i][j] = injec->E[2][i][j];
    }
  #elif NB_DIM==3
    if (idx<(2+nb_cell_per_shift_cst)*(len_y_cst+4)*(len_z_cst+4)){
      int i = int( idx/((len_y_cst+4)*(len_z_cst+4)) );
      int j = int( (idx - i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);
      #if !yee_cst
        Ba->B[0][len_x_cst+2-nb_cell_per_shift_cst+i][j][k] = injec->B[0][i][j][k];
        Ba->B[1][len_x_cst+2-nb_cell_per_shift_cst+i][j][k] = injec->B[1][i][j][k];
        Ba->B[2][len_x_cst+2-nb_cell_per_shift_cst+i][j][k] = injec->B[2][i][j][k];
        Bb->B[0][len_x_cst+2-nb_cell_per_shift_cst+i][j][k] = injec->B[0][i][j][k];
        Bb->B[1][len_x_cst+2-nb_cell_per_shift_cst+i][j][k] = injec->B[1][i][j][k];
        Bb->B[2][len_x_cst+2-nb_cell_per_shift_cst+i][j][k] = injec->B[2][i][j][k];
      #else
        Ba->B_s[0][len_x_cst+2-nb_cell_per_shift_cst+i][j][k] = injec->B[0][i][j][k];
        Ba->B_s[1][len_x_cst+2-nb_cell_per_shift_cst+i][j][k] = injec->B[1][i][j][k];
        Ba->B_s[2][len_x_cst+2-nb_cell_per_shift_cst+i][j][k] = injec->B[2][i][j][k];
        Bb->B_s[0][len_x_cst+2-nb_cell_per_shift_cst+i][j][k] = injec->B[0][i][j][k];
        Bb->B_s[1][len_x_cst+2-nb_cell_per_shift_cst+i][j][k] = injec->B[1][i][j][k];
        Bb->B_s[2][len_x_cst+2-nb_cell_per_shift_cst+i][j][k] = injec->B[2][i][j][k];
      #endif
      fields->E[0][len_x_cst+2-nb_cell_per_shift_cst+i][j][k] = injec->E[0][i][j][k];
      fields->E[1][len_x_cst+2-nb_cell_per_shift_cst+i][j][k] = injec->E[1][i][j][k];
      fields->E[2][len_x_cst+2-nb_cell_per_shift_cst+i][j][k] = injec->E[2][i][j][k];
    }
  #endif
}

__global__ void inject_dens_k
#if NB_DIM==2
(injector* injec, float density[len_x_cst+4][len_y_cst+4])
#elif NB_DIM==3
(injector* injec, float density[len_x_cst+4][len_y_cst+4][len_z_cst+4])
#endif
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  #if NB_DIM==2
    if (idx<(2+nb_cell_per_shift_cst)*(len_y_cst+4)){
      int i = int(idx/(len_y_cst+4));
      int j = idx - i*(len_y_cst+4);
      density[len_x_cst+2-nb_cell_per_shift_cst+i][j] = 1.;//injec->density[i][j];
    }
  #elif NB_DIM==3
    if (idx<(2+nb_cell_per_shift_cst)*(len_y_cst+4)*(len_z_cst+4)){
      int i = int( idx/((len_y_cst+4)*(len_z_cst+4)) );
      int j = int( (idx - i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);
      density[len_x_cst+2-nb_cell_per_shift_cst+i][j][k] = injec->density[i][j][k];
    }
  #endif
}

__global__ void inject_curr_k
#if NB_DIM==2
(injector* injec, float Ji[3][len_x_cst+4][len_y_cst+4])
#elif NB_DIM==3
(injector* injec, float Ji[3][len_x_cst+4][len_y_cst+4][len_z_cst+4])
#endif
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  #if NB_DIM==2
    if (idx<(2+nb_cell_per_shift_cst)*(len_y_cst+4)){
      int i = int(idx/(len_y_cst+4));
      int j = idx - i*(len_y_cst+4);
      Ji[0][len_x_cst+2-nb_cell_per_shift_cst+i][j] = injec->Ji[0][i][j];
      Ji[1][len_x_cst+2-nb_cell_per_shift_cst+i][j] = injec->Ji[1][i][j];
      Ji[2][len_x_cst+2-nb_cell_per_shift_cst+i][j] = injec->Ji[2][i][j];
    }
  #elif NB_DIM==3
    if (idx<(2+nb_cell_per_shift_cst)*(len_y_cst+4)*(len_z_cst+4)){
      int i = int( idx/((len_y_cst+4)*(len_z_cst+4)) );
      int j = int( (idx - i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);
      Ji[0][len_x_cst+2-nb_cell_per_shift_cst+i][j][k] = injec->Ji[0][i][j][k];
      Ji[1][len_x_cst+2-nb_cell_per_shift_cst+i][j][k] = injec->Ji[1][i][j][k];
      Ji[2][len_x_cst+2-nb_cell_per_shift_cst+i][j][k] = injec->Ji[2][i][j][k];
    }
  #endif
}

// __global__ void inject_E_const_ORF_k
// #if NB_DIM==2
// (float E[3][len_x_cst+4][len_y_cst+4], simu_param* sP)
// #elif NB_DIM==3
// (float E[3][len_x_cst+4][len_y_cst+4][len_z_cst+4], simu_param* sP)
// #endif
// {
//   int idx = threadIdx.x + blockIdx.x*blockDim.x;
//   #if NB_DIM==2
//     // printf("%i %.7e %.7e\n", idx, -sP->v_A, sP->v_A);
//     if (idx<(len_y_cst+4)){
//       for (int i=0; i<150; i++){
//         E[0][len_x_cst+3-i][idx] = 0.;
//         E[1][len_x_cst+3-i][idx] = -sP->e_mot*sP->v_obs*sP->B0_z;
//         E[2][len_x_cst+3-i][idx] =  sP->e_mot*sP->v_obs*sP->B0_y;
//       }
//     }
//   #elif NB_DIM==3
//     if(idx < (len_y_cst+4)*(len_z_cst+4)){
//       int j = floorf( idx/(len_z_cst+4) );
//       int k = idx - j*(len_z_cst+4);
//       for (int i=0; i<20; i++){
//         E[0][len_x_cst+3-i][j][k] = 0.;
//         E[1][len_x_cst+3-i][j][k] = -sP->e_mot*sP->v_obs*sP->B0_z;
//         E[2][len_x_cst+3-i][j][k] =  sP->e_mot*sP->v_obs*sP->B0_y;
//       }
//     }
//   #endif
// }
__global__ void inject_const_E_k(simu_fields* fields, simu_param* sP, bool pute)
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  #if NB_DIM==2
    if(idx < (len_y_cst+4)){
      for (int i=0; i<1+nb_cell_per_shift_cst; i++){
        if (pute){
          fields->E[0][len_x_cst+3-i][idx] = 0.;
          fields->E[1][len_x_cst+3-i][idx] = -sP->v_obs*sP->B0_z;
          fields->E[2][len_x_cst+3-i][idx] =  sP->v_obs*sP->B0_y;
        }
        else{
          fields->E[0][len_x_cst+3-i][idx] = 0.;
          fields->E[1][len_x_cst+3-i][idx] = 0.;
          fields->E[2][len_x_cst+3-i][idx] = 0.;
        }
      }
    }
  #elif NB_DIM==3
    if(idx < (len_y_cst+4)*(len_z_cst+4)){
      int j = int( idx/(len_z_cst+4) );
      int k = idx - j*(len_z_cst+4);
      for (int i=0; i<1+nb_cell_per_shift_cst; i++){
        if (pute){
          fields->E[0][len_x_cst+3-i][j][k] = 0.;
          fields->E[1][len_x_cst+3-i][j][k] = -sP->v_obs*sP->B0_z;
          fields->E[2][len_x_cst+3-i][j][k] =  sP->v_obs*sP->B0_y;
        }
        else{
          fields->E[0][len_x_cst+3-i][j][k] = 0.;
          fields->E[1][len_x_cst+3-i][j][k] = 0.;
          fields->E[2][len_x_cst+3-i][j][k] = 0.;
        }
      }
    }
  #endif

}

__global__ void inject_const_B_k(simu_B_field* B, simu_param* sP)
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  #if NB_DIM==2
    if(idx < (len_y_cst+4)){
      for (int i=0; i<1+nb_cell_per_shift_cst; i++){
      B->B[0][len_x_cst+3-i][idx] = sP->B0_x;
      B->B[1][len_x_cst+3-i][idx] = sP->B0_y;
      B->B[2][len_x_cst+3-i][idx] = sP->B0_z;

      #if yee_cst
        B->B_s[0][len_x_cst+3-i][idx] = sP->B0_x;
        B->B_s[1][len_x_cst+3-i][idx] = sP->B0_y;
        B->B_s[2][len_x_cst+3-i][idx] = sP->B0_z;
      #endif
      }
    }
  #elif NB_DIM==3
    if(idx < (len_y_cst+4)*(len_z_cst+4)){
      int j = int( idx/(len_z_cst+4) );
      int k = idx - j*(len_z_cst+4);
      for (int i=0; i<1+nb_cell_per_shift_cst; i++){
        B->B[0][len_x_cst+3-i][j][k] = sP->B0_x;
        B->B[1][len_x_cst+3-i][j][k] = sP->B0_y;
        B->B[2][len_x_cst+3-i][j][k] = sP->B0_z;

        #if yee_cst
          B->B_s[0][len_x_cst+3-i][idx][k] = sP->B0_x;
          B->B_s[1][len_x_cst+3-i][idx][k] = sP->B0_y;
          B->B_s[2][len_x_cst+3-i][idx][k] = sP->B0_z;
        #endif
      }
    }
  #endif

}
__global__ void inject_wave_B_k(simu_B_field* B, simu_param* sP,
                                simu_grid* grid, int idx_it)
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  #if NB_DIM==2
    if(idx < (len_y_cst+4)){
      for (int i=0; i<1+nb_cell_per_shift_cst+1; i++){
        // B->B[0][len_x_cst+3-i][idx] +=  .2/std::sqrt(.4)*cos(.2*grid->xGrid[i]+.2*grid->yGrid[idx]+6.*idx_it*sP->dt);
        // B->B[1][len_x_cst+3-i][idx] += -.2/std::sqrt(.4)*cos(.2*grid->xGrid[i]+.2*grid->yGrid[idx]+6.*idx_it*sP->dt);
        B->B[0][len_x_cst+3-i][idx] = sP->B0_x+cos(.6*idx_it*sP->dt);
        // B->B[1][len_x_cst+3-i][idx] = sP->B0_y+cos(6.*idx_it*sP->dt);
        // B->B[2][len_x_cst+3-i][idx] = sP->B0_z+cos(6.*idx_it*sP->dt);
        // printf("%i %i %.4e\n", len_x_cst+3-i, idx, B->B[1][len_x_cst+3-i][idx]);
        // B->B[2][len_x_cst+3-i][idx] += -cos(6.*idx_it*sP->dt);
      // B->B[0][len_x_cst+3-i][idx] = 0.;
      // B->B[1][len_x_cst+3-i][idx] = 1.*sin(.25*idx_it*sP->dt);
      // B->B[2][len_x_cst+3-i][idx] = 0.;
      }
    }
  #elif NB_DIM==3
    if(idx < (len_y_cst+4)*(len_z_cst+4)){
      int j = int( idx/(len_z_cst+4) );
      int k = idx - j*(len_z_cst+4);
      for (int i=0; i<1+nb_cell_per_shift_cst; i++){
        B->B[1][len_x_cst+3-i][j][k] = 1.*sin(.25*idx_it*sP->dt);
      }
    }
  #endif

}
__global__ void inject_vortex_B_k(simu_B_field* B, simu_param* sP,
                                simu_grid* grid, int idx_it)
{


  #if NB_DIM==2
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < (len_y_cst+4)){
      for (int i=0; i<1+nb_cell_per_shift_cst+1; i++){
        float x = -500 + (idx_it+i)*sP->dX/nb_it_per_shift_cst;
        float y = grid->yGrid[idx] - 250;
        float r = sqrtf(x*x + y*y);
        //
        B->B[0][len_x_cst+3-i][idx] =  y/(r) * .5*expf(-((r-50)*(r-50)/(25*25)));
        B->B[1][len_x_cst+3-i][idx] = -x/(r) * .5*expf(-((r-50)*(r-50)/(25*25)));
      }
    }

  // #elif NB_DIM==3
  //   if(idx < (len_y_cst+4)*(len_z_cst)){
  //     for (int i=0; i<1+nb_cell_per_shift_cst+1; i++){
  //       float x = -500 + (idx_it+i)*sP->dX/nb_it_per_shift_cst;
  //       float y = grid->yGrid[idx] - 250;
  //       float r = sqrtf(x*x + y*y);
  //       //
  //       B->B[0][len_x_cst+3-i][idx] =  y/(r) * .5*expf(-((r-50)*(r-50)/(25*25)));
  //       B->B[1][len_x_cst+3-i][idx] = -x/(r) * .5*expf(-((r-50)*(r-50)/(25*25)));
  //     }
  //   }

  #endif

}

__global__ void inject_dens_const_k
#if NB_DIM==2
(float density[len_x_cst+4][len_y_cst+4], simu_param* sP)
#elif NB_DIM==3
(float density[len_x_cst+4][len_y_cst+4][len_z_cst+4], simu_param* sP)
#endif
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  #if NB_DIM==2
    if (idx<(len_y_cst+4)){
      for (int i=0; i<2+nb_cell_per_shift_cst; i++){
        density[len_x_cst+3-i][idx] = 1.;
      }
    }
  #elif NB_DIM==3
    if(idx < (len_y_cst+4)*(len_z_cst+4)){
      int j = floorf( idx/(len_z_cst+4) );
      int k = idx - j*(len_z_cst+4);
      for (int i=0; i<2+nb_cell_per_shift_cst; i++){
        density[len_x_cst+3-i][j][k] = 1.;
      }
    }
  #endif
}

__global__ void inject_curr_const_k
#if NB_DIM==2
(float Ji[3][len_x_cst+4][len_y_cst+4], simu_param* sP)
#elif NB_DIM==3
(float Ji[3][len_x_cst+4][len_y_cst+4][len_z_cst+4], simu_param* sP)
#endif
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  #if NB_DIM==2
    if (idx<(len_y_cst+4)){
      for (int i=0; i<2+nb_cell_per_shift_cst; i++){
        #if ORF_cst
          Ji[0][len_x_cst+3-i][idx] = -sP->v_obs/1.; // 1. is the background density!
        #else
          Ji[0][len_x_cst+3-i][idx] = 0.;//-sP->v_obs/1.;//
        #endif
        Ji[1][len_x_cst+3-i][idx] = 0.;
        Ji[2][len_x_cst+3-i][idx] = 0.;
      }
    }
  #elif NB_DIM==3
    if(idx < (len_y_cst+4)*(len_z_cst+4)){
      int j = floorf( idx/(len_z_cst+4) );
      int k = idx - j*(len_z_cst+4);
      for (int i=0; i<2+nb_cell_per_shift_cst; i++){
        #if ORF_cst
          Ji[0][len_x_cst+3-i][j][k] = -sP->v_obs/1.; // 1. is the background density!
        #else
          Ji[0][len_x_cst+3-i][j][k] = 0.;//-sP->v_obs/1.;//
          // #if dipole_cst
          //   Ji[0][len_x_cst+3-i][j][k] = -sP->v_obs/1.;
          // #endif
        #endif
        Ji[1][len_x_cst+3-i][j][k] = 0.;
        Ji[2][len_x_cst+3-i][j][k] = 0.;
      }
    }
  #endif
}


__global__ void clear_counts_flux_num_k(simu_fields* fields)
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if(idx < nb_nodes_tot_cst){
    #if NB_DIM==2
      int i = int(idx/(len_y_cst+4));
      int j = idx - i*(len_y_cst+4);

      fields->counts[i][j] = 0.;
      fields->fluxNum[0][i][j] = 0.;
      fields->fluxNum[1][i][j] = 0.;
      fields->fluxNum[2][i][j] = 0.;
      fields->counts_pla[i][j] = 0.;
      fields->fluxNum_pla[0][i][j] = 0.;
      fields->fluxNum_pla[1][i][j] = 0.;
      fields->fluxNum_pla[2][i][j] = 0.;
    #elif NB_DIM==3
      int i = int( idx/((len_y_cst+4)*(len_z_cst+4)) );
      int j = int( (idx-i*(len_y_cst+4)*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*(len_y_cst+4)*(len_z_cst+4) - j*(len_z_cst+4);

      fields->counts[i][j][k] = 0.;
      fields->fluxNum[0][i][j][k] = 0.;
      fields->fluxNum[1][i][j][k] = 0.;
      fields->fluxNum[2][i][j][k] = 0.;
      fields->counts_pla[i][j][k] = 0.;
      fields->fluxNum_pla[0][i][j][k] = 0.;
      fields->fluxNum_pla[1][i][j][k] = 0.;
      fields->fluxNum_pla[2][i][j][k] = 0.;
    #endif
  }
}


#if obstacle_cst && NB_DIM==2
__global__ void get_boundary_position_k(simu_B_field* B, simu_fields* fields, int idx_it)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx<len_y_cst){

    int idx_save = int(idx_it/nb_it_per_shift_cst);

    // Setting a first value to downstream, if the threshold isn't surpassed anywhere;
    fields->bound_pos[idx_save][idx] = 0;

    for (int i=len_x_cst; i>0; i--){ // Searching from upstream!
      if (fields->density[i+2][idx]>=1.778){  // 1.778 = 10**.25, found to work best for run 31.
        fields->bound_pos[idx_save][idx] = i+2; // Taken on full len_x_cst+4 domain.
        break;
      }
    }

    #if debug_mode_cst
      if (idx_save>nb_it_max_cst/nb_it_per_shift_cst){
        printf("| Out of pos_bound.\n");
      }
    #endif
  }
}
#endif



#if NB_DIM==2
  __global__ void add_copy_guard_X_k(float G[len_x_cst+4][len_y_cst+4]){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < (len_y_cst+4)){

      G[0          ][idx]  += G[len_x_cst  ][idx];
      G[1          ][idx]  += G[len_x_cst+1][idx];
      G[2          ][idx]  += G[len_x_cst+2][idx];
      G[len_x_cst+1][idx]   = G[1          ][idx];
      G[len_x_cst+2][idx]   = G[2          ][idx];
      G[len_x_cst+3][idx]   = G[3          ][idx];

    }
  }
  __global__ void add_copy_guard_Y_k(float G[len_x_cst+4][len_y_cst+4]){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < (len_x_cst+4)){

      G[idx][0          ]  += G[idx][len_y_cst  ];
      G[idx][1          ]  += G[idx][len_y_cst+1];
      G[idx][2          ]  += G[idx][len_y_cst+2];
      G[idx][len_y_cst+1]   = G[idx][1          ];
      G[idx][len_y_cst+2]   = G[idx][2          ];
      G[idx][len_y_cst+3]   = G[idx][3          ];

    }
  }

  __global__ void copy_guard_X_k(float G[len_x_cst+4][len_y_cst+4]){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < (len_y_cst+4)){

      G[0          ][idx] = G[len_x_cst  ][idx];
      G[1          ][idx] = G[len_x_cst+1][idx];
      G[len_x_cst+2][idx] = G[2          ][idx];
      G[len_x_cst+3][idx] = G[3          ][idx];

    }
  }
  __global__ void copy_guard_Y_k(float G[len_x_cst+4][len_y_cst+4]){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < (len_x_cst+4)){

      G[idx][0          ]  = G[idx][len_y_cst  ];
      G[idx][1          ]  = G[idx][len_y_cst+1];
      G[idx][len_y_cst+2]  = G[idx][2          ];
      G[idx][len_y_cst+3]  = G[idx][3          ];

    }
  }

  __global__ void extrapolate_X_k(float G[len_x_cst+4][len_y_cst+4]){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx<len_y_cst+4){
      // if (fill_const){
        // G[0][idx] = .5;
        // G[1][idx] = .5;
        // G[2][idx] = .5;
        // G[0][idx] = G[len_x_cst+3][idx];
        // G[1][idx] = G[len_x_cst+3][idx];
        // G[2][idx] = G[len_x_cst+3][idx];
        // G[0][idx] = G[len_x_cst  ][idx];
        // G[1][idx] = G[len_x_cst+1][idx];
        // G[2][idx] = G[len_x_cst+2][idx];
        G[0][idx] = G[3][idx];
        G[1][idx] = G[3][idx];
        G[2][idx] = G[3][idx];

      // }
      // else {
      //   G[0][idx] = .001*G[3][idx];
      //   G[1][idx] = .01*G[3][idx];
      //   G[2][idx] = .1*G[3][idx];
      // }
      // if (signbit(G[4][idx]) && !signbit(G[3][idx])) {
      //   G[0][idx] *= -1;
      //   G[1][idx] *= -1;
      //   G[2][idx] *= -1;
      //   G[3][idx] *= -1;
      // }
      // if (signbit(G[3][idx]) && !signbit(G[4][idx])) {
      //   G[0][idx] *= -1;
      //   G[1][idx] *= -1;
      //   G[2][idx] *= -1;
      //   G[3][idx] *= -1;
      // }
    }
  }
  //
  __global__ void extrapolate_Y_up_k(float G[len_x_cst+4][len_y_cst+4])
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx<(len_x_cst+4)){
      G[idx][(len_y_cst+3)] = G[idx][len_y_cst];
      G[idx][(len_y_cst+2)] = G[idx][len_y_cst];
      G[idx][(len_y_cst+1)] = G[idx][len_y_cst];
    }
  }
  //
  __global__ void extrapolate_Y_down_k(float G[len_x_cst+4][len_y_cst+4])
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx<(len_x_cst+4)){
      G[idx][0] = G[idx][3];
      G[idx][1] = G[idx][3];
      G[idx][2] = G[idx][3];
    }
  }


  __global__ void fill_buff_down_y_k(simu_fields* fields, float G[len_x_cst+4][len_y_cst+4])
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if ( idx<(4*(len_x_cst+4)) ){
      int i = int(idx/4);
      int j = idx - i*4;
      fields->buff_send_1d_y[idx] = G[i][j];
    }
  }

  __global__ void add_buff_down_y_k(simu_fields* fields, float G[len_x_cst+4][len_y_cst+4])
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if ( idx<(4*(len_x_cst+4)) ){
      int i = int(idx/4);
      int j = idx - i*4;
      G[i][len_y_cst+j] += fields->buff_reci_1d_y[idx];
    }
  }

  __global__ void copy_buff_down_long_y_k(simu_fields* fields, float G[len_x_cst+4][len_y_cst+4])
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if ( idx<(4*(len_x_cst+4)) ){
      int i = int(idx/4);
      int j = idx - i*4;
      G[i][len_y_cst+j] = fields->buff_reci_1d_y[idx];
    }
  }
  __global__ void copy_buff_down_short_y_k(simu_fields* fields, float G[len_x_cst+4][len_y_cst+4])
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if ( idx<(4*(len_x_cst+4)) ){
      int i = int(idx/4);
      int j = idx - i*4;
      if (j>1){
        G[i][len_y_cst+j] = fields->buff_reci_1d_y[idx];
      }
    }
  }

  __global__ void fill_buff_up_y_k(simu_fields* fields, float G[len_x_cst+4][len_y_cst+4])
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if ( idx<(4*(len_x_cst+4)) ){
      int i = int(idx/4);
      int j = idx - i*4;
      fields->buff_send_1d_y[idx] = G[i][len_y_cst+j];
    }
  }

  __global__ void copy_buff_up_long_y_k(simu_fields* fields, float G[len_x_cst+4][len_y_cst+4])
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if ( idx<(4*(len_x_cst+4)) ){
      int i = int(idx/4);
      int j = idx - i*4;
      G[i][j] = fields->buff_reci_1d_y[idx];
    }
  }
  __global__ void copy_buff_up_short_y_k(simu_fields* fields, float G[len_x_cst+4][len_y_cst+4])
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if ( idx<(4*(len_x_cst+4)) ){
      int i = int(idx/4);
      int j = idx - i*4;
      if (j<2){
        G[i][j] = fields->buff_reci_1d_y[idx];
      }
    }
  }



#elif NB_DIM==3
  __global__ void add_copy_guard_X_k(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4])
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < (len_y_cst+4)*(len_z_cst+4)){
      int j = int( idx/(len_z_cst+4) );
      int k = idx - j*(len_z_cst+4);

      G[0          ][j][k] += G[len_x_cst  ][j][k];
      G[1          ][j][k] += G[len_x_cst+1][j][k];
      G[2          ][j][k] += G[len_x_cst+2][j][k];
      G[len_x_cst+1][j][k]  = G[1          ][j][k];
      G[len_x_cst+2][j][k]  = G[2          ][j][k];
      G[len_x_cst+3][j][k]  = G[3          ][j][k];
    }
  }
  __global__ void add_copy_guard_Y_k(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4])
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < (len_x_cst+4)*(len_z_cst+4)){
      int i = int( idx/(len_z_cst+4) );
      int k = idx - i*(len_z_cst+4);

      G[i][0          ][k] += G[i][len_y_cst  ][k];
      G[i][1          ][k] += G[i][len_y_cst+1][k];
      G[i][2          ][k] += G[i][len_y_cst+2][k];
      G[i][len_y_cst+1][k]  = G[i][1          ][k];
      G[i][len_y_cst+2][k]  = G[i][2          ][k];
      G[i][len_y_cst+3][k]  = G[i][3          ][k];

    }
  }
  __global__ void add_copy_guard_Z_k(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4])
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < (len_x_cst+4)*(len_y_cst+4)){
      int i = int( idx/(len_y_cst+4) );
      int j = idx - i*(len_y_cst+4);

      G[i][j][0          ] += G[i][j][len_z_cst  ];
      G[i][j][1          ] += G[i][j][len_z_cst+1];
      G[i][j][2          ] += G[i][j][len_z_cst+2];
      G[i][j][len_z_cst+1]  = G[i][j][1          ];
      G[i][j][len_z_cst+2]  = G[i][j][2          ];
      G[i][j][len_z_cst+3]  = G[i][j][3          ];

    }
  }

  __global__ void copy_guard_X_k(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4])
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < (len_y_cst+4)*(len_z_cst+4)){
      int j = int( idx/(len_z_cst+4) );
      int k = idx - j*(len_z_cst+4);

      G[0          ][j][k] = G[len_x_cst  ][j][k];
      G[1          ][j][k] = G[len_x_cst+1][j][k];
      G[len_x_cst+2][j][k] = G[2          ][j][k];
      G[len_x_cst+3][j][k] = G[3          ][j][k];

    }
  }
  __global__ void copy_guard_Y_k(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4])
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < (len_x_cst+4)*(len_z_cst+4)){
      int i = int( idx/(len_z_cst+4) );
      int k = idx - i*(len_z_cst+4);

      G[i][0          ][k]  = G[i][len_y_cst  ][k];
      G[i][1          ][k]  = G[i][len_y_cst+1][k];
      G[i][len_y_cst+2][k]  = G[i][2          ][k];
      G[i][len_y_cst+3][k]  = G[i][3          ][k];

    }
  }
  __global__ void copy_guard_Z_k(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4])
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < (len_x_cst+4)*(len_y_cst+4)){
      int i = int( idx/(len_y_cst+4) );
      int j = idx - i*(len_y_cst+4);
      //
      G[i][j][0          ]  = G[i][j][len_z_cst  ];
      G[i][j][1          ]  = G[i][j][len_z_cst+1];
      G[i][j][len_z_cst+2]  = G[i][j][2          ];
      G[i][j][len_z_cst+3]  = G[i][j][3          ];

    }
  }

  __global__ void extrapolate_X_k(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4])
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx<(len_y_cst+4)*(len_z_cst+4))
    {
      int j = int( idx/(len_z_cst+4) );
      int k = idx - j*(len_z_cst+4);
      //
      G[0][j][k] = G[3][j][k];
      G[1][j][k] = G[3][j][k];
      G[2][j][k] = G[3][j][k];
    }
  }
  //
  __global__ void extrapolate_Y_up_k(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4])
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx<(len_x_cst+4)*(len_z_cst+4)){
      int i = int( idx/(len_z_cst+4) );
      int k = idx - i*(len_z_cst+4);
      //
      G[i][(len_y_cst+3)][k] = G[i][len_y_cst][k];
      G[i][(len_y_cst+2)][k] = G[i][len_y_cst][k];
      G[i][(len_y_cst+1)][k] = G[i][len_y_cst][k];
    }
  }
  //
  __global__ void extrapolate_Y_down_k(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4])
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx<(len_x_cst+4)*(len_z_cst+4)){
      int i = int( idx/(len_z_cst+4) );
      int k = idx - i*(len_z_cst+4);
      //
      G[i][0][k] = G[i][3][k];
      G[i][1][k] = G[i][3][k];
      G[i][2][k] = G[i][3][k];
    }
  }

  __global__ void extrapolate_Z_up_k(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4])
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx<(len_x_cst+4)*(len_y_cst+4)){
      int i = int( idx/(len_y_cst+4) );
      int j = idx - i*(len_y_cst+4);
      //
      G[i][j][(len_z_cst+3)] = G[i][j][len_z_cst];
      G[i][j][(len_z_cst+2)] = G[i][j][len_z_cst];
      G[i][j][(len_z_cst+1)] = G[i][j][len_z_cst];
    }
  }
  //
  __global__ void extrapolate_Z_down_k(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4])
  {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx<(len_x_cst+4)*(len_y_cst+4)){
      int i = int( idx/(len_y_cst+4) );
      int j = idx - i*(len_y_cst+4);

      G[i][j][0] = G[i][j][3];
      G[i][j][1] = G[i][j][3];
      G[i][j][2] = G[i][j][3];
    }
  }






  __global__ void fill_buff_down_y_k(simu_fields* fields, float G[len_x_cst+4][len_y_cst+4][len_z_cst+4]){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if ( idx<((len_x_cst+4)*4*(len_z_cst+4)) ){
      int i = int( idx/(4*(len_z_cst+4)) );
      int j = int( (idx - i*4*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*4*(len_z_cst+4) - j*(len_z_cst+4);

      fields->buff_send_1d_y[idx] = G[i][j][k];
    }
  }

  __global__ void add_buff_down_y_k(simu_fields* fields, float G[len_x_cst+4][len_y_cst+4][len_z_cst+4]){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if ( idx<((len_x_cst+4)*4*(len_z_cst+4)) ){
      int i = int( idx/(4*(len_z_cst+4)) );
      int j = int( (idx - i*4*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*4*(len_z_cst+4) - j*(len_z_cst+4);

      G[i][len_y_cst+j][k] += fields->buff_reci_1d_y[idx];
    }
  }

  __global__ void copy_buff_down_long_y_k(simu_fields* fields, float G[len_x_cst+4][len_y_cst+4][len_z_cst+4]){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if ( idx<((len_x_cst+4)*4*(len_z_cst+4)) ){
      int i = int( idx/(4*(len_z_cst+4)) );
      int j = int( (idx - i*4*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*4*(len_z_cst+4) - j*(len_z_cst+4);

      G[i][len_y_cst+j][k] = fields->buff_reci_1d_y[idx];
    }
  }
  __global__ void copy_buff_down_short_y_k(simu_fields* fields, float G[len_x_cst+4][len_y_cst+4][len_z_cst+4]){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if ( idx<((len_x_cst+4)*4*(len_z_cst+4)) ){
      int i = int( idx/(4*(len_z_cst+4)) );
      int j = int( (idx - i*4*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*4*(len_z_cst+4) - j*(len_z_cst+4);

      if (j>1){
        G[i][len_y_cst+j][k] = fields->buff_reci_1d_y[idx];
      }
    }
  }

  __global__ void fill_buff_up_y_k(simu_fields* fields, float G[len_x_cst+4][len_y_cst+4][len_z_cst+4]){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if ( idx<((len_x_cst+4)*4*(len_z_cst+4)) ){
      int i = int( idx/(4*(len_z_cst+4)) );
      int j = int( (idx - i*4*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*4*(len_z_cst+4) - j*(len_z_cst+4);

      fields->buff_send_1d_y[idx] = G[i][len_y_cst+j][k];
    }
  }

  __global__ void copy_buff_up_long_y_k(simu_fields* fields, float G[len_x_cst+4][len_y_cst+4][len_z_cst+4]){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if ( idx<((len_x_cst+4)*4*(len_z_cst+4)) ){
      int i = int( idx/(4*(len_z_cst+4)) );
      int j = int( (idx - i*4*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*4*(len_z_cst+4) - j*(len_z_cst+4);

      G[i][j][k] = fields->buff_reci_1d_y[idx];
    }
  }
  __global__ void copy_buff_up_short_y_k(simu_fields* fields, float G[len_x_cst+4][len_y_cst+4][len_z_cst+4]){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if ( idx<((len_x_cst+4)*4*(len_z_cst+4)) ){
      int i = int( idx/(4*(len_z_cst+4)) );
      int j = int( (idx - i*4*(len_z_cst+4))/(len_z_cst+4) );
      int k = idx - i*4*(len_z_cst+4) - j*(len_z_cst+4);

      if (j<2){
        G[i][j][k] = fields->buff_reci_1d_y[idx];
      }
    }
  }


  __global__ void fill_buff_down_z_k(simu_fields* fields, float G[len_x_cst+4][len_y_cst+4][len_z_cst+4]){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if ( idx<((len_x_cst+4)*(len_y_cst+4)*4) ){
      int i = int( idx/((len_y_cst+4)*4) );
      int j = int( (idx - i*(len_y_cst+4)*4)/4 );
      int k = idx - i*(len_y_cst+4)*4 - j*4;

      fields->buff_send_1d_z[idx] = G[i][j][k];
    }
  }

  __global__ void add_buff_down_z_k(simu_fields* fields, float G[len_x_cst+4][len_y_cst+4][len_z_cst+4]){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if ( idx<((len_x_cst+4)*(len_y_cst+4)*4) ){
      int i = int( idx/((len_y_cst+4)*4) );
      int j = int( (idx - i*(len_y_cst+4)*4)/4 );
      int k = idx - i*(len_y_cst+4)*4 - j*4;

      G[i][j][len_z_cst+k] += fields->buff_reci_1d_z[idx];
    }
  }

  __global__ void copy_buff_down_long_z_k(simu_fields* fields, float G[len_x_cst+4][len_y_cst+4][len_z_cst+4]){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if ( idx<((len_x_cst+4)*(len_y_cst+4)*4) ){
      int i = int( idx/((len_y_cst+4)*4) );
      int j = int( (idx - i*(len_y_cst+4)*4)/4 );
      int k = idx - i*(len_y_cst+4)*4 - j*4;

      G[i][j][len_z_cst+k] = fields->buff_reci_1d_z[idx];
    }
  }
  __global__ void copy_buff_down_short_z_k(simu_fields* fields, float G[len_x_cst+4][len_y_cst+4][len_z_cst+4]){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if ( idx<((len_x_cst+4)*(len_y_cst+4)*4) ){
      int i = int( idx/((len_y_cst+4)*4) );
      int j = int( (idx - i*(len_y_cst+4)*4)/4 );
      int k = idx - i*(len_y_cst+4)*4 - j*4;

      if (k>1){
        G[i][j][len_z_cst+k] = fields->buff_reci_1d_z[idx];
      }
    }
  }

  __global__ void fill_buff_up_z_k(simu_fields* fields, float G[len_x_cst+4][len_y_cst+4][len_z_cst+4]){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if ( idx<((len_x_cst+4)*(len_y_cst+4)*4) ){
      int i = int( idx/((len_y_cst+4)*4) );
      int j = int( (idx - i*(len_y_cst+4)*4)/4 );
      int k = idx - i*(len_y_cst+4)*4 - j*4;

      fields->buff_send_1d_z[idx] = G[i][j][len_z_cst+k];
    }
  }

  __global__ void copy_buff_up_long_z_k(simu_fields* fields, float G[len_x_cst+4][len_y_cst+4][len_z_cst+4]){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if ( idx<((len_x_cst+4)*(len_y_cst+4)*4) ){
      int i = int( idx/((len_y_cst+4)*4) );
      int j = int( (idx - i*(len_y_cst+4)*4)/4 );
      int k = idx - i*(len_y_cst+4)*4 - j*4;

      G[i][j][k] = fields->buff_reci_1d_z[idx];
    }
  }
  __global__ void copy_buff_up_short_z_k(simu_fields* fields, float G[len_x_cst+4][len_y_cst+4][len_z_cst+4]){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if ( idx<((len_x_cst+4)*(len_y_cst+4)*4) ){
      int i = int( idx/((len_y_cst+4)*4) );
      int j = int( (idx - i*(len_y_cst+4)*4)/4 );
      int k = idx - i*(len_y_cst+4)*4 - j*4;

      if (k<2){
        G[i][j][k] = fields->buff_reci_1d_z[idx];
      }
    }
  }
#endif




__global__ void vorticity_k(simu_fields* fields, int nbNode, simu_param* sP)
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x;


  if(idx < nbNode){
    #if NB_DIM==2
      int i = floorf(idx/len_y_cst);
      int j = idx-i*len_y_cst;
      i += 2;
      j += 2;

      fields->vort[0][i][j] = 1./(2.*sP->dX)*( (fields->fluxNum[2][i][j+1]/fields->counts[i][j+1])-(fields->fluxNum[2][i][j-1]/fields->counts[i][j-1]) );
      fields->vort[1][i][j] = 1./(2.*sP->dX)*(-(fields->fluxNum[2][i+1][j]/fields->counts[i+1][j])+(fields->fluxNum[2][i-1][j]/fields->counts[i-1][j]) );
      fields->vort[2][i][j] = 1./(2.*sP->dX)*( (fields->fluxNum[1][i+1][j]/fields->counts[i+1][j])-(fields->fluxNum[1][i-1][j]/fields->counts[i-1][j])
                                              -(fields->fluxNum[0][i][j+1]/fields->counts[i][j+1])+(fields->fluxNum[0][i][j-1]/fields->counts[i][j-1]) );
    #endif

  }
}
//__________________________________________________________________________________________________________________









#endif
