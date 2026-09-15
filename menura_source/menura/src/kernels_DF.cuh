#ifndef __KERNELSDF_CUH_INCLUDED__   // if x.h hasn't been included yet...
#define __KERNELSDF_CUH_INCLUDED__

#if DF_cst

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "parameters.h"
#include "kernels_device.cuh"


// __global__ void moments_time_k(simu_DF* DF, simu_fields* fields, simu_grid* grid, int ind_it, simu_param* sP)
// {
//
//   int idx = threadIdx.x + blockIdx.x*blockDim.x ;
//
//   int i, l, m, n ;
//
//   if(idx < len_x_cst*len_v_cst*len_v_cst*len_v_cst){
//     i = int( idx/(len_v_cst*len_v_cst*len_v_cst) ) ;
//     l = int( (idx - i*len_v_cst*len_v_cst*len_v_cst)/(len_v_cst*len_v_cst) ) ;
//     m = int( (idx - i*len_v_cst*len_v_cst*len_v_cst - l*len_v_cst*len_v_cst)/len_v_cst ) ;
//     n = idx - i*len_v_cst*len_v_cst*len_v_cst - l*len_v_cst*len_v_cst - m*len_v_cst ;
//     i += 2 ;
//     l += 2 ;
//     m += 2 ;
//     n += 2 ;
//
//     atomicAdd(&fields->density[ind_it][i], DF->DF[i][l][m][n]) ;
//     atomicAdd(&fields->current[0][ind_it][i], grid->vGrid[l]*DF->DF[i][l][m][n]) ;
//     atomicAdd(&fields->current[1][ind_it][i], grid->vGrid[m]*DF->DF[i][l][m][n]) ;
//     atomicAdd(&fields->current[2][ind_it][i], grid->vGrid[n]*DF->DF[i][l][m][n]) ;
//   }
//
// }


__global__ void EB_fields_k(simu_DF* DF, simu_grid* grid, simu_param* sP, int idx_it)
{

  int idx = threadIdx.x + blockIdx.x*blockDim.x ;

  // int i ;

  if (idx<(len_x_cst+4)){

    // i = idx+2 ;

    //__________________________________________________________________________
    // Gyration around homogeneous constant B.
    //
    DF->B[0][idx] = 0.;
    DF->B[1][idx] = 0.;
    DF->B[2][idx] = 1.;
    //
    DF->E[0][idx] = 0.;
    DF->E[1][idx] = 0.;
    DF->E[2][idx] = 0.;
    //__________________________________________________________________________
    // Simplest electrostatic Landau.
    // DF->B[0][idx] = 1.;//sP->B0 ;
    // DF->B[1][idx] = 0. ;
    // DF->B[2][idx] = 0. ;
    // //
    // DF->E[0][idx] = sP->delta_E*sin(sP->k_wave*grid->xGrid[idx] - sP->omega_wave*idx_it*sP->dt) ;
    // DF->E[1][idx] = 0. ;
    // DF->E[2][idx] = 0. ;
    //__________________________________________________________________________
    // Oblique Whistler from Hsieh 2017, B along x.
    // float Psi = sP->omega*simuTime - sP->kvec_para*grid->xGrid[idx] ; // - sP->kvec_perp*p->ry[idx] ;
    // float den = ( cos(Psi)*cos(Psi) + sP->As*sP->As*(1-sP->Ap*tan(sP->theta))*(1-sP->Ap*tan(sP->theta))*sin(Psi)*sin(Psi) + tan(sP->theta)*tan(sP->theta)*cos(Psi)*cos(Psi) ) ;
    // den = pow(den, .5) ;
    // fields->B[1][i] = sP->deltaB/den ;
    // fields->B[2][i] = fields->B[1][i]*sP->As*(1-sP->Ap*tan(sP->theta)) ;
    // fields->B[0][i] = fields->B[1][i]*tan(sP->theta) ;
    // //
    // fields->E[1][i] = fields->B[1][i]*sP->vPhase_para*sP->As ;
    // fields->E[2][i] = fields->B[1][i]*sP->vPhase_para ;
    // fields->E[0][i] = fields->B[1][i]*sP->vPhase_para*sP->As*sP->Ap ;
    // //
    // fields->B[1][i] =  cos(Psi)*fields->B[1][i] ;
    // fields->B[2][i] =  sin(Psi)*fields->B[2][i] ;
    // fields->B[0][i] = -cos(Psi)*fields->B[0][i] + sP->B0 ;
    // fields->E[1][i] =  sin(Psi)*fields->E[1][i] ;
    // fields->E[2][i] = -cos(Psi)*fields->E[2][i] ;
    // fields->E[0][i] =  sin(Psi)*fields->E[0][i] ;
    //__________________________________________________________________________
    // Electrostatic, B0 out of plane (y), Ew perpendicular.
    // float x = grid->xGrid[idx];
    // float x_min = .25*grid->xMax + idx_it*sP->dt*sP->omega_wave/sP->k_wave;
    // //
    // DF->B[0][idx] = 0.;
    // DF->B[1][idx] = 0.;
    // DF->B[2][idx] = 1. + sP->delta_B*( sin(sP->k_wave* - sP->omega_wave*idx_it*sP->dt) );
    // //
    // DF->E[0][idx] = 0.;
    // DF->E[1][idx] = 0.;
    // DF->E[2][idx] = 0.;
    // if (x>x_min && x<x_min+sP->lambda_wave){
    //   DF->E[1][idx] = sP->delta_E*( sin(sP->k_wave*x - sP->omega_wave*idx_it*sP->dt) );
    // }
    // if (x>x_min-sP->lambda_wave/4 && x<x_min+3./4.*sP->lambda_wave){
    //   DF->E[0][idx] = sP->delta_E*( cos(sP->k_wave*x - sP->omega_wave*idx_it*sP->dt) );
    // }
    //__________________________________________________________________________
    // Electrostatic, B0 out of plane (y), Ew perpendicular.
    // fields->B[0][i] = sP->B0 ;
    // fields->B[1][i] = 0. ;
    // fields->B[2][i] = 0.;//sP->B0 ;
    // //
    // fields->E[0][i] = 0. ;
    // fields->E[1][i] = 0. ;
    // fields->E[2][i] = 0. ;
  }

}


__global__ void van_leer_rx_k(simu_DF* DF_in, simu_DF* DF_out, simu_grid* grid, simu_param* sP, state_solver* staSol)
{
  int idx = threadIdx.x + blockIdx.x*blockDim.x ;

  int i, l, m, n ;

  if(idx < len_x_cst*len_v_cst*len_v_cst*len_v_cst){
    i = int( idx/(len_v_cst*len_v_cst*len_v_cst) ) ;
    l = int( (idx - i*len_v_cst*len_v_cst*len_v_cst)/(len_v_cst*len_v_cst) ) ;
    m = int( (idx - i*len_v_cst*len_v_cst*len_v_cst - l*len_v_cst*len_v_cst)/len_v_cst ) ;
    n = idx - i*len_v_cst*len_v_cst*len_v_cst - l*len_v_cst*len_v_cst - m*len_v_cst ;
    i += 2 ;
    l += 2 ;
    m += 2 ;
    n += 2 ;

    float dlt = (grid->vGrid[l]*sP->dt/sP->dX) ;

    if (dlt>=1.){
      printf("van_leer_rx_k: dlt>=1. (%.4e)\n", dlt) ;
      staSol->break_solver = true ;
    }

    if (dlt >= 0){
      DF_out->DF[i][l][m][n] =         -dlt/4. *(1.-dlt) *DF_in->DF[i-2][l][m][n]
                               + (dlt + dlt/4. *(1.-dlt))*DF_in->DF[i-1][l][m][n]
                                 + (1 + dlt/4.)*(1.-dlt) *DF_in->DF[i  ][l][m][n]
                                       -dlt/4. *(1.-dlt) *DF_in->DF[i+1][l][m][n] ;
      // if (DF_out->DF[i][l][m][n]<0){
      //  printf("+delta rx, %i %i %i %i, %.4e, %.4e %.4e %.4e\n", i, l, m, n, DF_in->DF[i-2][l][m][n], DF_in->DF[i-1][l][m][n], DF_in->DF[i][l][m][n], DF_in->DF[i+1][l][m][n]);
      //  // DF_out->DF[i][l][m][n] = 0.;
      // }
    }
    else{
      dlt *= -1.;
      DF_out->DF[i][l][m][n] =         -dlt/4. *(1.-dlt) *DF_in->DF[i+2][l][m][n]
                               + (dlt + dlt/4. *(1.-dlt))*DF_in->DF[i+1][l][m][n]
                                 + (1 + dlt/4.)*(1.-dlt) *DF_in->DF[i  ][l][m][n]
                                       -dlt/4. *(1.-dlt) *DF_in->DF[i-1][l][m][n] ;
      // if (DF_out->DF[i][l][m][n]<0){
      //  printf("-delta rx, %i %i %i %i, %.4e, %.4e %.4e %.4e\n", i, l, m, n, DF_in->DF[i+2][l][m][n], DF_in->DF[i+1][l][m][n], DF_in->DF[i][l][m][n], DF_in->DF[i-1][l][m][n]);
      //  // DF_out->DF[i][l][m][n] = 0.;
      // }
    }

    // if (DF_out->DF[i][l][m][n]<0){
    //   DF_out->DF[i][l][m][n] = 0.;
    // }
  }
}

// __global__ void van_leer_coef_k(float dlt, float A[4])
// {
//   dlt = abs(dlt);
//   A[0] =     -1*dlt/4. *(1.-dlt) ;
//   A[1] = (dlt + dlt/4. *(1.-dlt));
//   A[2] =   (1 + dlt/4.)*(1.-dlt) ;
//   A[3] =       -dlt/4. *(1.-dlt) ;
// }

__global__ void van_leer_vx_k(simu_DF* DF_in, simu_DF* DF_out, simu_fields* fields, simu_grid* grid, simu_param* sP, float simuTime, state_solver* staSol){

  int idx = threadIdx.x + blockIdx.x*blockDim.x ;

  if(idx < len_x_cst*len_v_cst*len_v_cst*len_v_cst){
    int i, l, m, n ;
    i = int( idx/(len_v_cst*len_v_cst*len_v_cst) ) ;
    l = int( (idx - i*len_v_cst*len_v_cst*len_v_cst)/(len_v_cst*len_v_cst) ) ;
    m = int( (idx - i*len_v_cst*len_v_cst*len_v_cst - l*len_v_cst*len_v_cst)/len_v_cst ) ;
    n = idx - i*len_v_cst*len_v_cst*len_v_cst - l*len_v_cst*len_v_cst - m*len_v_cst ;
    i += 2 ;
    l += 2 ;
    m += 2 ;
    n += 2 ;

    float dlt = (DF_in->E[0][i] + grid->vGrid[m]*DF_in->B[2][i] - grid->vGrid[n]*DF_in->B[1][i])*sP->dt/sP->dV ;
    //

    if (dlt>=1.){
      printf("van_leer_vx_k: dlt>=1.") ;
      staSol->break_solver = true ;
    }
    if (dlt >= 0){
      // van_leer_coef_k<<< 1, 1 >>>(dlt, &A);
      float Am2, Am1, Ap0, Ap1;
      Am2 =     -1*dlt/4. *(1.-dlt) ;
      Am1 = (dlt + dlt/4. *(1.-dlt));
      Ap0 =   (1 + dlt/4.)*(1.-dlt) ;
      Ap1 =       -dlt/4. *(1.-dlt) ;
      // if (i==10 && l==42 && m==40 && n==24){
      //   DF_in->DF[i][l+1][m][n] = 1.1e-20;
      // }
      DF_out->DF[i][l][m][n] =   Am2*DF_in->DF[i][l-2][m][n]
                               + Am1*DF_in->DF[i][l-1][m][n]
                               + Ap0*DF_in->DF[i][l  ][m][n]
                               + Ap1*DF_in->DF[i][l+1][m][n] ;
      // if (i==10 && l==42 && m==40 && n==24){
      //   // if (DF_out->DF[i][l][m][n]<0){
      //     printf("%4e %.4e\n", DF_in->DF[i][l][m][n], DF_out->DF[i][l][m][n]);
      //     printf("%.4e\n", dlt);
      //     printf("%.4e %.4e %.4e %.4e %.4e\n", a0, a1, a2, a3, a0+a1+a2+a3);
      //     printf("%.4e %.4e %.4e %.4e\n",  DF_in->DF[i][l-2][m][n], DF_in->DF[i][l-1][m][n], DF_in->DF[i][l][m][n], DF_in->DF[i][l+1][m][n]);
      //     printf("%.4e %.4e %.4e %.4e\n", a0*DF_in->DF[i][l-2][m][n], a1*DF_in->DF[i][l-1][m][n], a2*DF_in->DF[i][l  ][m][n], a3*DF_in->DF[i][l+1][m][n]);
      //     printf("%.4e\n", a0*DF_in->DF[i][l-2][m][n] + a1*DF_in->DF[i][l-1][m][n] + a2*DF_in->DF[i][l  ][m][n] + a3*DF_in->DF[i][l+1][m][n]);
      //    // DF_out->DF[i][l][m][n] = 0.;
      //   // }
      // }
    }
    else{
      float Am1, Ap0, Ap1, Ap2;
      dlt *= -1 ;
      Am1 =       -dlt/4. *(1.-dlt) ;
      Ap0 =   (1 + dlt/4.)*(1.-dlt) ;
      Ap1 = (dlt + dlt/4. *(1.-dlt));
      Ap2 =     -1*dlt/4. *(1.-dlt) ;
      DF_out->DF[i][l][m][n] =   Ap2*DF_in->DF[i][l+2][m][n]
                               + Ap1*DF_in->DF[i][l+1][m][n]
                               + Ap0*DF_in->DF[i][l  ][m][n]
                               + Am1*DF_in->DF[i][l-1][m][n] ;
      // if (DF_out->DF[i][l][m][n]<0){
      //  printf("-delta vx %.4e, %i %.4e, %i %.4e, %i %.4e, %i %.4e\n", dlt, l+2, DF_in->DF[i][l+2][m][n], l+1, DF_in->DF[i][l+1][m][n], l, DF_in->DF[i][l][m][n], l-1, DF_in->DF[i][l-1][m][n]);
      //  // DF_out->DF[i][l][m][n] = 0.;
      // }
    }

    // if (DF_out->DF[i][l][m][n]<0){
    //   // printf("Negative DF, %.4e %.4e %.4e\n", DF_in->DF[i][l+1][m][n], DF_in->DF[i][l][m][n], DF_in->DF[i][l-1][m][n]);
    //   DF_out->DF[i][l][m][n] = 0.;
    // }
  }
}

__global__ void van_leer_vy_k(simu_DF* DF_in, simu_DF* DF_out, simu_fields* fields, simu_grid* grid, simu_param* sP, float simuTime, state_solver* staSol){

  int idx = threadIdx.x + blockIdx.x*blockDim.x ;

  int i, l, m, n ;

  if(idx < len_x_cst*len_v_cst*len_v_cst*len_v_cst){
    i = int( idx/(len_v_cst*len_v_cst*len_v_cst) ) ;
    l = int( (idx - i*len_v_cst*len_v_cst*len_v_cst)/(len_v_cst*len_v_cst) ) ;
    m = int( (idx - i*len_v_cst*len_v_cst*len_v_cst - l*len_v_cst*len_v_cst)/len_v_cst ) ;
    n = idx - i*len_v_cst*len_v_cst*len_v_cst - l*len_v_cst*len_v_cst - m*len_v_cst ;
    i += 2 ;
    l += 2 ;
    m += 2 ;
    n += 2 ;

    float dlt = .5*(DF_in->E[1][i] + grid->vGrid[n]*DF_in->B[0][i] - grid->vGrid[l]*DF_in->B[2][i])*sP->dt/sP->dV ;

    if (dlt>=1.){
      printf("van_leer_vy_k: dlt>=1.") ;
      staSol->break_solver = true ;
    }

    if (dlt >= 0){
      float Am2, Am1, Ap0, Ap1;
      Am2 =     -1*dlt/4. *(1.-dlt) ;
      Am1 = (dlt + dlt/4. *(1.-dlt));
      Ap0 =   (1 + dlt/4.)*(1.-dlt) ;
      Ap1 =       -dlt/4. *(1.-dlt) ;
      // if (i==10 && l==42 && m==40 && n==24){
      //   DF_in->DF[i][l+1][m][n] = 1.1e-20;
      // }
      DF_out->DF[i][l][m][n] =   Am2*DF_in->DF[i][l][m-2][n]
                               + Am1*DF_in->DF[i][l][m-1][n]
                               + Ap0*DF_in->DF[i][l][m  ][n]
                               + Ap1*DF_in->DF[i][l][m+1][n] ;
      // if (DF_out->DF[i][l][m][n]<0){
      //  printf("-delta vy, %i %i %i %i, %.4e, %.4e %.4e %.4e\n", i, l, m, n, DF_in->DF[i][l][m-2][n], DF_in->DF[i][l][m-1][n], DF_in->DF[i][l][m][n], DF_in->DF[i][l][m+1][n]);
      //  // DF_out->DF[i][l][m][n] = 0.;
      // }
    }
    else{
      float Am1, Ap0, Ap1, Ap2;
      dlt *= -1 ;
      Am1 =       -dlt/4. *(1.-dlt) ;
      Ap0 =   (1 + dlt/4.)*(1.-dlt) ;
      Ap1 = (dlt + dlt/4. *(1.-dlt));
      Ap2 =     -1*dlt/4. *(1.-dlt) ;
      DF_out->DF[i][l][m][n] =   Ap2*DF_in->DF[i][l][m+2][n]
                               + Ap1*DF_in->DF[i][l][m+1][n]
                               + Ap0*DF_in->DF[i][l][m  ][n]
                               + Am1*DF_in->DF[i][l][m-1][n] ;
      // if (DF_out->DF[i][l][m][n]<0){
      //  printf("-delta vy, %i %i %i %i, %.4e, %.4e %.4e %.4e\n", i, l, m, n, DF_in->DF[i][l][m+2][n], DF_in->DF[i][l][m+1][n], DF_in->DF[i][l][m][n], DF_in->DF[i][l][m-1][n]);
      //  // DF_out->DF[i][l][m][n] = 0.;
      // }
    }

    // if (DF_out->DF[i][l][m][n]<0){
    //   // printf("Negative DF, %.4e %.4e %.4e\n", DF_in->DF[i][l][m+1][n], DF_in->DF[i][l][m][n], DF_in->DF[i][l][m-1][n]);
    //   DF_out->DF[i][l][m][n] = 0.;
    // }
  }
}

__global__ void van_leer_vz_k(simu_DF* DF_in, simu_DF* DF_out, simu_fields* fields, simu_grid* grid, simu_param* sP, float simuTime, state_solver* staSol){

  int idx = threadIdx.x + blockIdx.x*blockDim.x ;

  int i, l, m, n ;

  if(idx < len_x_cst*len_v_cst*len_v_cst*len_v_cst){
    i = int( idx/(len_v_cst*len_v_cst*len_v_cst) ) ;
    l = int( (idx - i*len_v_cst*len_v_cst*len_v_cst)/(len_v_cst*len_v_cst) ) ;
    m = int( (idx - i*len_v_cst*len_v_cst*len_v_cst - l*len_v_cst*len_v_cst)/len_v_cst ) ;
    n = idx - i*len_v_cst*len_v_cst*len_v_cst - l*len_v_cst*len_v_cst - m*len_v_cst ;
    i += 2 ;
    l += 2 ;
    m += 2 ;
    n += 2 ;

    float dlt = (DF_in->E[2][i] + grid->vGrid[l]*DF_in->B[1][i] - grid->vGrid[m]*DF_in->B[0][i])*sP->dt/sP->dV ;

    if (dlt>=1.){
      printf("van_leer_vz_k: dlt>=1.") ;
      staSol->break_solver = true ;
    }
    if (dlt >= 0){
      float Am2, Am1, Ap0, Ap1;
      Am2 =     -1*dlt/4. *(1.-dlt) ;
      Am1 = (dlt + dlt/4. *(1.-dlt));
      Ap0 =   (1 + dlt/4.)*(1.-dlt) ;
      Ap1 =       -dlt/4. *(1.-dlt) ;
      // if (i==10 && l==42 && m==40 && n==24){
      //   DF_in->DF[i][l+1][m][n] = 1.1e-20;
      // }
      DF_out->DF[i][l][m][n] =   Am2*DF_in->DF[i][l][m][n-2]
                               + Am1*DF_in->DF[i][l][m][n-1]
                               + Ap0*DF_in->DF[i][l][m][n  ]
                               + Ap1*DF_in->DF[i][l][m][n+1] ;
      // if (DF_out->DF[i][l][m][n]<0){
      //  printf("Negative DF positive delta, vy, %.4e, %.4e %.4e %.4e\n", DF_in->DF[i][l][m][n-2], DF_in->DF[i][l][m][n-1], DF_in->DF[i][l][m][n], DF_in->DF[i][l][m][n+1]);
      //  // DF_out->DF[i][l][m][n] = 0.;
      // }
    }
    else{
      float Am1, Ap0, Ap1, Ap2;
      dlt *= -1 ;
      Am1 =       -dlt/4. *(1.-dlt) ;
      Ap0 =   (1 + dlt/4.)*(1.-dlt) ;
      Ap1 = (dlt + dlt/4. *(1.-dlt));
      Ap2 =     -1*dlt/4. *(1.-dlt) ;
      DF_out->DF[i][l][m][n] =   Ap2*DF_in->DF[i][l][m][n+2]
                               + Ap1*DF_in->DF[i][l][m][n+1]
                               + Ap0*DF_in->DF[i][l][m][n  ]
                               + Am1*DF_in->DF[i][l][m][n-1] ;
      // if (DF_out->DF[i][l][m][n]<0){
      //  printf("Negative DF negative delta, vy, %.4e, %.4e %.4e %.4e\n", DF_in->DF[i][l][m][n+2], DF_in->DF[i][l][m][n+1], DF_in->DF[i][l][m][n], DF_in->DF[i][l][m][n-1]);
      //  // DF_out->DF[i][l][m][n] = 0.;
      // }
    }

    // if (DF_out->DF[i][l][m][n]<0){
    //   // printf("Negative DF, %.4e %.4e %.4e\n", DF_in->DF[i][l][m][n+1], DF_in->DF[i][l][m][n], DF_in->DF[i][l][m][n-1]);
    //   DF_out->DF[i][l][m][n] = 0.;
    // }
  }
}
//
//
//
//
// __global__ void van_leer_vx_add_k(simu_DF* DF_in, simu_DF* DFb, simu_grid* grid, simu_param* sP, float simuTime){
//
//   int idx = threadIdx.x + blockIdx.x*blockDim.x ;
//
//   int i, l, m, n ;
//
//   if(idx < len_x_cst*len_v_cst*len_v_cst*len_v_cst){
//     i = int( idx/(len_v_cst*len_v_cst*len_v_cst) ) ;
//     l = int( (idx - i*len_v_cst*len_v_cst*len_v_cst)/(len_v_cst*len_v_cst) ) ;
//     m = int( (idx - i*len_v_cst*len_v_cst*len_v_cst - l*len_v_cst*len_v_cst)/len_v_cst ) ;
//     n = idx - i*len_v_cst*len_v_cst*len_v_cst - l*len_v_cst*len_v_cst - m*len_v_cst ;
//     i += 2 ;
//     l += 2 ;
//     m += 2 ;
//     n += 2 ;
//
//     // DFb->DF[i][l][m][n] = DFa->DF[i][l][m][n] ;
//
//     // float Ex  = 0. ; //sP->deltaE*cos(-sP->kvec_para*grid->xGrid[i] + sP->omega*simuTime) ;
//     // float Bz  = 0. ; //sP->deltaB*sin(sP->kvec_para*grid->xGrid[i] - sP->omega*simuTime) ;
//     // float By  = 0. ; //sP->deltaB*cos(sP->kvec_para*grid->xGrid[i] - sP->omega*simuTime) ;
//     // float dlt = q/mass_e*(Ex + grid->vGrid[m]*Bz - grid->vGrid[n]*By)*sP->dt/sP->dV ; //12.67*
//     float dlt = (1.e10*sP->dt/sP->dV) ;
//     // if (i==66 && l==26){
//     //   printf("%.4e\n", dlt) ;
//     // }
//     if (dlt>=1.){
//       printf("van_leer_vx_k: dlt>=1.") ;
//     }
//     if (dlt >= 0){
//       DFb->DF[i][l][m][n] +=         1./2.*(-dlt/4. *(1.-dlt) *DF_in->DF[i][l-2][m][n]
//                             + (dlt + dlt/4. *(1.-dlt))*DFa->DF[i][l-1][m][n]
//                               + (1 + dlt/4.)*(1.-dlt) *DFa->DF[i][l  ][m][n]
//                                 -    dlt/4. *(1.-dlt) *DFa->DF[i][l+1][m][n]) ;
//     }
//     else{
//       dlt *= -1 ;
//       DFb->DF[i][l][m][n] +=         1./2.*(-dlt/4. *(1.-dlt) *DFa->DF[i][l+2][m][n]
//                             + (dlt + dlt/4. *(1.-dlt))*DFa->DF[i][l+1][m][n]
//                               + (1 + dlt/4.)*(1.-dlt) *DFa->DF[i][l  ][m][n]
//                                 -    dlt/4. *(1.-dlt) *DFa->DF[i][l-1][m][n]) ;
//     }
//   }
// }






__global__ void copy_guard_rx_k(simu_DF* DF, simu_param* sP){

  int idx = threadIdx.x + blockIdx.x*blockDim.x ;

  int l, m, n ;

  if (idx < len_v_cst*len_v_cst*len_v_cst){
    l = int( idx/(len_v_cst*len_v_cst) ) ;
    m = int( (idx - l*len_v_cst*len_v_cst)/len_v_cst ) ;
    n = idx - l*(len_v_cst*len_v_cst) - m*len_v_cst ;
    l += 2 ;
    m += 2 ;
    n += 2 ;

    DF->DF[len_x_cst+2][l][m][n] = DF->DF[2][l][m][n] ;
    DF->DF[len_x_cst+3][l][m][n] = DF->DF[3][l][m][n] ;
    DF->DF[0][l][m][n] = DF->DF[len_x_cst][l][m][n] ;
    DF->DF[1][l][m][n] = DF->DF[len_x_cst+1][l][m][n] ;
  }
}


__global__ void transfer_guard_rx_k(simu_DF* DF_in, simu_DF* DF_out, simu_param* sP){

  int idx = threadIdx.x + blockIdx.x*blockDim.x ;

  int l, m, n ;

  if (idx < len_v_cst*len_v_cst*len_v_cst){
    l = int( idx/(len_v_cst*len_v_cst) );
    m = int( (idx - l*len_v_cst*len_v_cst)/len_v_cst );
    n = idx - l*len_v_cst*len_v_cst - m*len_v_cst;
    l += 2 ;
    m += 2 ;
    n += 2 ;

    DF_out->DF[len_x_cst+2][l][m][n] = DF_in->DF[len_x_cst+2][l][m][n] ;
    DF_out->DF[len_x_cst+3][l][m][n] = DF_in->DF[len_x_cst+3][l][m][n] ;
    DF_out->DF[0][l][m][n]          = DF_in->DF[0][l][m][n] ;
    DF_out->DF[1][l][m][n]          = DF_in->DF[1][l][m][n] ;
  }
}

__global__ void transfer_all_k(simu_DF* DF_in, simu_DF* DF_out, simu_grid* grid, simu_param* sP){

  int idx = threadIdx.x + blockIdx.x*blockDim.x ;

  int i, l, m, n ;

  if(idx < (len_x_cst+4)*len_v_cst*len_v_cst*len_v_cst){
    i = int( idx/(len_v_cst*len_v_cst*len_v_cst) ) ;
    l = int( (idx - i*len_v_cst*len_v_cst*len_v_cst)/(len_v_cst*len_v_cst) ) ;
    m = int( (idx - i*len_v_cst*len_v_cst*len_v_cst - l*len_v_cst*len_v_cst)/len_v_cst ) ;
    n = idx - i*len_v_cst*len_v_cst*len_v_cst - l*len_v_cst*len_v_cst - m*len_v_cst ;
    l += 2 ;
    m += 2 ;
    n += 2 ;

    DF_out->DF[i][l][m][n] = DF_in->DF[i][l][m][n] ;
  }
}

#endif
#endif
