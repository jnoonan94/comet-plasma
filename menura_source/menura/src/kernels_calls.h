#ifndef __KERNELCALLS_H_INCLUDED__   // if x.h hasn't been included yet...
#define __KERNELCALLS_H_INCLUDED__

#include "kernels_fields.cuh"
#include "kernels_particles.cuh"
#include "kernels_HK.cuh"
#include "kernels_particles_to_grid.cuh"
#include "kernels_tricks.cuh"

#include "mpi.h"



void count_rms(simu_B_field* B, simu_fields* fields, root_mean_sqr* rms, int idx_save,
                int threadsPerBlock, simu_param* sP_h, simu_param* sP_d)
{

  reset_rms_k<<< 1, 1>>>(rms);
  rms_k <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>> (rms, sP_d, B->B[0]);
  sum_rms_k <<< 1, 1 >>> (rms, sP_d, rms->B[0], idx_save);
  rms_k <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>> (rms, sP_d, B->B[1]);
  sum_rms_k <<< 1, 1 >>> (rms, sP_d, rms->B[1], idx_save);
  rms_k <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>> (rms, sP_d, B->B[2]);
  sum_rms_k <<< 1, 1 >>> (rms, sP_d, rms->B[2], idx_save);

  rms_k <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>> (rms, sP_d, fields->E[0]);
  sum_rms_k <<< 1, 1 >>> (rms, sP_d, rms->E[0], idx_save);
  rms_k <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>> (rms, sP_d, fields->E[1]);
  sum_rms_k <<< 1, 1 >>> (rms, sP_d, rms->E[1], idx_save);
  rms_k <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>> (rms, sP_d, fields->E[2]);
  sum_rms_k <<< 1, 1 >>> (rms, sP_d, rms->E[2], idx_save);

  rms_k <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>> (rms, sP_d, fields->Ji[0]);
  sum_rms_k <<< 1, 1 >>> (rms, sP_d, rms->Ji[0], idx_save);
  rms_k <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>> (rms, sP_d, fields->Ji[1]);
  sum_rms_k <<< 1, 1 >>> (rms, sP_d, rms->Ji[1], idx_save);
  rms_k <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>> (rms, sP_d, fields->Ji[2]);
  sum_rms_k <<< 1, 1 >>> (rms, sP_d, rms->Ji[2], idx_save);

  rms_k <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>> (rms, sP_d, fields->J_tot[0]);
  sum_rms_k <<< 1, 1 >>> (rms, sP_d, rms->J_tot[0], idx_save);
  rms_k <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>> (rms, sP_d, fields->J_tot[1]);
  sum_rms_k <<< 1, 1 >>> (rms, sP_d, rms->J_tot[1], idx_save);
  rms_k <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>> (rms, sP_d, fields->J_tot[2]);
  sum_rms_k <<< 1, 1 >>> (rms, sP_d, rms->J_tot[2], idx_save);

  rms_k <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>> (rms, sP_d, fields->vort[0]);
  sum_rms_k <<< 1, 1 >>> (rms, sP_d, rms->vort[0], idx_save);
  rms_k <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>> (rms, sP_d, fields->vort[1]);
  sum_rms_k <<< 1, 1 >>> (rms, sP_d, rms->vort[1], idx_save);
  rms_k <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>> (rms, sP_d, fields->vort[2]);
  sum_rms_k <<< 1, 1 >>> (rms, sP_d, rms->vort[2], idx_save);
}


//_________________________________________________________________________________________
//
// Field related kernel calls.
//

void scalar_boundaries
#if NB_DIM==2
(float G[len_x_cst+4][len_y_cst+4], simu_param* sP)
#elif NB_DIM==3
(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4], simu_param* sP)
#endif
{

  #if NB_DIM==2
    int n_x = len_y_cst+4;
    int n_y = len_x_cst+4;
  #elif NB_DIM==3
    int n_x = (len_y_cst+4)*(len_z_cst+4);
    int n_y = (len_x_cst+4)*(len_z_cst+4);
    int n_z = (len_x_cst+4)*(len_y_cst+4);
  #endif

  #if (obstacle_cst && !periodic_yz_cst)
    // Mercury.
    extrapolate_X_k   <<< 1 + n_x/tpb_cst, tpb_cst >>>(G);
    if (sP->mpi_nb_proc_y==1 || sP->mpi_rank_y==0){
      extrapolate_Y_down_k <<< 1 + n_y/tpb_cst, tpb_cst >>>(G);
    }
    if (sP->mpi_nb_proc_y==1 || sP->mpi_rank_y==sP->mpi_nb_proc_y-1){
      extrapolate_Y_up_k   <<< 1 + n_y/tpb_cst, tpb_cst >>>(G);
    }
    #if NB_DIM==3
      if (sP->mpi_nb_proc_z==1 || sP->mpi_rank_z==0){
        extrapolate_Z_down_k <<< 1 + n_z/tpb_cst, tpb_cst >>>(G);
      }
      if (sP->mpi_nb_proc_z==1 || sP->mpi_rank_z==sP->mpi_nb_proc_z-1){
        extrapolate_Z_up_k   <<< 1 + n_z/tpb_cst, tpb_cst >>>(G);
      }
    #endif
  #elif obstacle_cst
    // Comet and Moon.
    extrapolate_X_k   <<< 1 + n_x/tpb_cst, tpb_cst >>>(G);
    if (sP->mpi_nb_proc_y==1){
      copy_guard_Y_k<<< 1 + n_y/tpb_cst, tpb_cst >>>(G);
    }
    #if NB_DIM==3
      if (sP->mpi_nb_proc_z==1){
        copy_guard_Z_k<<< 1 + n_z/tpb_cst, tpb_cst >>>(G);
      }
    #endif
  #else
    // Plasma box, periodic in all directions.
    copy_guard_X_k<<< 1 + n_x/tpb_cst, tpb_cst >>>(G);
    if (sP->mpi_nb_proc_y==1){
      copy_guard_Y_k<<< 1 + n_y/tpb_cst, tpb_cst >>>(G);
    }
    #if NB_DIM==3
      if (sP->mpi_nb_proc_z==1){
        copy_guard_Z_k<<< 1 + n_z/tpb_cst, tpb_cst >>>(G);
      }
    #endif
  #endif

}

void scalar_boundaries_add
#if NB_DIM==2
(float G[len_x_cst+4][len_y_cst+4], simu_param* sP)
#elif NB_DIM==3
(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4], simu_param* sP)
#endif
{

  #if NB_DIM==2
    int n_x = len_y_cst+4;
    int n_y = len_x_cst+4;
  #elif NB_DIM==3
    int n_x = (len_y_cst+4)*(len_z_cst+4);
    int n_y = (len_x_cst+4)*(len_z_cst+4);
    int n_z = (len_x_cst+4)*(len_y_cst+4);
  #endif

  #if (obstacle_cst && !periodic_yz_cst)
    // Mercury, Mars. Not periodic along y.
    extrapolate_X_k   <<< 1 + n_x/tpb_cst, tpb_cst >>>(G);
    if (sP->mpi_nb_proc_y==1 || sP->mpi_rank_y==0){
      extrapolate_Y_down_k <<< 1 + n_y/tpb_cst, tpb_cst >>>(G);
    }
    if (sP->mpi_nb_proc_y==1 || sP->mpi_rank_y==sP->mpi_nb_proc_y-1){
      extrapolate_Y_up_k <<< 1 + n_y/tpb_cst, tpb_cst >>>(G);
    }
    #if NB_DIM==3
      if (sP->mpi_nb_proc_z==1 || sP->mpi_rank_z==0){
        extrapolate_Z_down_k <<< 1 + n_z/tpb_cst, tpb_cst >>>(G);
      }
      if (sP->mpi_nb_proc_z==1 || sP->mpi_rank_z==sP->mpi_nb_proc_z-1){
        extrapolate_Z_up_k <<< 1 + n_z/tpb_cst, tpb_cst >>>(G);
      }
    #endif
  #elif obstacle_cst
    // Comet and Moon.
    extrapolate_X_k   <<< 1 + n_x/tpb_cst, tpb_cst >>>(G);
    if (sP->mpi_nb_proc_y==1){
      add_copy_guard_Y_k<<< 1 + n_y/tpb_cst, tpb_cst >>>(G);
    }
    #if NB_DIM==3
      if (sP->mpi_nb_proc_z==1){
        add_copy_guard_Z_k<<< 1 + n_z/tpb_cst, tpb_cst >>>(G);
      }
    #endif
  #else
    // Plasma box, periodic in all directions.
    add_copy_guard_X_k<<< 1 + n_x/tpb_cst, tpb_cst >>>(G);
    if (sP->mpi_nb_proc_y==1){
      add_copy_guard_Y_k<<< 1 + n_y/tpb_cst, tpb_cst >>>(G);
    }
    #if NB_DIM==3
      if (sP->mpi_nb_proc_z==1){
        add_copy_guard_Z_k<<< 1 + n_z/tpb_cst, tpb_cst >>>(G);
      }
    #endif
  #endif

}


void MPI_receive_send_guard_cells_down_y(simu_fields* fields, simu_param* sP, MPI_Comm comm)
{

  MPI_Request down_req;
  MPI_Status stat;
  //
  int down_proc_y = (sP->mpi_rank_y-1+sP->mpi_nb_proc_y)%sP->mpi_nb_proc_y;
  int up_proc_y   = (sP->mpi_rank_y+1)%sP->mpi_nb_proc_y;
  int down_proc_lin = down_proc_y*sP->mpi_nb_proc_z + sP->mpi_rank_z;
  int up_proc_lin   =   up_proc_y*sP->mpi_nb_proc_z + sP->mpi_rank_z;
  //
  cudaDeviceSynchronize();
  MPI_Irecv(fields->buff_reci_1d_y, sP->len_buff_field_y, MPI_FLOAT, up_proc_lin,   1, comm, &down_req);
  MPI_Send( fields->buff_send_1d_y, sP->len_buff_field_y, MPI_FLOAT, down_proc_lin, 1, comm);
  MPI_Wait(&down_req, &stat);
  cudaDeviceSynchronize();

}
void MPI_receive_send_guard_cells_up_y(simu_fields* fields, simu_param* sP, MPI_Comm comm)
{

  MPI_Request up_req;
  MPI_Status stat;
  //
  int down_proc_y = (sP->mpi_rank_y-1+sP->mpi_nb_proc_y)%sP->mpi_nb_proc_y;
  int up_proc_y   = (sP->mpi_rank_y+1)%sP->mpi_nb_proc_y;
  int down_proc_lin = down_proc_y*sP->mpi_nb_proc_z + sP->mpi_rank_z;
  int up_proc_lin   =   up_proc_y*sP->mpi_nb_proc_z + sP->mpi_rank_z;
  //
  cudaDeviceSynchronize();
  MPI_Irecv(fields->buff_reci_1d_y, sP->len_buff_field_y, MPI_FLOAT, down_proc_lin, 1, comm, &up_req);
  MPI_Send( fields->buff_send_1d_y, sP->len_buff_field_y, MPI_FLOAT, up_proc_lin,   1, comm);
  MPI_Wait(&up_req, &stat);
  cudaDeviceSynchronize();

}
#if NB_DIM==3
  void MPI_receive_send_guard_cells_down_z(simu_fields* fields, simu_param* sP, MPI_Comm comm)
  {

    MPI_Request down_req;
    MPI_Status stat;
    //
    int down_proc_z = (sP->mpi_rank_z-1+sP->mpi_nb_proc_z)%sP->mpi_nb_proc_z;
    int up_proc_z   = (sP->mpi_rank_z+1)%sP->mpi_nb_proc_z;
    int down_proc_lin = sP->mpi_rank_y*sP->mpi_nb_proc_z + down_proc_z;
    int up_proc_lin   = sP->mpi_rank_y*sP->mpi_nb_proc_z + up_proc_z;
    //
    cudaDeviceSynchronize();
    MPI_Irecv(fields->buff_reci_1d_z, sP->len_buff_field_z, MPI_FLOAT, up_proc_lin,   1, comm, &down_req);
    MPI_Send( fields->buff_send_1d_z, sP->len_buff_field_z, MPI_FLOAT, down_proc_lin, 1, comm);
    MPI_Wait(&down_req, &stat);
    cudaDeviceSynchronize();

  }
  void MPI_receive_send_guard_cells_up_z(simu_fields* fields, simu_param* sP, MPI_Comm comm)
  {

    MPI_Request up_req;
    MPI_Status stat;
    //
    int down_proc_z = (sP->mpi_rank_z-1+sP->mpi_nb_proc_z)%sP->mpi_nb_proc_z;
    int up_proc_z   = (sP->mpi_rank_z+1)%sP->mpi_nb_proc_z;
    int down_proc_lin = sP->mpi_rank_y*sP->mpi_nb_proc_z + down_proc_z;
    int up_proc_lin   = sP->mpi_rank_y*sP->mpi_nb_proc_z + up_proc_z;
    //
    cudaDeviceSynchronize();
    MPI_Irecv(fields->buff_reci_1d_z, sP->len_buff_field_z, MPI_FLOAT, down_proc_lin, 1, comm, &up_req);
    MPI_Send( fields->buff_send_1d_z, sP->len_buff_field_z, MPI_FLOAT, up_proc_lin,   1, comm);
    MPI_Wait(&up_req, &stat);
    cudaDeviceSynchronize();

  }
#endif


void MPI_communicate_scalar
#if NB_DIM==2
(float G[len_x_cst+4][len_y_cst+4], simu_fields* fields,
    simu_param* sP, MPI_Comm comm, bool copy_long=true)
#elif NB_DIM==3
(float G[len_x_cst+4][len_y_cst+4][len_z_cst+4], simu_fields* fields,
    simu_param* sP, MPI_Comm comm, bool copy_long=true)
#endif
{
  //
  #if (!periodic_yz_cst)
    //
    if (sP->mpi_nb_proc_y>1){
      fill_buff_down_y_k <<< 1+sP->len_buff_field_y/tpb_cst, tpb_cst >>> (fields, G);
      MPI_receive_send_guard_cells_down_y(fields, sP, comm);
      if (sP->mpi_rank_y<(sP->mpi_nb_proc_y-1)){
        if (copy_long){
          add_buff_down_y_k  <<< 1+sP->len_buff_field_y/tpb_cst, tpb_cst >>> (fields, G);
        }
        else {
          copy_buff_down_short_y_k<<< 1+sP->len_buff_field_y/tpb_cst, tpb_cst >>> (fields, G);
        }
      }
      //
      fill_buff_up_y_k   <<< 1+sP->len_buff_field_y/tpb_cst, tpb_cst >>> (fields, G);
      MPI_receive_send_guard_cells_up_y(fields, sP, comm);
      if (sP->mpi_rank_y>0){
        if (copy_long){
          copy_buff_up_long_y_k<<< 1+sP->len_buff_field_y/tpb_cst, tpb_cst >>> (fields, G);
        }
        else {
          copy_buff_up_short_y_k<<< 1+sP->len_buff_field_y/tpb_cst, tpb_cst >>> (fields, G);
        }
      }
    }
    //
    #if NB_DIM==3
      if (sP->mpi_nb_proc_z>1){
        fill_buff_down_z_k <<< 1+sP->len_buff_field_z/tpb_cst, tpb_cst >>> (fields, G);
        MPI_receive_send_guard_cells_down_z(fields, sP, comm);
        if (sP->mpi_rank_z<(sP->mpi_nb_proc_z-1)){
          if (copy_long){
            add_buff_down_z_k  <<< 1+sP->len_buff_field_z/tpb_cst, tpb_cst >>> (fields, G);
          }
          else {
            copy_buff_down_short_z_k<<< 1+sP->len_buff_field_z/tpb_cst, tpb_cst >>> (fields, G);
          }
        }
        //
        fill_buff_up_z_k   <<< 1+sP->len_buff_field_z/tpb_cst, tpb_cst >>> (fields, G);
        MPI_receive_send_guard_cells_up_z(fields, sP, comm);
        if (sP->mpi_rank_z>0){
          if (copy_long){
            copy_buff_up_long_z_k<<< 1+sP->len_buff_field_z/tpb_cst, tpb_cst >>> (fields, G);
          }
          else {
            copy_buff_up_short_z_k<<< 1+sP->len_buff_field_z/tpb_cst, tpb_cst >>> (fields, G);
          }
        }
      }
    #endif

  #else
    //
    if (sP->mpi_nb_proc_y>1){
      fill_buff_down_y_k <<< 1+sP->len_buff_field_y/tpb_cst, tpb_cst >>> (fields, G);
      MPI_receive_send_guard_cells_down_y(fields, sP, comm);
      if (copy_long){
        add_buff_down_y_k  <<< 1+sP->len_buff_field_y/tpb_cst, tpb_cst >>> (fields, G);
      }
      else {
        copy_buff_down_short_y_k<<< 1+sP->len_buff_field_y/tpb_cst, tpb_cst >>> (fields, G);
      }
      //
      fill_buff_up_y_k   <<< 1+sP->len_buff_field_y/tpb_cst, tpb_cst >>> (fields, G);
      MPI_receive_send_guard_cells_up_y(fields, sP, comm);
      if (copy_long){
        copy_buff_up_long_y_k<<< 1+sP->len_buff_field_y/tpb_cst, tpb_cst >>> (fields, G);
      }
      else {
        copy_buff_up_short_y_k<<< 1+sP->len_buff_field_y/tpb_cst, tpb_cst >>> (fields, G);
      }
    }
    //
    #if NB_DIM==3
      if (sP->mpi_nb_proc_z>1){
        fill_buff_down_z_k <<< 1+sP->len_buff_field_z/tpb_cst, tpb_cst >>> (fields, G);
        MPI_receive_send_guard_cells_down_z(fields, sP, comm);
        if (copy_long){
          add_buff_down_z_k  <<< 1+sP->len_buff_field_z/tpb_cst, tpb_cst >>> (fields, G);
        }
        else {
          copy_buff_down_short_z_k<<< 1+sP->len_buff_field_z/tpb_cst, tpb_cst >>> (fields, G);
        }
        //
        fill_buff_up_z_k   <<< 1+sP->len_buff_field_z/tpb_cst, tpb_cst >>> (fields, G);
        MPI_receive_send_guard_cells_up_z(fields, sP, comm);
        if (copy_long){
          copy_buff_up_long_z_k<<< 1+sP->len_buff_field_z/tpb_cst, tpb_cst >>> (fields, G);
        }
        else {
          copy_buff_up_short_z_k<<< 1+sP->len_buff_field_z/tpb_cst, tpb_cst >>> (fields, G);
        }
      }
    #endif

  #endif

}









void boundaries_density
#if NB_DIM==2
(simu_fields* fields, float density[len_x_cst+4][len_y_cst+4],
    injector* injec, simu_param* sP_h, simu_param* sP_d, MPI_Comm comm, int idx_it)
#elif NB_DIM==3
(simu_fields* fields, float density[len_x_cst+4][len_y_cst+4][len_z_cst+4],
    injector* injec, simu_param* sP_h, simu_param* sP_d, MPI_Comm comm, int idx_it)
#endif
{

  if (sP_h->mpi_nb_proc_tot>1){
    MPI_communicate_scalar(density, fields, sP_h, comm, true);
  }
  scalar_boundaries_add(density, sP_h);

  #if (obstacle_cst && !inject_turb_cst && !decay_turb_cst)
    #if NB_DIM==2
      inject_dens_const_k<<< 1 + (len_y_cst+4)/tpb_cst, tpb_cst >>>(density, sP_d);
    #elif NB_DIM==3
      inject_dens_const_k<<< 1 + (len_y_cst+4)*(len_z_cst+4)/tpb_cst, tpb_cst >>>(density, sP_d);
    #endif
  #elif inject_turb_cst
    #if NB_DIM==2
      inject_dens_k  <<< 1 + (2+nb_cell_per_shift_cst)*(len_y_cst+4)/tpb_cst, tpb_cst >>>(injec, density);
    #elif NB_DIM==3
      inject_dens_k  <<< 1 + (2+nb_cell_per_shift_cst)*(len_y_cst+4)*(len_z_cst+4)/tpb_cst, tpb_cst >>>(injec, density);
    #endif
  #endif

}


void boundaries_curr
#if NB_DIM==2
(simu_fields* fields, float Ji[3][len_x_cst+4][len_y_cst+4],
    injector* injec, simu_param* sP_h, simu_param* sP_d, MPI_Comm comm)
#elif NB_DIM==3
(simu_fields* fields, float Ji[3][len_x_cst+4][len_y_cst+4][len_z_cst+4],
    injector* injec, simu_param* sP_h, simu_param* sP_d, MPI_Comm comm)
#endif
{

  if (sP_h->mpi_nb_proc_tot>1){
    MPI_communicate_scalar(Ji[0], fields, sP_h, comm, true);
    MPI_communicate_scalar(Ji[1], fields, sP_h, comm, true);
    MPI_communicate_scalar(Ji[2], fields, sP_h, comm, true);
  }
  scalar_boundaries_add(Ji[0], sP_h);
  scalar_boundaries_add(Ji[1], sP_h);
  scalar_boundaries_add(Ji[2], sP_h);

  #if (obstacle_cst && !inject_turb_cst && !decay_turb_cst)
    #if NB_DIM==2
      inject_curr_const_k<<< 1+(len_y_cst+4)/tpb_cst, tpb_cst >>>(Ji, sP_d);
    #elif NB_DIM==3
      inject_curr_const_k<<< 1+(len_y_cst+4)*(len_z_cst+4)/tpb_cst, tpb_cst >>>(Ji, sP_d);
    #endif
  #elif inject_turb_cst
    #if NB_DIM==2
      inject_curr_k<<< 1+(2+nb_cell_per_shift_cst)*(len_y_cst+4)/tpb_cst, tpb_cst >>>(injec, Ji);
    #elif NB_DIM==3
      inject_curr_k<<< 1+(2+nb_cell_per_shift_cst)*(len_y_cst+4)*(len_z_cst+4)/tpb_cst, tpb_cst >>>(injec, Ji);
    #endif
  #endif
}


void boundaries_EB
#if NB_DIM==2
  (simu_fields* fields, float G[3][len_x_cst+4][len_y_cst+4], simu_param* sP, MPI_Comm comm)
#elif NB_DIM==3
  (simu_fields* fields, float G[3][len_x_cst+4][len_y_cst+4][len_z_cst+4], simu_param* sP, MPI_Comm comm)
#endif
{
  if (sP->mpi_nb_proc_tot>1){
    MPI_communicate_scalar(G[0], fields, sP, comm, false);
    MPI_communicate_scalar(G[1], fields, sP, comm, false);
    MPI_communicate_scalar(G[2], fields, sP, comm, false);
  }
  scalar_boundaries(G[0], sP);
  scalar_boundaries(G[1], sP);
  scalar_boundaries(G[2], sP);
}















void moments_mapping(particles* pa, simu_fields* fields, simu_grid* grid, injector* injec,
                     simu_param* sP_d, simu_param* sP_h, int idx_it, bool moments_b, int tag)
{
  clear_counts_flux_num_k    <<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(fields);
  part2grid_counts_flux_num_k<<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(pa, grid, fields, sP_d, idx_it, tag);
  // float delta_x = 0.;
  if (!moments_b){
    add_counts_k<<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(fields, sP_d);
    if (tag!=999){
      add_curr_k  <<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(fields, sP_d);
    }
    boundaries_density(fields, fields->density, injec, sP_h, sP_d, MPI_COMM_WORLD, idx_it);
    if (tag!=999){
      boundaries_curr(fields, fields->Ji, injec, sP_h, sP_d, MPI_COMM_WORLD);
    }
  }
  else if (moments_b){
    add_counts_b_k<<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(fields, sP_d);
    add_curr_b_k  <<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(fields, sP_d, idx_it);
    boundaries_density(fields, fields->density_b, injec, sP_h, sP_d, MPI_COMM_WORLD, idx_it);
    boundaries_curr(fields, fields->Ji_b, injec, sP_h, sP_d, MPI_COMM_WORLD);
    // #if !ORF_cst
    //   delta_x = sP_h->dt*sP_h->v_obs;
    // #endif
  }
  // identify_region_k<<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(fields, grid, sP_d, delta_x);
}



void ohm(simu_B_field* B, simu_fields* fields, simu_grid* grid, int idx_it,
         simu_param* sP_h, simu_param* sP_d, MPI_Comm comm,
         int ohm_ID)
{




  #if !yee_cst
    curl_B_k      <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(B, fields, sP_d);
  #else
    #if NB_DIM==2
      curl_B_yee_k  <<< 1+(len_x_cst+2)*(len_y_cst+2)/tpb_cst, tpb_cst >>>(B, fields, sP_d);
      unstag_J_tot_k<<< 1+(len_x_cst+3)*(len_y_cst+3)/tpb_cst, tpb_cst >>>(fields);
      unstag_B_k    <<< 1+(len_x_cst+3)*(len_y_cst+3)/tpb_cst, tpb_cst >>>(B);
      boundaries_EB(fields, fields->J_tot, sP_h, MPI_COMM_WORLD);
      boundaries_EB(fields, B->B, sP_h, MPI_COMM_WORLD);
    #elif NB_DIM==3
      curl_B_yee_k  <<< 1+(len_x_cst+2)*(len_y_cst+2)*(len_z_cst+2)/tpb_cst, tpb_cst >>>(B, fields, sP_d);
      unstag_J_tot_k<<< 1+(len_x_cst+3)*(len_y_cst+3)*(len_z_cst+3)/tpb_cst, tpb_cst >>>(fields);
    #endif
  #endif
  //
  if (ohm_ID!=0){
    pressure_k<<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(fields, fields->density_b, sP_d);
  }

  if (ohm_ID==0){
    float delta_x = .5*sP_h->dt*sP_h->v_obs;
    identify_region_k<<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(fields, grid, sP_d, delta_x);
    //
    ohm_k     <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(fields, B, grid, fields->density, fields->Ji, idx_it, sP_d, ohm_ID);
    #if (!ORF_cst && dipole_cst)
      #if NB_DIM==2
        inject_const_E_k<<< 1 + (len_y_cst+4)/tpb_cst, tpb_cst >>>(fields, sP_d, true);
      #elif NB_DIM==3
        inject_const_E_k<<< 1 + (len_y_cst+4)*(len_z_cst+4)/tpb_cst, tpb_cst >>>(fields, sP_d, true);
      #endif
    #endif
  }
  else if (ohm_ID==1){
    float delta_x = sP_h->dt*sP_h->v_obs;
    identify_region_k<<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(fields, grid, sP_d, delta_x);
    //
    ohm_k     <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(fields, B, grid, fields->density_b, fields->Ji_b, idx_it, sP_d, ohm_ID);
    #if (!ORF_cst && dipole_cst)
      #if NB_DIM==2
        inject_const_E_k<<< 1 + (len_y_cst+4)/tpb_cst, tpb_cst >>>(fields, sP_d, false);
      #elif NB_DIM==3
        inject_const_E_k<<< 1 + (len_y_cst+4)*(len_z_cst+4)/tpb_cst, tpb_cst >>>(fields, sP_d, false);
      #endif
    #endif
  }
  else if (ohm_ID==2){
    float delta_x = sP_h->dt*sP_h->v_obs;
    identify_region_k<<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(fields, grid, sP_d, delta_x);
    //
    ohm_k     <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(fields, B, grid, fields->density_b, fields->Ji, idx_it, sP_d, ohm_ID);
    #if (!ORF_cst && dipole_cst)
      #if NB_DIM==2
        inject_const_E_k<<< 1 + (len_y_cst+4)/tpb_cst, tpb_cst >>>(fields, sP_d, false);
      #elif NB_DIM==3
        inject_const_E_k<<< 1 + (len_y_cst+4)*(len_z_cst+4)/tpb_cst, tpb_cst >>>(fields, sP_d, false);
      #endif
    #endif
  }

  boundaries_EB(fields, fields->E, sP_h, MPI_COMM_WORLD);
  //
  #if (NB_DIM==2 && obstacle_cst)
    smooth_downstream_k      <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)/tpb_cst, tpb_cst >>>(fields->E[0], fields, sP_d);
    copy_smooth_downstream_k <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)/tpb_cst, tpb_cst >>>(fields->E[0], fields, sP_d);
    smooth_downstream_k      <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)/tpb_cst, tpb_cst >>>(fields->E[1], fields, sP_d);
    copy_smooth_downstream_k <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)/tpb_cst, tpb_cst >>>(fields->E[1], fields, sP_d);
    smooth_downstream_k      <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)/tpb_cst, tpb_cst >>>(fields->E[2], fields, sP_d);
    copy_smooth_downstream_k <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)/tpb_cst, tpb_cst >>>(fields->E[2], fields, sP_d);
    boundaries_EB(fields, fields->E, sP_h, MPI_COMM_WORLD);
  #elif (NB_DIM==3 && obstacle_cst)
    smooth_downstream_k      <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)*(len_z_cst+2)/tpb_cst, tpb_cst >>>(fields->E[0], fields, sP_d);
    copy_smooth_downstream_k <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)*(len_z_cst+2)/tpb_cst, tpb_cst >>>(fields->E[0], fields, sP_d);
    smooth_downstream_k      <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)*(len_z_cst+2)/tpb_cst, tpb_cst >>>(fields->E[1], fields, sP_d);
    copy_smooth_downstream_k <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)*(len_z_cst+2)/tpb_cst, tpb_cst >>>(fields->E[1], fields, sP_d);
    smooth_downstream_k      <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)*(len_z_cst+2)/tpb_cst, tpb_cst >>>(fields->E[2], fields, sP_d);
    copy_smooth_downstream_k <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)*(len_z_cst+2)/tpb_cst, tpb_cst >>>(fields->E[2], fields, sP_d);
    boundaries_EB(fields, fields->E, sP_h, MPI_COMM_WORLD);
  #endif

}

void faraday(simu_B_field* B_in, simu_B_field* B_out, simu_fields* fields, float sub_dt,
             simu_param* sP_d, simu_param* sP_h, simu_grid* grid,
             float delta_x_in, float delta_x_out)
{

  // compute_potential_k<<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(B_in, grid, sP_d, delta_x_in);
  // compute_dipole_k<<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(fields->vort, B_in, fields, grid, sP_d);
  //
  // compute_potential_k<<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(B_out, grid, sP_d, delta_x_out);
  // compute_dipole_k<<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(B_out->B_dip, B_out, fields, grid, sP_d);

  #if !yee_cst
    faraday_k<<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(B_in, B_out, fields, sub_dt, sP_d, grid);
  #else
    #if NB_DIM==2
      stag_E_k     <<< 1+(len_x_cst+3)*(len_y_cst+3)/tpb_cst, tpb_cst >>>(fields);
      faraday_yee_k<<< 1+(len_x_cst+2)*(len_y_cst+2)/tpb_cst, tpb_cst >>>(B_in, B_out, fields, sub_dt, sP_d, grid);
      unstag_B_k   <<< 1+(len_x_cst+3)*(len_y_cst+3)/tpb_cst, tpb_cst >>>(B_out);
    #elif NB_DIM==3
      stag_E_k     <<< 1+(len_x_cst+3)*(len_y_cst+3)*(len_z_cst+3)/tpb_cst, tpb_cst >>>(fields);
      faraday_yee_k<<< 1+(len_x_cst+2)*(len_y_cst+2)*(len_z_cst+2)/tpb_cst, tpb_cst >>>(B_in, B_out, fields, sub_dt, sP_d, grid);
      unstag_B_k   <<< 1+(len_x_cst+3)*(len_y_cst+3)*(len_z_cst+3)/tpb_cst, tpb_cst >>>(B_out);
    #endif
    boundaries_EB(fields, B_out->B_s, sP_h, MPI_COMM_WORLD);
  #endif
  //
  boundaries_EB(fields, B_out->B, sP_h, MPI_COMM_WORLD);
  //
  #if (NB_DIM==2 && obstacle_cst)
    smooth_downstream_k      <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)/tpb_cst, tpb_cst >>>(B_out->B[0], fields, sP_d);
    copy_smooth_downstream_k <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)/tpb_cst, tpb_cst >>>(B_out->B[0], fields, sP_d);
    smooth_downstream_k      <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)/tpb_cst, tpb_cst >>>(B_out->B[1], fields, sP_d);
    copy_smooth_downstream_k <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)/tpb_cst, tpb_cst >>>(B_out->B[1], fields, sP_d);
    smooth_downstream_k      <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)/tpb_cst, tpb_cst >>>(B_out->B[2], fields, sP_d);
    copy_smooth_downstream_k <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)/tpb_cst, tpb_cst >>>(B_out->B[2], fields, sP_d);
    boundaries_EB(fields, B_out->B, sP_h, MPI_COMM_WORLD);
  #elif (NB_DIM==3 && obstacle_cst)
    smooth_downstream_k      <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)*(len_z_cst+2)/tpb_cst, tpb_cst >>>(B_out->B[0], fields, sP_d);
    copy_smooth_downstream_k <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)*(len_z_cst+2)/tpb_cst, tpb_cst >>>(B_out->B[0], fields, sP_d);
    smooth_downstream_k      <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)*(len_z_cst+2)/tpb_cst, tpb_cst >>>(B_out->B[1], fields, sP_d);
    copy_smooth_downstream_k <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)*(len_z_cst+2)/tpb_cst, tpb_cst >>>(B_out->B[1], fields, sP_d);
    smooth_downstream_k      <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)*(len_z_cst+2)/tpb_cst, tpb_cst >>>(B_out->B[2], fields, sP_d);
    copy_smooth_downstream_k <<< 1+sP_h->width_smooth_downstream*(len_y_cst+2)*(len_z_cst+2)/tpb_cst, tpb_cst >>>(B_out->B[2], fields, sP_d);
    boundaries_EB(fields, B_out->B, sP_h, MPI_COMM_WORLD);
  #endif

}




void check_smooth(simu_fields* fields, simu_B_field* B, simu_param* sP_h, simu_param* sP_d,
                  simu_grid* grid, state_solver* sta_sol, int idx_it)
{

  #if smooth_patch_cst
    //
    bool smooth = false;  // If smooth is true, there is at least one spot to be smoothed.
    int nb_pass = 0;  // Total number of smoothed patch for this iteration.
    #if NB_DIM==2
      int nb_nodes = (len_x_cst+2)*(len_y_cst+2);  // +2 because smoothing order 0 (1 neighbour each side).
      int nb_nodes_patch = len_patch_cst*len_patch_cst;
    #elif NB_DIM==3
      int nb_nodes = (len_x_cst+2)*(len_y_cst+2)*(len_z_cst+2);
      int nb_nodes_patch = len_patch_cst*len_patch_cst*len_patch_cst;
    #endif
    //
    // deriv_order_0_k<<< 1 + nb_nodes_cst/tpb_cst, tpb_cst>>>(fields);
    search_smooth_k <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(fields, sP_d);
    cudaMemcpy(&smooth, &fields->smooth_field, sizeof(bool), cudaMemcpyDeviceToHost);
    //
    while (smooth){
      nb_pass++;
      //
      // smooth_field_all_k  <<< 1 + nb_nodes_tot_cst/tpb_cst,       tpb_cst >>>(fields->E[0], fields);
      smooth_patch_k      <<< 1 + nb_nodes_patch/tpb_cst, tpb_cst >>>(fields->E[0], fields);
      copy_smooth_patch_k <<< 1 + nb_nodes_patch/tpb_cst, tpb_cst >>>(fields->E[0], fields, idx_it);
      // smooth_field_all_k  <<< 1 + nb_nodes_tot_cst/tpb_cst,       tpb_cst >>>(fields->E[1], fields);
      smooth_patch_k      <<< 1 + nb_nodes_patch/tpb_cst, tpb_cst >>>(fields->E[1], fields);
      copy_smooth_patch_k <<< 1 + nb_nodes_patch/tpb_cst, tpb_cst >>>(fields->E[1], fields, idx_it);
      // smooth_field_all_k  <<< 1 + nb_nodes_tot_cst/tpb_cst,       tpb_cst >>>(fields->E[2], fields);
      smooth_patch_k      <<< 1 + nb_nodes_patch/tpb_cst, tpb_cst >>>(fields->E[2], fields);
      copy_smooth_patch_k <<< 1 + nb_nodes_patch/tpb_cst, tpb_cst >>>(fields->E[2], fields, idx_it);
      //
      // smooth_field_all_k  <<< 1 + nb_nodes_tot_cst/tpb_cst,       tpb_cst >>>(B->B[0], fields);
      smooth_patch_k      <<< 1 + nb_nodes_patch/tpb_cst, tpb_cst >>>(B->B[0], fields);
      copy_smooth_patch_k <<< 1 + nb_nodes_patch/tpb_cst, tpb_cst >>>(B->B[0], fields, idx_it);
      // smooth_field_all_k  <<< 1 + nb_nodes_tot_cst/tpb_cst,       tpb_cst >>>(B->B[1], fields);
      smooth_patch_k      <<< 1 + nb_nodes_patch/tpb_cst, tpb_cst >>>(B->B[1], fields);
      copy_smooth_patch_k <<< 1 + nb_nodes_patch/tpb_cst, tpb_cst >>>(B->B[1], fields, idx_it);
      // smooth_field_all_k  <<< 1 + nb_nodes_tot_cst/tpb_cst,       tpb_cst >>>(B->B[2], fields);
      smooth_patch_k      <<< 1 + nb_nodes_patch/tpb_cst, tpb_cst >>>(B->B[2], fields);
      copy_smooth_patch_k <<< 1 + nb_nodes_patch/tpb_cst, tpb_cst >>>(B->B[2], fields, idx_it);
      //
      save_smooth_patch_k<<< 1, 1 >>>(fields, grid, idx_it);
      //
      smooth = false;
      cudaMemcpy(&fields->smooth_field, &smooth, sizeof(bool), cudaMemcpyHostToDevice);
      //
      // deriv_order_0_k<<< 1 + nb_nodes_cst/tpb_cst, tpb_cst>>>(fields);
      search_smooth_k <<< 1+nb_nodes/tpb_cst, tpb_cst >>>(fields, sP_d);
      cudaMemcpy(&smooth, &fields->smooth_field, sizeof(bool), cudaMemcpyDeviceToHost);
      // std::cout << "| Smoothing patch idx_it " << idx_it << std::endl;
      if (nb_pass>=100){
        smooth = false;
      }
    }
    // std::cout << "yo " << nb_pass << std::endl;
    boundaries_EB(fields, fields->E, sP_h, MPI_COMM_WORLD);
    // boundaries_EB(fields, B->B, sP_h, MPI_COMM_WORLD);

    if (nb_pass>=100){
      bool brok = true;
      int id = 9;
      cudaMemcpy(&sta_sol->break_solver, &brok, sizeof(bool), cudaMemcpyHostToDevice);
      cudaMemcpy(&sta_sol->error_ID,     &id,   sizeof(int),  cudaMemcpyHostToDevice);
    }
  #endif

}









void increment_ind_slice(simu_tank* tank_h)
{
  tank_h->idx_x++;
  if (tank_h->idx_x==int(len_x_cst/nb_cell_per_shift_cst)){
    tank_h->idx_x = 0;
  }
}

void copy_injector(particles* p, simu_tank* tank_h, injector* injec_h,
                   int nb_part_in_slice, simu_param* sP, simu_grid* grid, int idx_it)
{
  int idx_part_tank;
  int idx_ord;

  #if debug_mode_cst
    if (nb_part_in_slice>injec_size_cst){
      printf("Buffer too small indeed!\n"); // Already guarded in load_particle() actually...(?)
    }
  #endif


  #if NB_DIM==2
    for (int idx_part_inj=0; idx_part_inj<nb_part_in_slice; idx_part_inj++){
      idx_ord = tank_h->indFirstPartSlice[tank_h->idx_x] + idx_part_inj;
      idx_part_tank = tank_h->idx_ordered[idx_ord];
      if ( idx_ord > sP->nb_part_tank){
        std::cout << "| Out of particles tank, copy_injector." << std::endl;
      }
      if (idx_part_tank > pool_size_cst){
        std::cout << "| Out of particles pool, copy_injector." << std::endl;
      }
      if (idx_part_inj > injec_size_cst){
        std::cout << "| Out of particles tank, copy_injector." << std::endl;
      }
      injec_h->rx[idx_part_inj] = p->rx[idx_part_tank] + (len_x_cst-nb_cell_per_shift_cst*(tank_h->idx_x+1))*sP->dX;
      injec_h->ry[idx_part_inj] = p->ry[idx_part_tank];
      injec_h->vx[idx_part_inj] = p->vx[idx_part_tank];
      injec_h->vy[idx_part_inj] = p->vy[idx_part_tank];
      injec_h->vz[idx_part_inj] = p->vz[idx_part_tank];
      if (injec_h->rx[idx_part_inj]<0 || injec_h->rx[idx_part_inj]>grid->xMax){
        std::cout << "| Particle outside of x domain, copy_injector. idx_part_inj: " << idx_part_tank << std::endl;
      }
      if (injec_h->ry[idx_part_inj]<0 || injec_h->ry[idx_part_inj]>grid->yMax){
        std::cout << "| Particle outside of y domain, copy_injector. idx_part_inj: " << idx_part_tank << std::endl;
      }
    }

    for (int i=0; i<2+nb_cell_per_shift_cst; i++){
      for (int j=0; j<len_y_cst+4; j++){
        injec_h->B [0][i][j] = tank_h->B [0][(tank_h->idx_x*nb_cell_per_shift_cst)+2+i][j];
        injec_h->B [1][i][j] = tank_h->B [1][(tank_h->idx_x*nb_cell_per_shift_cst)+2+i][j];
        injec_h->B [2][i][j] = tank_h->B [2][(tank_h->idx_x*nb_cell_per_shift_cst)+2+i][j];
        injec_h->E [0][i][j] = tank_h->E [0][(tank_h->idx_x*nb_cell_per_shift_cst)+2+i][j];
        injec_h->E [1][i][j] = tank_h->E [1][(tank_h->idx_x*nb_cell_per_shift_cst)+2+i][j];
        injec_h->E [2][i][j] = tank_h->E [2][(tank_h->idx_x*nb_cell_per_shift_cst)+2+i][j];
        injec_h->Ji[0][i][j] = tank_h->Ji[0][(tank_h->idx_x*nb_cell_per_shift_cst)+2+i][j];
        injec_h->Ji[1][i][j] = tank_h->Ji[1][(tank_h->idx_x*nb_cell_per_shift_cst)+2+i][j];
        injec_h->Ji[2][i][j] = tank_h->Ji[2][(tank_h->idx_x*nb_cell_per_shift_cst)+2+i][j];
        injec_h->density[i][j] = tank_h->density[(tank_h->idx_x*nb_cell_per_shift_cst)+2+i][j];
      }
    }
  #elif NB_DIM==3
    for (int idx_part_inj=0; idx_part_inj<nb_part_in_slice; idx_part_inj++){
      idx_ord = tank_h->indFirstPartSlice[tank_h->idx_x] + idx_part_inj;
      idx_part_tank = tank_h->idx_ordered[idx_ord];
      if ( idx_ord > sP->nb_part_tank){
        std::cout << "| Out of particles tank, copy_injector." << std::endl;
      }
      if (idx_part_tank > pool_size_cst){
        std::cout << "| Out of particles pool, copy_injector." << std::endl;
      }
      if (idx_part_inj > injec_size_cst){
        std::cout << "| Out of particles tank, copy_injector." << std::endl;
      }
      injec_h->rx[idx_part_inj] = p->rx[idx_part_tank] + (len_x_cst-nb_cell_per_shift_cst*(tank_h->idx_x+1))*sP->dX;
      injec_h->ry[idx_part_inj] = p->ry[idx_part_tank];
      injec_h->rz[idx_part_inj] = p->rz[idx_part_tank];
      injec_h->vx[idx_part_inj] = p->vx[idx_part_tank];
      injec_h->vy[idx_part_inj] = p->vy[idx_part_tank];
      injec_h->vz[idx_part_inj] = p->vz[idx_part_tank];
    }
    for (int i=0; i<2+nb_cell_per_shift_cst; i++){
      for (int j=0; j<len_y_cst+4; j++){
        for (int k=0; k<len_z_cst+4; k++){
          injec_h->B [0][i][j][k] = tank_h->B [0][(tank_h->idx_x*nb_cell_per_shift_cst)+2+i][j][k];
          injec_h->B [1][i][j][k] = tank_h->B [1][(tank_h->idx_x*nb_cell_per_shift_cst)+2+i][j][k];
          injec_h->B [2][i][j][k] = tank_h->B [2][(tank_h->idx_x*nb_cell_per_shift_cst)+2+i][j][k];
          injec_h->E [0][i][j][k] = tank_h->E [0][(tank_h->idx_x*nb_cell_per_shift_cst)+2+i][j][k];
          injec_h->E [1][i][j][k] = tank_h->E [1][(tank_h->idx_x*nb_cell_per_shift_cst)+2+i][j][k];
          injec_h->E [2][i][j][k] = tank_h->E [2][(tank_h->idx_x*nb_cell_per_shift_cst)+2+i][j][k];
          injec_h->Ji[0][i][j][k] = tank_h->Ji[0][(tank_h->idx_x*nb_cell_per_shift_cst)+2+i][j][k];
          injec_h->Ji[1][i][j][k] = tank_h->Ji[1][(tank_h->idx_x*nb_cell_per_shift_cst)+2+i][j][k];
          injec_h->Ji[2][i][j][k] = tank_h->Ji[2][(tank_h->idx_x*nb_cell_per_shift_cst)+2+i][j][k];
          injec_h->density[i][j][k] = tank_h->density[(tank_h->idx_x*nb_cell_per_shift_cst)+2+i][j][k];
        }
      }
    }
  #endif

}





//_________________________________________________________________________________________
//
// Particles related kernel calls.
//
void receive_send_part_down_y(buff_part* b_p, simu_param* sP, house_keeping* h_k, MPI_Comm comm, int idx_it)
{

  MPI_Request down_req;
  MPI_Status stat;
  //
  int down_proc_y = (sP->mpi_rank_y-1+sP->mpi_nb_proc_y)%sP->mpi_nb_proc_y;
  int up_proc_y   = (sP->mpi_rank_y+1)%sP->mpi_nb_proc_y;
  int down_proc_lin = down_proc_y*sP->mpi_nb_proc_z + sP->mpi_rank_z;
  int up_proc_lin   =   up_proc_y*sP->mpi_nb_proc_z + sP->mpi_rank_z;

  // buff_part* buff_part_h = new buff_part;

  cudaDeviceSynchronize();
  // cudaMemcpy(buff_part_h, b_p, sizeof(*buff_part_h), cudaMemcpyDeviceToHost);
  // int nb_part_send = buff_part_h->next_idx_comm;
  int nb_part_send;
  int nb_part_rece ;
  cudaMemcpy(&nb_part_send, &b_p->next_idx_comm, sizeof(int), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  MPI_Irecv(&nb_part_rece, 1, MPI_INT, up_proc_lin  , 1, comm, &down_req);
  MPI_Send( &nb_part_send, 1, MPI_INT, down_proc_lin, 1, comm);
  MPI_Wait(&down_req, &stat);
  update_nb_part_rece_k<<<1, 1>>>(b_p, nb_part_rece);
  if (idx_it%rate_save_t_cst==0){
    HK_nb_part_send_down_k<<<1, 1>>>(h_k, nb_part_send, int(idx_it/rate_save_t_cst));
    cudaDeviceSynchronize();
  }

  if (nb_part_rece>buff_size_cst){
    printf("Too many particles to receive down for buffer.\n");
  }
  if (nb_part_send>buff_size_cst){
    printf("Too many particles to send down for buffer.\n");
  }

  if (nb_part_rece>0){
    #if NB_DIM==2
      MPI_Irecv(b_p->buff_rece, 6*nb_part_rece, MPI_FLOAT, up_proc_lin, 1, comm, &down_req);
    #elif NB_DIM==3
      MPI_Irecv(b_p->buff_rece, 7*nb_part_rece, MPI_FLOAT, up_proc_lin, 1, comm, &down_req);
    #endif
  }
  if (nb_part_send>0){
    #if NB_DIM==2
      MPI_Send( b_p->buff_send, 6*nb_part_send, MPI_FLOAT, down_proc_lin, 1, comm);
    #elif NB_DIM==3
      MPI_Send( b_p->buff_send, 7*nb_part_send, MPI_FLOAT, down_proc_lin, 1, comm);
    #endif
  }
  if (nb_part_rece>0){
    MPI_Wait(&down_req, &stat);
  }

  // delete(buff_part_h);

}

void receive_send_part_up_y(buff_part* b_p, simu_param* sP, house_keeping* h_k, MPI_Comm comm, int idx_it)
{

  MPI_Request up_req;
  MPI_Status stat;
  //
  int down_proc_y = (sP->mpi_rank_y-1+sP->mpi_nb_proc_y)%sP->mpi_nb_proc_y;
  int up_proc_y   = (sP->mpi_rank_y+1)%sP->mpi_nb_proc_y;
  int down_proc_lin = down_proc_y*sP->mpi_nb_proc_z + sP->mpi_rank_z;
  int up_proc_lin   =   up_proc_y*sP->mpi_nb_proc_z + sP->mpi_rank_z;

  // buff_part* buff_part_h = new buff_part;

  cudaDeviceSynchronize();
  // cudaMemcpy(buff_part_h, b_p, sizeof(*buff_part_h), cudaMemcpyDeviceToHost);
  // int nb_part_send = buff_part_h->next_idx_comm;
  int nb_part_send;
  int nb_part_rece ;
  cudaMemcpy(&nb_part_send, &b_p->next_idx_comm, sizeof(int), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  MPI_Irecv(&nb_part_rece, 1, MPI_INT, down_proc_lin, 1, comm, &up_req);
  MPI_Send( &nb_part_send, 1, MPI_INT, up_proc_lin  , 1, comm);
  MPI_Wait(&up_req, &stat);
  update_nb_part_rece_k<<<1, 1>>>(b_p, nb_part_rece);
  if (idx_it%rate_save_t_cst==0){
    HK_nb_part_send_up_k<<<1, 1>>>(h_k, nb_part_send, int(idx_it/rate_save_t_cst));
    cudaDeviceSynchronize();
  }

  if (nb_part_rece>buff_size_cst){
    printf("Too many particles to receive up for buffer.\n");
  }
  if (nb_part_send>buff_size_cst){
    printf("Too many particles to send up for buffer.\n");
  }

  if (nb_part_rece>0){
    #if NB_DIM==2
      MPI_Irecv(b_p->buff_rece, 6*nb_part_rece, MPI_FLOAT, down_proc_lin, 1, comm, &up_req);
    #elif NB_DIM==3
      MPI_Irecv(b_p->buff_rece, 7*nb_part_rece, MPI_FLOAT, down_proc_lin, 1, comm, &up_req);
    #endif
  }
  if (nb_part_send>0){
    #if NB_DIM==2
      MPI_Send( b_p->buff_send, 6*nb_part_send, MPI_FLOAT, up_proc_lin, 1, comm);
    #elif NB_DIM==3
      MPI_Send( b_p->buff_send, 7*nb_part_send, MPI_FLOAT, up_proc_lin, 1, comm);
    #endif
  }
  if (nb_part_rece>0){
    MPI_Wait(&up_req, &stat);
  }

  // delete(buff_part_h);

}

void receive_send_part_down_z(buff_part* b_p, simu_param* sP, house_keeping* h_k, MPI_Comm comm, int idx_it)
{

  MPI_Request down_req;
  MPI_Status stat;
  //
  int down_proc_z = (sP->mpi_rank_z-1+sP->mpi_nb_proc_z)%sP->mpi_nb_proc_z;
  int up_proc_z   = (sP->mpi_rank_z+1)%sP->mpi_nb_proc_z;
  int down_proc_lin = sP->mpi_rank_y*sP->mpi_nb_proc_z + down_proc_z;
  int up_proc_lin   = sP->mpi_rank_y*sP->mpi_nb_proc_z + up_proc_z;

  // buff_part* buff_part_h = new buff_part;

  cudaDeviceSynchronize();
  // cudaMemcpy(buff_part_h, b_p, sizeof(*buff_part_h), cudaMemcpyDeviceToHost);
  // int nb_part_send = buff_part_h->next_idx_comm;
  int nb_part_send;
  int nb_part_rece ;
  cudaMemcpy(&nb_part_send, &b_p->next_idx_comm, sizeof(int), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  MPI_Irecv(&nb_part_rece, 1, MPI_INT, up_proc_lin  , 1, comm, &down_req);
  MPI_Send( &nb_part_send, 1, MPI_INT, down_proc_lin, 1, comm);
  MPI_Wait(&down_req, &stat);
  update_nb_part_rece_k<<<1, 1>>>(b_p, nb_part_rece);
  if (idx_it%rate_save_t_cst==0){
    HK_nb_part_send_down_k<<<1, 1>>>(h_k, nb_part_send, int(idx_it/rate_save_t_cst));
    cudaDeviceSynchronize();
  }

  if (nb_part_rece>buff_size_cst){
    printf("Too many particles to receive down for buffer.\n");
  }
  if (nb_part_send>buff_size_cst){
    printf("Too many particles to send down for buffer.\n");
  }

  if (nb_part_rece>0){
    #if NB_DIM==2
      MPI_Irecv(b_p->buff_rece, 6*nb_part_rece, MPI_FLOAT, up_proc_lin, 1, comm, &down_req);
    #elif NB_DIM==3
      MPI_Irecv(b_p->buff_rece, 7*nb_part_rece, MPI_FLOAT, up_proc_lin, 1, comm, &down_req);
    #endif
  }
  if (nb_part_send>0){
    #if NB_DIM==2
      MPI_Send( b_p->buff_send, 6*nb_part_send, MPI_FLOAT, down_proc_lin, 1, comm);
    #elif NB_DIM==3
      MPI_Send( b_p->buff_send, 7*nb_part_send, MPI_FLOAT, down_proc_lin, 1, comm);
    #endif
  }
  if (nb_part_rece>0){
    MPI_Wait(&down_req, &stat);
  }

  // delete(buff_part_h);

}

void receive_send_part_up_z(buff_part* b_p, simu_param* sP, house_keeping* h_k, MPI_Comm comm, int idx_it)
{

  MPI_Request up_req;
  MPI_Status stat;
  //
  int down_proc_z = (sP->mpi_rank_z-1+sP->mpi_nb_proc_z)%sP->mpi_nb_proc_z;
  int up_proc_z   = (sP->mpi_rank_z+1)%sP->mpi_nb_proc_z;
  int down_proc_lin = sP->mpi_rank_y*sP->mpi_nb_proc_z + down_proc_z;
  int up_proc_lin   = sP->mpi_rank_y*sP->mpi_nb_proc_z + up_proc_z;

  // buff_part* buff_part_h = new buff_part;

  cudaDeviceSynchronize();
  // cudaMemcpy(buff_part_h, b_p, sizeof(*buff_part_h), cudaMemcpyDeviceToHost);
  // int nb_part_send = buff_part_h->next_idx_comm;
  int nb_part_send;
  int nb_part_rece ;
  cudaMemcpy(&nb_part_send, &b_p->next_idx_comm, sizeof(int), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  MPI_Irecv(&nb_part_rece, 1, MPI_INT, down_proc_lin, 1, comm, &up_req);
  MPI_Send( &nb_part_send, 1, MPI_INT, up_proc_lin  , 1, comm);
  MPI_Wait(&up_req, &stat);
  update_nb_part_rece_k<<<1, 1>>>(b_p, nb_part_rece);
  if (idx_it%rate_save_t_cst==0){
    HK_nb_part_send_up_k<<<1, 1>>>(h_k, nb_part_send, int(idx_it/rate_save_t_cst));
    cudaDeviceSynchronize();
  }

  if (nb_part_rece>buff_size_cst){
    printf("Too many particles to receive up for buffer.\n");
  }
  if (nb_part_send>buff_size_cst){
    printf("Too many particles to send up for buffer.\n");
  }

  if (nb_part_rece>0){
    #if NB_DIM==2
      MPI_Irecv(b_p->buff_rece, 6*nb_part_rece, MPI_FLOAT, down_proc_lin, 1, comm, &up_req);
    #elif NB_DIM==3
      MPI_Irecv(b_p->buff_rece, 7*nb_part_rece, MPI_FLOAT, down_proc_lin, 1, comm, &up_req);
    #endif
  }
  if (nb_part_send>0){
    #if NB_DIM==2
      MPI_Send( b_p->buff_send, 6*nb_part_send, MPI_FLOAT, up_proc_lin, 1, comm);
    #elif NB_DIM==3
      MPI_Send( b_p->buff_send, 7*nb_part_send, MPI_FLOAT, up_proc_lin, 1, comm);
    #endif
  }
  if (nb_part_rece>0){
    MPI_Wait(&up_req, &stat);
  }

  // delete(buff_part_h);

}

void comm_part(particles* p, simu_grid* grid, buff_part* b_p,
               simu_param* sP_h,simu_param* sP_d, house_keeping* h_k,
               MPI_Comm comm, int idx_it)
{

  if (sP_h->mpi_nb_proc_tot > 1){
    reset_idx_comm_k <<< 1, 1 >>>(b_p);
    search_idx_down_y_k<<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(p, grid, b_p, sP_d, idx_it);
    receive_send_part_down_y(b_p, sP_h, h_k, comm, idx_it);
    reset_idx_free_k <<< 1, 1 >>>(b_p);
    update_idx_free_k<<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(p, b_p);
    copy_buff_part_k <<< 1+buff_size_cst/tpb_cst, tpb_cst>>>(p, b_p);
    //
    reset_idx_comm_k <<< 1, 1 >>>(b_p);
    search_idx_up_y_k  <<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(p, grid, b_p, sP_d);
    receive_send_part_up_y(b_p, sP_h, h_k, comm, idx_it);
    reset_idx_free_k <<< 1, 1 >>>(b_p);
    update_idx_free_k<<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(p, b_p);
    copy_buff_part_k <<< 1+buff_size_cst/tpb_cst, tpb_cst>>>(p, b_p);
    //
    #if NB_DIM==3
      if (sP_h->mpi_nb_proc_z > 1){
        reset_idx_comm_k <<< 1, 1 >>>(b_p);
        search_idx_down_z_k<<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(p, grid, b_p, sP_d, idx_it);
        receive_send_part_down_z(b_p, sP_h, h_k, comm, idx_it);
        reset_idx_free_k <<< 1, 1 >>>(b_p);
        update_idx_free_k<<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(p, b_p);
        copy_buff_part_k <<< 1+buff_size_cst/tpb_cst, tpb_cst>>>(p, b_p);
        //
        reset_idx_comm_k <<< 1, 1 >>>(b_p);
        search_idx_up_z_k  <<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(p, grid, b_p, sP_d);
        receive_send_part_up_z(b_p, sP_h, h_k, comm, idx_it);
        reset_idx_free_k <<< 1, 1 >>>(b_p);
        update_idx_free_k<<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(p, b_p);
        copy_buff_part_k <<< 1+buff_size_cst/tpb_cst, tpb_cst>>>(p, b_p);
      }
    #endif
  }

}



#endif
