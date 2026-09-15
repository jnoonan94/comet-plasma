#include <stdio.h>
#include <iostream>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "mpi.h"

#include <sys/types.h>
#include <sys/stat.h>

#include "kernels_fields.cuh"
#include "kernels_HK.cuh"



void pre_checks()
{

  struct stat info;
  const char* pathname = "";

  bool pass_checks = true;

  char* local_rank_env = getenv("SLURM_LOCALID");
  int local_rank;
  if (local_rank_env) {
    local_rank = atoi(local_rank_env);
  }
  else{
    local_rank=0;
  }

  if (local_rank==0){
    std::cout << "._____________________________________________________________________________________\n|\n| Initial checks. " << std::endl;
  }

  pathname = "./products/HK";
  if ( stat( pathname, &info ) != 0 ){
    std::cout <<  "|\n| Cannot access " << pathname << std::endl;
    pass_checks = false;
  }
  pathname = "./products/probes";
  if ( stat( pathname, &info ) != 0 ){
    std::cout <<  "|\n| Cannot access " << pathname << std::endl;
    pass_checks = false;
  }

  if (nb_cell_per_shift_cst>nb_it_per_shift_cst){
    std::cout << "| Cometary ions too fast for the resolution: nb_cell_per_shift_cst > nb_it_per_shift_cst!" << std::endl;
    pass_checks = false;
  }

  if (rate_save_particles_cst%rate_save_field_cst!=0){
    std::cout << "| Fields and particles will not be saved at the same iterations." << std::endl;
    pass_checks = false;
  }

  if (decay_turb_cst && !periodic_yz_cst){
    std::cout << "| Decaying turbulence run but not periodic along y and z." << std::endl;
    pass_checks = false;
  }


  if (pass_checks){
    std::cout << "| All good." << std::endl;
  }
  else {
    sortie();
  }

  // if (local_rank==0){
  //   std::cout << "|_____________________________________________________________________________________" << std::endl;
  // }

}


void initialisation_cuda()
{
    /* http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-mpi-cuda-aware-gpudirect-eng.html */
    char* local_rank_env;
    int local_rank;
    cudaError_t cudaRet;

     /* Recovery of the local rank of the process via the environment variable
        set by Slurm, as  MPI_Comm_rank cannot be used here because this routine
        is used BEFORE the initialisation of MPI */
    local_rank_env = getenv("SLURM_LOCALID");


    if (local_rank_env) {
      local_rank = atoi(local_rank_env);
      // if (local_rank==0){
      //   std::cout << "._____________________________________________________________________________________" << std::endl;
      // }
      /* Define the GPU to use for each MPI process */
      cudaRet = cudaSetDevice(local_rank);
      std::cout << "| Setting device number " << local_rank << std::endl;
      if(cudaRet != cudaSuccess) {//CUDA_SUCCESS) {
          std::cout  << "| Erreur: cudaSetDevice has failed" << std::endl;
          exit(1);
      }
      if (local_rank==0){
        std::cout << "|_____________________________________________________________________________________" << std::endl;
      }
    }
    // else {
    //     printf("Error : impossible to determine the local rank of the process\n");
    //     exit(1);
    // }

    // Old:
    // int mpi_local_rank, mpi_local_size;
    // MPI_Comm mpi_local_comm;
    // MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank, MPI_INFO_NULL, &mpi_local_comm);
    // MPI_Comm_size(mpi_local_comm, &mpi_local_size);
    // MPI_Comm_rank(mpi_local_comm, &mpi_local_rank);
    // //
    // cudaSetDevice(mpi_local_rank%mpi_local_size);
}


void debug_menura(int idx_it, int ind_it_save, int str_ind_save, bool exit_leo, particles* pa_h, particles* pa_d,
               simu_fields* fields_h, simu_fields* fields_d, simu_B_field* B_h, simu_B_field* B_d,
               simu_param* simu_param_d, simu_param* simu_param_h, house_keeping* HK_h,
               trajectory* traj_h, trajectory* traj_d, moments* mom_h, moments* mom_d,
               root_mean_sqr* rms_h, root_mean_sqr* rms_d, bool light=false)
{

  cudaDeviceSynchronize();

  if (idx_it==ind_it_save){
    std::cout << "| Saving debug " << str_ind_save << std::endl;

    if (light){
      cudaMemcpy(fields_h, fields_d, sizeof(*fields_h), cudaMemcpyDeviceToHost);
      cudaMemcpy(B_h,      B_d,      sizeof(*B_h),      cudaMemcpyDeviceToHost);

      output_fields(fields_h, str_ind_save, simu_param_h);
      output_density_species(fields_h, str_ind_save, simu_param_h);
      output_B_field(B_h, str_ind_save, simu_param_h);
    }
    else{
      // ohm_components_k<<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(fields_d, B_d, idx_it, simu_param_d);
      cudaMemcpy(pa_h,     pa_d,     sizeof(*pa_h),     cudaMemcpyDeviceToHost);
      cudaMemcpy(fields_h, fields_d, sizeof(*fields_h), cudaMemcpyDeviceToHost);
      cudaMemcpy(B_h,      B_d,      sizeof(*B_h),      cudaMemcpyDeviceToHost);
      cudaMemcpy(traj_h,   traj_d,   sizeof(*traj_h),   cudaMemcpyDeviceToHost);
      cudaMemcpy(mom_h,    mom_d,    sizeof(*mom_h),    cudaMemcpyDeviceToHost);
      cudaMemcpy(rms_h,    rms_d,    sizeof(*rms_h),    cudaMemcpyDeviceToHost);

      output_HK(HK_h, traj_h, mom_h, rms_h, simu_param_h, str_ind_save);
      output_fields(fields_h, str_ind_save, simu_param_h);
      output_density_species(fields_h, str_ind_save, simu_param_h);
      output_ohm_components(B_h, fields_h, simu_param_h, str_ind_save);
      output_B_field(B_h, str_ind_save, simu_param_h);
      // output_particles(pa_h, str_ind_save, simu_param_h);
    }

    if (exit_leo) { // && simu_param_h->mpi_nb_proc==1
      sortie();
    }
  }
}


void check_state(state_solver* sta_sol, simu_param* s_p, int error_ID=0, cudaError_t err=cudaSuccess)
{

  if (err!=cudaSuccess){
    sta_sol->error_ID = error_ID;
    sta_sol->cuda_error_str = cudaGetErrorString(err);
  }

  if (sta_sol->error_ID != 0){
    std::cout << "| Rank y " << s_p->mpi_rank_y << " rank z " << s_p->mpi_rank_z
              << " " << sta_sol->error_str2[sta_sol->error_ID-1]
              << " " << sta_sol->cuda_error_str << std::endl;
    exit(EXIT_FAILURE);
  }
}



void check_status(simu_param* sP, state_solver* sta_sol_h,
                       state_solver* sta_sol_d, MPI_Comm comm,
                       int idx_it, int tag=0)
{

  cudaMemcpy(sta_sol_h, sta_sol_d, sizeof(*sta_sol_h), cudaMemcpyDeviceToHost);

  if (sta_sol_h->error_ID >= 8){
    std::cout << "| ye Tag " << tag << ", it " << idx_it << ", error ID: " << sta_sol_h->error_ID << " " << sta_sol_h->error_str2[sta_sol_h->error_ID-1] << std::endl;
  }

  cudaError_t err;
  err = cudaGetLastError();

  if (err!=cudaSuccess){
    sta_sol_h->break_solver = true;
    std::cout << "|\n| check_status, tag " << tag << ", error at iteration nb. " << idx_it
              << ", rank y " << sP->mpi_rank_y
              << ", rank z " << sP->mpi_rank_z
              << ": " << cudaGetErrorString(err) << std::endl;
  }

  // CAREFUL! If for some reason not all processes are calling check_status, the run is locked!
  MPI_Allreduce(&sta_sol_h->break_solver, &sta_sol_h->break_solver, 1, MPI_C_BOOL, MPI_LOR, comm);


  if (sta_sol_h->break_solver){
    size_t free, total;
    int id;
    cudaGetDevice(&id);
    cudaError_t err2;
    err2 = cudaMemGetInfo(&free, &total);
    if (err2!=cudaSuccess){
      std::cout << "| check_status, tag " << tag << ", it " << idx_it << ", couldn't retrieve memory info." << std::endl;
    }
    else {
      std::cout << "| check_status, tag " << tag << ", it " << idx_it << ", available/free memory (Mb): " << total/1000000 << " " << free/1000000 << std::endl;
    }

    // if (sP->mpi_rank==0){
    //   std::cout << "|_____________________________________________________________________________________\n\n\n";
    // }
    //
    // // std::cout << "Rank " << s_p->mpi_rank << " " << sta_sol->error_str2[sta_sol->error_ID-1] << " " << sta_sol->cuda_error_str << std::endl;
    // sortie();
  }
}


void check_memory_device(int rank, int tag)
{

  size_t free, total;

  int id;
  cudaGetDevice( &id );
  cudaError_t err;
  err = cudaMemGetInfo( &free, &total );
  if (err!=cudaSuccess){
    std::cout << "._____________________________________________________________________________________\n|\n| Tag "
              << "| Couldn't retrieve memory info."
              << "\n|_____________________________________________________________________________________" << std::endl;
  }
  else{
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0){
      std::cout << "._____________________________________________________________________________________" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "| Tag " << tag << ", MPI rank " << rank << ", Device " << id << ", memory: free=" << free/1000000
              << ", total=" << total/1000000 << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0){
      std::cout << "|_____________________________________________________________________________________" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  // int dev_cnt = 0;
  // cudaGetDeviceCount( &dev_cnt );
  // printf( "rank %d, cnt %d\n", mpi_rank, dev_cnt );
  //
  // cudaDeviceProp prop;
  // for (int dev = 0; dev < dev_cnt; ++dev) {
  //     cudaGetDeviceProperties( &prop, dev );
  //     printf( "rank %d, dev %d, prop %s, pci %d, %d, %d\n",
  //             mpi_rank, dev,
  //             prop.name,
  //             prop.pciBusID,
  //             prop.pciDeviceID,
  //             prop.pciDomainID );
  // }
  // std::cout << "\n._______________________________________________________________________\n|\n";
  // std::cout << "| tag " << tag << " rank " << rank << std::endl;
  // std::cout << "| nb GPUs " << num_gpus << std::endl;
  // for ( int gpu_id = 0; gpu_id < num_gpus; gpu_id++ ) {
      // cudaSetDevice( gpu_id );
}

void check_grids(simu_B_field* B, simu_fields* fields, state_solver* sta_sol_h,
                 state_solver* sta_sol_d, int indIt, int nb_nodes_tot, int tag,
                 simu_param* sP_h, simu_param* sP_d)
{
  reset_state_solver_k<<< 1, 1 >>>(sta_sol_d);
  check_grids_k<<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(B, fields, sta_sol_d, indIt, nb_nodes_tot_cst, tag, sP_d);
  cudaMemcpy(sta_sol_h, sta_sol_d, sizeof(*sta_sol_h), cudaMemcpyDeviceToHost);
  if (sta_sol_h->error_ID != 0){
    std::cout << "| " << sta_sol_h->error_str2[sta_sol_h->error_ID-1] << std::endl;
  }
}



//__________________________________________________________________________________________________________________

void sortie()
{
  int id;
  cudaDeviceSynchronize();
  cudaGetDevice(&id);
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "|\n| Device " << id <<  " Off." << std::endl;
  std::cout << "|_____________________________________________________________________________________\n" << std::endl;
  // int error = 1;
  // MPI_Bcast(&error, 1, MPI_INT, simu_param_h->mpi_rank, MPI_COMM_WORLD);
  MPI_Finalize();
  exit(EXIT_FAILURE);
}
