#include <stdio.h>
#include <iostream>
#include <fstream>

#include "parameters.h"
#include "structures.h"
#include "functions_i.h"
#include "functions_o.h"
#include "utils.h"
#include "kernels_calls.h"
#include "kernels_particles.cuh"
#include "kernels_fields.cuh"
#include "kernels_HK.cuh"
#include "kernels_DF.cuh"
#include "output_parameters.h"

#include "mpi.h"
#include <cuda.h>
#include <cuda_runtime.h>



//__________________________________________________________________________________________________________________
//
//  Main routine.
//
int main(int argc, char *argv[]){
  /**
  Yeet.
  */


  //____________________________________________________________________________
  // Setting up Cuda.
  pre_checks();


  //____________________________________________________________________________
  // Setting up Cuda.
  initialisation_cuda();


  //____________________________________________________________________________
  // Setting up MPI.
  MPI_Init(&argc, &argv);
  //
  int mpi_rank_lin, mpi_nb_proc_tot;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_lin);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_nb_proc_tot);
  //
  MPI_Barrier(MPI_COMM_WORLD);


  /*! Each process needs to have a different seed, to not produce the same random
  numbers at the same places.*/
  srand(mpi_rank_lin+seed_val_cst);



  int totalSizeMallocDevice = (sizeof(particles)
                              +sizeof(simu_grid)
                              +sizeof(simu_fields)
                              +sizeof(simu_B_field)*2
                              +sizeof(trajectory)
                              +sizeof(probes)
                              +sizeof(moments)
                              +sizeof(fields_time_space)
                              +sizeof(root_mean_sqr)
                              +sizeof(state_solver)
                              +sizeof(injector)
                              +sizeof(house_keeping)
                              +sizeof(buff_part))/1000000;



  //____________________________________________________________________________
  // Declaring objects and variables.
  simu_param* simu_param_h = new simu_param;
  simu_param* simu_param_d = new simu_param;
  int wrong_nb_proc = simu_param_h->init_rank_order(mpi_rank_lin, mpi_nb_proc_tot);
  if (wrong_nb_proc==1){
    sortie();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (simu_param_h->mpi_rank_lin==0){
    printf("\n._______________________________________________________________________\n|\n");
    if (simu_param_h->mpi_nb_proc_tot==1){
      printf("| Menura launched with 1 single process.\n");
      printf("| %i MBytes allocated on that GPU.\n|\n", totalSizeMallocDevice);
    }
    else {
      printf("| Menura launched with %i processes.\n", simu_param_h->mpi_nb_proc_tot);
      printf("| %i MBytes allocated per GPU.\n|\n", totalSizeMallocDevice);
    }
    #if DF_cst
    printf("| %i MBytes for the distribution function\n", sizeof(particles)/1000000);
    #else
    printf("| %i MBytes for particles\n", sizeof(particles)/1000000);
    #endif
    printf("| %i MBytes for fields\n", (sizeof(simu_fields)+2*sizeof(simu_B_field))/1000000);
    printf("| %i MBytes for fields space time\n", sizeof(fields_time_space)/1000000);
    printf("| %i MBytes for probes\n", sizeof(probes)/1000000);
    printf("|_______________________________________________________________________\n\n");

    size_t free, total;
    int id;
    cudaGetDevice(&id);
    cudaMemGetInfo(&free, &total);
    if (totalSizeMallocDevice>total/1000000){
      std::cout << "\nTotal memory to be allocated is too large " << totalSizeMallocDevice << " Mb vs. " << total << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);



  // simu_B_field_2* B_h = new simu_B_field_2(simu_param_h);
  // simu_B_field_2* B_d = new simu_B_field_2(simu_param_h);

  cudaError_t err;
  //
  state_solver* sta_sol_h = new state_solver;
  state_solver* sta_sol_d = new state_solver;
  //
  particles* pa_h        = new particles;
  particles* pa_d        = new particles;
  //
  #if DF_cst
    simu_DF* DF_h        = new simu_DF;
    simu_DF* DFa_d        = new simu_DF;
    simu_DF* DFb_d        = new simu_DF;
  #endif
  //
  buff_part* buff_part_h = new buff_part;
  buff_part* buff_part_d = new buff_part;
  buff_part_h->next_idx_comm = 999;
  //
  simu_grid *grid_h      = new simu_grid;
  simu_grid *grid_d      = new simu_grid;
  //
  simu_fields *fields_h  = new simu_fields;
  simu_fields *fields_d  = new simu_fields;
  //
  simu_B_field *Ba_h     = new simu_B_field;
  simu_B_field *Ba_d     = new simu_B_field;
  simu_B_field *Bb_h     = new simu_B_field;
  simu_B_field *Bb_d     = new simu_B_field;
  //
  trajectory *traj_h     = new trajectory;
  trajectory *traj_d     = new trajectory;
  //
  probes *prob_h     = new probes;
  probes *prob_d     = new probes;
  //
  moments *mom_h         = new moments;
  moments *mom_d         = new moments;
  //
  fields_time_space *fields_t_s_h = new fields_time_space;
  fields_time_space *fields_t_s_d = new fields_time_space;
  //
  root_mean_sqr *rms_h    = new root_mean_sqr;
  root_mean_sqr *rms_d    = new root_mean_sqr;
  //
  house_keeping* HK_h     = new house_keeping;
  house_keeping* HK_d     = new house_keeping;
  //
  curandState* device_states_d;
  float* random_floats_d;
  int nb_random_max = 0;
  //
  simu_tank *tank_h = new simu_tank;
  //
  injector *injec_h = new injector;
  injector *injec_d = new injector;
  //
  //
  float t = 0.;
  int idx_it = 0;
  int idx_save = 0;
  int nb_active_part_sw = 0;
  int nb_active_part_pla = 0;
  #if (obstacle_cst && !ORF_cst)
    int nb_part_in_slice;
  #endif


  //____________________________________________________________________________
  // Allocating memory on the device.
  //
  // check_memory_device(mpi_rank, 0);
  err = cudaMalloc((void**)&pa_d,         sizeof(*pa_d));
  check_state(sta_sol_h, simu_param_h, 1, err);
  err = cudaMalloc((void**)&simu_param_d, sizeof(*simu_param_d));
  check_state(sta_sol_h, simu_param_h, 2, err);
  err = cudaMalloc((void**)&grid_d,       sizeof(*grid_d));
  check_state(sta_sol_h, simu_param_h, 3, err);
  err = cudaMalloc((void**)&fields_d,     sizeof(*fields_d));
  check_state(sta_sol_h, simu_param_h, 4, err);
  // err = cudaMalloc((void**)&B_d,            sizeof(*B_d));
  // check_state(sta_sol_h, simu_param_h, 5, err);
  err = cudaMalloc((void**)&Ba_d,         sizeof(*Ba_d));
  check_state(sta_sol_h, simu_param_h, 5, err);
  err = cudaMalloc((void**)&Bb_d,         sizeof(*Bb_d));
  check_state(sta_sol_h, simu_param_h, 6, err);
  err = cudaMalloc((void**)&traj_d,       sizeof(*traj_d));
  check_state(sta_sol_h, simu_param_h, 7, err);
  err = cudaMalloc((void**)&prob_d,       sizeof(*prob_d));
  check_state(sta_sol_h, simu_param_h, 7, err);
  err = cudaMalloc((void**)&mom_d,        sizeof(*mom_d));
  check_state(sta_sol_h, simu_param_h, 8, err);
  err = cudaMalloc((void**)&fields_t_s_d, sizeof(*fields_t_s_d));
  check_state(sta_sol_h, simu_param_h, 9, err);
  err = cudaMalloc((void**)&rms_d,        sizeof(*rms_d));
  check_state(sta_sol_h, simu_param_h, 10, err);
  err = cudaMalloc((void**)&HK_d,         sizeof(*HK_d));
  check_state(sta_sol_h, simu_param_h, 13, err);
  err = cudaMalloc((void**)&sta_sol_d,    sizeof(*sta_sol_d));
  check_state(sta_sol_h, simu_param_h, 14, err);
  err = cudaMalloc((void**)&buff_part_d,  sizeof(*buff_part_d));
  check_state(sta_sol_h, simu_param_h, 15, err);
  err = cudaMalloc((void**)&injec_d,      sizeof(*injec_d));
  check_state(sta_sol_h, simu_param_h, 16, err);
  #if DF_cst
  err = cudaMalloc((void**)&DFa_d,      sizeof(*DFa_d));
  err = cudaMalloc((void**)&DFb_d,      sizeof(*DFb_d));
  #endif
  check_memory_device(mpi_rank_lin, 1);
  MPI_Barrier(MPI_COMM_WORLD);



  //____________________________________________________________________________
  /*! Initialisation of fields and particles.
  */
  if (simu_param_h->mpi_rank_lin==0){
    std::cout << "\n._______________________________________________________________________\n|" << std::endl;
    std::cout << "| Initialisation of the fields and particles..." << std::endl;
  }
  init_house_keeping(mom_h, traj_h, rms_h, simu_param_h, HK_h, sta_sol_h);
  init_grid(grid_h, simu_param_h);
  #if NB_DIM==2
    init_probes(prob_h, simu_param_h, grid_h);
  #endif
  //
  #if inject_pla_cst
    init_exosphere(buff_part_h, grid_h, simu_param_h, MPI_COMM_WORLD);
  #elif ionosphere_cst
    init_ionosphere(buff_part_h, grid_h, simu_param_h, MPI_COMM_WORLD);
  #endif
  //
  #if inject_turb_cst && !restart_cst
    init_B(Bb_h, simu_param_h);
    load_B_field(Ba_h, tank_h, simu_param_h, true,
                 path_inputs_str_cst, idx_it_inject_cst);
    load_fields(fields_h, tank_h, simu_param_h, true,
                 path_inputs_str_cst, idx_it_inject_cst);
    load_particles(pa_h, grid_h, tank_h, simu_param_h, MPI_COMM_WORLD, true,
                   path_inputs_str_cst, idx_it_inject_cst, true);
  #elif inject_turb_cst && restart_cst
    init_B(Bb_h, simu_param_h);
    /*! First, the restart fields and particles are loaded to the host and immediately
    to the device, to make room for the turbulent injection. */
    load_B_field(Ba_h, tank_h, simu_param_h, false,
                 path_inputs_restart_cst, idx_it_restart_cst);
    load_fields(fields_h, tank_h, simu_param_h, false,
                 path_inputs_restart_cst, idx_it_restart_cst);
    load_particles(pa_h, grid_h, tank_h, simu_param_h, MPI_COMM_WORLD, false,
                   path_inputs_restart_cst, idx_it_restart_cst, false);
    cudaMemcpy(fields_d,     fields_h,     sizeof(*fields_d),     cudaMemcpyHostToDevice);
    cudaMemcpy(Ba_d,         Ba_h,         sizeof(*Ba_d),         cudaMemcpyHostToDevice);
    cudaMemcpy(pa_d,         pa_h,         sizeof(*pa_d),         cudaMemcpyHostToDevice);
    std::cout << "| Restart loaded." <<std::endl;
    /*! The following are meant to fill the tank only, not the devices variables! */
    load_B_field(Ba_h, tank_h, simu_param_h, true,
                 path_inputs_str_cst, idx_it_inject_cst);
    load_fields(fields_h, tank_h, simu_param_h, true,
                 path_inputs_str_cst, idx_it_inject_cst);
    load_particles(pa_h, grid_h, tank_h, simu_param_h, MPI_COMM_WORLD, true,
                   path_inputs_str_cst, idx_it_inject_cst, true);
  #elif !inject_turb_cst && restart_cst
    init_B(Bb_h, simu_param_h);
    /*! First, the restart fields and particles are loaded to the host and immediately
    to the device, to make room for the turbulent injection. */
    load_B_field(Ba_h, tank_h, simu_param_h, false,
                 path_inputs_restart_cst, idx_it_restart_cst);
    load_fields(fields_h, tank_h, simu_param_h, false,
                 path_inputs_restart_cst, idx_it_restart_cst);
    load_particles(pa_h, grid_h, tank_h, simu_param_h, MPI_COMM_WORLD, false,
                   path_inputs_restart_cst, idx_it_restart_cst, false);
    cudaMemcpy(fields_d,     fields_h,     sizeof(*fields_d),     cudaMemcpyHostToDevice);
    cudaMemcpy(Ba_d,         Ba_h,         sizeof(*Ba_d),         cudaMemcpyHostToDevice);
    cudaMemcpy(pa_d,         pa_h,         sizeof(*pa_d),         cudaMemcpyHostToDevice);
    std::cout << "| Restart loaded." <<std::endl;
    /*! The following are meant to fill the tank only, not the devices variables! */
    // load_B_field(Ba_h, tank_h, simu_param_h, true,
    //              path_inputs_str_cst, idx_it_inject_cst);
    // load_fields(fields_h, tank_h, simu_param_h, true,
    //              path_inputs_str_cst, idx_it_inject_cst);
    // load_particles(pa_h, grid_h, tank_h, simu_param_h, MPI_COMM_WORLD, true,
    //                path_inputs_str_cst, idx_it_inject_cst, true);
  #else
    init_fields(fields_h, simu_param_h);
    #if decay_turb_cst
      init_fluctuations(Ba_h, pa_h, grid_h, simu_param_h, MPI_COMM_WORLD);
    #else
      init_B(Ba_h, simu_param_h);
      init_B(Bb_h, simu_param_h);
      init_part_node(pa_h, grid_h, simu_param_h);
    #endif
    //
    // init_B(Ba_h, simu_param_h);
    // init_alfven_fluctuations(Ba_h, simu_param_h);
    // init_part_global(pa_h, grid_h, simu_param_h);
    // init_part_column(pa_h, grid_h, simu_param_h);
    // init_part_2_stream(pa_h, grid_h, simu_param_h);
    // init_part_acoustic(pa_h, grid_h, simu_param_h);
  #endif
  //
  #if dipole_cst
    init_B_dipole(Ba_h, simu_param_h, grid_h);
    init_B_dipole(Bb_h, simu_param_h, grid_h);
  #endif
  #if DF_cst
    init_DF(DF_h, grid_h, simu_param_h);
  #endif
  if (simu_param_h->mpi_rank_lin == 0){
    std::cout << "|_______________________________________________________________________\n" << std::endl;
  }

  output_grid(grid_h, simu_param_h);
  output_fields(fields_h, -1, simu_param_h);
  output_B_field(Ba_h, -1, simu_param_h);
  // output_particles(pa_h, -1, simu_param_h);
  // output_DF(DF_h, simu_param_h, -1);
  //
  // sortie();
  //
  if (simu_param_h->mpi_rank_lin == 0){
    printout_parameters(simu_param_h);
    output_parameters(simu_param_h);
  }

  //____________________________________________________________________________
  // Dealing with random number generators and random numbers.
  //
  nb_random_max = std::max(9*int(buff_size_cst), 3*simu_param_h->nb_part_add_pla);
  nb_random_max = std::max(nb_random_max, 3*int(nb_nodes_cst));
  //
  int size_alloc_max =  std::max(int(injec_size_cst), simu_param_h->nb_part_add_pla);
  size_alloc_max = std::max(size_alloc_max, int(1.1*simu_param_h->sum_pla_proba));
  if (size_alloc_max > buff_size_cst){
    std::cout << "\n._______________________________________________________________________\n|\n";
    std::cout << "| The buffer size is too small (rank lin "<< simu_param_h->mpi_rank_lin << "):\n";
    std::cout << "|    buff_size_cst: " << buff_size_cst  << std::endl;
    std::cout << "|   injec_size_cst: " << injec_size_cst << std::endl;
    std::cout << "|  nb_part_add_pla: " << simu_param_h->nb_part_add_pla << std::endl;
    std::cout << "|    sum_pla_proba: " << simu_param_h->sum_pla_proba << std::endl;
    std::cout << "|_______________________________________________________________________\n\n";
    exit(EXIT_FAILURE);
  }
  //
  err = cudaMalloc((void**)&device_states_d, nb_random_max*sizeof(curandState));
  check_state(sta_sol_h, simu_param_h, 11, err);
  err = cudaMalloc((void**)&random_floats_d, nb_random_max*sizeof(float));
  check_state(sta_sol_h, simu_param_h, 12, err);
  setup_rand_k<<< 1+nb_random_max/tpb_cst, tpb_cst>>>(device_states_d, simu_param_h->mpi_rank_lin, nb_random_max);




  //____________________________________________________________________________
  // Copying initialised variables on the GPU.
  //
  cudaMemcpy(simu_param_d, simu_param_h, sizeof(*simu_param_d), cudaMemcpyHostToDevice);
  if (!restart_cst){
    cudaMemcpy(pa_d,         pa_h,         sizeof(*pa_d),         cudaMemcpyHostToDevice);
    cudaMemcpy(fields_d,     fields_h,     sizeof(*fields_d),     cudaMemcpyHostToDevice);
    cudaMemcpy(Ba_d,         Ba_h,         sizeof(*Ba_d),         cudaMemcpyHostToDevice);
  }
  cudaMemcpy(grid_d,       grid_h,       sizeof(*grid_d),       cudaMemcpyHostToDevice);
  cudaMemcpy(Bb_d,         Bb_h,         sizeof(*Bb_d),         cudaMemcpyHostToDevice);
  cudaMemcpy(traj_d,       traj_h,       sizeof(*traj_d),       cudaMemcpyHostToDevice);
  cudaMemcpy(prob_d,       prob_h,       sizeof(*prob_d),       cudaMemcpyHostToDevice);
  cudaMemcpy(mom_d,        mom_h,        sizeof(*mom_d),        cudaMemcpyHostToDevice);
  cudaMemcpy(fields_t_s_d, fields_t_s_h, sizeof(*fields_t_s_d), cudaMemcpyHostToDevice);
  cudaMemcpy(rms_d,        rms_h,        sizeof(*rms_d),        cudaMemcpyHostToDevice);
  cudaMemcpy(HK_d,         HK_h,         sizeof(*HK_d),         cudaMemcpyHostToDevice);
  cudaMemcpy(sta_sol_d,    sta_sol_h,    sizeof(*sta_sol_d),    cudaMemcpyHostToDevice);
  cudaMemcpy(buff_part_d,  buff_part_h,  sizeof(*buff_part_d),  cudaMemcpyHostToDevice);
  #if DF_cst
  cudaMemcpy(DFa_d,        DF_h,         sizeof(*DFa_d),         cudaMemcpyHostToDevice);
  cudaMemcpy(DFb_d,        DF_h,         sizeof(*DFb_d),         cudaMemcpyHostToDevice);
  #endif
  //
  check_memory_device(mpi_rank_lin, 2);

  MPI_Barrier(MPI_COMM_WORLD);


  if (0){
    // Studying initial div(B).
    #if yee_cst
      unstag_B_k   <<< 1+(len_x_cst+3)*(len_y_cst+3)/tpb_cst, tpb_cst >>>(Ba_d);
    #endif
    #if !yee_cst
      div_B_k <<< 1 + nb_nodes_cst/tpb_cst, tpb_cst >>>(Ba_d, fields_d, simu_param_d);
    #else
      #if NB_DIM==2
        div_B_yee_k <<< 1 + (len_x_cst+3)*(len_y_cst+3)/tpb_cst, tpb_cst >>>(Ba_d, fields_d, simu_param_d);
      #elif NB_DIM==3
        div_B_yee_k <<< 1 + (len_x_cst+3)*(len_y_cst+3)*(len_z_cst+3)/tpb_cst, tpb_cst >>>(Ba_d, fields_d, simu_param_d);
      #endif
    #endif
    std::cout << "| Saving debug -1!" << std::endl;
    debug_menura(idx_it, 0, -1, true, pa_h, pa_d,
                 fields_h, fields_d, Ba_h, Ba_d,
                 simu_param_d, simu_param_h, HK_h,
                 traj_h, traj_d, mom_h, mom_d,
                 rms_h, rms_d);
  }





  //______________________________________________________________________________________
  // Main loop.
  //
  time_t timer;
  time(&timer);
  if (simu_param_h->mpi_rank_lin == 0){
    std::cout << "\n.-____________________________________________________________________________________\n|" << std::endl;
  }
  //

  while ( (idx_it < nb_it_max_cst) && (!sta_sol_h->break_solver) ){


    //_Step -1, injections._____________________________________________________
    #if (obstacle_cst && !ORF_cst)

      set_centre_k<<< 1, 1 >>>(grid_d, simu_param_d, idx_it);

      if (idx_it%nb_it_per_shift_cst==0){
        //
        if (inject_turb_cst){
          /*! Fields, particles and obstacles first need to be shifted from
          * nb_cell_per_shift_cst nodes. */
          shift_particles_k<<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(pa_d, simu_param_d, grid_d, idx_it);
          copy_fields_k<<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(Ba_d, Bb_d, fields_d);
          shift_field_k<<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(Bb_d, Ba_d, fields_d);
          //
          if (idx_it>0){
            /*! Here we re-inject the same particles as previous injection,
            * but one slice downstream, to mitigate "thermal diffusion" issues.
            * Further detail in shift_particles_k . */
            reset_idx_free_k <<< 1, 1 >>>(buff_part_d);
            update_idx_free_k<<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(pa_d, buff_part_d);
            inject_part_k<<< 1+nb_part_in_slice/tpb_cst, tpb_cst >>>(injec_d, pa_d, buff_part_d, grid_d, simu_param_d, nb_part_in_slice, true);
          }
          /* Number of particles to add, from the right slice of the tank:*/
          nb_part_in_slice = tank_h->sliceWidth[tank_h->idx_x];
          copy_injector(pa_h, tank_h, injec_h, nb_part_in_slice, simu_param_h, grid_h, idx_it);
          err = cudaMemcpy(injec_d, injec_h, sizeof(*injec_d), cudaMemcpyHostToDevice);
          reset_idx_free_k <<< 1, 1 >>>(buff_part_d);
          update_idx_free_k<<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(pa_d, buff_part_d);
          inject_part_k<<< 1+nb_part_in_slice/tpb_cst, tpb_cst >>>(injec_d, pa_d, buff_part_d, grid_d, simu_param_d, nb_part_in_slice, false);
          increment_ind_slice(tank_h);
          //
          #if NB_DIM==2
            inject_E_B_k<<< 1+(2+nb_cell_per_shift_cst)*(len_y_cst+4)/tpb_cst, tpb_cst >>>(injec_d, Ba_d, Bb_d, fields_d);//tank_h->indRow);
          #elif NB_DIM==3
            inject_E_B_k<<< 1+(2+nb_cell_per_shift_cst)*(len_y_cst+4)*(len_z_cst+4)/tpb_cst, tpb_cst >>>(injec_d, Ba_d, Bb_d, fields_d);//tank_h->indRow);
          #endif
        }
        else {
            shift_particles_k<<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(pa_d, simu_param_d, grid_d, idx_it);
            rand_uniform_k<<< 1+nb_random_max/tpb_cst, tpb_cst >>>(random_floats_d, device_states_d, nb_random_max);
            reset_idx_free_k<<< 1, 1 >>>(buff_part_d);
            update_idx_free_k<<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(pa_d, buff_part_d);
            add_part_SW_node_k<<< 1+(2*nb_cell_per_shift_cst*simu_param_h->nb_part_add_sw)/tpb_cst, tpb_cst >>>(pa_d, buff_part_d, grid_d, random_floats_d, simu_param_d, 0.);
            //
            copy_fields_k<<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(Ba_d, Bb_d, fields_d);
            shift_field_k<<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(Bb_d, Ba_d, fields_d);
            #if NB_DIM==2
              inject_const_B_k<<< 1+(len_y_cst+4)/tpb_cst, tpb_cst >>>(Ba_d, simu_param_d);
              // inject_wave_B_k <<< 1+(len_y_cst+4)/tpb_cst, tpb_cst >>>(Ba_d, simu_param_d, grid_d, idx_it);
              inject_vortex_B_k <<< 1+(len_y_cst+4)/tpb_cst, tpb_cst >>>(Ba_d, simu_param_d, grid_d, idx_it);
            #elif NB_DIM==3
              inject_const_B_k<<< 1+(len_y_cst+4)*(len_z_cst+4)/tpb_cst, tpb_cst >>>(Ba_d, simu_param_d);
              // inject_wave_B_k <<< 1+(len_y_cst+4)*(len_z_cst+4)/tpb_cst, tpb_cst >>>(Ba_d, simu_param_d, idx_it);
            #endif
        }
      }
    #endif
    //
    #if (inject_pla_cst || ionosphere_cst)
      if (simu_param_h->nb_part_add_pla>0.){
        rand_uniform_k    <<< 1+nb_random_max/tpb_cst, tpb_cst >>>(random_floats_d, device_states_d, nb_random_max);
        reset_idx_free_k  <<< 1, 1 >>>(buff_part_d);
        update_idx_free_k <<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(pa_d, buff_part_d);
        add_part_exo_node_k <<< 1+simu_param_h->nb_part_add_pla/tpb_cst, tpb_cst >>>(pa_d, buff_part_d, grid_d, random_floats_d, idx_it, simu_param_d);
      }
      rand_uniform_k    <<< 1+nb_random_max/tpb_cst, tpb_cst >>>(random_floats_d, device_states_d, nb_random_max);
      reset_idx_free_k  <<< 1, 1 >>>(buff_part_d);
      update_idx_free_k <<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(pa_d, buff_part_d);
      reset_idx_free_k  <<< 1, 1 >>>(buff_part_d);
      add_proba_exo_k   <<< 1+nb_nodes_cst/tpb_cst , tpb_cst >>>(pa_d, buff_part_d, grid_d, random_floats_d, idx_it, simu_param_d, sta_sol_d, HK_d);
    #endif





    #if solve_EB_cst
    //________________________________________________________________________________________________________________
    // The solver per se. All kernels calls are done in kernels_calls.h for readability
    // of this main loop.
    //
    // Step 0, mapping the particles on the grid at mixed time r(n) & v(n+1/2)._
    moments_mapping(pa_d, fields_d, grid_d, injec_d, simu_param_d, simu_param_h, idx_it, false, 1);
    //
    // Step 1, position push.___________________________________________________
    boris_pos_k<<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(pa_d, grid_d, simu_param_d);
    //
    #if ORF_cst
     rand_uniform_k       <<< 1+nb_random_max/tpb_cst, tpb_cst >>>(random_floats_d, device_states_d, nb_random_max) ;
     reset_idx_free_k     <<< 1, 1 >>>(buff_part_d) ;
     update_idx_free_k    <<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(pa_d, buff_part_d) ;
     inject_SW_ORF_k<<< 1+simu_param_h->nb_part_add_sw/tpb_cst, tpb_cst >>>(pa_d, buff_part_d, grid_d, random_floats_d, simu_param_d) ;
    #endif
    //
    comm_part(pa_d, grid_d, buff_part_d, simu_param_h, simu_param_d, HK_d, MPI_COMM_WORLD, idx_it);
    //
    // Step 2, mapping the particles on the grid at mixed time r(n+1) & v(n+1/2).
    moments_mapping(pa_d, fields_d, grid_d, injec_d, simu_param_d, simu_param_h, idx_it, true, 2);
    //
    // Step 3, average moments._________________________________________________
    average_moments_k<<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(fields_d);
    //
    // Step 4, B(n+1/2) <- B(n-1/2)_____________________________________________
    #if (dipole_cst && !ORF_cst)
      float delta_x = .5*simu_param_h->dt*simu_param_h->v_obs;
      #if NB_DIM==2
        compute_potential_k<<< 1+(len_x_cst+4)*(len_y_cst+4)*5/tpb_cst, tpb_cst >>>(Ba_d, grid_d, simu_param_d, delta_x);
      #elif NB_DIM==3
        compute_potential_k<<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(Ba_d, grid_d, simu_param_d, delta_x);
      #endif
      compute_dipole_k<<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(Ba_d->B_dip, Ba_d, fields_d, grid_d, simu_param_d);
      copy_dipole_k<<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(Ba_d, Bb_d);
    #endif
    //
    ohm(Ba_d, fields_d, grid_d, idx_it, simu_param_h, simu_param_d, MPI_COMM_WORLD, 0);
    faraday(Ba_d, Bb_d, fields_d, simu_param_h->sub_dt, simu_param_d, simu_param_h, grid_d, 0., simu_param_h->sub_dx);
    check_smooth(fields_d, Ba_d, simu_param_h, simu_param_d, grid_d, sta_sol_d, idx_it);
    for (int idx_sub=0; idx_sub<int((simu_param_h->nb_sub_cycles-1)/2); idx_sub++){
       ohm(Bb_d, fields_d, grid_d, idx_it, simu_param_h, simu_param_d, MPI_COMM_WORLD, 0);
       faraday(Ba_d, Ba_d, fields_d, 2*simu_param_h->sub_dt, simu_param_d, simu_param_h, grid_d, 2*idx_sub*simu_param_h->sub_dx, (2*(idx_sub+1))*simu_param_h->sub_dx);
       //
       ohm(Ba_d, fields_d, grid_d, idx_it, simu_param_h, simu_param_d, MPI_COMM_WORLD, 0);
       faraday(Bb_d, Bb_d, fields_d, 2*simu_param_h->sub_dt, simu_param_d, simu_param_h, grid_d, (2*idx_sub+1)*simu_param_h->sub_dx, (2*(idx_sub+1)+1)*simu_param_h->sub_dx);
    }
    check_smooth(fields_d, Ba_d, simu_param_h, simu_param_d, grid_d, sta_sol_d, idx_it);
    ohm(Bb_d, fields_d, grid_d, idx_it, simu_param_h, simu_param_d, MPI_COMM_WORLD, 0);
    faraday(Ba_d, Ba_d, fields_d, simu_param_h->sub_dt, simu_param_d, simu_param_h, grid_d, simu_param_h->dX-simu_param_h->sub_dx, simu_param_h->dX);
    predict_correct_k<<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(Ba_d, Bb_d);
    boundaries_EB(fields_d, Ba_d->B, simu_param_h, MPI_COMM_WORLD);
    check_smooth(fields_d, Ba_d, simu_param_h, simu_param_d, grid_d, sta_sol_d, idx_it);
    //
    // Step 5, estimate of E(n+1) <- J+, rho(n+1)_______________________________
    #if (dipole_cst && !ORF_cst)
      delta_x = simu_param_h->dt*simu_param_h->v_obs;// + (idx_it%nb_it_per_shift_cst)*simu_param_h->dX;
      #if NB_DIM==2
        compute_potential_k<<< 1+(len_x_cst+4)*(len_y_cst+4)*5/tpb_cst, tpb_cst >>>(Ba_d, grid_d, simu_param_d, delta_x);
      #elif NB_DIM==3
        compute_potential_k<<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(Ba_d, grid_d, simu_param_d, delta_x);
      #endif
      compute_dipole_k<<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(Ba_d->B_dip, Ba_d, fields_d, grid_d, simu_param_d);
    #endif
    ohm(Ba_d, fields_d, grid_d, idx_it, simu_param_h, simu_param_d, MPI_COMM_WORLD, 1);
    //
    // Step 6, advance ion current._____________________________________________
    current_advance_k<<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(fields_d, Ba_d, simu_param_d, idx_it);
    //
    // Step 7, E(n+1) <- Ji(n+1), rho(n+1)_______________________________
    ohm(Ba_d, fields_d, grid_d, idx_it, simu_param_h, simu_param_d, MPI_COMM_WORLD, 2);
    #if ( (ORF_cst && dipole_cst) )
      if (1 && idx_it%10==0){
        // copy_stag_k  <<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(fields_d->Ji, fields_d);
        // stag_k  <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(fields_d->Ji, fields_d);
        // destag_k<<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(fields_d->Ji, fields_d, grid_d);
        copy_stag_k  <<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(Ba_d->B, fields_d);
        stag_k  <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(Ba_d->B, fields_d);
        destag_k<<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(Ba_d->B, fields_d, grid_d);
        // copy_stag_scalar_k  <<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(fields_d->density, fields_d);
        // stag_scalar_k  <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(fields_d->density, fields_d);
        // destag_scalar_k<<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(fields_d->density, fields_d);
        copy_stag_k  <<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(fields_d->E, fields_d);
        stag_k  <<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(fields_d->E, fields_d);
        destag_k<<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(fields_d->E, fields_d, grid_d);
      }
    #endif
    //
    // Step 8___________________________________________________________________
    boris_vel_k<<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(pa_d, fields_d, Ba_d, grid_d, simu_param_d);
    //
    // End of the solver.
    //________________________________________________________________________________________________________________















    #elif (!solve_EB_cst && !DF_cst)
    boris_pos_k<<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(pa_d, grid_d, simu_param_d);
    boris_vel_test_part_k<<< 1+pool_size_cst/tpb_cst, tpb_cst >>>(pa_d, fields_d, Ba_d, grid_d, simu_param_d, idx_it);
    if (idx_it%rate_save_field_cst==0){
      analytical_EB_k<<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(Ba_d, fields_d, simu_param_d, grid_d, idx_it);
      moments_mapping(pa_d, fields_d, grid_d, injec_d, simu_param_d, simu_param_h, idx_it, true, 2);
      average_moments_k<<< 1+nb_nodes_tot_cst/tpb_cst, tpb_cst >>>(fields_d);
    }
    //
    //
    //
    #elif (!solve_EB_cst && DF_cst)
    van_leer_rx_k      <<< 1+(len_x_cst*len_v_cst*len_v_cst*len_v_cst)/tpb_cst, tpb_cst >>>(DFa_d, DFb_d, grid_d, simu_param_d, sta_sol_d) ;
    copy_guard_rx_k    <<< 1+     (len_v_cst*len_v_cst*len_v_cst)/tpb_cst, tpb_cst >>>(DFb_d, simu_param_d) ;
    // transfer_guard_rx_k<<< 1+     (len_v_cst*len_v_cst*len_v_cst)/tpb_cst, tpb_cst >>>(DFb_d, DFa_d, simu_param_d) ;
    //
    EB_fields_k <<< 1+(len_x_cst+4)/tpb_cst, tpb_cst >>>(DFa_d, grid_d, simu_param_d, idx_it) ;
    EB_fields_k <<< 1+(len_x_cst+4)/tpb_cst, tpb_cst >>>(DFb_d, grid_d, simu_param_d, idx_it) ;
    //
    van_leer_vx_k<<< 1+(len_x_cst*len_v_cst*len_v_cst*len_v_cst)/tpb_cst, tpb_cst >>>(DFb_d, DFa_d, fields_d, grid_d, simu_param_d, t, sta_sol_d) ;
    van_leer_vy_k<<< 1+(len_x_cst*len_v_cst*len_v_cst*len_v_cst)/tpb_cst, tpb_cst >>>(DFa_d, DFb_d, fields_d, grid_d, simu_param_d, t, sta_sol_d) ;
    van_leer_vz_k<<< 1+(len_x_cst*len_v_cst*len_v_cst*len_v_cst)/tpb_cst, tpb_cst >>>(DFb_d, DFa_d, fields_d, grid_d, simu_param_d, t, sta_sol_d) ;
    van_leer_vy_k<<< 1+(len_x_cst*len_v_cst*len_v_cst*len_v_cst)/tpb_cst, tpb_cst >>>(DFa_d, DFb_d, fields_d, grid_d, simu_param_d, t, sta_sol_d) ;
    transfer_all_k<<< 1+((len_x_cst+4)*len_v_cst*len_v_cst*len_v_cst)/tpb_cst, tpb_cst >>>(DFb_d, DFa_d, grid_d, simu_param_d);
    //
    // if (idx_it==0){
    //   cudaMemcpy(DF_h,   DFa_d,   sizeof(*DF_h),   cudaMemcpyDeviceToHost);
    //   output_DF(DF_h, simu_param_h, 999);
    //   sortie();
    // }
    if (idx_it%rate_save_particles_cst==0){
      cudaMemcpy(DF_h,   DFa_d,   sizeof(*DF_h),   cudaMemcpyDeviceToHost);
      output_DF(DF_h, simu_param_h, idx_it);
      // sortie();
    }
    #endif








    //_Probe work.______________________________________________________________
    push_probes_k <<< 1 + nb_probes_cst/tpb_cst, tpb_cst >>>(prob_d, simu_param_d, grid_d, idx_it);
    probe_fields_k<<< 1 + nb_probes_cst/tpb_cst, tpb_cst >>>(prob_d, fields_d, Ba_d, grid_d, simu_param_d, idx_it);






    // if (1 && idx_it>2470){
    //   // std::cout << "| debug_menura 999, it. " << idx_it << ", rank lin " << simu_param_h->mpi_rank_lin << std::endl;
    //   moments_mapping(pa_d, fields_d, grid_d, injec_d, simu_param_d, simu_param_h, idx_it, false, 1);
    //   debug_menura(idx_it, idx_it, idx_it, false, pa_h, pa_d,
    //                fields_h, fields_d, Ba_h, Ba_d,
    //                simu_param_d, simu_param_h, HK_h,
    //                traj_h, traj_d, mom_h, mom_d,
    //                rms_h, rms_d, false);
    // }

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);


    #if !DF_cst
      check_grids(Ba_d, fields_d, sta_sol_h, sta_sol_d, idx_it, nb_nodes_tot_cst, 3, simu_param_h, simu_param_d);
    #endif
    check_status(simu_param_h, sta_sol_h, sta_sol_d, MPI_COMM_WORLD, idx_it, 0);

    //_House-keeping & Saving intermediate states.______________________________
    #if obstacle_cst && NB_DIM==2
    if (idx_it%nb_it_per_shift_cst){
      get_boundary_position_k<<< 1+len_y_cst/tpb_cst, tpb_cst >>>(Ba_d, fields_d, idx_it);
    }
    #endif
    if (idx_it%rate_save_t_cst==0 && idx_it < nb_it_max_cst){
      idx_save = int(idx_it/rate_save_t_cst);
      // update_traj_k             <<< 1, 1 >>>(pa_d, traj_d, idx_save, 0);
      HK_run_time_k             <<< 1, 1 >>>(HK_d, idx_save, t, (time(NULL)-timer));
      count_rms(Ba_d, fields_d, rms_d, idx_save, tpb_cst, simu_param_h, simu_param_d);
      moments_particles_k       <<< 1+pool_size_cst/tpb_cst,  tpb_cst >>>(pa_d, mom_d, idx_save, simu_param_d, HK_d);
      #if !yee_cst
        div_B_k <<< 1 + nb_nodes_cst/tpb_cst, tpb_cst >>>(Ba_d, fields_d, simu_param_d);
      #else
        #if NB_DIM==2
          div_B_yee_k <<< 1 + (len_x_cst+3)*(len_y_cst+3)/tpb_cst, tpb_cst >>>(Ba_d, fields_d, simu_param_d);
        #elif NB_DIM==3
          div_B_yee_k <<< 1 + (len_x_cst+3)*(len_y_cst+3)*(len_z_cst+3)/tpb_cst, tpb_cst >>>(Ba_d, fields_d, simu_param_d);
        #endif
      #endif
      moments_grids_k           <<< 1+nb_nodes_cst/tpb_cst,   tpb_cst >>>(Ba_d, fields_d, mom_d, idx_save);
      divide_moments_k          <<< 1, 1 >>>(mom_d, idx_save, simu_param_d);

      update_fields_time_space_k<<< 1+len_save_x_cst/tpb_cst, tpb_cst >>>(fields_d, Ba_d, fields_t_s_d, idx_save, simu_param_d);
      //
      cudaMemcpy(HK_h,   HK_d,   sizeof(*HK_h),   cudaMemcpyDeviceToHost);
      cudaMemcpy(traj_h, traj_d, sizeof(*traj_h), cudaMemcpyDeviceToHost);
      cudaMemcpy(prob_h, prob_d, sizeof(*prob_h), cudaMemcpyDeviceToHost);
      cudaMemcpy(mom_h,  mom_d,  sizeof(*mom_h),  cudaMemcpyDeviceToHost);
      cudaMemcpy(rms_h,  rms_d,  sizeof(*rms_h),  cudaMemcpyDeviceToHost);
      cudaMemcpy(fields_t_s_h,  fields_t_s_d,  sizeof(*fields_t_s_h),  cudaMemcpyDeviceToHost);
      output_HK(HK_h, traj_h, mom_h, rms_h, simu_param_h, idx_save);
      output_fields_time_space(fields_t_s_h, idx_it, simu_param_h);
      // output_probes(prob_h, simu_param_h);
      // //
      if ( (simu_param_h->mpi_rank_lin == 0) && (idx_it%(1*rate_save_t_cst)==0)){
        cudaMemcpy(&nb_active_part_sw, &HK_d->nbPartSW[idx_save], sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&nb_active_part_pla, &HK_d->nbPartPla[idx_save], sizeof(int), cudaMemcpyDeviceToHost);
        printf("| Iteration: %4i, time: %7.2f (elapsed run time: %4i s, remaining: %7.2f s)\n",
                  idx_it, t, (time(NULL)-timer),
                  float(nb_it_max_cst-idx_it)*float(time(NULL)-timer)/float(idx_it));
        printf("|                              %3.f %% of the particle pool used on rank 0.\n",
                  100*float(nb_active_part_sw+nb_active_part_pla)/float(pool_size_cst) );
      }
    }
    //__________________________________________________________________________
    if (1 && (idx_it%rate_save_field_cst==0)) {
      cudaMemcpy(fields_h,   fields_d,   sizeof(*fields_h),   cudaMemcpyDeviceToHost);
      cudaMemcpy(Ba_h,       Ba_d,       sizeof(*Ba_h),       cudaMemcpyDeviceToHost);
      //
      output_fields(fields_h, idx_it, simu_param_h);
      // output_density_species(fields_h, idx_it, simu_param_h);
      output_B_field(Ba_h, idx_it, simu_param_h);
    }
    //
    //__________________________________________________________________________
    if (1 && obstacle_cst && restart_cst && (idx_it%20==0) && idx_it>=0 && idx_it<=8000){
      cudaMemcpy(fields_h,   fields_d,   sizeof(*fields_h),   cudaMemcpyDeviceToHost);
      cudaMemcpy(Ba_h,       Ba_d,       sizeof(*Ba_h),       cudaMemcpyDeviceToHost);
      //
      output_fields(fields_h, idx_it, simu_param_h, false);
      output_B_field(Ba_h, idx_it, simu_param_h);
    }
    // if (0 && idx_it>=0 && idx_it%rate_save_field_cst==0) {
    //   // ohm_components_k<<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(fields_d, Ba_d, idx_it, simu_param_d);
    //   cudaMemcpy(fields_h,   fields_d,   sizeof(*fields_h),   cudaMemcpyDeviceToHost);
    //   output_ohm_components(Ba_h, fields_h, simu_param_h, idx_it);
    // }
    if (0 && (idx_it>70)) {
      // ohm_components_k<<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(fields_d, Ba_d, idx_it, simu_param_d);
      cudaMemcpy(fields_h,   fields_d,   sizeof(*fields_h),   cudaMemcpyDeviceToHost);
      cudaMemcpy(Ba_h,       Ba_d,       sizeof(*Ba_h),       cudaMemcpyDeviceToHost);
      //
      output_fields(fields_h, idx_it, simu_param_h);
      output_density_species(fields_h, idx_it, simu_param_h);
      output_B_field(Ba_h, idx_it, simu_param_h);
      // output_ohm_components(Ba_h, fields_h, simu_param_h, idx_it);
    }
    //
    if (0 && (idx_it>2460) && (idx_it<2490)) {
      // ohm_components_k<<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(fields_d, Ba_d, idx_it, simu_param_d);
      cudaMemcpy(fields_h,   fields_d,   sizeof(*fields_h),   cudaMemcpyDeviceToHost);
      cudaMemcpy(Ba_h,       Ba_d,       sizeof(*Ba_h),       cudaMemcpyDeviceToHost);
      //
      output_fields(fields_h, idx_it, simu_param_h);
      output_density_species(fields_h, idx_it, simu_param_h);
      output_B_field(Ba_h, idx_it, simu_param_h);
      // output_ohm_components(Ba_h, fields_h, simu_param_h, idx_it);
    }
    //
    if (1 && idx_it>=8000 && decay_turb_cst && !DF_cst && idx_it%rate_save_particles_cst==0 ) {
      cudaMemcpy(pa_h, pa_d, sizeof(*pa_h), cudaMemcpyDeviceToHost);
      //
      output_particles(pa_h, idx_it, simu_param_h);
    }
    if (1 && (idx_it==4000)) {
      cudaMemcpy(pa_h, pa_d, sizeof(*pa_h), cudaMemcpyDeviceToHost);
      //
      output_particles(pa_h, idx_it, simu_param_h);
    }
    //______________________________________________________________________________


    //________________________________________________________________
    // Check simulation's state.
    cudaDeviceSynchronize();
    #if !DF_cst
      check_grids(Ba_d, fields_d, sta_sol_h, sta_sol_d, idx_it, nb_nodes_tot_cst, 9, simu_param_h, simu_param_d);
    #endif
    check_status(simu_param_h, sta_sol_h, sta_sol_d, MPI_COMM_WORLD, idx_it, 1);
    //__________________________________________________________________________

    //________________________________________________________________
    // Advancement in time.
    t += simu_param_h->dt;
    idx_it++;
    // MPI_Barrier(MPI_COMM_WORLD); // So every process takes time to hear if any other process broadcasted a break_solver true.
    //______________________________________________________________________________

  } // End of main-loop.


  // ohm_components_k<<< 1+nb_nodes_cst/tpb_cst, tpb_cst >>>(fields_d, Ba_d, idx_it, simu_param_d);
  if (simu_param_h->mpi_rank_lin == 0){
    printf("|\n| Solver (%i it.): %.2ld s.\n|_____________________________________________________________________________________\n\n\n", idx_it, time(NULL)-timer);
  }
  //______________________________________________________________________________________



  //____________________________________________________________________________
  // Bringing all variables back to the CPU and saving them.
  if (!restart_cst){
    cudaMemcpy(pa_h,         pa_d,         sizeof(*pa_h),         cudaMemcpyDeviceToHost);
  }
  cudaMemcpy(fields_h,     fields_d,     sizeof(*fields_h),     cudaMemcpyDeviceToHost);
  cudaMemcpy(Ba_h,         Ba_d,         sizeof(*Ba_h),         cudaMemcpyDeviceToHost);
  cudaMemcpy(traj_h,       traj_d,       sizeof(*traj_h),       cudaMemcpyDeviceToHost);
  cudaMemcpy(prob_h,       prob_d,       sizeof(*prob_h),       cudaMemcpyDeviceToHost);
  cudaMemcpy(mom_h,        mom_d,        sizeof(*mom_h),        cudaMemcpyDeviceToHost);
  cudaMemcpy(fields_t_s_h, fields_t_s_d, sizeof(*fields_t_s_h), cudaMemcpyDeviceToHost);
  cudaMemcpy(rms_h,        rms_d,        sizeof(*rms_h),        cudaMemcpyDeviceToHost);
  cudaMemcpy(HK_h,         HK_d,         sizeof(*HK_h),         cudaMemcpyDeviceToHost);
  //
  // output_probes(prob_h, simu_param_h);
  output_B_field(Ba_h, idx_it, simu_param_h);
  output_fields(fields_h, idx_it, simu_param_h);
  output_density_species(fields_h, idx_it, simu_param_h);
  output_parameters(simu_param_h);
  output_grid(grid_h, simu_param_h);
  // output_ohm_components(Ba_h, fields_h, simu_param_h, idx_it);
  output_HK(HK_h, traj_h, mom_h, rms_h, simu_param_h, idx_it);
  // output_ohm_components(Ba_h, fields_h, simu_param_h, idx_it);
  output_fields_time_space(fields_t_s_h, idx_it, simu_param_h);
  if (!restart_cst){
    output_particles(pa_h, idx_it, simu_param_h);
  }
  //____
  cudaFree(pa_d);
  cudaFree(simu_param_d);
  cudaFree(grid_d);
  cudaFree(fields_d);
  cudaFree(Ba_d);
  cudaFree(Bb_d);
  cudaFree(traj_d);
  cudaFree(mom_d);
  cudaFree(rms_d);
  cudaFree(random_floats_d);
  cudaFree(device_states_d);
  cudaFree(fields_t_s_d);
  cudaFree(HK_d);
  cudaFree(sta_sol_d);
  cudaFree(buff_part_d);
  cudaFree(injec_d);
  //____
  delete(pa_h);
  delete(simu_param_h);
  delete(grid_h);
  delete(fields_h);
  delete(Ba_h);
  delete(Bb_h);
  delete(traj_h);
  delete(mom_h);
  delete(rms_h);
  // delete(fields_t_s_h);
  // delete(HK_h);
  // delete(sta_sol_h);

  printf(".\n");

  MPI_Finalize();
  return EXIT_SUCCESS;
}
//
//
//__________________________________________________________________________________________________________________
