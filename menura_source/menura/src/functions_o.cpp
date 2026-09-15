#include <cmath>
#include <time.h>

#include "parameters.h"
#include "structures.h"
#include "functions_o.h"
#include "cnpy.h"
#include <iostream>
#include <fstream>
#include <algorithm>

#include "mpi.h"

template <typename T> int sgn(T val)
{
    return (T(0) < val) - (val < T(0)) ;
}






//__________________________________________________________________________________________________________________
//
//  Modules for data handling.
//
void output_HK(house_keeping* HK, trajectory* traj, moments* mom, root_mean_sqr* rms, simu_param* sP, int idx_save)
{

  //_______________________
  std::string fileName = "products/HK/energy_elec_rank_" + std::to_string(sP->mpi_rank_y) + "_"
                         + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, mom->E_elec, {len_save_t_cst}, "w") ;
  fileName = "products/HK/energy_mag_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, mom->E_mag, {len_save_t_cst}, "w") ;
  fileName = "products/HK/energy_kin_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, mom->E_kin, {len_save_t_cst}, "w") ;

  float *flat_B     = new float[3*len_save_t_cst] ;
  float *flat_E     = new float[3*len_save_t_cst] ;
  float *flat_Ji    = new float[3*len_save_t_cst] ;
  float *flat_J_tot = new float[3*len_save_t_cst] ;
  float *flat_vort  = new float[3*len_save_t_cst] ;
  float *flat_B_mean      = new float[3*len_save_t_cst] ;
  float *flat_B_var       = new float[3*len_save_t_cst] ;
  float *flat_Ji_mean     = new float[3*len_save_t_cst] ;
  float *flat_vel_part    = new float[3*len_save_t_cst] ;
  float *flat_vel_th_part = new float[3*len_save_t_cst] ;
  int idx_lin ;
  for (int h=0; h<3; h++){
    for (int t=0; t<len_save_t_cst; t++){
      idx_lin = h*len_save_t_cst + t ;
      flat_B    [idx_lin] = rms->B    [h][t] ;
      flat_E    [idx_lin] = rms->E    [h][t] ;
      flat_Ji   [idx_lin] = rms->Ji   [h][t] ;
      flat_J_tot[idx_lin] = rms->J_tot[h][t] ;
      flat_vort [idx_lin] = rms->vort [h][t] ;
      //
      flat_B_mean     [idx_lin] = mom->B_mean     [h][t] ;
      flat_B_var      [idx_lin] = mom->B_var      [h][t] ;
      flat_Ji_mean    [idx_lin] = mom->Ji_mean    [h][t] ;
      flat_vel_part   [idx_lin] = mom->vel_part   [h][t] ;
      flat_vel_th_part[idx_lin] = mom->vel_th_part[h][t] ;
    }
  }
  fileName = "products/HK/rms_B_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, flat_B, {3, len_save_t_cst}, "w") ;
  fileName = "products/HK/rms_E_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, flat_E, {3, len_save_t_cst}, "w") ;
  fileName = "products/HK/rms_Ji_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, flat_Ji, {3, len_save_t_cst}, "w") ;
  fileName = "products/HK/rms_J_tot_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, flat_J_tot, {3, len_save_t_cst}, "w") ;
  fileName = "products/HK/rms_vort_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, flat_vort, {3, len_save_t_cst}, "w") ;
  fileName = "products/HK/B_mean_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, flat_B_mean, {3, len_save_t_cst}, "w") ;
  fileName = "products/HK/B_var_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, flat_B_var, {3, len_save_t_cst}, "w") ;
  fileName = "products/HK/Ji_mean_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, flat_Ji_mean, {3, len_save_t_cst}, "w") ;
  fileName = "products/HK/vel_part_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, flat_vel_part, {3, len_save_t_cst}, "w") ;
  fileName = "products/HK/vel_th_part_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, flat_vel_th_part, {3, len_save_t_cst}, "w") ;


  //_______________________
  fileName = "products/HK/active_part_0_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, HK->nbPartSW, {len_save_t_cst}, "w") ;
  fileName = "products/HK/active_part_1_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, HK->nbPartPla, {len_save_t_cst}, "w") ;
  //
  fileName = "products/HK/nb_part_comm_up_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, HK->nb_part_comm_up, {len_save_t_cst}, "w") ;
  fileName = "products/HK/nb_part_comm_down_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, HK->nb_part_comm_down, {len_save_t_cst}, "w") ;
  fileName = "products/HK/nb_pla_proba_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, HK->nb_pla_proba, {len_save_t_cst}, "w") ;
  //
  if (sP->mpi_rank_lin==0){
    fileName = "products/HK/simu_time.npy" ;
    cnpy::npy_save(fileName, HK->simuTime, {len_save_t_cst}, "w") ;
    fileName = "products/HK/run_time.npy" ;
    cnpy::npy_save(fileName, HK->runTime, {len_save_t_cst}, "w") ;
  }

  fileName = "products/HK/dens_tot_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
             + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName,  &mom->dens_tot[0], {len_save_t_cst}, "w") ;
  fileName = "products/HK/div_B_mean_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName,  &mom->div_B_mean[0], {len_save_t_cst}, "w") ;
  fileName = "products/HK/div_B_var_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName,  &mom->div_B_var[0], {len_save_t_cst}, "w") ;
  fileName = "products/HK/div_B_max_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName,  &mom->div_B_max[0], {len_save_t_cst}, "w") ;
}

void output_particles(particles* p, int indIt, simu_param* sP)
{

  /// It appears that cnpy cannot output files larger than some unknown size.
  ///   Say this size is 2 Go. This limit would be reached for over a million particles.
  ///   We save as many files smaller than 2 Gb as hundreds of millions of particles.
  ///   1e6 particles -> 19.1 Mb
  long unsigned int nb_valid_part_tot = 0 ;
  for (int n=0; n<pool_size_cst; n++){
    if (p->active[n] == true){
      nb_valid_part_tot++ ;
    }
  }
  //
  if (nb_valid_part_tot==0){
    std::cout << "| NO VALID PARTICLE TO BE OUTPUT!" << std::endl;
  }

  int nb_part_per_file = 100000000;
  int nb_files = int(std::ceil(nb_valid_part_tot/float(nb_part_per_file))) ; // 100000000
  int ind_file = 0 ;
  int ind_part_out = 0 ; /// Current index where to write a valid particle in the flat array.
                         /// Comes back to zero every  time a file is written out.
  int nb_part_out ; /// Number of particles in one particular file.
                    /// Either nb_part_per_file, or the remaining number of valid particles.
  int nb_part_out_tot = 0 ; /// Total number of valid particles already written out.
  int part_flat_size ;


  /// How many particles in the first (maybe only) file:
  if (ind_file<(nb_files-1)){
    nb_part_out = nb_part_per_file ;
    #if NB_DIM==2
      part_flat_size = 6*nb_part_per_file ;
    #elif NB_DIM==3
      part_flat_size = 7*nb_part_per_file ;
    #endif
  }
  else {
    nb_part_out = (nb_valid_part_tot-(nb_part_per_file*(nb_files-1))) ;
    #if NB_DIM==2
      part_flat_size = 6*nb_part_out ;
    #elif NB_DIM==3
      part_flat_size = 7*nb_part_out ;
    #endif
  }
  float *part_flat = new float[part_flat_size] ;

  /// Iterating over all particles in memory, valid or not, with index n:
  uint n=0 ;

  std::string file_name = "products/nb_particle_files_it" + std::to_string(indIt)
                          + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                          + std::to_string(sP->mpi_rank_z) + ".txt";
  std::ofstream thefile ;
  thefile.open (file_name) ;
  thefile << nb_files ;
  thefile.close() ;

  while ( nb_part_out_tot < nb_valid_part_tot ){

    if (p->active[n] == true){
      #if NB_DIM==2
        part_flat[ind_part_out*6]   = p->rx[n] ;
        part_flat[ind_part_out*6+1] = p->ry[n] ;
        part_flat[ind_part_out*6+2] = p->vx[n] ;
        part_flat[ind_part_out*6+3] = p->vy[n] ;
        part_flat[ind_part_out*6+4] = p->vz[n] ;
        part_flat[ind_part_out*6+5] = float(p->ID[n]) ;
      #elif NB_DIM==3
        part_flat[ind_part_out*7]   = p->rx[n] ;
        part_flat[ind_part_out*7+1] = p->ry[n] ;
        part_flat[ind_part_out*7+2] = p->rz[n] ;
        part_flat[ind_part_out*7+3] = p->vx[n] ;
        part_flat[ind_part_out*7+4] = p->vy[n] ;
        part_flat[ind_part_out*7+5] = p->vz[n] ;
        part_flat[ind_part_out*7+6] = float(p->ID[n]) ;
      #endif
      ind_part_out++ ;
      nb_part_out_tot++ ;
    }
    //
    n++ ;

    /// Gathered enough particles for that file:
    if (ind_part_out == nb_part_out){
      /// Write out the file:
      file_name = "products/particles_" + std::to_string(ind_file) + "_it" + std::to_string(indIt)
                  + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                  + std::to_string(sP->mpi_rank_z) + ".npy";
      #if NB_DIM==2
        cnpy::npy_save(file_name, part_flat, {(long unsigned int) (part_flat_size/6.), 6}, "w") ;
      #elif NB_DIM==3
        cnpy::npy_save(file_name, part_flat, {(long unsigned int) (part_flat_size/7.), 7}, "w") ;
      #endif
      //
      delete(part_flat) ;

      file_name = "products/nb_particles_" + std::to_string(ind_file) + "_it" + std::to_string(indIt)
                  + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                  + std::to_string(sP->mpi_rank_z) + ".txt";
      std::ofstream thefile ;
      thefile.open (file_name) ;
      thefile << nb_part_out ;
      thefile.close() ;

      /// Increase file index and number of particle to write in the next file:
      ind_file ++ ;
      if (ind_file<(nb_files-1)){
        nb_part_out = nb_part_per_file ;
        #if NB_DIM==2
          part_flat_size = 6*nb_part_per_file ;
        #elif NB_DIM==3
          part_flat_size = 7*nb_part_per_file ;
        #endif
      }
      else {
        nb_part_out = (nb_valid_part_tot-(nb_part_per_file*(nb_files-1))) ;
        #if NB_DIM==2
          part_flat_size = 6*nb_part_out ;
        #elif NB_DIM==3
          part_flat_size = 7*nb_part_out ;
        #endif
      }
      //
      ind_part_out = 0 ;
      part_flat = new float[part_flat_size] ;
    }

  }


}

void output_fields(simu_fields* fields, int indIt, simu_param* sP, bool full)
{

  #if NB_DIM==2

    std::string fileName = "";
    //
    fileName = "products/dens_spec1_it" + std::to_string(indIt)
                           + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                           + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName,  &fields->counts_pla[0][0], {len_x_cst+4,len_y_cst+4}, "w");
    //
    if (full){
      fileName = "products/dens_it" + std::to_string(indIt)
                             + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                             + std::to_string(sP->mpi_rank_z) + ".npy";
      cnpy::npy_save(fileName,  &fields->density[0][0], {len_x_cst+4,len_y_cst+4}, "w");
      //
      fileName = "products/dens_spec0_it" + std::to_string(indIt)
                             + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                             + std::to_string(sP->mpi_rank_z) + ".npy";
      cnpy::npy_save(fileName,  &fields->counts[0][0], {len_x_cst+4,len_y_cst+4}, "w");
      //
      fileName = "products/curr_it" + std::to_string(indIt)
                  + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                  + std::to_string(sP->mpi_rank_z) + ".npy";
      cnpy::npy_save(fileName, &fields->Ji[0][0][0], {3,len_x_cst+4,len_y_cst+4}, "w");
      //
      fileName = "products/curr_tot_it" + std::to_string(indIt)
                  + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                  + std::to_string(sP->mpi_rank_z) + ".npy";
      cnpy::npy_save(fileName, &fields->J_tot[0][0][0], {3,len_x_cst+4,len_y_cst+4}, "w");
      //
      fileName = "products/E_it" + std::to_string(indIt)
                  + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                  + std::to_string(sP->mpi_rank_z) + ".npy";
      cnpy::npy_save(fileName, &fields->E[0][0][0], {3,len_x_cst+4,len_y_cst+4}, "w");
      //
      fileName = "products/region_ID_it" + std::to_string(indIt)
                  + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                  + std::to_string(sP->mpi_rank_z) + ".npy";
      cnpy::npy_save(fileName, &fields->region_ID[0][0], {len_x_cst+4,len_y_cst+4}, "w");
      //
      fileName = "products/smoothed_patches_rank_"
                  + std::to_string(sP->mpi_rank_y) + "_"
                  + std::to_string(sP->mpi_rank_z) + ".npy";
      cnpy::npy_save(fileName, &fields->save_smooth[0][0], {3, smooth_patch_save_len_cst}, "w");
      //
      #if obstacle_cst
        fileName = "products/boundary_position_rank_"
                    + std::to_string(sP->mpi_rank_y) + "_"
                    + std::to_string(sP->mpi_rank_z) + ".npy";
        cnpy::npy_save(fileName, &fields->bound_pos[0][0], {int(nb_it_max_cst/nb_it_per_shift_cst), len_y_cst}, "w");
      #endif
    }
  //
  #elif NB_DIM==3

    std::string fileName = "products/dens_it" + std::to_string(indIt)
                + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, &fields->density_b[0][0][0], {len_x_cst+4, len_y_cst+4, len_z_cst+4}, "w") ;

    // fileName = "products/smooth_it" + std::to_string(indIt)
    //             + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
    //             + std::to_string(sP->mpi_rank_z) + ".npy";
    // cnpy::npy_save(fileName, &fields->smooth[0][0][0], {len_x_cst+4, len_y_cst+4, len_z_cst+4}, "w") ;

    fileName = "products/curr_it" + std::to_string(indIt)
                + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, &fields->Ji[0][0][0][0], {3, len_x_cst+4, len_y_cst+4, len_z_cst+4}, "w");
    //
    fileName = "products/curr_tot_it" + std::to_string(indIt)
                + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, &fields->J_tot[0][0][0][0], {3, len_x_cst+4, len_y_cst+4, len_z_cst+4}, "w");
    //
    fileName = "products/E_it" + std::to_string(indIt)
                + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, &fields->E[0][0][0][0], {3, len_x_cst+4, len_y_cst+4, len_z_cst+4}, "w");
    //
    fileName = "products/region_ID_it_" + std::to_string(indIt)
                + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, &fields->region_ID[0][0][0], {len_x_cst+4,len_y_cst+4,len_z_cst+4}, "w");
    //
    fileName = "products/smoothed_patches_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, &fields->save_smooth[0][0], {4, smooth_patch_save_len_cst}, "w");

  #endif
}

void output_fields_time_space(fields_time_space* fields_t_s, int indIt, simu_param* sP)
{


  float *densFlat = new float[len_save_t_cst*len_save_x_cst] ;
  float *EFlat    = new float[3*len_save_t_cst*len_save_x_cst] ;
  float *BFlat    = new float[3*len_save_t_cst*len_save_x_cst] ;
  // float *BFlat_across = new float[len_save_t_cst*len_save_y_cst] ;
  int linInd ;

  for (int ind_t=0; ind_t<len_save_t_cst; ind_t++){
    for (int i=0; i<len_save_x_cst; i++){
          linInd = ind_t*len_save_x_cst + i ;
          densFlat[linInd] = fields_t_s->density[ind_t][i] ;
    }
  }

  for (int h=0; h<3; h++){
    for (int ind_t=0; ind_t<len_save_t_cst; ind_t++){
      for (int i=0; i<len_save_x_cst; i++){
            linInd = h*len_save_x_cst*len_save_t_cst + ind_t*len_save_x_cst + i ;
            EFlat[linInd]    = fields_t_s->E[h][ind_t][i] ;
            BFlat[linInd]    = fields_t_s->B[h][ind_t][i] ;
      }
    }
  }

  char numstr[21] ;
  std::string fileName = "products/dens_time_space_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, &fields_t_s->density[0][0], {len_save_t_cst, len_save_x_cst}, "w") ;

  fileName = "products/E_time_space_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, EFlat, {3, len_save_t_cst, len_save_x_cst}, "w");
  fileName = "products/B_time_space_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, BFlat, {3, len_save_t_cst, len_save_x_cst}, "w");
  // fileName = "products/B_across__time_space.npy" ;
  // cnpy::npy_save(fileName, BFlat_across,    {len_save_t_cst, len_save_y_cst}, "w");
  delete(densFlat) ;
  delete(EFlat) ;
  delete(BFlat) ;
  // delete(BFlat_across) ;
}

void output_probes(probes* prob, simu_param* sP)
{
  #if NB_DIM==2
    for (int p=0; p<nb_probes_cst; p++){
      if (prob->active[p]){
        std::string fileName = "products/probes/probes_positions_x_ID_" + std::to_string(p) + ".npy";
        cnpy::npy_save(fileName,  &prob->rx[p][0], {nb_it_max_cst}, "w") ;
        fileName = "products/probes/probes_positions_y_ID_" + std::to_string(p) + ".npy";
        cnpy::npy_save(fileName,  &prob->ry[p][0], {nb_it_max_cst}, "w") ;
        //
        fileName = "products/probes/probes_E_ID_" + std::to_string(p) + ".npy";
        cnpy::npy_save(fileName,  &prob->E[p][0][0], {3, nb_it_max_cst}, "w") ;
        //
        fileName = "products/probes/probes_B_ID_" + std::to_string(p) + ".npy";
        cnpy::npy_save(fileName,  &prob->B[p][0][0], {3, nb_it_max_cst}, "w") ;
        //
        fileName = "products/probes/probes_density_ID_" + std::to_string(p) + ".npy";
        cnpy::npy_save(fileName,  &prob->density[p][0], {nb_it_max_cst}, "w") ;
        //
        fileName = "products/probes/probes_Ji_ID_" + std::to_string(p) + ".npy";
        cnpy::npy_save(fileName,  &prob->Ji[p][0][0], {3, nb_it_max_cst}, "w") ;
        //
        fileName = "products/probes/probes_Jtot_ID_" + std::to_string(p) + ".npy";
        cnpy::npy_save(fileName,  &prob->J_tot[p][0][0], {3, nb_it_max_cst}, "w") ;
      }
    }
  #elif NB_DIM==3
    // std::string fileName = "products/probes/probes_positions_x_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
    //             + std::to_string(sP->mpi_rank_z) + ".npy";
    // cnpy::npy_save(fileName,  &prob->rx[0][0][0], {nb_probes_x_cst, nb_probes_y_cst, nb_probes_z_cst}, "w") ;
    // fileName = "products/probes/probes_positions_y_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
    //             + std::to_string(sP->mpi_rank_z) + ".npy";
    // cnpy::npy_save(fileName,  &prob->ry[0][0][0], {nb_probes_x_cst, nb_probes_y_cst, nb_probes_z_cst}, "w") ;
    // fileName = "products/probes/probes_positions_z_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
    //             + std::to_string(sP->mpi_rank_z) + ".npy";
    // cnpy::npy_save(fileName,  &prob->rz[0][0][0], {nb_probes_x_cst, nb_probes_y_cst, nb_probes_z_cst}, "w") ;
    // //
    // fileName = "products/probes/probes_E_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
    //             + std::to_string(sP->mpi_rank_z) + ".npy";
    // cnpy::npy_save(fileName,  &prob->E[0][0][0][0][0], {3, nb_probes_x_cst, nb_probes_y_cst, nb_probes_z_cst, nb_it_max_cst}, "w") ;
    // //
    // fileName = "products/probes/probes_B_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
    //             + std::to_string(sP->mpi_rank_z) + ".npy";
    // cnpy::npy_save(fileName,  &prob->B[0][0][0][0][0], {3, nb_probes_x_cst, nb_probes_y_cst, nb_probes_z_cst, nb_it_max_cst}, "w") ;
    //
  #endif
}

void output_density_species(simu_fields* fields, int ind_it, simu_param* sP)
{

  #if NB_DIM==2
    float *densSW  = new float[(len_x_cst+4)*(len_y_cst+4)] ;
    float *densCom = new float[(len_x_cst+4)*(len_y_cst+4)] ;
    int linInd ;

    for (int i=0; i<(len_x_cst+4); i++){
      for (int j=0; j<(len_y_cst+4); j++){
          linInd = i*(len_y_cst+4) + j ;
          densSW[linInd]  = fields->counts[i][j] ; //sP->w_sw*
          densCom[linInd] = fields->counts_pla[i][j] ; //sP->w_pla*
      }
    }

    std::string fileName = "products/dens_spec0_it" + std::to_string(ind_it)
                           + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                           + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, densSW, {len_x_cst+4, len_y_cst+4}, "w") ;

    fileName = "products/dens_spec1_it" + std::to_string(ind_it)
                + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, densCom, {len_x_cst+4, len_y_cst+4}, "w");

    delete(densSW) ;
    delete(densCom) ;

  #elif NB_DIM==3
    float *densSW  = new float[(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4)] ;
    float *densCom = new float[(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4)] ;
    int linInd ;

    for (int i=0; i<(len_x_cst+4); i++){
      for (int j=0; j<(len_y_cst+4); j++){
        for (int k=0; k<(len_z_cst+4); k++){
          linInd = i*(len_y_cst+4)*(len_z_cst+4) + j*(len_z_cst+4) + k ;
          densSW[linInd]  = fields->counts[i][j][k] ; //sP->w_sw*
          densCom[linInd] = fields->counts_pla[i][j][k] ; //sP->w_pla*
        }
      }
    }

    std::string fileName = "products/dens_spec0_it" + std::to_string(ind_it)
                + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, densSW, {len_x_cst+4, len_y_cst+4, len_z_cst+4}, "w") ;

    fileName = "products/dens_spec1_it" + std::to_string(ind_it)
                + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, densCom, {len_x_cst+4, len_y_cst+4, len_z_cst+4}, "w");

    delete(densSW) ;
    delete(densCom) ;

  #endif

}

void output_ohm_components(simu_B_field* B, simu_fields* fields, simu_param* sP, int ind_it)
{

  #if NB_DIM==2
    float *E_mot_flat = new float[3*(len_x_cst+4)*(len_y_cst+4)] ;
    float *E_hal_flat = new float[3*(len_x_cst+4)*(len_y_cst+4)] ;
    float *E_amb_flat = new float[3*(len_x_cst+4)*(len_y_cst+4)] ;
    float *E_res_flat = new float[3*(len_x_cst+4)*(len_y_cst+4)] ;
    int linInd ;

    for (int h=0; h<3; h++){
      for (int i=0; i<(len_x_cst+4); i++){
        for (int j=0; j<(len_y_cst+4); j++){
            linInd = h*(len_x_cst+4)*(len_y_cst+4) + i*(len_y_cst+4) + j ;
            E_mot_flat[linInd] = fields->E_mot[h][i][j] ;
            E_hal_flat[linInd] = fields->E_hal[h][i][j] ;
            E_amb_flat[linInd] = fields->E_amb[h][i][j] ;
            E_res_flat[linInd] = fields->E_res[h][i][j] ;
        }
      }
    }

    std::string fileName = "products/E_mot_it" + std::to_string(ind_it)
                           + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                           + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, E_mot_flat, {3,len_x_cst+4,len_y_cst+4}, "w");

    fileName = "products/E_hal_it" + std::to_string(ind_it)
                + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, E_hal_flat, {3,len_x_cst+4,len_y_cst+4}, "w");

    fileName = "products/E_amb_it" + std::to_string(ind_it)
                + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, E_amb_flat, {3,len_x_cst+4,len_y_cst+4}, "w");

    fileName = "products/E_res_it" + std::to_string(ind_it)
                + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, E_res_flat, {3,len_x_cst+4,len_y_cst+4}, "w");

  #elif NB_DIM==3
    float *E_mot_flat = new float[3*(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4)] ;
    float *E_hal_flat = new float[3*(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4)] ;
    float *E_amb_flat = new float[3*(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4)] ;
    int linInd ;

    for (int h=0; h<3; h++){
      for (int i=0; i<(len_x_cst+4); i++){
        for (int j=0; j<(len_y_cst+4); j++){
          for (int k=0; k<(len_z_cst+4); k++){
            linInd = h*(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4) + i*(len_y_cst+4)*(len_z_cst+4) + j*(len_z_cst+4) + k ;
            E_mot_flat[linInd] = fields->E_mot[h][i][j][k] ;
            E_hal_flat[linInd] = fields->E_hal[h][i][j][k] ;
            E_amb_flat[linInd] = fields->E_amb[h][i][j][k] ;
          }
        }
      }
    }

    std::string fileName = "products/E_mot_it" + std::to_string(ind_it)
                + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, E_mot_flat, {3,len_x_cst+4, len_y_cst+4, len_z_cst+4}, "w");

    fileName = "products/E_hal_it" + std::to_string(ind_it)
                + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, E_hal_flat, {3,len_x_cst+4, len_y_cst+4, len_z_cst+4}, "w");

    fileName = "products/E_amb_it" + std::to_string(ind_it)
                + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, E_amb_flat, {3,len_x_cst+4, len_y_cst+4, len_z_cst+4}, "w");

  #endif

}

void output_B_field(simu_B_field* B, int ind_it, simu_param* sP)
{

  #if NB_DIM==2
    float *BFlat = new float[3*(len_x_cst+4)*(len_y_cst+4)] ;
    int linInd ;
    // std::vector<float> arr1((len_x_cst+4)*len_y_cst);
    for (int h=0; h<3; h++){
      for (int i=0; i<(len_x_cst+4); i++){
        for (int j=0; j<(len_y_cst+4); j++){
            linInd = h*(len_x_cst+4)*(len_y_cst+4) + i*(len_y_cst+4) + j ;
            #if !yee_cst
              BFlat[linInd] = B->B[h][i][j] ;
            #else
              BFlat[linInd] = B->B[h][i][j] ;
            #endif
            // #if dipole_cst
            //   // std::cout << B->B_dip[h][i][j] << std::endl ;
            //   BFlat[linInd] += B->B_dip[h][i][j] ;
            // #endif
        }
      }
    }
    // std::cout << B->B[1][50][50] << std::endl;

    std::string fileName = "products/B_it" + std::to_string(ind_it)
                + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, BFlat, {3, len_x_cst+4, len_y_cst+4}, "w");
    delete(BFlat) ;

    #if dipole_cst
      fileName = "products/B_dip_it" + std::to_string(ind_it)
                  + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                  + std::to_string(sP->mpi_rank_z) + ".npy";
      cnpy::npy_save(fileName,  &B->B_dip[0][0][0], {3, len_x_cst+4, len_y_cst+4}, "w") ;
    #endif

    #if yee_cst
      fileName = "products/B_s_it" + std::to_string(ind_it)
                  + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                  + std::to_string(sP->mpi_rank_z) + ".npy";
      cnpy::npy_save(fileName,  &B->B_s[0][0][0], {3, len_x_cst+4, len_y_cst+4}, "w") ;
    #endif

  #elif NB_DIM==3
    float *BFlat = new float[3*(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4)] ;
    int linInd ;
    // std::vector<float> arr1((len_x_cst+4)*len_y_cst);
    for (int h=0; h<3; h++){
      for (int i=0; i<(len_x_cst+4); i++){
        for (int j=0; j<(len_y_cst+4); j++){
          for (int k=0; k<(len_z_cst+4); k++){
            linInd = h*(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4) + i*(len_y_cst+4)*(len_z_cst+4) + j*(len_z_cst+4) + k ;
            #if !yee_cst
              BFlat[linInd] = B->B[h][i][j][k] ;
            #else
              BFlat[linInd] = B->B[h][i][j][k] ;
            #endif
            // #if dipole_cst
            //   BFlat[linInd] += B->B_dip[h][i][j][k] ;
            // #endif
          }
        }
      }
    }

    std::string fileName = "products/B_it" + std::to_string(ind_it)
                + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, BFlat, {3, len_x_cst+4, len_y_cst+4, len_z_cst+4}, "w");
    delete(BFlat) ;

    #if dipole_cst
      fileName = "products/B_dip_it" + std::to_string(ind_it)
                  + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                  + std::to_string(sP->mpi_rank_z) + ".npy";
      cnpy::npy_save(fileName,  &B->B_dip[0][0][0][0], {3, len_x_cst+4, len_y_cst+4, len_z_cst+4}, "w") ;
    #endif

    #if yee_cst
      fileName = "products/B_s_it" + std::to_string(ind_it)
                  + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                  + std::to_string(sP->mpi_rank_z) + ".npy";
      cnpy::npy_save(fileName,  &B->B_s[0][0][0][0], {3, len_x_cst+4, len_y_cst+4, len_z_cst+4}, "w") ;
    #endif

  #endif
}

#if DF_cst
void output_DF(simu_DF* DF, simu_param* sP, int idx_it)
{
  std::string fileName = "products/DF_it" + std::to_string(idx_it)
                         + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                         + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, &DF->DF[0][0][0][0], {len_x_cst+4, len_v_cst+4, len_v_cst+4, len_v_cst+4}, "w") ;
  //
  fileName = "products/B_DF_it" + std::to_string(idx_it)
              + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, &DF->B[0][0], {3, len_x_cst+4}, "w") ;
  //
  fileName = "products/E_DF_it" + std::to_string(idx_it)
              + "_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
              + std::to_string(sP->mpi_rank_z) + ".npy";
  cnpy::npy_save(fileName, &DF->E[0][0], {3, len_x_cst+4}, "w") ;
}
#endif

void output_grid(simu_grid* grid, simu_param* sP)
{


  #if NB_DIM==2
    float *grid_flat = new float[(len_x_cst+4)*(len_y_cst+4)] ;
    int linInd ;

    for (int i=0; i<(len_x_cst+4); i++){
      for (int j=0; j<(len_y_cst+4); j++){
          linInd = i*(len_y_cst+4) + j ;
          grid_flat[linInd]   = grid->rsq[i][j] ;
      }
    }

    std::string fileName = "products/grid_rsq_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                           + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, grid_flat, {len_x_cst+4, len_y_cst+4}, "w") ;
    delete(grid_flat) ;

    fileName = "products/grid_x_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, grid->xGrid, {len_x_cst+4}, "w") ;
    fileName = "products/grid_y_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, grid->yGrid, {len_y_cst+4}, "w") ;

  #elif NB_DIM==3
    float *grid_flat = new float[(len_x_cst+4)*(len_y_cst+4)*(len_z_cst+4)] ;
    int linInd ;

    for (int i=0; i<(len_x_cst+4); i++){
      for (int j=0; j<(len_y_cst+4); j++){
        for (int k=0; k<(len_z_cst+4); k++){
          linInd = i*(len_y_cst+4)*(len_z_cst+4) + j*len_z_cst + k ;
          grid_flat[linInd]   = grid->rsq[i][j][k] ;
        }
      }
    }

    std::string fileName = "products/grid_rsq_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                           + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, grid_flat, {len_x_cst+4, len_y_cst+4, len_z_cst+4}, "w") ;
    delete(grid_flat) ;

    fileName = "products/grid_x_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, grid->xGrid, {len_x_cst+4}, "w") ;
    fileName = "products/grid_y_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, grid->yGrid, {len_y_cst+4}, "w") ;
    fileName = "products/grid_z_rank_" + std::to_string(sP->mpi_rank_y)+ "_"
                + std::to_string(sP->mpi_rank_z) + ".npy";
    cnpy::npy_save(fileName, grid->zGrid, {len_z_cst+4}, "w") ;

  #endif


  #if DF_cst
  fileName = "products/vGrid.npy";
  cnpy::npy_save(fileName, &grid->vGrid[0], {len_v_cst+4}, "w") ;
  #endif

}
