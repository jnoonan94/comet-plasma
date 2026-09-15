#ifndef __OUPUT_PARAMETERS_H_INCLUDED__   // if x.h hasn't been included yet...
#define __OUPUT_PARAMETERS_H_INCLUDED__

#include "parameters.h"

void printout_parameters(simu_param* sP)
{

  printf("\n._____________________________________________________________________________________\n|\n");
  printf("|                Dimensionality: %iD\n|\n", NB_DIM);
  printf("|       Proton plasma frequency: %9.4f 1/s\n", sP->omega_i);
  printf("|          Proton gyrofrequency: %9.4f 1/s\n", sP->omega_ci);
  printf("|\n");
  printf("|   Proton inertial length d_i : %9.4f km\n", sP->d_i*1e-3);
  printf("| Electron inertial length d_e : %9.4f km\n", sP->d_e*1e-3);
  printf("|        Debye length lambda_D : %9.4f km\n", sP->lambda_D*1e-3);
  printf("|           Gyro-radius (v_th) : %9.4f \n", sP->v_thi/sP->v_A);
  printf("|\n");
  printf("|           Proton temperature : %9.4f K \n", sP->Ti_inf);
  printf("|         Electron temperature : %9.4f K \n", sP->Te_inf);
  printf("|\n");
  printf("|                 Alfven speed : %9.4f km/s\n", sP->v_A*1e-3);
  printf("|                  Sound speed : %9.4f km/s\n", sP->v_s*1e-3);
  printf("|           Magnetosonic speed : %9.4f km/s\n", sP->v_ms*1e-3);
  printf("|            Ion thermal speed : %9.4f km/s\n", sP->v_thi*1e-3);
  printf("|       Electron thermal speed : %9.4f km/s\n", sP->v_the*1e-3);
  printf("|\n");
  printf("|                       Beta_p : %9.4f \n", sP->Beta_p);
  printf("|                       Beta_e : %9.4f \n", sP->Beta_e);
  printf("|\n");
  printf("|                      eta_res : %9.4f \n", sP->eta_res_norm);
  printf("|                  eta_hyp_res : %9.4f \n", sP->eta_hyp_res);
  if (decay_turb_cst){
    printf("|\n");
    printf("| Decaying turbulence run, fluctuations sqr %.2f \n", sP->fluctu_sqrt);
  }
  if (obstacle_cst){
    printf("|\n");
    printf("| Obstacle simulation. \n" );
    printf("|         Obstacle speed (km/s): %9.4f km/s \n", sP->v_obs*sP->v_A*1e-3);
    printf("|          Obstacle speed (v_A): %9.4f  \n", sP->v_obs);
    if (solid_body_cst){
      printf("|         Obstacle radius (d_i): %9.4f  \n", sP->r_obs);
      printf("|          Obstacle radius (km): %9.4f  \n", sP->r_obs_SI*1e-3);
    }
  }
  if (inject_pla_cst){
    printf("| Cometary simulation. \n" );
    printf("|              Gyro-radius (km): %9.4f \n", sP->Z_pla*m_i*sP->v_obs*sP->v_A/(e*sP->B0_SI) );
    printf("|             Gyro-radius (d_i): %9.4f \n", sP->Z_pla*sP->v_obs);
  }
  printf("|\n|\n");
  printf("| Particle-per-node:             %i \n", nb_part_node_cst);
  printf("| Particle weight spec 0 (sw):   %.2e\n", sP->w_sw);
  if (inject_pla_cst){
    printf("| Particle weight spec 1 (pla):  %.2e\n", sP->w_pla);
  }
  printf("|\n");
  printf("| Particle pool size:            %i\n", pool_size_cst);
  printf("| Particle buffer size:          %i\n", buff_size_cst);
  if (inject_turb_cst){
    printf("| Turbulent injection. \n" );
    printf("| Tank size:                   %i\n", tank_size_cst);
    printf("| Injector size:               %i\n", injec_size_cst);
  }
  printf("|\n");
  printf("| Box length (%i x %i nodes):    %.4e (%.4f proton inertial lengths d_i)\n", len_x_cst, len_y_cst, len_x_cst*sP->dX, len_x_cst*sP->dX);
  printf("|\n");
  printf("| Node spacing: %.3e (%.4f d_i).\n", sP->dX, sP->dX);
  printf("| dx >> %.1e (d_e) \n", sP->d_e/sP->d_i);
  printf("|\n");
  printf("| Time step: %.3e (%.4f gyroperiod).\n", sP->dt, sP->dt/(2.*PI));
  printf("| dt < %.1e (CFL) \n", 1/(sqrt(sP->nb_dim)*PI) * (sP->dX*sP->dX) ) ;
  // std::cout << dt*dt/(2*dX*dX) << std::endl;
  printf("| dt < %.1e (No cell jump at v_thi) \n", sP->dX/(sP->v_thi/sP->v_A));
  printf("| dt < %.1e (No cell jump at v_s) \n", sP->dX/(sP->v_s/sP->v_A));
  printf("| dt < %.1e (No cell jump at v_A) \n", sP->dX);
  printf("| dt < %.1e (vacuum resistivity) \n", sP->dX*sP->dX/(2*sP->eta_res_vac));
  if (ORF_cst){
    printf("| dt < %.1e (No cell jump) \n", sP->dX/sP->v_obs);
  }
  if (solid_body_cst){
    printf("| dt < %.1e (obstacle resistivity) \n", sP->dX*sP->dX/(2*sP->eta_res_obs));
  }
  printf("|\n|_____________________________________________________________________________________\n\n\n");

}

void output_parameters(simu_param* sP)
{
  if (sP->mpi_rank_lin==0){
    std::string file_name = "products/parameters.txt"; // + std::to_string(indIt) + "_rank" + std::to_string(sP->mpi_rank) + ".txt";
    std::ofstream thefile;
    thefile.open (file_name);
    thefile << "NB_DIM                " << NB_DIM                 << "\n";
    thefile << "seed_val_cst          " << seed_val_cst           << "\n";
    thefile << "tpb_cst               " << tpb_cst                << "\n";
    thefile << "nb_it_max_cst         " << nb_it_max_cst          << "\n";
    thefile << "rate_save_t_cst       " << rate_save_t_cst        << "\n";
    thefile << "len_save_t_cst        " << len_save_t_cst         << "\n";
    thefile << "len_x_cst             " << len_x_cst              << "\n";
    thefile << "len_y_cst             " << len_y_cst              << "\n";
    #if NB_DIM==3
    thefile << "len_z_cst             " << len_z_cst              << "\n";
    #endif
    thefile << "rate_save_x_cst       " << rate_save_x_cst        << "\n";
    thefile << "len_save_x_cst        " << len_save_x_cst         << "\n";
    thefile << "rate_save_y_cst       " << rate_save_y_cst        << "\n";
    thefile << "len_save_y_cst        " << len_save_y_cst         << "\n";
    thefile << "nb_nodes_cst          " << nb_nodes_cst           << "\n";
    thefile << "nb_nodes_tot_cst      " << nb_nodes_tot_cst       << "\n";
    thefile << "nb_part_node_cst      " << nb_part_node_cst       << "\n";
    thefile << "pool_size_cst         " << pool_size_cst          << "\n";
    thefile << "tank_size_cst         " << tank_size_cst          << "\n";
    thefile << "buff_size_cst         " << buff_size_cst          << "\n";
    thefile << "injec_size_cst        " << injec_size_cst         << "\n";
    thefile << "nb_part_add_pla_cst   " << nb_part_add_pla_cst    << "\n";
    thefile << "nb_probes_cst         " << nb_probes_cst          << "\n";
    thefile << "smooth_patch_save_len_cst " << smooth_patch_save_len_cst << "\n";
    thefile << "yee_cst               " << yee_cst                << "\n";
    thefile << "decay_turb_cst        " << decay_turb_cst         << "\n";
    thefile << "obstacle_cst          " << obstacle_cst           << "\n";
    thefile << "ORF_cst               " << ORF_cst                << "\n";
    thefile << "inject_pla_cst        " << inject_pla_cst         << "\n";
    thefile << "inject_turb_cst       " << inject_turb_cst        << "\n";
    thefile << "solid_body_cst        " << solid_body_cst         << "\n";
    thefile << "dipole_cst            " << dipole_cst             << "\n";
    thefile << "ionosphere_cst        " << ionosphere_cst         << "\n";
    thefile << "idx_it_inject_cst     " << idx_it_inject_cst      << "\n";
    thefile << "nb_it_per_shift_cst   " << nb_it_per_shift_cst    << "\n";
    thefile << "nb_cell_per_shift_cst " << nb_cell_per_shift_cst  << "\n";
    //
    thefile << "n0_SI                 " << sP->n0_SI                  << "\n";
    thefile << "B0_SI                 " << sP->B0_SI                  << "\n";
    thefile << "Ti_inf                " << sP->Ti_inf                 << "\n";
    thefile << "Te_inf                " << sP->Te_inf                 << "\n";
    thefile << "e_mot                 " << sP->e_mot                  << "\n";
    thefile << "e_hal                 " << sP->e_hal                  << "\n";
    thefile << "e_amb                 " << sP->e_amb                  << "\n";
    thefile << "poly_ind              " << sP->poly_ind               << "\n";
    thefile << "eta_res_norm          " << sP->eta_res_norm           << "\n";
    thefile << "eta_sm                " << sP->eta_sm                 << "\n";
    thefile << "eta_hyp_res           " << sP->eta_hyp_res            << "\n";
    thefile << "eta_res_vac           " << sP->eta_res_vac            << "\n";
    thefile << "eta_res_obs           " << sP->eta_res_obs            << "\n";
    thefile << "omega_i               " << sP->omega_i                << "\n";
    thefile << "omega_ci              " << sP->omega_ci               << "\n";
    thefile << "v_A                   " << sP->v_A                    << "\n";
    thefile << "t0                    " << sP->t0                     << "\n";
    thefile << "x0                    " << sP->x0                     << "\n";
    thefile << "q0                    " << sP->q0                     << "\n";
    thefile << "m0                    " << sP->m0                     << "\n";
    thefile << "p0                    " << sP->p0                     << "\n";
    thefile << "B0_x                  " << sP->B0_x                   << "\n";
    thefile << "B0_y                  " << sP->B0_y                   << "\n";
    thefile << "B0_z                  " << sP->B0_z                   << "\n";
    thefile << "Q                     " << sP->Q                      << "\n";
    thefile << "nu_i                  " << sP->nu_i                   << "\n";
    thefile << "nu_d                  " << sP->nu_d                   << "\n";
    thefile << "u0                    " << sP->u0                     << "\n";
    thefile << "r_obs                 " << sP->r_obs                  << "\n";
    thefile << "v_obs                 " << sP->v_obs                  << "\n";
    thefile << "centre_x              " << sP->centre_x               << "\n";
    thefile << "centre_y              " << sP->centre_y               << "\n";
    #if NB_DIM==3
    thefile << "centre_z              " << sP->centre_z               << "\n";
    #endif
    thefile << "R_gyr_norm            " << sP->R_gyr_norm             << "\n";
    thefile << "R_gyr                 " << sP->R_gyr                  << "\n";
    thefile << "fluctu_sqrt           " << sP->fluctu_sqrt            << "\n";
    thefile << "d_i                   " << sP->d_i                    << "\n";
    thefile << "omega_e               " << sP->omega_e                << "\n";
    thefile << "omega_ce              " << sP->omega_ce               << "\n";
    thefile << "d_e                   " << sP->d_e                    << "\n";
    thefile << "v_thi                 " << sP->v_thi                  << "\n";
    thefile << "v_the                 " << sP->v_the                  << "\n";
    thefile << "v_s                   " << sP->v_s                    << "\n";
    thefile << "v_ms                  " << sP->v_ms                   << "\n";
    thefile << "lambda_D              " << sP->lambda_D               << "\n";
    thefile << "Beta_p                " << sP->Beta_p                 << "\n";
    thefile << "Beta_e                " << sP->Beta_e                 << "\n";
    thefile << "xLen                  " << sP->xLen                   << "\n";
    thefile << "yLen                  " << sP->yLen                   << "\n";
    thefile << "dX                    " << sP->dX                     << "\n";
    thefile << "dX_i                  " << sP->dX_i                   << "\n";
    thefile << "nb_dim                " << sP->nb_dim                 << "\n";
    thefile << "dt                    " << sP->dt                     << "\n";
    thefile << "tMax                  " << sP->tMax                   << "\n";
    thefile << "nb_wave               " << sP->nb_wave                << "\n";
    thefile << "lambda_wave           " << sP->lambda_wave            << "\n";
    thefile << "k_wave                " << sP->k_wave                 << "\n";
    thefile << "omega_wave            " << sP->omega_wave             << "\n";
    thefile << "period_wave           " << sP->period_wave            << "\n";
    thefile << "nb_sub_cycles         " << sP->nb_sub_cycles          << "\n";
    thefile << "sub_dt                " << sP->sub_dt                 << "\n";
    thefile << "rateSaveIt            " << sP->rateSaveIt             << "\n";
    thefile << "nb_nodes              " << sP->nb_nodes               << "\n";
    thefile << "nb_nodes_tot          " << sP->nb_nodes_tot           << "\n";
    thefile << "nb_part_add_sw        " << sP->nb_part_add_sw         << "\n";
    thefile << "nb_part_add_pla       " << sP->nb_part_add_pla        << "\n";
    thefile << "sum_pla_proba         " << sP->sum_pla_proba          << "\n";
    thefile << "w_sw                  " << sP->w_sw                   << "\n";
    thefile << "w_pla                 " << sP->w_pla                  << "\n";
    thefile << "Z_pla                 " << sP->Z_pla                  << "\n";
    thefile << "dens_min              " << sP->dens_min               << "\n";
    thefile << "mpi_rank_y            " << sP->mpi_rank_y             << "\n";
    thefile << "mpi_nb_proc_y         " << sP->mpi_nb_proc_y          << "\n";
    thefile << "len_buff_field_y      " << sP->len_buff_field_y       << "\n";
    thefile << "mpi_rank_z            " << sP->mpi_rank_z             << "\n";
    thefile << "mpi_nb_proc_z         " << sP->mpi_nb_proc_z          << "\n";
    #if NB_DIM==3
      thefile << "len_buff_field_z      " << sP->len_buff_field_z       << "\n";
    #endif
    thefile << "mpi_nb_proc_tot       " << sP->mpi_nb_proc_tot        << "\n";
    thefile << "nb_part_tank          " << sP->nb_part_tank           << "\n";

    thefile.close();
  }
}

#endif
