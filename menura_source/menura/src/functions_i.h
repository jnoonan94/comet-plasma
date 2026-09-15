#ifndef __FUNCTIONS_I_H_INCLUDED__   // if x.h hasn't been included yet...
#define __FUNCTIONS_I_H_INCLUDED__

#include "parameters.h"
#include "mpi.h"

float normalSample(float mean, float standardDeviation);
void shuffleOwn(int *arr, size_t n);
void init_house_keeping(moments* mom, trajectory* traj, root_mean_sqr* rms,
                        simu_param* sP, house_keeping* h_k, state_solver* sta_sol);
void init_grid(simu_grid* grid, simu_param* sP);
void init_grid_test(simu_grid* grid, simu_param* sP);
void init_fields(simu_fields* fields, simu_param* sP);
void init_probes(probes* prob, simu_param* sP, simu_grid* grid);
void init_B(simu_B_field* B, simu_param* sP);
#if DF_cst
void init_DF(simu_DF* DF, simu_grid* grid, simu_param* sP);
#endif
void init_B_2(simu_B_field_2* B, simu_param* sP);
void init_B_dipole(simu_B_field* B, simu_param* sP, simu_grid* grid);
void init_alfven_fluctuations(simu_B_field* B, simu_param* sP);
void init_part_node(particles* p, simu_grid* grid, simu_param* sP);
void init_part_global(particles* p, simu_grid* grid, simu_param* sP);
void init_part_2_stream(particles* p, simu_grid* grid, simu_param* sP);
void init_part_acoustic(particles* p, simu_grid* grid, simu_param* sP);
void correctBulk(particles* p, simu_param* sP);
void addFluctuations(simu_B_field* B, particles* p, simu_grid* grid, simu_param* sP, MPI_Comm comm);
void init_fluctuations(simu_B_field* B, particles* p, simu_grid* grid, simu_param* sP, MPI_Comm comm);
void initFluctuations(simu_B_field* B, particles* p, simu_grid* grid, simu_param* sP);
void initFluctuationsNew(simu_B_field* B, particles* p, simu_grid* grid, simu_param* sP);
void init_exosphere(buff_part* b_p, simu_grid* grid, simu_param* sP, MPI_Comm comm);
void init_ionosphere(buff_part* b_p, simu_grid* grid, simu_param* sP, MPI_Comm comm);
//
//
void load_B_field(simu_B_field* B, simu_tank* tank, simu_param* sP,
                  bool fill_tank, std::string path_inputs, int idx_load);
void load_fields(simu_fields* fields, simu_tank* tank, simu_param* sP,
                 bool fill_tank, std::string path_inputs, int idx_load);
void load_particles(particles* p, simu_grid* grid, simu_tank* tank, simu_param* sP,
                    MPI_Comm comm, bool fill_tank, std::string path_inputs, int idx_load,
                    bool old_file);


#endif
