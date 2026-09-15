#ifndef __FUNCTIONS_O_H_INCLUDED__   // if x.h hasn't been included yet...
#define __FUNCTIONS_O_H_INCLUDED__

#include "parameters.h"
// #include <curand.h>
// #include <curand_kernel.h>

void output_HK(house_keeping* HK, trajectory* traj, moments* mom, root_mean_sqr* rms, simu_param* sP, int idx_save);
void output_particles(particles* p, int indIt, simu_param* sP);
void output_fields(simu_fields* fields, int indIt, simu_param* sP, bool full=true);
void output_fields_time_space(fields_time_space* fields_t_s, int indIt, simu_param* sP);
void output_probes(probes* prob, simu_param* sP);
void output_density_species(simu_fields* fields, int ind_it, simu_param* sP);
void output_ohm_components(simu_B_field* B, simu_fields* fields, simu_param* sP, int ind_it);
void output_B_field(simu_B_field* B, int indIt, simu_param* sP);
#if DF_cst
void output_DF(simu_DF* DF, simu_param* sP, int idx_it);
#endif
void output_grid(simu_grid* grid, simu_param* sP);
void outputFiles(particles* p, simu_grid* grid, float curr[3][len_x_cst][len_y_cst], float dens[len_x_cst][len_y_cst],
                 float E[3][len_x_cst][len_y_cst], float B[3][len_x_cst][len_y_cst], int indIt, float energies [12][nb_it_max_cst]);
void sortie();

#endif
