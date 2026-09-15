#include <fstream>

#include "parameters.h"
#include "structures.h"
#include "output_parameters.h"





int main(int argc, char *argv[]){

  // std::cout << "\nSize of float: " << sizeof(float) << std::endl ;
  // std::cout << "Size of int: " << sizeof(int) << std::endl ;
  // std::cout << "Size of bool: " << sizeof(bool) << std::endl ;

  // simu_param sp ;
  simu_param* sp = new simu_param;
  printout_parameters(sp) ;

  int totalSizeMallocDevice = (sizeof(particles)
                              +sizeof(simu_grid)
                              +sizeof(simu_fields)
                              +sizeof(simu_B_field)*2
                              +sizeof(trajectory)
                              +sizeof(moments)
                              +sizeof(fields_time_space)
                              +sizeof(root_mean_sqr)
                              +sizeof(state_solver)
                              +sizeof(injector)
                              +sizeof(house_keeping)
                              +sizeof(buff_part))/1000000 ;

  std::cout << "\n._____________________________\n|\n| Total size: " << totalSizeMallocDevice <<" Mb.\n|_____________________________\n\n" << std::endl ;
  return 0 ;

}
// gcc output_parameters.cpp -o yo -lstdc++ -lm
