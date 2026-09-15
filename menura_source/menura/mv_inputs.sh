idx_it=$1
if [ -n "$idx_it" ]
then
  cp -v products/B_it$idx_it*            inputs/
  cp -v products/E_it$idx_it*            inputs/
  cp -v products/dens_it$idx_it*         inputs/
  cp -v products/curr_it$idx_it*         inputs/
  cp -v products/nb_particles_*_it$idx_it* inputs/
  cp -v products/particles_*_it$idx_it*    inputs/
  cp -v products/nb_particle_files_it$idx_it*.txt inputs/
else
  echo Please provide the iteration index of the files to be copied.
fi
