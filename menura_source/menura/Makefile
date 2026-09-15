##################################################################
# compilers and compilation options

CC=mpic++
CFLAGS=# no option

NC=nvcc
NFLAGS=-gencode arch=compute_35,code=sm_35 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70

# link to external libraries
LINK=-lcuda -lcudart -lz
##################################################################

# executable name
EXE1=menura

# list of sources
SRC1=cnpy.cpp functions_i.cpp functions_o.cpp menura.cu

# lists of dependencies (.h files)
cnpyDEP=
functions_iDEP=parameters.h structures.h functions_i.h functions_o.h
functions_oDEP=parameters.h structures.h functions_o.h
menuraDEP=parameters.h structures.h functions_i.h functions_o.h utils.h kernels_calls.h kernels_particles.cuh kernels_particles_to_grid.cuh kernels_fields.cuh kernels_HK.cuh kernels_DF.cuh kernels_tricks.cuh

#########################################################
# creating the list of object files

OBJa1=$(SRC1:.cpp=.o) # replacing .cpp .o for each file of the list
OBJb1=$(OBJa1:.cu=.o) # replacing .cu by .o for each file of the list
OBJ1=$(addprefix obj/,$(OBJb1)) # adding the prefix obj/ to all files of the list

# creating the list of dependencies for each file
cnpyHDEP=$(addprefix src/,$(cnpyDEP)) # adding the prefix src/ to all .h files of the list
functions_iHDEP=$(addprefix src/,$(functions_iDEP))
functions_oHDEP=$(addprefix src/,$(functions_oDEP))
menuraHDEP=$(addprefix src/,$(menuraDEP))
#########################################################

# programs compiled when typing "make"
all: $(EXE1)

# compilation rule
$(EXE1): $(OBJ1)
	$(CC) $^ -o $@ $(LINK)

################################################################
# compilation of the obj files

# from .cu files
#obj/%.o: src/%.cu
#	$(NC) $(NFLAGS) -c $< -o $@

# from .cpp files
#obj/%.o: src/%.cpp
#	$(CC) $(CFLAGS) -c $< -o $@

# Instead of generic compilation rules (as commented above)
# it is better to explicitely mention all dependencies:

obj/menura.o: src/menura.cu $(menuraHDEP)
	$(NC) $(NFLAGS) -c $< -o $@

obj/functions_i.o: src/functions_i.cpp $(functions_iHDEP)
	$(CC) $(CFLAGS) -c $< -o $@

obj/functions_o.o: src/functions_o.cpp $(functions_oHDEP)
	$(CC) $(CFLAGS) -c $< -o $@

obj/cnpy.o: src/cnpy.cpp $(cnpyHDEP)
	$(CC) $(CFLAGS) -c $< -o $@

#########################################################
# deleting compilation files

clean:
	rm -f *.x obj/*.o

################################################################
