import os
print('gcc src/output_parameters.cpp -o op_tmp -lstdc++ -lm')
os.system( 'gcc src/output_parameters.cpp -o op_tmp -lstdc++ -lm' )
os.system('./op_tmp')
os.system('rm op_tmp')
