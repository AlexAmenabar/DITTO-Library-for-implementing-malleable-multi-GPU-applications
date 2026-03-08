bnd run --devel nvcc -g toy_app_malleable.cpp DITO_API.cpp DDM.cpp cudaDTM.cpp toy_app_malleable.cu -Xcompiler -fopenmp -o toy_app -lpthread 
