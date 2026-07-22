bnd run --devel nvcc \
    -O1 -g \
    -Iinclude -Isrc \
    testApps/toy_app_malleable.cpp \
    src/DITO_API.cpp \
    src/DDM.cpp \
    src/cudaDTM.cpp \
    src/RMS.cpp \
    src/jobQueue.cpp \
    src/eventQueue.cpp \
    src/RCF.cpp \
    src/SCH.cpp \
    testApps/toy_app_malleable.cu \
    src/mainWorkload.cpp \
    -Xcompiler="-fopenmp -Wall -fsanitize=address -fno-omit-frame-pointer" \
    -o bin/scheduler \
    -lpthread \
    -lnvidia-ml \
    -lnccl \
    -DUTILIZATION