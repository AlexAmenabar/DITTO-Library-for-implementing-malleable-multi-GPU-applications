#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

int main(int argc, char **argv){
    

    int *arr = (int*)calloc(10, sizeof(int));

    for(int i = 0; i<10; i++){

        arr[i] = i;
    }
    size_t size = sizeof(int);


    void *varr = (void*)calloc(10, sizeof(int));

    for(int i = 0; i<10; i++){

        // get position
        void *dst = (char*)varr + i * size;

        // copy
        memcpy(dst, (char*)arr + i * size, size);
    }


    // print
    for(int i = 0; i<10; i++){

        printf(" %d, ", arr[i]);
    }
    printf("\n");

    // copy byte to byte
    for(int i = 0; i<10; i++){
        printf(" %d, ", ((int*)(varr))[i]);
    }


    printf("\n Finished!");
}
