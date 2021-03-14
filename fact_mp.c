#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(void) {
    int n = 10;
    int factorial[n];
    factorial[1] = 1;

    int *proda;
    #pragma omp parallel
    {
        int ithread = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        #pragma omp single
        {
            proda = malloc(nthreads * sizeof *proda);
            proda[0] = 1;
        }
        int prod = 1;
        #pragma omp for schedule(static) nowait
        for (int i=2; i<n; i++) {
            prod *= i;
            factorial[i] = prod;
        }
        proda[ithread+1] = prod;
        #pragma omp barrier
        int offset = 1;
        for(int i=0; i<(ithread+1); i++) offset *= proda[i];
        #pragma omp for schedule(static)
        for(int i=1; i<n; i++) factorial[i] *= offset;
    }
    free(proda);

    for(int i=1; i<n; i++) printf("%d\n", factorial[i]); putchar('\n'); 
}