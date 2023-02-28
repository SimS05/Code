#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#define width 800
#define height 600
#define i 1000

int main(int argc, char** argv) {
    clock_t start, end;
    double cpu_used;
    start = clock();
    
    int rank;
    int size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start = MPI_Wtime();

    int tasks = width * height;
    int tasks_p = ceil(tasks / (double)size);

    int start_task = rank * tasks_p;
    int end_task = fmin(start_task + tasks_p, tasks);

    int mandelbrot[height][width] = {0};
    for (int task = start_task; task < end_task; task++) {
        int a = task % width;
        int b = task / width;

        double x = (a - width/2.0)*4.0/width;
        double y = (b - height/2.0)*4.0/width;
        double z = 0, z1 = 0;

      
        for (int i = 0; i < i; i++) {
            double z_new = z*z – z1*z1 + x;
            double z_new1 = 2*z*z1 + y;
            z = z_new;
            z1 = z_new1;

            if (z*z+ z1*z1 > 4) {
                break;
            }
        }
        mandelbrot[b][a] = i;
    }

    // Gather computed results to root process
    int* counts = malloc(size * sizeof(int));
    int* d = malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        int tasks_i = i == size - 1 ? tasks - i * tasks_p : tasks_p;
        counts[i] = tasks_i;
        d[i] = i * tasks_p;
    }

    int* m = NULL;
    if (rank == 0) {
        m = malloc(tasks * sizeof(int));
    }
    MPI_Gatherv(&(mandelbrot[0][start_task]), end_task - start_task, MPI_INT,
                m, counts, d, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE* file = fopen("mandelbrot.ppm", "wb");
        fprintf(file, "P6 %d %d 255\n", width, height);
        for (int i = 0; i < tasks; i++) {
            int c = m[i] * 255 / i;
            fputc(c, file);
            fputc(c, file);
            fputc(c, file);
        }
        fclose(file);
        free(m);

        double end = MPI_Wtime();
        printf("Execution time: %.2f seconds\n", end - start);
    }

    MPI_Finalize();
    
    end = clock();
    cpu_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CPU time used: %f seconds\n", cpu_used);
    
    return 0;
}
