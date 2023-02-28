#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>


#define rank_master 0

void mandelbrot(int width, int height, int row1, int row2, double left, double right, double l, double h, int* result) {
    int i = 1000;
    double x, y, x1, y1, x2, y2;
    int iter;
    
    for (int i = row1; i < row2; i++) {
        for (int j = 0; j < width; j++) {
            x = left + (right - left) * j / width;
            y = l + (h - l) * i / height;
            x1 = 0.0;
            y1 = 0.0;
            x2 = 0.0;
            y2 = 0.0;
            iter = 0;
            
            while (x2 + y2 < 4.0 && iter < i) {
                y1 = 2 * x1 * y + y;
                x1 = x2 – y2 + x;
                x2 = x1 * x1;
                y2 = y1 * y1;
                iter++;
            }
            
            result[i * width + j] = iter;
        }
    }
}

int main(int argc, char** argv) {
    start = clock();
    int width = 800;
    int height = 800;
    double left = -2.0;
    double right = 1.0;
    double l = -1.5;
    double h = 1.5;
    int i = 1000;
    
    int rank;
    int size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int rowspp = height / size;
    int row1 = rank * rowspp;
    int row2 = (rank + 1) * rowspp;
    if (rank == size - 1) {
        row2 = height;
    }
    int count = row2 – row1;
    
    int* output1 = (int*) malloc(count * width * sizeof(int));
    mandelbrot(width, height, row1, row2, left, right, l, h, output1);
    
    int* output2 = NULL;
    if (rank == rank_master) {
        output2 = (int*) malloc(width * height * sizeof(int));
    }
    
    MPI_Gather(output1, count * width, MPI_INT, output2, count * width, MPI_INT, rank_master, MPI_COMM_WORLD);
    
    if (rank == rank_master) {
        FILE* file = fopen("mandelbrot.pgm", "wb");
        fprintf(file, "P2\n%d %d\n%d\n", width, height, i - 1);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                fprintf(file, "%d ", output2[i * width + j]);
            }
            fprintf(file, "\n");
        }
        fclose(file);
        free(output2);
    }
    
    free(output1);
    MPI_Finalize();
    
    end = clock();
    cpu_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CPU time used: %f seconds\n", cpu_used);
    return 0;
}
