// ---- Avoid circular dependencies
#ifndef GLOBALS_H_EIM
#define GLOBALS_H_EIM

#ifdef __cplusplus

extern "C"
{
#endif

#include <stdint.h>
#include <stdio.h>

// Macro for accessing a 3D flattened array (b x g x c)
// #define Q_3D(q, bIdx, gIdx, cIdx, G, C) ((q)[((bIdx) * (G) * (C)) + ((gIdx) * (C)) + (cIdx)])
#define Q_3D(q, bIdx, gIdx, cIdx, G, C) ((q)[(bIdx) * (G) * (C) + (cIdx) * (G) + (gIdx)])
#define MATRIX_AT(matrix, i, j) (matrix.data[(j) * (matrix.rows) + (i)])
#define MATRIX_AT_PTR(matrix, i, j) (matrix->data[(j) * (matrix->rows) + (i)])

    // All of the helper functions are made towards double type matrices
    typedef struct
    {
        double *data; // Pointer to matrix data array (col-major order)
        int rows;     // Number of rows
        int cols;     // Number of columns
    } Matrix;

#ifdef __cplusplus
}
#endif
#endif // GLOBALS_H
