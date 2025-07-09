#include "main.h"
#include "globals.h"
#include "utils_matrix.h"
#include <R.h>
#include <R_ext/BLAS.h>
#include <R_ext/Memory.h>
#include <R_ext/RS.h> /* for R_Calloc/R_Free, F77_CALL */
#include <Rinternals.h>
#include <dirent.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef Calloc
#define Calloc(n, type) ((type *)R_chk_calloc((size_t)(n), sizeof(type)))
#endif

#ifndef Free
#define Free(p) R_chk_free((void *)(p))
#endif

#ifndef CLOCK_MONOTONIC_RAW
#define CLOCK_MONOTONIC_RAW 4
#endif
#ifndef BLAS_INT
#define BLAS_INT int
#endif
#undef I

// Calculates a D x G X C tensor with the probabilities of each district
// If
Matrix *getProbability(Matrix *X, Matrix *W, Matrix *V, Matrix *beta, Matrix *alpha)
{

    int D = V->rows;
    int A = V->cols;
    int Cminus1 = alpha->rows;
    int C = Cminus1 + 1;
    int G = beta->rows;

    // ---- Generate needed matrices
    Matrix *toReturn = Calloc(D, Matrix);
    Matrix alphaTransposed = transposeMatrix(alpha);

    // ---- Multiply V and alpha transposed
    Matrix VxA = matrixMultiplication(V, &alphaTransposed);

    // ---- Exponentiate
    for (int d = 0; d < D; d++)
    { // --- For each district
        toReturn[d] = createMatrix(G, C);
        for (int g = 0; g < G; g++)
        { // --- For each group
            double sum = 0.0;
            for (int c = 0; c < Cminus1; c++)
            { // --- For each candidate
                // Obtain the exponential of the linear combination
                double u = MATRIX_AT_PTR(beta, g, c) + MATRIX_AT(VxA, d, c);
                double ex = exp(u);
                MATRIX_AT(toReturn[d], g, c) = exp(u);
                sum += ex;
            }

            // Base line candidate
            MATRIX_AT(toReturn[d], g, Cminus1) = 1;
            sum += 1;

            for (int c = 0; c < C; c++)
            { // --- For each candidate
                // Normalize
                MATRIX_AT(toReturn[d], g, c) /= sum;
            }
        }
    }
    // Free matrices
    freeMatrix(&alphaTransposed);
    freeMatrix(&VxA);
    return toReturn; // Replace with actual return value
}

Matrix *E_step(Matrix *X, Matrix *W, Matrix *V, Matrix *beta, Matrix *alpha)
{
    // ---- Get the probabilities
    Matrix *probabilities = getProbability(X, W, V, beta, alpha);
}

Matrix EM_Algorithm(Matrix *X, Matrix *W, Matrix *V, Matrix *beta, Matrix *alpha, const int maxiter,
                    const double maxtime, const double param_threshold, const double ll_threshold, const bool verbose)
{
}
