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

// Calculates a B x G X C tensor with the probabilities of each district
// If
Matrix *getProbability(Matrix *X, Matrix *W, Matrix *V, Matrix *beta, Matrix *alpha)
{

    int B = V->rows;
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

    int B = V->rows;
    int A = V->cols;
    int Cminus1 = alpha->rows;
    int C = Cminus1 + 1;
    int G = beta->rows;

    // ---- Get the probabilities
    Matrix *probabilities = getProbability(X, W, V, beta, alpha);

    // ---- Get S_bc
    Matrix S_bc = createMatrix(B, C);
    double *W_row = (double *)Calloc(G, double);
    double *S_row = (double *)Calloc(C, double);
    for (int b = 0; b < B; b++)
    { // --- For each ballot box
      // Get the bth row of W
        memcpy(W_row, &W->data[b * G], G * sizeof(double));

        // Multiply
        vectorMatrixMultiplication_inplace(W_row, &probabilities[b], S_row); // --- Length C

        // Copy the output to S_bc matrix
        memcpy(&S_bc.data[b * C], S_row, C * sizeof(double));
    }
    Free(W_row);
    Free(S_row);

    // ---- Get q_bgc
    Matrix *q_bgc = Calloc(B, Matrix);

    for (int b = 0; b < B; b++)
    { // --- For each ballot box
        q_bgc[b] = createMatrix(G, C);
        for (int g = 0; g < G; g++)
        { // --- For each group
            double denominator = 0.0;
            double values[C];
            for (int c = 0; c < C; c++)
            { // --- For each candidate
                double n = MATRIX_AT(probabilities[b], g, c) * MATRIX_AT_PTR(X, b, c);
                double d = MATRIX_AT(S_bc, b, c) - MATRIX_AT(probabilities[b], g, c);
                double nd = n / d;
                values[c] = nd;
                denominator += nd;
            }
            for (int c = 0; c < C; c++) // --- For each candidate
                MATRIX_AT(q_bgc[b], g, c) = values[c] / denominator;
        }
    }
    freeMatrix(&S_bc);

    return q_bgc;
}

Matrix EM_Algorithm(Matrix *X, Matrix *W, Matrix *V, Matrix *beta, Matrix *alpha, const int maxiter,
                    const double maxtime, const double param_threshold, const double ll_threshold, const bool verbose)
{
    Matrix *q_bgc = E_step(X, W, V, beta, alpha);
}
