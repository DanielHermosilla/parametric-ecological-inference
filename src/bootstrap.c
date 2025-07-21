#include "bootstrap.h"
#include "globals.h"
#include "main.h"
#include "utils_matrix.h"
#include <R.h>
#include <R_ext/BLAS.h>
#include <R_ext/Memory.h>
#include <R_ext/RS.h> /* for R_Calloc/R_Free, F77_CALL */
#include <Rinternals.h>
#include <dirent.h>
#include <float.h>
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

void iterMat(const Matrix *originalX, const Matrix *originalW, const Matrix *originalV, Matrix *newX, Matrix *newW,
             Matrix *newV, const int *indexArr, int indexStart)
{
    // The amount of ballot boxes
    int ballotBoxes = originalW->rows;
    for (int b = 0; b < ballotBoxes; b++)
    {
        int sampledIndex = indexArr[indexStart + b];
        // For the 'w' matrix
        for (int g = 0; g < originalW->cols; g++)
        { // --- For each group given a ballot box
            MATRIX_AT_PTR(newW, b, g) = MATRIX_AT_PTR(originalW, sampledIndex, g);
        }
        // For the 'x' matrix
        for (int c = 0; c < originalX->cols; c++)
        { // --- For each candidate given a ballot box
            MATRIX_AT_PTR(newX, b, c) = MATRIX_AT_PTR(originalX, sampledIndex, c);
        }
        // For the 'v' matrix
        for (int v = 0; v < originalX->cols; v++)
        { // --- For each candidate given a ballot box
            MATRIX_AT_PTR(newV, b, v) = MATRIX_AT_PTR(originalV, sampledIndex, v);
        }
    }
}

Matrix standardDeviations(Matrix *bootstrapResults, Matrix *sumMatrix, int totalIter)
{

    // Get the mean for each component
    for (int i = 0; i < sumMatrix->rows; i++)
    {
        for (int j = 0; j < sumMatrix->cols; j++)
        {
            MATRIX_AT_PTR(sumMatrix, i, j) /= totalIter;
        }
    }

    Matrix sdMatrix = createMatrix(sumMatrix->rows, sumMatrix->cols);

    // Get the summatory (x_i - \mu)^2
    for (int h = 0; h < totalIter; h++)
    {
        // Yields the summatory for each dimension
        for (int i = 0; i < sdMatrix.rows; i++)
        {
            for (int j = 0; j < sdMatrix.cols; j++)
            {
                double diff = MATRIX_AT(bootstrapResults[h], i, j) - MATRIX_AT_PTR(sumMatrix, i, j);
                MATRIX_AT(sdMatrix, i, j) += diff * diff;
            }
        }
        freeMatrix(&bootstrapResults[h]);
    }

    // Make the division and get the square root
    for (int i = 0; i < sdMatrix.rows; i++)
    {
        for (int j = 0; j < sdMatrix.cols; j++)
        {
            double val = sqrt(MATRIX_AT(sdMatrix, i, j) / (totalIter - 1));
            MATRIX_AT(sdMatrix, i, j) = val == 0 ? NAN : val;
        }
    }
    return sdMatrix;
}

// Matrix *EM_Algorithm(Matrix *X, Matrix *W, Matrix *V, Matrix *beta, Matrix *alpha, const int maxiter,
//                     const double maxtime, const double ll_threshold, const int maxnewton, const bool verbose,
//                    double *out_elapsed, int *total_iterations)
void bootstrap(Matrix *X, Matrix *W, Matrix *V, Matrix *beta, Matrix *alpha, const int bootiter, const int maxiter,
               const double maxtime, const double ll_threshold, const int maxnewton, const bool verbose,
               double *out_elapsed, int *total_iterations, Matrix *sdBetas, Matrix *sdAlpha)
{

    // ---- Initial variables
    int bdim = W->rows;
    int samples = bdim * bootiter;
    int matsize = W->cols * X->rows;

    // ---- Generate the indices for bootstrap ---- //
    int *indices = Calloc(bdim * bootiter, int);
    // For each bootstrap replicate i
sampling:
    for (int i = 0; i < bdim * bootiter; i++)
    {
        indices[i] = (int)(unif_rand() * bdim);
    }
    // Check that every index is not the same
    for (int i = 1; i < bdim * bootiter; i++)
    {
        if (indices[i] != indices[i - 1])
            break;
        if (i == bdim * bootiter - 1)
        {
            goto sampling;
        }
    }

    // We want to avoid the case where the same ballot box is drawn FOR EACH placement
    // This has a probability of 1/b^b. Maybe this calculation could be avoided at 6 > ballot boxes,
    // since then it becomes practically 0
    // ---...--- //

    // ---- Execute the bootstrap algorithm ---- //
    Matrix sumMatBeta = createMatrix(beta->cols, beta->rows);
    Matrix *resultsBeta = Calloc(bootiter, Matrix);
    Matrix sumMatAlpha = createMatrix(alpha->cols, alpha->rows);
    Matrix *resultsAlpha = Calloc(bootiter, Matrix);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < bootiter; i++)
    {
        if (verbose && bootiter > 20 && (i % (bootiter / 20) == 0)) // Print every 5% (20 intervals)
        {
            double progress = (double)i / bootiter * 100;
            Rprintf("%.0f%% of iterations completed.\n", progress);
        }
        // ---- Declare variables for the current iteration
        Matrix iterX = createMatrix(X->rows, X->cols);
        Matrix iterW = createMatrix(W->rows, W->cols);
        Matrix iterV = createMatrix(V->rows, V->cols);
        Matrix BetaR = copMatrix(beta);
        Matrix AlphaR = copMatrix(alpha);
        iterMat(X, W, V, &iterX, &iterW, &iterV, indices, i * bdim);

        // Declare EM variables, they're not used in this case...
        // It could be useful to yield a mean if the user wants to (logLL mean?)
        double elapsed = 0.0;
        int total_iter = 0;
        Matrix *finalProb = EM_Algorithm(&iterX, &iterW, &iterV, &BetaR, &AlphaR, maxiter, maxtime, ll_threshold,
                                         maxnewton, false, &elapsed, &total_iter);
        // Sum each value so later we can get the mean
        for (int j = 0; j < BetaR.cols; j++)
        {
            for (int k = 0; k < BetaR.rows; k++)
            {
                MATRIX_AT(sumMatBeta, j, k) += MATRIX_AT(BetaR, j, k);
            }
        }

        resultsBeta[i] = BetaR;

        for (int j = 0; j < AlphaR.cols; j++)
        {
            for (int k = 0; k < AlphaR.rows; k++)
            {
                MATRIX_AT(sumMatAlpha, j, k) += MATRIX_AT(AlphaR, j, k);
            }
        }

        resultsAlpha[i] = AlphaR;
        // memcpy(&results[i * matsize], resultP.data, matsize * sizeof(double));

        // ---- Release loop allocated variables ---- //
        // freeMatrix(&iterP);
        freeMatrix(finalProb); // Check, for a possible segmentation fault
        freeMatrix(&iterX);
        freeMatrix(&iterW);
        // ---...--- //
    }
    *sdBetas = standardDeviations(resultsBeta, &sumMatBeta, bootiter);
    *sdAlpha = standardDeviations(resultsAlpha, &sumMatAlpha, bootiter);

    if (verbose)
    {
        Rprintf("Bootstrapping finished!\nThe estimated standard deviation matrix for beta is:\n");
        printMatrix(sdBetas);
        Rprintf("\nThe estimated standard deviation matrix for alpha is:\n");
        printMatrix(sdAlpha);
    }

    Free(indices);
    freeMatrix(&sumMatBeta);
    freeMatrix(&sumMatAlpha);
    for (int i = 0; i < bootiter; i++)
    {
        freeMatrix(&resultsBeta[i]);
        freeMatrix(&resultsAlpha[i]);
    }
    Free(resultsAlpha);
    Free(resultsBeta);
}
