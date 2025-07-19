#include "main.h"
#include "globals.h"
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

#ifndef CLOCK_MONOTONIC_RAW
#define CLOCK_MONOTONIC_RAW 4
#endif
#ifndef BLAS_INT
#define BLAS_INT int
#endif
#undef I

// Calculates a B x G X C tensor with the probabilities of each district
Matrix *getProbability(Matrix *V, Matrix *beta, Matrix *alpha)
{

    int B = V->rows;
    int A = V->cols;
    int Cminus1 = alpha->rows;
    int C = Cminus1 + 1;
    int G = beta->rows;

    // ---- Generate needed matrices
    Matrix *toReturn = Calloc(B, Matrix);
    Matrix alphaTransposed = transposeMatrix(alpha);

    // ---- Multiply V and alpha transposed
    Matrix VxA = matrixMultiplication(V, &alphaTransposed);

    // ---- Exponentiate
    for (int d = 0; d < B; d++)
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
    return toReturn;
}

Matrix *E_step(Matrix *X, Matrix *W, Matrix *V, Matrix *beta, Matrix *alpha)
{

    int B = V->rows;
    int A = V->cols;
    int Cminus1 = alpha->rows;
    int C = Cminus1 + 1;
    int G = beta->rows;

    // ---- Get the probabilities
    Matrix *probabilities = getProbability(V, beta, alpha);

    // ---- Get S_bc
    Matrix S_bc = createMatrix(B, C);
    // double *W_row = (double *)Calloc(G, double);
    // double *S_row = (double *)Calloc(C, double);
    // double *W_buf = Calloc(G, double);
    double *W_buf[G];
    for (int b = 0; b < B; b++)
    { // --- For each ballot box
      // Get the bth row of W
      // vectorMatrixMultiplication_inplace(W_buf, &probabilities[b], S_ptr); // --- Length C
        for (int g = 0; g < G; g++)
        {
            W_buf[g] = &MATRIX_AT_PTR(W, b, g);
        }
        for (int c = 0; c < C; c++)
        {
            double acc = 0;
            for (int g = 0; g < G; g++)
            {
                acc += *W_buf[g] * MATRIX_AT(probabilities[b], g, c);
            }
            MATRIX_AT(S_bc, b, c) = acc;
        }

        // Copy the output to S_bc matrix

        // memcpy(&S_bc.data[b * C], S_row, C * sizeof(double));

        // double *S_row = getRow(&S_bc, b);
        // memcpy(W_row, &W->data[b * G], G * sizeof(double));
        // double *W_row = getRow(W, b);
        // double *S_ptr = getRow(&S_bc, b); // length C

        // Multiply
        // vectorMatrixMultiplication_inplace(W_buf, &probabilities[b], S_ptr); // --- Length C
    }
    // Free(W_buf);
    // Free(S_row);

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
        freeMatrix(&probabilities[b]);
    }

    freeMatrix(&S_bc);
    Free(probabilities);
    return q_bgc;
}

double objective_function(Matrix *W, Matrix *V, Matrix *alpha, Matrix *beta, Matrix *q_bgc)
{

    int B = V->rows;
    int A = V->cols;
    int Cminus1 = alpha->rows;
    int C = Cminus1 + 1;
    int G = beta->rows;

    double loss = 0.0;

    // --- Get probabilities
    Matrix *probabilities = getProbability(V, beta, alpha);

    // --- Get the dot product
    for (int b = 0; b < B; b++)
    { // --- For each ballot box
        for (int g = 0; g < G; g++)
        { // --- For each group
            // Must be a continuos pointer, hence, the macro can't be used
            /*
            double *q_ptr = &q_bgc[b].data[g * C];
            double *logp_ptr = &probabilities[b].data[g * C];
            double dot = matrixDotProduct(q_ptr, logp_ptr, C);
            */
            double dot = 0.0;
            for (int c = 0; c < C; c++)
            {
                double q = MATRIX_AT(q_bgc[b], g, c);
                double p = MATRIX_AT(probabilities[b], g, c);
                dot += q * log(fmax(p, 1e-12));
            }
            loss -= MATRIX_AT_PTR(W, b, g) * dot; // Check if it is to sum or to substract
        }
        freeMatrix(&probabilities[b]);
    }
    Free(probabilities);
    return loss;
}

void compute_gradients(const Matrix *W, Matrix *V, Matrix *alpha, Matrix *beta, Matrix *q_bgc, Matrix *grad_alpha_out,
                       Matrix *grad_beta_out)
{
    int B = V->rows;
    int A = V->cols;
    int Cminus1 = alpha->rows;
    int C = Cminus1 + 1;
    int G = beta->rows;

    // --- Get probabilities
    Matrix *p_bgc = getProbability(V, beta, alpha);

    for (int g = 0; g < G; g++)
    {
        for (int c = 0; c < Cminus1; c++)
        {
            double sum1 = 0;
            double sum2 = 0;
            for (int b = 0; b < B; b++)
            {
                double w = MATRIX_AT_PTR(W, b, g);
                sum1 += w * MATRIX_AT(q_bgc[b], g, c);
                sum2 += w * MATRIX_AT(p_bgc[b], g, c);
            }
            MATRIX_AT_PTR(grad_beta_out, g, c) = sum1 - sum2;
        }
    }

    for (int c = 0; c < Cminus1; c++)
    {
        for (int a = 0; a < A; a++)
        {
            double sum1 = 0;
            double sum2 = 0;
            for (int b = 0; b < B; b++)
            {
                for (int g = 0; g < G; g++)
                {
                    double w = MATRIX_AT_PTR(W, b, g);
                    double q = MATRIX_AT(q_bgc[b], g, c);
                    double v = MATRIX_AT_PTR(V, b, a);
                    double p = MATRIX_AT(p_bgc[b], g, c);
                    sum1 += w * q * v;
                    sum2 += w * p * v;
                }
            }
            MATRIX_AT_PTR(grad_alpha_out, c, a) = sum1 - sum2;
        }
    }
    // after filling grad_alpha_out and grad_beta_out:
    for (int b = 0; b < B; b++)
    {
        freeMatrix(&p_bgc[b]);
    }
    Free(p_bgc);
}

/*
void compute_hessian(const Matrix *W,     // BxG
                     const Matrix *V,     // BxA   (k_ba)
                     const Matrix *alpha, // (C-1)xA
                     const Matrix *beta,  // GxC   (we only use 1..C-1)
                     Matrix *H_out)       // (dα+dβ) × (dα+dβ) – PRE-zeroed
{
    // --- Get the dimensions
    const int B = V->rows;
    const int A = V->cols;
    const int Cm = alpha->rows; // Cminus1 = C-1
    const int C = Cm + 1;
    const int G = beta->rows;

    const int d_alpha = Cm * A; // flattened \alpha block size
    const int d_beta = G * Cm;  // flattened \beta block size

    // --- Preallocate the Hessian blocks
    Matrix H_aa = createMatrix(d_alpha, d_alpha);
    Matrix H_ab = createMatrix(d_alpha, d_beta);
    Matrix H_ba = createMatrix(d_beta, d_alpha);
    Matrix H_bb = createMatrix(d_beta, d_beta);

    // --- Get the probabilities
    Matrix *p_bgc = getProbability((Matrix *)V, (Matrix *)beta, (Matrix *)alpha);
    // -----------------------------------------------------------------------

    // =====================   \alpha - \alpha   ======================================
    for (int i = 0; i < d_alpha; ++i)
    { // --- For each coordinate in the flattened \alpha block
        int a_i = i / Cm;
        int c_i = i % Cm;
        for (int j = 0; j < d_alpha; ++j)
        { // --- For each coordinate in the flattened \alpha block
            int a_j = j / Cm;
            int c_j = j % Cm;

            double sum = 0.0;
            for (int b = 0; b < B; ++b)
            { // --- For each ballot box
                double k_ba = MATRIX_AT_PTR(V, b, a_i);
                double k_ba_ = MATRIX_AT_PTR(V, b, a_j);

                for (int g = 0; g < G; ++g)
                { // --- For each group
                    double w = MATRIX_AT_PTR(W, b, g);
                    double p_c = MATRIX_AT(p_bgc[b], g, c_i);
                    double p_c_ = MATRIX_AT(p_bgc[b], g, c_j);

                    double t1 = (c_i == c_j ? -w * k_ba * k_ba_ * p_c : 0.0);
                    double t2 = w * k_ba * k_ba_ * p_c * p_c_;
                    sum += t1 + t2;
                }
            }
            MATRIX_AT(H_aa, i, j) = sum;
        }
    }

    // =====================   \beta - \alpha   ======================================
    for (int i = 0; i < d_alpha; ++i)
    { // --- For each \alpha coordinate
        int a_i = i / Cm;
        int c_i = i % Cm;

        for (int j = 0; j < d_beta; ++j)
        { // --- For each \beta coordinate
            int c_j = j / G;
            int g_j = j % G;

            double sum = 0.0;
            for (int b = 0; b < B; ++b)
            { // --- For each ballot box
                double w = MATRIX_AT_PTR(W, b, g_j);
                double k_ba = MATRIX_AT_PTR(V, b, a_i);
                double p_ci = MATRIX_AT(p_bgc[b], g_j, c_i);
                double p_cj = MATRIX_AT(p_bgc[b], g_j, c_j);

                double t1 = (c_i == c_j ? -w * k_ba * p_ci : 0.0);
                double t2 = w * k_ba * p_ci * p_cj;
                sum += t1 + t2;
            }
            MATRIX_AT(H_ab, i, j) = sum;
            MATRIX_AT(H_ba, j, i) = sum; // Simmetric
        }
    }

    // =====================   \beta – \beta   ======================================
    for (int i = 0; i < d_beta; ++i)
    { // --- For each beta coordinate
        int c_i = i / G;
        int g_i = i % G;

        for (int j = 0; j < d_beta; ++j)
        { // --- For each beta coordinate
            int c_j = j / G;
            int g_j = j % G;

            if (g_i != g_j)
            {
                MATRIX_AT(H_bb, i, j) = 0.0; // Not actually necessary, for clarity
                continue;
            }

            double sum = 0.0;
            for (int b = 0; b < B; ++b)
            { // --- For each ballot box
                double w = MATRIX_AT_PTR(W, b, g_i);
                double p_ci = MATRIX_AT(p_bgc[b], g_i, c_i);
                double p_cj = MATRIX_AT(p_bgc[b], g_j, c_j);

                double t1 = (c_i == c_j ? -w * p_ci : 0.0);
                double t2 = w * p_ci * p_cj;
                sum += t1 + t2;
            }
            MATRIX_AT(H_bb, i, j) = sum;
        }
    }

    // Assemble everything in ROW-MAJOR order, this is VERY important!
    const int D = d_alpha + d_beta;
    for (int i = 0; i < D; ++i)
    { // For each coordinate in the flattened Hessian
        for (int j = 0; j < D; ++j)
        { // For each coordinate in the flattened Hessian
            double val;
            // We save the hessian values as negative
            if (i < d_alpha && j < d_alpha)
            {
                // \alpha - \alpha
                val = -MATRIX_AT(H_aa, i, j);
            }
            else if (i < d_alpha && j >= d_alpha)
            {
                // \alpha - \beta
                val = -MATRIX_AT(H_ab, i, j - d_alpha);
            }
            else if (i >= d_alpha && j < d_alpha)
            {
                // \beta - \alpha
                val = -MATRIX_AT(H_ba, i - d_alpha, j);
            }
            else
            {
                // \beta \beta
                val = -MATRIX_AT(H_bb, i - d_alpha, j - d_alpha);
            }
            // Assign them as row major
            H_out->data[i * D + j] = val;
        }
    }

    // Clean up
    for (int b = 0; b < B; ++b)
        freeMatrix(&p_bgc[b]);
    Free(p_bgc);

    freeMatrix(&H_aa);
    freeMatrix(&H_ab);
    freeMatrix(&H_ba);
    freeMatrix(&H_bb);
}
*/

void compute_hessian(const Matrix *W,     // B×G
                     const Matrix *V,     // B×A
                     const Matrix *alpha, // (C-1)×A
                     const Matrix *beta,  // G×C  (we only use cols 0..C-2)
                     Matrix *H_out)       // (d_alpha+d_beta)×(d_alpha+d_beta) – PRE-zeroed
{
    const int B = V->rows;
    const int A = V->cols;
    const int Cm = alpha->rows; // C-1
    const int C = Cm + 1;
    const int G = beta->rows;
    const int d_alpha = Cm * A;
    const int d_beta = G * Cm;
    const int D = d_alpha + d_beta;

    /* 1) zero out the entire Hessian */
    memset(H_out->data, 0, D * D * sizeof(double));

    /* 2) compute p_bgc once */
    Matrix *p_bgc = getProbability((Matrix *)V, (Matrix *)beta, (Matrix *)alpha);

    size_t n_iters = (size_t)B * G * Cm * Cm;
#ifdef _OPENMP
#pragma omp parallel for collapse(4) if (n_iters > 100) schedule(static)
#endif
    for (int b = 0; b < B; b++)
    {
        for (int g = 0; g < G; g++)
        {
            double w = MATRIX_AT_PTR(W, b, g);

            for (int ci = 0; ci < Cm; ci++)
            {
                double p_ci = MATRIX_AT(p_bgc[b], g, ci);

                for (int cj = 0; cj < Cm; cj++)
                {
                    double p_cj = MATRIX_AT(p_bgc[b], g, cj);
                    // negative of second‐derivative of log‑prob
                    double base = p_ci * p_cj + (ci == cj ? -p_ci : 0.0);
                    double factor = -w * base;

                    // alpha–alpha block
                    for (int ai = 0; ai < A; ai++)
                    {
                        double v_ai = MATRIX_AT_PTR(V, b, ai);
                        int row_alpha = ai * Cm + ci;
                        for (int aj = 0; aj < A; aj++)
                        {
                            double v_aj = MATRIX_AT_PTR(V, b, aj);
                            int col_alpha = aj * Cm + cj;
                            MATRIX_AT_PTR(H_out, row_alpha, col_alpha) += v_ai * v_aj * factor;
                        }
                    }

                    // alpha–beta and beta–alpha blocks
                    for (int ai = 0; ai < A; ai++)
                    {
                        double v_ai = MATRIX_AT_PTR(V, b, ai);
                        int row_alpha = ai * Cm + ci;
                        int col_beta = d_alpha + g * Cm + cj;
                        double upd = v_ai * factor;
                        MATRIX_AT_PTR(H_out, row_alpha, col_beta) += upd;
                        MATRIX_AT_PTR(H_out, col_beta, row_alpha) += upd;
                    }

                    // beta–beta block
                    {
                        int row_beta = d_alpha + g * Cm + ci;
                        int col_beta = d_alpha + g * Cm + cj;
                        MATRIX_AT_PTR(H_out, row_beta, col_beta) += factor;
                    }
                }
            }
        }
    }

    // cleanup
    for (int b = 0; b < B; ++b)
    {
        freeMatrix(&p_bgc[b]);
    }
    Free(p_bgc);
}

// ----- HELPER FUNCTION ----- //
// Packs grad_alpha (C–1 x A, column‐major) followed by grad_beta (G x C–1, column‐major)
// both in column‐major order, into the flat vector g[0..D-1].
static void pack_gradients(const Matrix *grad_alpha, // (C–1)×A
                           const Matrix *grad_beta,  // G×(C–1)
                           double *g                 // length = Cminus1*A + G*Cminus1
)
{
    int Cminus1 = grad_alpha->rows;
    int A = grad_alpha->cols;
    int G = grad_beta->rows;
    int idx = 0;

    // \alpha block
    for (int a = 0; a < A; a++)
    { // --- For each attribute
        for (int c = 0; c < Cminus1; c++)
        { // --- For each candidate
            g[idx++] = MATRIX_AT_PTR(grad_alpha, c, a);
        }
    }

    // \beta block
    for (int c = 0; c < Cminus1; c++)
    { // --- For each candidate
        for (int gi = 0; gi < G; gi++)
        { // --- For each group
            g[idx++] = MATRIX_AT_PTR(grad_beta, gi, c);
        }
    }
}

// Unpacks a flat D‐vector v[] back into two matrices, in the same order.
static void unpack_step(const double *v,    // length = Cminus1*A + G*Cminus1
                        Matrix *grad_alpha, // (C–1)×A
                        Matrix *grad_beta   // G×(C–1)
)
{
    int Cminus1 = grad_alpha->rows;
    int A = grad_alpha->cols;
    int G = grad_beta->rows;
    int idx = 0;

    // \alpha block
    for (int a = 0; a < A; ++a)
    { // --- For each attribute
        for (int c = 0; c < Cminus1; ++c)
        { // --- For each candidate
            MATRIX_AT_PTR(grad_alpha, c, a) = v[idx++];
        }
    }

    // \beta block
    for (int c = 0; c < Cminus1; ++c)
    { // --- For each candidate
        for (int gi = 0; gi < G; ++gi)
        { // --- For each group
            MATRIX_AT_PTR(grad_beta, gi, c) = v[idx++];
        }
    }
}

int Newton_damped(Matrix *W,      // B×G weights
                  Matrix *V,      // B×A covariates
                  Matrix **q_bgc, // array of B matrices G×C
                  Matrix *alpha0, // initial α (C-1×A)
                  Matrix *beta0,  // initial β  (G×C)
                  double tol, int max_iter, double alpha_bs, double beta_bs,
                  Matrix *alpha_out, // outputs (same dims as alpha0)
                  Matrix *beta_out   // outputs (same dims as beta0)
)
{
    int B = V->rows;
    int A = V->cols;
    int Cminus1 = alpha0->rows;
    int C = Cminus1 + 1;
    int G = beta0->rows;
    int D = Cminus1 * A + G * Cminus1;

    // ---- Clone initial parameters
    Matrix *alpha = copMatrixPtr(alpha0);
    Matrix *beta = copMatrixPtr(beta0);

    // ---- Allocate temporaries
    double *gvec = (double *)Calloc(D, double);
    double *vvec = (double *)Calloc(D, double);
    Matrix dalpha = createMatrix(Cminus1, A);
    Matrix dbeta = createMatrix(G, Cminus1);
    Matrix H = createMatrix(D, D);

    int iter;
    for (iter = 0; iter < max_iter; iter++)
    {
        // ---- Evaluate loss, gradient, Hessian at the current points
        // Loss
        double f0 = objective_function(W, V, alpha, beta, *q_bgc);

        // Gradients
        Matrix grad_alpha = createMatrix(Cminus1, A);
        Matrix grad_beta = createMatrix(G, Cminus1);
        compute_gradients(W, V, alpha, beta, *q_bgc, &grad_alpha, &grad_beta);
        pack_gradients(&grad_alpha, &grad_beta, gvec);

        // Hessian (prezero H)
        size_t D2 = (size_t)D * (size_t)D;
        memset(H.data, 0, D2 * sizeof(double));
        compute_hessian(W, V, alpha, beta, &H);
        freeMatrix(&grad_alpha);
        freeMatrix(&grad_beta);

        // Damping of the hessian
        double min_diag = DBL_MAX;
        for (int i = 0; i < D; i++)
        {
            min_diag = fmin(min_diag, MATRIX_AT(H, i, i));
        }
        // Choose a tiny epsilon
        double eps = 1e-6 * (fabs(min_diag) + 1.0);
        // If the smallest diagonal is below zero, shift by (–min_diag + \epsilon),
        // otherwise just add \epsilon to make it strictly positive
        double shift = (min_diag < 0.0 ? -min_diag + eps : eps);
        for (int i = 0; i < D; i++)
        {
            MATRIX_AT(H, i, i) += shift;
        }
        // ---...--- //

        // Solve H v = -g, for approximating it with Taylor expansion
        Matrix Hcopy = copMatrix(&H);
        solve_linear_system(D, Hcopy.data, gvec, vvec);
        for (int i = 0; i < D; i++)
        {
            vvec[i] = -vvec[i];
        }
        unpack_step(vvec, &dalpha, &dbeta);

        // Convergence check
        double g_inf = 0;
        for (int i = 0; i < D; i++)
        {
            g_inf = fmax(g_inf, fabs(gvec[i]));
        }
        if (g_inf < tol)
            break;

        // Armijo backtracking line search
        double t = 1.0; // We start with t = 1
        // compute g*v, the direction
        double gv = 0;
        for (int i = 0; i < D; i++)
            gv += gvec[i] * vvec[i];

        while (1)
        {
            // Trial parameters
            Matrix *alpha_t = copMatrixPtr(alpha);
            Matrix *beta_t = copMatrixPtr(beta);
            // alpha_t = alpha_t + t * dalpha
            for (int i = 0; i < Cminus1; i++)
                for (int j = 0; j < A; j++)
                    MATRIX_AT_PTR(alpha_t, i, j) += t * MATRIX_AT(dalpha, i, j);
            // beta_t = beta_t + t * dbeta
            for (int i = 0; i < G; i++)
                for (int j = 0; j < Cminus1; j++)
                    MATRIX_AT_PTR(beta_t, i, j) += t * MATRIX_AT(dbeta, i, j);

            double f_trial = objective_function(W, V, alpha_t, beta_t, *q_bgc);
            freeMatrix(alpha_t);
            freeMatrix(beta_t);
            if (f_trial <= f0 + alpha_bs * t * gv)
            {
                break;
            }
            t *= beta_bs;
            if (t < 1e-4)
                break;
        }

        // Update parameters. i.e, alpha and beta
        for (int i = 0; i < Cminus1; i++)
            for (int j = 0; j < A; j++)
                MATRIX_AT_PTR(alpha, i, j) += t * MATRIX_AT(dalpha, i, j);
        for (int i = 0; i < G; i++)
            for (int j = 0; j < Cminus1; j++)
                MATRIX_AT_PTR(beta, i, j) += t * MATRIX_AT(dbeta, i, j);

    } // --- Newton iteration finishes

    // Copy results out
    size_t alpha_elems = alpha->rows * alpha->cols;
    size_t alpha_bytes = alpha_elems * sizeof(double);
    memcpy(alpha_out->data, alpha->data, alpha_bytes);

    size_t beta_elems = beta->rows * beta->cols;
    size_t beta_bytes = beta_elems * sizeof(double);
    memcpy(beta_out->data, beta->data, beta_bytes);

    // Cleanup
    freeMatrix(alpha);
    freeMatrix(beta);
    freeMatrix(&dalpha);
    freeMatrix(&dbeta);
    freeMatrix(&H);
    Free(gvec);
    Free(vvec);

    return iter + 1;
}

double compute_ll_multinomial(const Matrix *X, // BxC
                              const Matrix *W, // BxG
                              Matrix *q_bgc,   // BxGxC
                              Matrix *V,       // BxA
                              Matrix *alpha,   // (C-1)xA  (plus a baseline row)
                              Matrix *beta     // GxC
)
{
    int B = X->rows;
    int C = X->cols;
    int A = V->cols;
    int G = W->cols;

    double total_ll = 0.0;

    Matrix *p_bgc = getProbability(V, beta, alpha);

    for (int b = 0; b < B; ++b)
    {
        // compute denominator = sum_g w_bg[b,g]
        double wsum = 0;
        for (int g = 0; g < G; ++g)
            wsum += MATRIX_AT_PTR(W, b, g);
        double denom = wsum + 1e-12;

        // multinomial factorial term
        double xb = 0;
        for (int c = 0; c < C; ++c)
        {
            double x = MATRIX_AT_PTR(X, b, c);
            total_ll -= lgamma(x + 1.0);
            xb += x;
        }
        total_ll += lgamma(xb + 1.0);

        // data term \sum x_bc · log(p_bc)
        for (int c = 0; c < C; ++c)
        {
            double marg = 0;
            for (int g = 0; g < G; ++g)
            {
                marg += MATRIX_AT_PTR(W, b, g) * MATRIX_AT(p_bgc[b], g, c);
            }
            // normalize
            double pbc = marg / denom;
            // clamp before log
            total_ll += MATRIX_AT_PTR(X, b, c) * log(fmax(pbc, 1e-12));
        }
    }

    return total_ll;
}

void M_step(Matrix *X, Matrix *W, Matrix *V, Matrix *q_bgc, Matrix *alpha, Matrix *beta, const double tol,
            const int maxnewton, const bool verbose)
{
    int newton_iterations = Newton_damped(W, V, &q_bgc, alpha, beta, tol, maxnewton, 0.01, 0.5, alpha, beta);

    if (verbose)
    {
        Rprintf("The newton algorithm was made in %d iterations\n", newton_iterations);
    }
}

Matrix *EM_Algorithm(Matrix *X, Matrix *W, Matrix *V, Matrix *beta, Matrix *alpha, const int maxiter,
                     const double maxtime, const double ll_threshold, const int maxnewton, const bool verbose,
                     double *out_elapsed, int *total_iterations)
{
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0); // Start timer
    double current_ll = -DBL_MAX;
    double new_ll = -DBL_MAX;
    for (int iter = 0; iter < maxiter; iter++)
    {
        *total_iterations += 1;
        Matrix *q_bgc = E_step(X, W, V, beta, alpha);
        M_step(X, W, V, q_bgc, alpha, beta, 0.01, maxnewton, verbose);
        new_ll = compute_ll_multinomial(X, W, q_bgc, V, alpha, beta);
        Free(q_bgc);

        if (verbose)
        {
            Rprintf("Iteration %d: log-likelihood = %.4f\n", iter + 1, new_ll);
        }

        // Check for convergence
        if (fabs(new_ll - current_ll) <= ll_threshold)
        {
            if (verbose)
            {
                Rprintf("Converged after %d iterations.\n", iter + 1);
            }
            break;
        }
        current_ll = new_ll;
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

    // Compute elapsed seconds
    double sec = (double)(t1.tv_sec - t0.tv_sec);
    double nsec = (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    *out_elapsed = sec + nsec;

    Matrix *finalProbability = getProbability(V, beta, alpha);
    return finalProbability; // Return the final probabilities
}
