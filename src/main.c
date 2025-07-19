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

typedef struct
{
    int B, G, C, A, D;
    Matrix *prob;  // length B, each G x C
    Matrix S_bc;   // B x C
    Matrix *q_bgc; // length B, each G x C
    Matrix alpha;
    Matrix beta;
    Matrix grad_alpha; // (C-1) x A
    Matrix grad_beta;  // G x (C-1)
    Matrix H;          // D x D Hessian
    double *gvec;      // length D
    double *vvec;      // length D
} EMBuffers;

void init_EMBuffers(EMBuffers *buf, int B, int G, int Cminus1, int A)
{
    buf->B = B;
    buf->G = G;
    buf->C = Cminus1 + 1;
    buf->A = A;
    buf->D = Cminus1 * A + G * Cminus1;

    // Preallocate probability tensor
    buf->prob = (Matrix *)Calloc(B, Matrix);
    for (int b = 0; b < B; ++b)
    {
        buf->prob[b] = createMatrix(G, buf->C);
    }

    // Preallocate S_bc
    buf->S_bc = createMatrix(B, buf->C);

    // Preallocate q_bgc
    buf->q_bgc = (Matrix *)Calloc(B, Matrix);
    for (int b = 0; b < B; ++b)
    {
        buf->q_bgc[b] = createMatrix(G, buf->C);
    }

    // Gradients and Hessian

    buf->grad_alpha = createMatrix(Cminus1, A);
    buf->grad_beta = createMatrix(G, Cminus1);
    buf->H = createMatrix(buf->D, buf->D);

    // Vectors
    buf->gvec = (double *)Calloc(buf->D, double);
    buf->vvec = (double *)Calloc(buf->D, double);
}

void free_EMBuffers(EMBuffers *buf)
{
    for (int b = 0; b < buf->B; ++b)
    {
        if (buf->prob != NULL)
            freeMatrix(&buf->prob[b]);
        if (buf->q_bgc != NULL)
            freeMatrix(&buf->q_bgc[b]);
    }
    Free(buf->prob);
    Free(buf->q_bgc);

    freeMatrix(&buf->S_bc);

    freeMatrix(&buf->H);
    freeMatrix(&buf->grad_alpha);
    freeMatrix(&buf->grad_beta);
    Free(buf->gvec);
    Free(buf->vvec);
}

// Calculates a B x G X C tensor with the probabilities of each district
/*
void getProbability(EMBuffers *buf, Matrix *V, Matrix *alpha, Matrix *beta)
{
    int B = buf->B, G = buf->G, Cminus1 = alpha->rows, C = buf->C;

    // ---- Generate needed matrices
    Matrix alphaTransposed = transposeMatrix(alpha);

    // ---- Multiply V and alpha transposed
    Matrix VxA = matrixMultiplication(V, &alphaTransposed);

    // ---- Exponentiate
    for (int b = 0; b < B; b++)
    { // --- For each district
        for (int g = 0; g < G; g++)
        { // --- For each group
            double sum = 0.0;
            for (int c = 0; c < Cminus1; c++)
            { // --- For each candidate
                // Obtain the exponential of the linear combination
                double u = MATRIX_AT_PTR(beta, g, c) + MATRIX_AT(VxA, b, c);
                double ex = exp(u);
                MATRIX_AT(buf->prob[b], g, c) = exp(u);
                sum += ex;
            }

            // Base line candidate
            MATRIX_AT(buf->prob[b], g, Cminus1) = 1;
            sum += 1;

            for (int c = 0; c < C; c++)
            { // --- For each candidate
                // Normalize
                MATRIX_AT(buf->prob[b], g, c) /= sum;
            }
        }
    }
    // Free matrices
    freeMatrix(&alphaTransposed);
    freeMatrix(&VxA);
}
*/
// Compute and normalize buf->prob[b][g][c] = softmax_c( beta[g,c] + (V×αᵀ)[b,c] )
// buf->C must be α->rows+1
void getProbability(EMBuffers *buf,
                    Matrix *V,           // B×A
                    const Matrix *alpha, // (C-1)×A
                    const Matrix *beta)  // G×(C-1)
{
    int B = buf->B;
    int G = buf->G;
    int Cminus1 = alpha->rows;
    int C = buf->C; // = Cminus1+1

    // 1) compute V × αᵀ
    Matrix alphaT = transposeMatrix(alpha);
    Matrix VxA = matrixMultiplication(V, &alphaT);

    // 2) exponentiate & normalize into buf->prob
    for (int b = 0; b < B; ++b)
    {
        for (int g = 0; g < G; ++g)
        {
            double sum = 0.0;
            // first Cminus1 candidates
            for (int c = 0; c < Cminus1; ++c)
            {
                double u = MATRIX_AT_PTR(beta, g, c) + MATRIX_AT(VxA, b, c);
                double ex = exp(u);
                MATRIX_AT(buf->prob[b], g, c) = ex;
                sum += ex;
            }
            // baseline candidate
            MATRIX_AT(buf->prob[b], g, Cminus1) = 1.0;
            sum += 1.0;

            // normalize all C entries
            for (int c = 0; c < C; ++c)
            {
                MATRIX_AT(buf->prob[b], g, c) /= sum;
            }
        }
    }

    // 3) cleanup temporaries
    freeMatrix(&alphaT);
    freeMatrix(&VxA);
}

void E_step(Matrix *X, Matrix *W, Matrix *V, EMBuffers *buf)
{

    int B = buf->B, G = buf->G, C = buf->C;

    // ---- Get the probabilities
    getProbability(buf, V, &buf->alpha, &buf->beta);

    for (int b = 0; b < B; ++b)
    {
        for (int c = 0; c < C; ++c)
        {
            double acc = 0.0;
            for (int g = 0; g < G; ++g)
            {
                acc += MATRIX_AT_PTR(W, b, g) * MATRIX_AT(buf->prob[b], g, c);
            }
            MATRIX_AT(buf->S_bc, b, c) = acc;
        }
    }

    // --- Compute q_bgc
    for (int b = 0; b < B; ++b)
    {
        for (int g = 0; g < G; ++g)
        {
            double denom = 0.0;
            // --- first compute numerators into local array
            for (int c = 0; c < C; ++c)
            {
                double n = MATRIX_AT(buf->prob[b], g, c) * MATRIX_AT_PTR(X, b, c);
                double d = MATRIX_AT(buf->S_bc, b, c) - MATRIX_AT(buf->prob[b], g, c);
                double v = n / d;
                MATRIX_AT(buf->q_bgc[b], g, c) = v;
                denom += v;
            }
            // --- normalize
            for (int c = 0; c < C; ++c)
            {
                MATRIX_AT(buf->q_bgc[b], g, c) /= denom;
            }
        }
    }
}

double objective_function(Matrix *W, Matrix *V, EMBuffers *buf, const Matrix *alpha_eval, const Matrix *beta_eval)
{

    int B = buf->B, G = buf->G, C = buf->C;

    double loss = 0.0;

    // --- Get probabilities
    getProbability(buf, V, alpha_eval, beta_eval);

    // --- Get the dot product
    for (int b = 0; b < B; b++)
    { // --- For each ballot box
        for (int g = 0; g < G; g++)
        { // --- For each group
            // Must be a continuos pointer, hence, the macro can't be used
            double dot = 0.0;
            for (int c = 0; c < C; c++)
            {
                double q = MATRIX_AT(buf->q_bgc[b], g, c);
                double p = MATRIX_AT(buf->prob[b], g, c);
                dot += q * log(fmax(p, 1e-12));
            }
            loss -= MATRIX_AT_PTR(W, b, g) * dot; // Check if it is to sum or to substract
        }
    }
    return loss;
}

void compute_gradients(const Matrix *W, Matrix *V, EMBuffers *buf, const Matrix *alpha_eval, const Matrix *beta_eval)
{

    int B = buf->B, G = buf->G, Cminus1 = buf->C - 1, A = buf->A;

    // --- Get probabilities
    getProbability(buf, V, alpha_eval, beta_eval);

    for (int g = 0; g < G; g++)
    {
        for (int c = 0; c < Cminus1; c++)
        {
            double sum1 = 0;
            double sum2 = 0;
            for (int b = 0; b < B; b++)
            {
                double w = MATRIX_AT_PTR(W, b, g);
                sum1 += w * MATRIX_AT(buf->q_bgc[b], g, c);
                sum2 += w * MATRIX_AT(buf->prob[b], g, c);
            }
            MATRIX_AT(buf->grad_beta, g, c) = sum1 - sum2;
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
                    double q = MATRIX_AT(buf->q_bgc[b], g, c);
                    double v = MATRIX_AT_PTR(V, b, a);
                    double p = MATRIX_AT(buf->prob[b], g, c);
                    sum1 += w * q * v;
                    sum2 += w * p * v;
                }
            }
            MATRIX_AT(buf->grad_alpha, c, a) = sum1 - sum2;
        }
    }
}

void compute_hessian(const Matrix *W, // B×G
                     Matrix *V,       // B×A
                     EMBuffers *buf)
{

    int B = buf->B, G = buf->G, Cm = buf->C - 1, A = buf->A, D = buf->D;
    int d_alpha = Cm * A;
    int d_beta = G * Cm;
    /* 1) zero out the entire Hessian */
    memset(buf->H.data, 0, D * D * sizeof(double));

    /* 2) compute p_bgc once */
    getProbability(buf, V, &buf->alpha, &buf->beta);

    size_t n_iters = (size_t)B * G * Cm * Cm;
#ifdef _OPENMP
#pragma omp parallel for collapse(4) if (n_iters > 500) schedule(static)
#endif
    for (int b = 0; b < B; b++)
    {
        for (int g = 0; g < G; g++)
        {
            double w = MATRIX_AT_PTR(W, b, g);

            for (int ci = 0; ci < Cm; ci++)
            {
                double p_ci = MATRIX_AT(buf->prob[b], g, ci);

                for (int cj = 0; cj < Cm; cj++)
                {
                    double p_cj = MATRIX_AT(buf->prob[b], g, cj);
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
                            MATRIX_AT(buf->H, row_alpha, col_alpha) += v_ai * v_aj * factor;
                        }
                    }

                    // alpha–beta and beta–alpha blocks
                    for (int ai = 0; ai < A; ai++)
                    {
                        double v_ai = MATRIX_AT_PTR(V, b, ai);
                        int row_alpha = ai * Cm + ci;
                        int col_beta = d_alpha + g * Cm + cj;
                        double upd = v_ai * factor;
                        MATRIX_AT(buf->H, row_alpha, col_beta) += upd;
                        MATRIX_AT(buf->H, col_beta, row_alpha) += upd;
                    }

                    // beta–beta block
                    {
                        int row_beta = d_alpha + g * Cm + ci;
                        int col_beta = d_alpha + g * Cm + cj;
                        MATRIX_AT(buf->H, row_beta, col_beta) += factor;
                    }
                }
            }
        }
    }
}

// ----- HELPER FUNCTION ----- //
// Packs grad_alpha (C–1 x A, column‐major) followed by grad_beta (G x C–1, column‐major)
// both in column‐major order, into the flat vector g[0..D-1].
static void pack_gradients(EMBuffers *buf)
{
    int Cminus1 = buf->grad_alpha.rows;
    int A = buf->grad_alpha.cols;
    int G = buf->grad_beta.rows;
    int idx = 0;

    // \alpha block
    for (int a = 0; a < A; a++)
    { // --- For each attribute
        for (int c = 0; c < Cminus1; c++)
        { // --- For each candidate
            buf->gvec[idx++] = MATRIX_AT(buf->grad_alpha, c, a);
        }
    }

    // \beta block
    for (int c = 0; c < Cminus1; c++)
    { // --- For each candidate
        for (int gi = 0; gi < G; gi++)
        { // --- For each group
            buf->gvec[idx++] = MATRIX_AT(buf->grad_beta, gi, c);
        }
    }
}

// Unpacks a flat D‐vector v[] back into two matrices, in the same order.
static void unpack_step(EMBuffers *buf)
{
    int Cminus1 = buf->grad_alpha.rows;
    int A = buf->grad_alpha.cols;
    int G = buf->grad_beta.rows;
    int idx = 0;

    // \alpha block
    for (int a = 0; a < A; ++a)
    { // --- For each attribute
        for (int c = 0; c < Cminus1; ++c)
        { // --- For each candidate
            MATRIX_AT(buf->grad_alpha, c, a) = buf->vvec[idx++];
        }
    }

    // \beta block
    for (int c = 0; c < Cminus1; ++c)
    { // --- For each candidate
        for (int gi = 0; gi < G; ++gi)
        { // --- For each group
            MATRIX_AT(buf->grad_beta, gi, c) = buf->vvec[idx++];
        }
    }
}

int Newton_damped(Matrix *W, // B×G weights
                  Matrix *V, // B×A covariates
                  EMBuffers *buf, double tol, int max_iter, double alpha_bs, double beta_bs)
{
    int B = V->rows;
    int A = V->cols;
    int Cminus1 = buf->alpha.rows;
    int C = Cminus1 + 1;
    int G = buf->beta.rows;
    int D = Cminus1 * A + G * Cminus1;

    // ---- Clone initial parameters
    Matrix *alpha = copMatrixPtr(&buf->alpha);
    Matrix *beta = copMatrixPtr(&buf->beta);

    // ---- Allocate temporaries
    // double *gvec = (double *)Calloc(D, double);
    // double *vvec = (double *)Calloc(D, double);
    // Matrix dalpha = createMatrix(Cminus1, A);
    // Matrix dbeta = createMatrix(G, Cminus1);
    // Matrix H = createMatrix(D, D);

    int iter;
    for (iter = 0; iter < max_iter; iter++)
    {
        // ---- Evaluate loss, gradient, Hessian at the current points
        // Loss
        double f0 = objective_function(W, V, buf, alpha, beta);

        // Gradients
        compute_gradients(W, V, buf, alpha, beta);
        pack_gradients(buf);

        // Hessian (prezero H)
        size_t D2 = (size_t)D * (size_t)D;
        memset(buf->H.data, 0, D2 * sizeof(double));
        compute_hessian(W, V, buf);

        // Damping of the hessian
        double min_diag = DBL_MAX;
        for (int i = 0; i < D; i++)
        {
            min_diag = fmin(min_diag, MATRIX_AT(buf->H, i, i));
        }
        // Choose a tiny epsilon
        double eps = 1e-6 * (fabs(min_diag) + 1.0);
        // If the smallest diagonal is below zero, shift by (–min_diag + \epsilon),
        // otherwise just add \epsilon to make it strictly positive
        double shift = (min_diag < 0.0 ? -min_diag + eps : eps);
        for (int i = 0; i < D; i++)
        {
            MATRIX_AT(buf->H, i, i) += shift;
        }
        // ---...--- //

        // Solve H v = -g, for approximating it with Taylor expansion
        Matrix Hcopy = copMatrix(&buf->H);
        solve_linear_system(D, Hcopy.data, buf->gvec, buf->vvec);
        for (int i = 0; i < D; i++)
        {
            buf->vvec[i] = -buf->vvec[i];
        }
        unpack_step(buf);

        // Convergence check
        double g_inf = 0;
        for (int i = 0; i < D; i++)
        {
            g_inf = fmax(g_inf, fabs(buf->gvec[i]));
        }
        if (g_inf < tol)
            break;

        // Armijo backtracking line search
        double t = 1.0; // We start with t = 1
        // compute g*v, the direction
        double gv = 0;
        for (int i = 0; i < D; i++)
            gv += buf->gvec[i] * buf->vvec[i];

        while (1)
        {
            // Trial parameters
            Matrix *alpha_t = copMatrixPtr(alpha);
            Matrix *beta_t = copMatrixPtr(beta);
            // alpha_t = alpha_t + t * dalpha
            for (int i = 0; i < Cminus1; i++)
                for (int j = 0; j < A; j++)
                    MATRIX_AT_PTR(alpha_t, i, j) += t * MATRIX_AT(buf->grad_alpha, i, j);
            // beta_t = beta_t + t * dbeta
            for (int i = 0; i < G; i++)
                for (int j = 0; j < Cminus1; j++)
                    MATRIX_AT_PTR(beta_t, i, j) += t * MATRIX_AT(buf->grad_beta, i, j);

            double f_trial = objective_function(W, V, buf, alpha_t, beta_t);
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
                MATRIX_AT_PTR(alpha, i, j) += t * MATRIX_AT(buf->grad_alpha, i, j);
        for (int i = 0; i < G; i++)
            for (int j = 0; j < Cminus1; j++)
                MATRIX_AT_PTR(beta, i, j) += t * MATRIX_AT(buf->grad_beta, i, j);

    } // --- Newton iteration finishes

    // Copy results out
    size_t alpha_elems = alpha->rows * alpha->cols;
    size_t alpha_bytes = alpha_elems * sizeof(double);
    memcpy(buf->alpha.data, alpha->data, alpha_bytes);

    size_t beta_elems = beta->rows * beta->cols;
    size_t beta_bytes = beta_elems * sizeof(double);
    memcpy(buf->beta.data, beta->data, beta_bytes);

    // Cleanup
    freeMatrix(alpha);
    freeMatrix(beta);

    return iter + 1;
}

double compute_ll_multinomial(const Matrix *X, // BxC
                              const Matrix *W, // BxG
                              Matrix *V,       // BxA
                              EMBuffers *buf)
{
    int B = X->rows;
    int C = X->cols;
    int A = V->cols;
    int G = W->cols;

    double total_ll = 0.0;

    getProbability(buf, V, &buf->alpha, &buf->beta);

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
                marg += MATRIX_AT_PTR(W, b, g) * MATRIX_AT(buf->prob[b], g, c);
            }
            // normalize
            double pbc = marg / denom;
            // clamp before log
            total_ll += MATRIX_AT_PTR(X, b, c) * log(fmax(pbc, 1e-12));
        }
    }

    return total_ll;
}

void M_step(Matrix *X, Matrix *W, Matrix *V, EMBuffers *buf, const double tol, const int maxnewton, const bool verbose)
{
    int newton_iterations = Newton_damped(W, V, buf, tol, maxnewton, 0.01, 0.5);

    if (verbose)
    {
        Rprintf("The newton algorithm was made in %d iterations\n", newton_iterations);
    }
}

Matrix *EM_Algorithm(Matrix *X, Matrix *W, Matrix *V, Matrix *beta, Matrix *alpha, const int maxiter,
                     const double maxtime, const double ll_threshold, const int maxnewton, const bool verbose,
                     double *out_elapsed, int *total_iterations)
{
    int B = V->rows;
    int A = V->cols;
    int Cm = alpha->rows;
    int G = beta->rows;

    // Initialize buffers
    EMBuffers buf;
    init_EMBuffers(&buf, B, G, Cm, A);
    buf.alpha = copMatrix(alpha);
    buf.beta = copMatrix(beta);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0); // Start timer
    double current_ll = -DBL_MAX;
    double new_ll = -DBL_MAX;
    for (int iter = 0; iter < maxiter; iter++)
    {
        *total_iterations += 1;
        E_step(X, W, V, &buf);
        M_step(X, W, V, &buf, 0.01, maxnewton, verbose);
        new_ll = compute_ll_multinomial(X, W, V, &buf);

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

    size_t na = buf.alpha.rows * buf.alpha.cols;
    memcpy(alpha->data, buf.alpha.data, na * sizeof(double));
    size_t nb = buf.beta.rows * buf.beta.cols;
    memcpy(beta->data, buf.beta.data, nb * sizeof(double));

    // Compute elapsed seconds
    double sec = (double)(t1.tv_sec - t0.tv_sec);
    double nsec = (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9;
    *out_elapsed = sec + nsec;

    Matrix *finalProb = buf.prob;
    // detach buf.prob so we don't free it:
    buf.prob = NULL;
    free_EMBuffers(&buf);

    // Matrix *finalProbability = getProbability(V, beta, alpha);
    return finalProb; // Return the final probabilities
}
