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
// If
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
    Matrix *probabilities = getProbability(V, beta, alpha);

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

    // --- Get log probabilities
    Matrix *probabilities = getProbability(V, beta, alpha);
    for (int b = 0; b < B; b++)
    { // --- For each ballot box
        for (int g = 0; g < G; g++)
        { // --- For each group
            for (int c = 0; c < C; c++)
            { // --- For each candidate
                MATRIX_AT(probabilities[b], g, c) = log(fmax(MATRIX_AT(probabilities[b], g, c), 1e-12));
            }
        }
    }

    // --- Get the dot product
    for (int b = 0; b < B; b++)
    { // --- For each ballot box
        for (int g = 0; g < G; g++)
        { // --- For each group
            // Must be a continuos pointer, hence, the macro can't be used
            double *q_ptr = &q_bgc[b].data[g * C];
            double *logp_ptr = &probabilities[b].data[g * C];
            double dot = matrixDotProduct(q_ptr, logp_ptr, C);
            loss -= MATRIX_AT_PTR(W, b, g) * dot;
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

    for (int b = 0; b < B; b++)
    { // --- For each ballot box
        for (int g = 0; g < G; g++)
        { // --- For each group
            double w = MATRIX_AT_PTR(W, b, g);
            for (int c = 0; c < Cminus1; c++)
            { // --- For each candidate
                double diff = MATRIX_AT(q_bgc[b], g, c) - MATRIX_AT(p_bgc[b], g, c);
                // --- Beta gradient
                MATRIX_AT_PTR(grad_beta_out, g, c) += w * diff;
                // --- Alpha gradient over a
                for (int a = 0; a < A; a++)
                {
                    MATRIX_AT_PTR(grad_alpha_out, c, a) += w * diff * MATRIX_AT_PTR(V, b, a);
                }
            }
            // baseline column c=Cminus1: grad_beta = 0 by constraint, skip
        }
        freeMatrix(&p_bgc[b]);
    }
    Free(p_bgc);
}

// Cmpute the Hessian matrix for the optimization problem
void compute_hessian(const Matrix *W,     // B×G
                     Matrix *V,           // B×A
                     Matrix *alpha,       // (C–1)×A
                     Matrix *beta,        // G×C
                     const Matrix *q_bgc, // array of B matrices G×C
                     Matrix *H_out        // (D×D), prezeroed
)
{
    int B = V->rows;
    int A = V->cols;
    int Cminus1 = alpha->rows;
    int C = Cminus1 + 1;
    int G = beta->rows;

    int d_alpha = Cminus1 * A; // This is the alpha block of the hessian

    // --- Get the probabilities
    Matrix *p_bgc = getProbability(V, beta, alpha);

    // -- Accumulate second derivatives over b,g
    for (int b = 0; b < B; b++)
    { // For each ballot box
        for (int g = 0; g < G; g++)
        { // For each group
            double w = MATRIX_AT_PTR(W, b, g);
            for (int c = 0; c < Cminus1; c++)
            { // For each candidate
                // Get its probability and conditional probability
                double p_c = MATRIX_AT(p_bgc[b], g, c);
                double q_c = MATRIX_AT(q_bgc[b], g, c);

                // ---- We fill the \partial\beta\beta block ---- //
                // $ -w \cdot [ \delta_{c,k} \cdot (q_c - p_c) - p_c \cdot p_k ] $
                for (int k = 0; k < Cminus1; k++)
                { // For each candidate
                    // Get the kth candidate probability
                    double p_k = MATRIX_AT(p_bgc[b], g, k);
                    // If it's the same candidate as the outer loop, compute the second derivative
                    double d2bb = -w * ((c == k ? (q_c - p_c) : 0.0) - p_c * p_k);
                    // This is just for indexing purposes
                    int row_b = d_alpha + g * Cminus1 + c;
                    int col_b = d_alpha + g * Cminus1 + k;
                    // Accumulate the second derivate term, it's a summatory
                    MATRIX_AT_PTR(H_out, row_b, col_b) += d2bb;
                }

                // ---- Beta alpha and alpha beta blocks ---- //
                // $ -w \cdot v_{b,a} \cdot (q_c - p_c) $
                for (int a = 0; a < A; a++)
                { // For each attribute
                    double v_ba = MATRIX_AT_PTR(V, b, a);
                    double d2ba = -w * v_ba * (q_c - p_c);
                    // This is just for indexing purposes
                    int row_b = d_alpha + g * Cminus1 + c;
                    int col_a = c * A + a;
                    // The entry is symmetric, so we fill both positions
                    MATRIX_AT_PTR(H_out, row_b, col_a) += d2ba;
                    MATRIX_AT_PTR(H_out, col_a, row_b) += d2ba;
                }

                // ---- Alpha alpha block ---- //
                // $ w[p_c \cdot p_k \cdot v_{b,a} - \delta_{c,k}(q_c - p_c) \cdot v_{b,a} \cdot v_{b,o}] $
                for (int k = 0; k < Cminus1; k++)
                { // For each candidate
                    // Get the probability of the kth candidate
                    double p_k = MATRIX_AT(p_bgc[b], g, k);
                    for (int a = 0; a < A; a++)
                    { // For each attribute
                        // Get the ponderation of its attribute
                        double v_ba = MATRIX_AT_PTR(V, b, a);
                        for (int o = 0; o < A; o++)
                        { // For each attribute
                            double v_bo = MATRIX_AT_PTR(V, b, o);
                            double term1 = p_c * p_k * v_ba * v_bo;
                            double term2 = (c == k ? (q_c - p_c) * v_ba * v_bo : 0.0);
                            double d2aa = w * (term1 - term2);
                            // For indexing purposes
                            int row_a = c * A + a;
                            int col_o = k * A + o;
                            MATRIX_AT_PTR(H_out, row_a, col_o) += d2aa;
                        }
                    }
                }
            }
        }
        freeMatrix(&p_bgc[b]);
    }
    Free(p_bgc);
}

// ----- HELPER FUNCTION ----- //
// Packs grad_alpha (C-1×A) and grad_beta (G×(C-1)) into a vector g (length D)
static void pack_gradients(const Matrix *grad_alpha, const Matrix *grad_beta, double *g)
{
    int Cminus1 = grad_alpha->rows;
    int A = grad_alpha->cols;
    int G = grad_beta->rows;

    // α block: row-major
    int idx = 0;
    for (int c = 0; c < Cminus1; c++)
    {
        for (int a = 0; a < A; a++)
        {
            g[idx++] = MATRIX_AT_PTR(grad_alpha, c, a);
        }
    }
    // β block:
    for (int g_i = 0; g_i < G; g_i++)
    {
        for (int c = 0; c < Cminus1; c++)
        {
            g[idx++] = MATRIX_AT_PTR(grad_beta, g_i, c);
        }
    }
}

// ----- HELPER FUNCTION ----- //
// Unpacks a step vector v (length D) into dalpha, dbeta
static void unpack_step(const double *v, Matrix *dalpha, Matrix *dbeta)
{
    int Cminus1 = dalpha->rows;
    int A = dalpha->cols;
    int G = dbeta->rows;
    int idx = 0;
    for (int c = 0; c < Cminus1; c++)
    {
        for (int a = 0; a < A; a++)
        {
            MATRIX_AT_PTR(dalpha, c, a) = v[idx++];
        }
    }
    for (int g_i = 0; g_i < G; g_i++)
    {
        for (int c = 0; c < Cminus1; c++)
        {
            MATRIX_AT_PTR(dbeta, g_i, c) = v[idx++];
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
        compute_hessian(W, V, alpha, beta, *q_bgc, &H);

        freeMatrix(&grad_alpha);
        freeMatrix(&grad_beta);

        // ---- Damping of Hessian, in case it gets undefined on its diagonal (has to be semidefinite positive)
        double diag_max = 0;
        for (int i = 0; i < D; i++)
        {
            diag_max = fmax(diag_max, MATRIX_AT(H, i, i));
        }
        double eta = pow(tol, 0.5 /*τ*/);
        for (int i = 0; i < D; i++)
        {
            for (int j = 0; j < D; j++)
            {
                double Hij = MATRIX_AT(H, i, j);
                double Iij = (i == j ? 1.0 : 0.0);
                MATRIX_AT(H, i, j) = (1 - eta) * Hij + eta * diag_max * Iij;
            }
        }
        // ---...--- //

        // Solve H v = -g, for approximating it with Taylor expansion
        solve_linear_system(D, H.data, gvec, vvec);

        // Unpack step into delta alpha, delta beta
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
        // compute g·v, the direction
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
                break;
            t *= beta_bs;
            if (t < 1e-10)
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

    // β: (G × Cminus1)
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
    for (int b = 0; b < B; b++)
    {
        freeMatrix(&(*q_bgc)[b]);
    }

    return iter + 1;
}

double compute_ll_multinomial(const Matrix *X, // B×C
                              const Matrix *W, // B×G
                              Matrix *V,       // B×A
                              Matrix *alpha,   // (C-1)×A  (plus a baseline row)
                              Matrix *beta     // G×C
)
{
    int B = X->rows;
    int C = X->cols;
    int A = V->cols;
    int G = W->cols;

    double total_ll = 0.0;

    // --- Get probabVy
    Matrix *p_bgc = getProbability(V, beta, alpha); // B×G×C

    // --- Get the log-likelihood in one go
    for (int b = 0; b < B; b++)
    {
        for (int c = 0; c < C; c++)
        {
            double x_bc = MATRIX_AT_PTR(X, b, c);
            total_ll -= lgamma(x_bc + 1);
            ballot_votes[b] += x_bc;
            total_ll += x_bc * log(fmax(MATRIX_AT_PTR(p_bgc, b, c), 1e-12));
        }
        total_ll += lgamma(ballot_votes[b] + 1);
        freeMatrix(&p_bgc[b]);
    }
    Free(p_bgc);

    return total_ll;
}

void M_step(Matrix *X, Matrix *W, Matrix *V, Matrix *q_bgc, Matrix *alpha, Matrix *beta, const double tol,
            const int maxnewton, const bool verbose)
{
    int newton_iterations = Newton_damped(W, V, &q_bgc, alpha, beta, tol, maxnewton, 0.01, 0.5, alpha, beta);

    if (verbose)
    {
        Rprintf("The newton algorithm was made in %d seconds\n", newton_iterations);
    }
}

Matrix *EM_Algorithm(Matrix *X, Matrix *W, Matrix *V, Matrix *beta, Matrix *alpha, const int maxiter,
                     const double maxtime, const double param_threshold, const double ll_threshold, const int maxnewton,
                     const bool verbose)
{
    double current_ll = -DBL_MAX;
    double new_ll = -DBL_MAX;
    for (int iter = 0; iter < maxiter; iter++)
    {
        Matrix *q_bgc = E_step(X, W, V, beta, alpha);
        M_step(X, W, V, q_bgc, alpha, beta, 0.001, maxnewton, verbose);
        Free(q_bgc);
        new_ll = compute_ll_multinomial(X, W, V, alpha, beta);

        if (verbose)
        {
            Rprintf("Iteration %d: log-likelihood = %.3f\n", iter + 1, new_ll);
        }

        // Check for convergence
        if (fabs(new_ll - current_ll) < ll_threshold)
        {
            if (verbose)
            {
                Rprintf("Converged after %d iterations.\n", iter + 1);
            }
            break;
        }
        current_ll = new_ll;
    }

    Matrix *finalProbability = getProbability(V, beta, alpha);
    return finalProbability; // Return the final probabilities
}
