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
    // double *W_row = (double *)Calloc(G, double);
    // double *S_row = (double *)Calloc(C, double);
    double *W_buf = Calloc(G, double);
    for (int b = 0; b < B; b++)
    { // --- For each ballot box
      // Get the bth row of W
        for (int g = 0; g < G; g++)
        {
            W_buf[g] = MATRIX_AT_PTR(W, b, g);
        }
        // memcpy(W_row, &W->data[b * G], G * sizeof(double));
        // double *W_row = getRow(W, b);
        double *S_ptr = getRow(&S_bc, b); // length C

        // Multiply
        // vectorMatrixMultiplication_inplace(W_buf, &probabilities[b], S_ptr); // --- Length C
        for (int c = 0; c < C; c++)
        {
            double acc = 0;
            for (int g = 0; g < G; g++)
            {
                acc += W_buf[g] * MATRIX_AT(probabilities[b], g, c);
            }
            MATRIX_AT(S_bc, b, c) = acc;
        }

        // Copy the output to S_bc matrix
        // memcpy(&S_bc.data[b * C], S_row, C * sizeof(double));
        // double *S_row = getRow(&S_bc, b);
    }
    Free(W_buf);
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
                dot += q * p;
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

    /*
    for (int b = 0; b < B; b++)
    { // --- For each ballot box
        for (int g = 0; g < G; g++)
        { // --- For each group
            double w = MATRIX_AT_PTR(W, b, g);
            for (int c = 0; c < Cminus1; c++)
            { // --- For each candidate
                double diff = MATRIX_AT(q_bgc[b], g, c) - MATRIX_AT(p_bgc[b], g, c);
                // double diff = MATRIX_AT(p_bgc[b], g, c) - MATRIX_AT(q_bgc[b], g, c);

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
    */
    // after filling grad_alpha_out and grad_beta_out:

    for (int b = 0; b < B; b++)
    {
        freeMatrix(&p_bgc[b]);
    }
    Free(p_bgc);
    double sum_ga = 0, sum_gb = 0;
    for (int c = 0; c < Cminus1; c++)
    {
        for (int a = 0; a < A; a++)
            sum_ga += MATRIX_AT_PTR(grad_alpha_out, c, a);
    }
    for (int g = 0; g < G; g++)
    {
        for (int c = 0; c < Cminus1; c++)
            sum_gb += MATRIX_AT_PTR(grad_beta_out, g, c);
    }
    Rprintf("DEBUG grads: sum(grad_alpha)=%f  sum(grad_beta)=%f\n", sum_ga, sum_gb);
}

// Cmpute the Hessian matrix for the optimization problem
/*
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
                    double d2bb = -w * ((c == k ? (p_c) : 0.0) - p_c * p_k);
                    // This is just for indexing purposes
                    // int row_b = d_alpha + c * G + g;
                    // int col_b = d_alpha + k * G + g;
                    int row_b = d_alpha + c * G + g;
                    int col_b = d_alpha + k * G + g;
                    MATRIX_AT_PTR(H_out, row_b, col_b) += d2bb;
                }

                // ---- Beta alpha and alpha beta blocks ---- //
                // $ -w \cdot v_{b,a} \cdot (q_c - p_c) $
                for (int a = 0; a < A; a++)
                { // For each attribute
                    double v_ba = MATRIX_AT_PTR(V, b, a);
                    double d2ba = -w * v_ba * (q_c - p_c);
                    // This is just for indexing purposes
                    // int row_b = d_alpha + c * G + g;
                    // int col_a = a * Cminus1 + c;
                    int row_b = d_alpha + c * G + g; // column-major index of β₍g,c₎
                    int col_a = a * Cminus1 + c;
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
                            int row_a = a * Cminus1 + c;
                            int col_o = o * Cminus1 + k;
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
*/
/* --------------  H = ∂²Q  ---------------------------------------------- */
void compute_hessian(const Matrix *W,     /* B×G */
                     const Matrix *V,     /* B×A   (k_ba)               */
                     const Matrix *alpha, /* (C-1)×A */
                     const Matrix *beta,  /* G×C   (we only use 1..C-1) */
                     Matrix *H_out)       /* (dα+dβ) × (dα+dβ) – PRE-zeroed */
{
    /* ---------- dimensions & derived sizes -------------------------------- */
    const int B = V->rows;
    const int A = V->cols;
    const int Cm = alpha->rows; /* Cminus1 = C-1                      */
    const int C = Cm + 1;
    const int G = beta->rows;

    const int d_alpha = Cm * A; /* flattened α block size             */
    const int d_beta = G * Cm;  /* flattened β block size             */

    /* ---------- 1. pre-allocate the 4 blocks (they start at zero) ---------- */
    Matrix H_aa = createMatrix(d_alpha, d_alpha);
    Matrix H_ab = createMatrix(d_alpha, d_beta);
    Matrix H_ba = createMatrix(d_beta, d_alpha);
    Matrix H_bb = createMatrix(d_beta, d_beta);

    /* ---------- 2. p_bgc[b] ≡ probabilities for mesa b --------------------- */
    Matrix *p_bgc = getProbability((Matrix *)V, (Matrix *)beta, (Matrix *)alpha);
    /* ----------------------------------------------------------------------- */

    /* =====================   α – α   ====================================== */
    /*  ∂²Q/∂α_{c′,a′}∂α_{c,a}  */
    for (int i = 0; i < d_alpha; ++i)
    {
        int a_i = i / Cm;
        int c_i = i % Cm;
        for (int j = 0; j < d_alpha; ++j)
        {
            int a_j = j / Cm;
            int c_j = j % Cm;

            double sum = 0.0;
            for (int b = 0; b < B; ++b)
            {
                double k_ba = MATRIX_AT_PTR(V, b, a_i);
                double k_ba_ = MATRIX_AT_PTR(V, b, a_j);

                for (int g = 0; g < G; ++g)
                {
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

    /* =====================   β – α   ====================================== */
    /*  ∂²Q/∂β_{g,c′}∂α_{c,a}  */
    for (int i = 0; i < d_alpha; ++i)
    {
        int a_i = i / Cm;
        int c_i = i % Cm;

        for (int j = 0; j < d_beta; ++j)
        {
            int c_j = j / G;
            int g_j = j % G;

            double sum = 0.0;
            for (int b = 0; b < B; ++b)
            {
                double w = MATRIX_AT_PTR(W, b, g_j);
                double k_ba = MATRIX_AT_PTR(V, b, a_i);
                double p_ci = MATRIX_AT(p_bgc[b], g_j, c_i);
                double p_cj = MATRIX_AT(p_bgc[b], g_j, c_j);

                double t1 = (c_i == c_j ? -w * k_ba * p_ci : 0.0);
                double t2 = w * k_ba * p_ci * p_cj;
                sum += t1 + t2;
            }
            MATRIX_AT(H_ab, i, j) = sum; /* αβ  */
            MATRIX_AT(H_ba, j, i) = sum; /* βα  (symmetry) */
        }
    }

    /* =====================   β – β   ====================================== */
    /*  ∂²Q/∂β_{g′,c′}∂β_{g,c}  */
    for (int i = 0; i < d_beta; ++i)
    {
        int c_i = i / G; /* candidate runs slowest            */
        int g_i = i % G; /* group runs  fastest               */

        for (int j = 0; j < d_beta; ++j)
        {
            int c_j = j / G;
            int g_j = j % G;

            if (g_i != g_j)
            { /* derivative is zero if g≠g′ */
                MATRIX_AT(H_bb, i, j) = 0.0;
                continue;
            }

            double sum = 0.0;
            for (int b = 0; b < B; ++b)
            {
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

    /* =====================   assemble into H_out   ======================== */
    /* upper-left αα */
    //  for (int i = 0; i < d_alpha; ++i)
    //     for (int j = 0; j < d_alpha; ++j)
    //        MATRIX_AT_PTR(H_out, i, j) = -MATRIX_AT(H_aa, i, j);

    /* upper-right αβ */
    // for (int i = 0; i < d_alpha; ++i)
    //    for (int j = 0; j < d_beta; ++j)
    //       MATRIX_AT_PTR(H_out, i, d_alpha + j) = -MATRIX_AT(H_ab, i, j);

    /* lower-left βα  */
    // for (int i = 0; i < d_beta; ++i)
    //    for (int j = 0; j < d_alpha; ++j)
    //       MATRIX_AT_PTR(H_out, d_alpha + i, j) = -MATRIX_AT(H_ba, i, j);

    /* lower-right ββ */
    // for (int i = 0; i < d_beta; ++i)
    //    for (int j = 0; j < d_beta; ++j)
    //       MATRIX_AT_PTR(H_out, d_alpha + i, d_alpha + j) = -MATRIX_AT(H_bb, i, j);
    const int D = d_alpha + d_beta;
    for (int i = 0; i < D; ++i)
    {
        for (int j = 0; j < D; ++j)
        {
            double val;
            if (i < d_alpha && j < d_alpha)
            {
                // α–α
                val = -MATRIX_AT(H_aa, i, j);
            }
            else if (i < d_alpha && j >= d_alpha)
            {
                // α–β
                val = -MATRIX_AT(H_ab, i, j - d_alpha);
            }
            else if (i >= d_alpha && j < d_alpha)
            {
                // β–α
                val = -MATRIX_AT(H_ba, i - d_alpha, j);
            }
            else
            {
                // β–β
                val = -MATRIX_AT(H_bb, i - d_alpha, j - d_alpha);
            }
            // asigna directamente en row-major:
            H_out->data[i * D + j] = val;
        }
    }
    /* =====================   clean-up   =================================== */
    for (int b = 0; b < B; ++b)
        freeMatrix(&p_bgc[b]);
    Free(p_bgc);

    freeMatrix(&H_aa);
    freeMatrix(&H_ab);
    freeMatrix(&H_ba);
    freeMatrix(&H_bb);
}
// ----- HELPER FUNCTION ----- //
// Packs grad_alpha (C–1 × A, column‐major) followed by grad_beta (G × C–1, column‐major)
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

    // 1) α-block: loop over columns a=0..A-1, then rows c=0..Cminus1-1
    for (int a = 0; a < A; ++a)
    {
        for (int c = 0; c < Cminus1; ++c)
        {
            g[idx++] = MATRIX_AT_PTR(grad_alpha, c, a);
        }
    }

    // 2) β-block: loop over “candidate” c=0..Cminus1-1, then groups g=0..G-1
    for (int c = 0; c < Cminus1; ++c)
    {
        for (int gi = 0; gi < G; ++gi)
        {
            g[idx++] = MATRIX_AT_PTR(grad_beta, gi, c);
        }
    }
    // at the end idx == Cminus1*A + G*Cminus1
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

    // 1) α-block: columns then rows
    for (int a = 0; a < A; ++a)
    {
        for (int c = 0; c < Cminus1; ++c)
        {
            MATRIX_AT_PTR(grad_alpha, c, a) = v[idx++];
        }
    }

    // 2) β-block: candidate‐index then group
    for (int c = 0; c < Cminus1; ++c)
    {
        for (int gi = 0; gi < G; ++gi)
        {
            MATRIX_AT_PTR(grad_beta, gi, c) = v[idx++];
        }
    }
    // idx ends up = total dimension
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
        // --- right after pack_gradients(&grad_alpha,&grad_beta,gvec) ---
        Rprintf("DEBUG pack_gradients: Cminus1=%d, A=%d, G=%d, D(total)=%d\n", Cminus1, A, G, D);

        /* print gvec with explicit mapping hints */
        for (int idx = 0; idx < D; ++idx)
        {
            if (idx < Cminus1 * A) /* α-block ------------------- */
            {
                int a = idx / Cminus1;
                int c = idx % Cminus1;
                Rprintf(" gvec[%2d] = % .6e   // alpha[%d,%d]   should be "
                        "grad_alpha[%d,%d]\n",
                        idx, gvec[idx], c, a, c, a);
            }
            else /* β-block ------------------- */
            {
                int beta_idx = idx - Cminus1 * A;
                int c = beta_idx / G;
                int g_i = beta_idx % G;
                Rprintf(" gvec[%2d] = % .6e   // beta[%d,%d]    should be "
                        "grad_beta[%d,%d]\n",
                        idx, gvec[idx], g_i, c, g_i, c);
            }
        }
        Rprintf("\nDEBUG gvec[0..%d]  (flat order α first, then β):\n", D - 1);
        for (int i = 0; i < D; ++i)
            Rprintf("%+.6e ", gvec[i]);
        Rprintf("\n\n");

        // Hessian (prezero H)
        size_t D2 = (size_t)D * (size_t)D;
        memset(H.data, 0, D2 * sizeof(double));

        compute_hessian(W, V, alpha, beta, &H);
        // --- right after compute_hessian(...,&H) but before any damping/solving ---
        Rprintf("DEBUG Hessian blocks (size %d×%d):\n", D, D);
        Rprintf(" alpha‐block size = %d×%d   (rows/cols %d..%d)\n", Cminus1 * A, Cminus1 * A, G * Cminus1,
                G * Cminus1 + Cminus1 * A - 1);
        Rprintf(" beta‐block  size = %d×%d   (rows/cols 0..%d)\n", G * Cminus1, G * Cminus1, G * Cminus1 - 1);

        /* ββ block ------------------------------------------------------------- */
        int d_alpha = (Cminus1)*A; /* = (C-1)·A  */
        int d_beta = G * Cminus1;  /* = G·(C-1)  */
        int off = d_beta;

        Rprintf("  ββ  (rows/cols 0..%d) – should vanish when g≠g':\n", off - 1);
        for (int i = 0; i < off; ++i)
        {
            for (int j = 0; j < off; ++j)
            {
                /* identify (g,c) of each flat position for easier eyeballing */
                int gi = i % G, ci = i / G;
                int gj = j % G, cj = j / G;
                Rprintf("% .3e", MATRIX_AT(H, i, j));
                if (gj != gi)
                    Rprintf("*"); /* mark entries that *should* be ≈0 */
                Rprintf(" ");
            }
            Rprintf("\n");
        }
        Rprintf("  ( * marks rows/cols where g≠g' and value should be ~0 )\n\n");

        /* βα / αβ cross block -------------------------------------------------- */
        Rprintf("  βα / αβ  (β rows 0..%d , α cols %d..%d) – symmetrical:\n", off - 1, off, D - 1);
        for (int i = 0; i < off; ++i)
        {
            for (int j = off; j < D; ++j)
                Rprintf("% .3e ", MATRIX_AT(H, i, j));
            Rprintf("\n");
        }
        Rprintf("\n");

        /* αα block ------------------------------------------------------------- */
        Rprintf("  αα  (rows/cols %d..%d) – should be symmetric:\n", off, D - 1);
        for (int ii = 0; ii < d_alpha; ++ii)
        {
            int i = off + ii;
            for (int jj = 0; jj < d_alpha; ++jj)
            {
                int j = off + jj;
                Rprintf("% .3e ", MATRIX_AT(H, i, j));
            }
            Rprintf("\n");
        }
        Rprintf("\n");
        printMatrix(&H);

        freeMatrix(&grad_alpha);
        freeMatrix(&grad_beta);

        // ---- Damping of Hessian, in case it gets undefined on its diagonal (has to be semidefinite positive)
        /*
        double diag_max = 0;
        for (int i = 0; i < D; i++)
        {
            diag_max = fmax(diag_max, MATRIX_AT(H, i, i));
        }
        double eta = pow(tol, 0.5);
        for (int i = 0; i < D; i++)
        {
            for (int j = 0; j < D; j++)
            {
                double Hij = MATRIX_AT(H, i, j);
                double Iij = (i == j ? 1.0 : 0.0);
                MATRIX_AT(H, i, j) = (1 - eta) * Hij + eta * diag_max * Iij;
            }
        }
        double diag_min = MATRIX_AT(H, 0, 0);
        for (int i = 1; i < D; ++i)
            diag_min = fmin(diag_min, MATRIX_AT(H, i, i));

        // 2. Define un umbral deseado
        double eps = tol; // p.ej. tol = 1e-6

        // 3. Si hace falta, añade delta a la diagonal
        if (diag_min < eps)
        {
            double delta = eps - diag_min;
            for (int i = 0; i < D; ++i)
            {
                MATRIX_AT(H, i, i) += delta;
            }
        }
        */
        double min_diag = DBL_MAX;
        for (int i = 0; i < D; i++)
        {
            min_diag = fmin(min_diag, MATRIX_AT(H, i, i));
        }

        // Choose a tiny ε relative to your scale
        double eps = 1e-6 * (fabs(min_diag) + 1.0);

        // If the smallest diagonal is below zero, shift by (–min_diag + ε),
        // otherwise just add ε to make it strictly positive
        double shift = (min_diag < 0.0 ? -min_diag + eps : eps);

        for (int i = 0; i < D; i++)
        {
            MATRIX_AT(H, i, i) += shift;
        }

        // ---...--- //
        // right after compute_hessian(...,&H):
        // Rprintf("DEBUG: H[0..4,0..4] block:\n");
        // printMatrix(&H);
        // Rprintf("   EXPECTED: H symmetric, diag entries ≥ 0 after damping\n\n");

        // Solve H v = -g, for approximating it with Taylor expansion
        Matrix Hcopy = copMatrix(&H);
        solve_linear_system(D, Hcopy.data, gvec, vvec);
        Rprintf("The vvec[0..9] after solve_linear_system:\n");
        for (int i = 0; i < D; i++)
        {
            vvec[i] = -vvec[i];
            Rprintf("%.4f, ", vvec[i]);
        }
        Rprintf("\n");

        // right after solve_linear_system and before negating if any:
        // Rprintf("DEBUG: raw Newton step vvec[0..9]:\n");
        // Unpack step into delta alpha, delta beta
        unpack_step(vvec, &dalpha, &dbeta);
        Rprintf("DEBUG  dalpha[0,0] = %+.6e   // should be vvec[0]\n", MATRIX_AT(dalpha, 0, 0));
        Rprintf("DEBUG  dbeta [0,0] = %+.6e   // should be vvec[%d]\n", MATRIX_AT(dbeta, 0, 0), Cminus1 * A);
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
            {
                Rprintf("The function value is %.3f, the initial value is %.3f, and the gv is %.3f\n", f_trial, f0, gv);
                break;
            }
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

        // Rprintf("  update: alpha[0,0]=%g, beta[0,0]=%g\n", MATRIX_AT_PTR(alpha, 0, 0), MATRIX_AT_PTR(beta, 0, 0));
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
    // for (int b = 0; b < B; b++)
    //{
    //    freeMatrix(&(*q_bgc)[b]);
    //}

    return iter + 1;
}

/*
double compute_ll_multinomial(const Matrix *X, // B×C
                              const Matrix *W, // B×G
                              Matrix *q_bgc,   // B×G×C
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

    Matrix *p_bgc = getProbability(V, beta, alpha);

    Matrix p_bc = createMatrix(B, C);
    for (int b = 0; b < B; ++b)
    {
        double w_sum = 0;
        for (int g = 0; g < G; ++g)
            w_sum += MATRIX_AT_PTR(W, b, g);
        double denom = w_sum + 1e-12;

        for (int c = 0; c < C; ++c)
        {
            double marg = 0;
            for (int g = 0; g < G; ++g)
                marg += MATRIX_AT_PTR(W, b, g) * MATRIX_AT(p_bgc[b], g, c);
            MATRIX_AT(p_bc, b, c) = marg / denom;
        }
        freeMatrix(&p_bgc[b]);
    }
    Free(p_bgc);

    double *x_sum = (double *)Calloc(B, double);

    for (int b = 0; b < B; ++b)
    {
        // 4a) accumulate counts, subtract ∑ ln(x_bc!)
        for (int c = 0; c < C; ++c)
        {
            double x = MATRIX_AT_PTR(X, b, c);
            x_sum[b] += x;
            total_ll -= lgamma(x + 1.0);
        }
        // 4b) add ln(x_b!) term
        total_ll += lgamma(x_sum[b] + 1.0);

        // 4c) add data term ∑ x_bc · ln(p_bc)
        for (int c = 0; c < C; ++c)
        {
            double p = fmax(MATRIX_AT(p_bc, b, c), 1e-12);
            double x = MATRIX_AT_PTR(X, b, c);
            total_ll += x * log(p);
        }
    }

    freeMatrix(&p_bc);
    Free(x_sum);

    return total_ll;
}
*/

double compute_ll_multinomial(const Matrix *X, // B×C
                              const Matrix *W, // B×G
                              Matrix *q_bgc,   // B×G×C
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

    Matrix *p_bgc = getProbability(V, beta, alpha);

    for (int b = 0; b < B; ++b)
    {
        // 1) compute denominator = sum_g w_bg[b,g]
        double wsum = 0;
        for (int g = 0; g < G; ++g)
            wsum += MATRIX_AT_PTR(W, b, g);
        double denom = wsum + 1e-12;

        // 2) (optional) multinomial factorial term
        double xb = 0;
        for (int c = 0; c < C; ++c)
        {
            double x = MATRIX_AT_PTR(X, b, c);
            total_ll -= lgamma(x + 1.0);
            xb += x;
        }
        total_ll += lgamma(xb + 1.0);

        // 3) data term ∑_c x_bc · log(p_bc)
        for (int c = 0; c < C; ++c)
        {
            double marg = 0;
            for (int g = 0; g < G; ++g)
            {
                marg += MATRIX_AT_PTR(W, b, g) * MATRIX_AT(p_bgc[b], g, c);
            }
            // normalize exactly as Python
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
    Rprintf("Before the newton method, the parameters are:\n");
    Rprintf("Alpha:\n");
    printMatrix(alpha);
    int newton_iterations = Newton_damped(W, V, &q_bgc, alpha, beta, tol, maxnewton, 0.01, 0.5, alpha, beta);
    Rprintf("After the newton method, the parameters are:\n");
    Rprintf("Alpha:\n");
    printMatrix(alpha);

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
    printMatrix(X);
    printMatrix(W);
    for (int iter = 0; iter < maxiter; iter++)
    {
        // Rprintf("The updated alpha is %f\n", MATRIX_AT_PTR(alpha, 0, 0));
        *total_iterations += 1;
        Matrix *q_bgc = E_step(X, W, V, beta, alpha);
        M_step(X, W, V, q_bgc, alpha, beta, 0.001, maxnewton, verbose);
        new_ll = compute_ll_multinomial(X, W, q_bgc, V, alpha, beta);
        Free(q_bgc);

        if (verbose)
        {
            Rprintf("Iteration %d: log-likelihood = %.3f\n", iter + 1, new_ll);
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
