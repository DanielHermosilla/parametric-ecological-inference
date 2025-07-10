#include "wrapper.h"
#include <R.h>
#include <R_ext/Random.h>
#include <Rcpp.h>
#include <Rinternals.h>
#include <cstring>

#ifndef Calloc
#define Calloc(n, type) ((type *)R_chk_calloc((size_t)(n), sizeof(type)))
#endif
#ifndef Free
#define Free(p) R_chk_free((void *)(p))
#endif

// Convert an Rcpp::NumericMatrix into our C Matrix struct
static Matrix convertToMatrix(const Rcpp::NumericMatrix &mat)
{
    int rows = mat.nrow(), cols = mat.ncol();
    double *data = (double *)malloc(rows * cols * sizeof(double));
    if (!data)
        Rcpp::stop("Failed to allocate memory in convertToMatrix");
    std::memcpy(data, mat.begin(), rows * cols * sizeof(double));
    return {data, rows, cols};
}

// [[Rcpp::export]]
Rcpp::List EMAlgorithmC(Rcpp::NumericMatrix X, Rcpp::NumericMatrix W, Rcpp::NumericMatrix V, Rcpp::NumericMatrix beta,
                        Rcpp::NumericMatrix alpha, int maxiter, double maxtime, double param_threshold,
                        double ll_threshold, int maxnewton, bool verbose)
{
    // --- 1) Marshal inputs into C structs
    Matrix XR = convertToMatrix(X);
    Matrix WR = convertToMatrix(W);
    Matrix VR = convertToMatrix(V);
    Matrix BetaR = convertToMatrix(beta);
    Matrix AlphaR = convertToMatrix(alpha);

    // --- 2) Call the core EM routine, timing internally
    double elapsed = 0.0;
    int total_iterations = 0;
    Matrix *finalProb = EM_Algorithm(&XR, &WR, &VR, &BetaR, &AlphaR, maxiter, maxtime, param_threshold, ll_threshold,
                                     maxnewton, verbose, &elapsed, &total_iterations);

    // --- 3) Copy back the probability matrix to Rcpp::NumericMatrix
    Rcpp::NumericMatrix Rprob(finalProb->rows, finalProb->cols);
    std::memcpy(Rprob.begin(), finalProb->data, finalProb->rows * finalProb->cols * sizeof(double));

    // --- 4) Clean up C allocations
    freeMatrix(finalProb);
    Free(finalProb);
    free(XR.data);
    free(WR.data);
    free(VR.data);
    free(BetaR.data);
    free(AlphaR.data);

    // --- 5) Return as an R list
    return Rcpp::List::create(Rcpp::_["probabilities"] = Rprob, Rcpp::_["total_time"] = elapsed,
                              Rcpp::_["total_iterations"] = total_iterations);
}
