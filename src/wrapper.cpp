#include "wrapper.h"
#include "main.h"
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

//' @export
// [[Rcpp::export]]
Rcpp::List EMAlgorithmC(Rcpp::NumericMatrix X, Rcpp::NumericMatrix W, Rcpp::NumericMatrix V, Rcpp::NumericMatrix beta,
                        Rcpp::NumericMatrix alpha, int maxiter, double maxtime, double ll_threshold, int maxnewton,
                        bool verbose)
{
    // 1) marshal inputs
    Matrix XR = convertToMatrix(X);
    Matrix WR = convertToMatrix(W);
    Matrix VR = convertToMatrix(V);
    Matrix BetaR = convertToMatrix(beta);
    Matrix AlphaR = convertToMatrix(alpha);

    // 2) run EM, capture time & iterations
    double elapsed = 0.0;
    int total_iter = 0;
    Matrix *finalProb = EM_Algorithm(&XR, &WR, &VR, &BetaR, &AlphaR, maxiter, maxtime, ll_threshold, maxnewton, verbose,
                                     &elapsed, &total_iter);

    // 3) build R array for finalProb
    //    we assume EM_Algorithm returns an array of VR.rows slices
    int S = VR.rows;           // number of “slices” (mesa or district count)
    int R = finalProb[0].rows; // rows per slice
    int C = finalProb[0].cols; // cols per slice

    Rcpp::NumericVector probArr(R * C * S);
    probArr.attr("dim") = Rcpp::IntegerVector::create(R, C, S);

    for (int s = 0; s < S; ++s)
    {
        for (int i = 0; i < R; ++i)
        {
            for (int j = 0; j < C; ++j)
            {
                // R uses column-major order [i + nrow*(j + ncol*s)]
                int idx = i + R * (j + C * s);
                probArr[idx] = MATRIX_AT(finalProb[s], i, j);
            }
        }
        freeMatrix(&finalProb[s]);
    }
    Free(finalProb);

    // 4) copy out final Beta, Alpha
    Rcpp::NumericMatrix Rbeta(BetaR.rows, BetaR.cols);
    std::memcpy(Rbeta.begin(), BetaR.data, sizeof(double) * BetaR.rows * BetaR.cols);

    Rcpp::NumericMatrix Ralpha(AlphaR.rows, AlphaR.cols);
    std::memcpy(Ralpha.begin(), AlphaR.data, sizeof(double) * AlphaR.rows * AlphaR.cols);

    // 5) clean up all C allocations
    freeMatrix(&XR);
    std::free(XR.data);
    freeMatrix(&WR);
    std::free(WR.data);
    freeMatrix(&VR);
    std::free(VR.data);
    std::free(BetaR.data);
    std::free(AlphaR.data);

    // 6) return everything
    return Rcpp::List::create(Rcpp::_["probabilities"] = probArr, Rcpp::_["beta"] = Rbeta, Rcpp::_["alpha"] = Ralpha,
                              Rcpp::_["total_time"] = elapsed, Rcpp::_["total_iterations"] = total_iter);
}
