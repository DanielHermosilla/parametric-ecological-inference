#include "wrapper.h"
#include <R.h>
#include <R_ext/Random.h>
#include <Rcpp.h>
#include <Rinternals.h>
#include <vector>

#ifndef Calloc
#define Calloc(n, type) ((type *)R_chk_calloc((size_t)(n), sizeof(type)))
#endif

#ifndef Free
#define Free(p) R_chk_free((void *)(p))
#endif

// ---- Run EM Algorithm ---- //
// [[Rcpp::export]]
Rcpp::List EMAlgorithmC(Rcpp::NumericMatrix X, Rcpp::NumericMatrix W, Rcpp::NumericMatrix V, Rcpp::NumericMatrix beta,
                        Rcpp::NumericMatrix alpha, Rcpp::IntegerVector maxiter, Rcpp::NumericVector maxtime,
                        Rcpp::NumericVector param_threshold, Rcpp::NumericVector ll_threshold,
                        Rcpp::IntegerVector maxnewton, Rcpp::LogicalVector verbose)
{
}
