#ifndef MAIN_H_EIM
#define MAIN_H_EIM

#ifdef __cplusplus

extern "C"
{
#endif

#include "globals.h"
#include "utils_matrix.h"

    Matrix *EM_Algorithm(Matrix *X, Matrix *W, Matrix *V, Matrix *beta, Matrix *alpha, const int maxiter,
                         const double maxtime, const double ll_threshold, const int maxnewton, const bool verbose,
                         double *out_elapsed, int *total_iterations);
#ifdef __cplusplus
}
#endif
#endif // UTIL_H
