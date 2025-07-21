#ifndef BOOTSTRAP_H_EIP
#define BOOTSTRAP_H_EIP

#ifdef __cplusplus

extern "C"
{
#endif

#include "globals.h"
#include "utils_matrix.h"

    void bootstrap(Matrix *X, Matrix *W, Matrix *V, Matrix *beta, Matrix *alpha, const int bootiter, const int maxiter,
                   const double maxtime, const double ll_threshold, const int maxnewton, const bool verbose,
                   double *out_elapsed, int *total_iterations, Matrix *sdBetas, Matrix *sdAlpha);
#ifdef __cplusplus
}
#endif
#endif // UTIL_H
