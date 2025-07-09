#ifndef WRAPPER_H_EIM
#define WRAPPER_H_EIM

/* From CRAN guide to packages:
 *Macros defined by the compiler/OS can cause problems. Identifiers starting with an underscore followed by an
 *upper-case letter or another underscore are reserved for system macros and should not be used in portable code
 *(including not as guards in C/C++ headers). Other macros, typically upper-case, may be defined by the compiler or
 *system headers and can cause problems. Some of these can be avoided by defining _POSIX_C_SOURCE before including any
 *system headers, but it is better to only use all-upper-case names which have a unique prefix such as the package name.
 */

#ifdef __cplusplus
extern "C"
{
#endif
#include "utils_matrix.h"
#ifdef __cplusplus
}
#endif
#include <Rcpp.h>

#endif // WRAPPER_H
