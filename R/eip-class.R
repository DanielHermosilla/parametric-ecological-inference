library(jsonlite)

#' S3 object for the Expectaction-Maximization Algorithm
#'
#' @param X A `(b x c)` matrix representing candidate votes per ballot box.
#'
#' @param W A `(b x g)` matrix representing group votes per ballot box.
#'
#' @param V A `(b x a)` matrix with the attributes for each ballot box
#'
#' @param json_path A path to a JSON file containing `X`, `W`, and `V` fields, stored as nested arrays. It may contain additional fields with other attributes, which will be added to the returned object.
#'
#' @export
eip <- function(X = NULL, W = NULL, V = NULL, json_path = NULL) {
    # Case when a json is provided
    if (json_path != NULL) {
        if (!all(is.null(X), is.null(W), is.null(V))) {
            stop("If you supply `json_path`, you must NOT supply X, W, or V by hand.")
        }
        params <- fromJSON(json_path)
    } else {
        params <- c(X, W, V)
    }
    # Validate each value handed
    .validate_eip(params)
    class(params) <- "eip"
    return(params)
}

#' @export
run_em <- function(object = NULL,
                   X = NULL,
                   W = NULL,
                   V = NULL,
                   json_path = NULL,
                   method = "mult",
                   initial_prob = "group_proportional",
                   maxiter = 1000,
                   maxtime = 3600,
                   param_threshold = 0.001,
                   ll_threshold = as.double(-Inf),
                   seed = NULL,
                   verbose = FALSE,
                   ...) {
    all_params <- lapply(as.list(match.call(expand.dots = TRUE)), eval, parent.frame())
    .validate_compute(all_params) # nolint
}
