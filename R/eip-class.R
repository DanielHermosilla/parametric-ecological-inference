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
    if (!is.null(json_path)) {
        if (!all(is.null(X), is.null(W), is.null(V))) {
            stop("If you supply `json_path`, you must NOT supply X, W, or V by hand.")
        }
        params <- jsonlite::fromJSON(json_path)
    } else {
        params <- list("X" = X, "W" = W, "V" = V)
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
                   beta = NULL,
                   alpha = NULL,
                   maxiter = 1000,
                   maxtime = 3600,
                   ll_threshold = as.double(-Inf),
                   maxnewton = 100,
                   seed = NULL,
                   verbose = FALSE,
                   ...) {
    all_params <- lapply(as.list(match.call(expand.dots = TRUE)), eval, parent.frame())
    # .validate_compute(all_params) # nolint

    if (is.null(object)) {
        object <- eip(X, W, V, json_path)
    } else if (!inherits(object, "eip")) {
        stop("run_em: The object must be initialized with the eip() function.")
    }

    num_candidates <- ncol(object$X)
    num_groups <- ncol(object$W)
    num_attributes <- ncol(object$V)
    num_ballot_boxes <- nrow(object$X)

    # Create the initial alpha and beta if not provided
    if (is.null(beta)) {
        beta <- matrix(0, nrow = num_groups, ncol = num_candidates - 1)
    }
    if (is.null(alpha)) {
        alpha <- matrix(0, nrow = num_candidates - 1, ncol = num_attributes)
    }

    # Call the algorithm from C
    resulting_values <- EMAlgorithmC(
        as.matrix(object$X),
        as.matrix(object$W),
        as.matrix(object$V),
        as.matrix(beta),
        as.matrix(alpha),
        maxiter,
        maxtime,
        ll_threshold,
        maxnewton,
        verbose
    )

    # Append the results
    for (nm in names(resulting_values)) {
        object[[nm]] <- resulting_values[[nm]]
    }

    class(object) <- "eip"
    return(object)
}
