#' @export
print.eip <- function(x, ...) {
    object <- x
    cat("eip ecological inference model\n")
    # Determine if truncation is needed
    truncated_X <- (nrow(object$X) > 5)
    truncated_W <- (nrow(object$W) > 5)

    cat("Candidates' vote matrix (X) [b x c]:\n")
    print(object$X[1:min(5, nrow(object$X)), ], drop = FALSE) # nolint
    if (truncated_X) cat(".\n.\n.\n") else cat("\n")

    cat("Group-level voter matrix (W) [b x g]:\n")
    print(object$W[1:min(5, nrow(object$W)), , drop = FALSE]) # nolint
    if (truncated_W) cat(".\n.\n.\n") else cat("\n")

    if (!is.null(object$beta)) {
        cat("Estimated beta parameters [c x g]\n")
        print(object$beta, drop = FALSE)
        cat("\n")
    }

    if (!is.null(object$alpha)) {
        cat("Estimated alpha parameters [c x a]\n")
        print(object$alpha, drop = FALSE)
        cat("\n")
    }
}

#' @export
summary.eip <- function(object, ...) {
    # Generates the list with the core attribute.
    object_core_attr <- list(
        candidates = ncol(object$X),
        groups = ncol(object$W)
    )

    # A list with attributes to display if the EM is computed.
    if (!is.null(object$prob)) {
        object_run_em_attr <- list(
            prob = object$prob,
            beta = object$beta,
            alpha = object$alpha
        )
    }

    final_list <- c(object_core_attr, object_run_em_attr)
    final_list
}
