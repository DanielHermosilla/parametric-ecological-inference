.validate_eip <- function(args) {
    # Check for must-have arguments
    must_have <- c("X", "W", "E", "V")
    provided <- names(args)
    missing <- setdiff(must_have, provided)

    if (length(missing) > 0) {
        stop("Missing arguments: ", paste(missing, collapse = ", "))
    }

    # Check for disallowed arguments
    allowed_params <- c(must_have, "beta", "alpha")
    unexpected <- setdiff(provided, allowed_params)
    if (length(unexpected) > 0) {
        stop("Unexpected arguments: ", paste(unexpected, collapse = ", "))
    }

    if (!is.matrix(args$X)) stop("`X` must be a matrix, but is a ", class(args$X)[1])
    if (!is.matrix(args$W)) stop("`W` must be a matrix, but is a ", class(args$W)[1])
    if (!is.matrix(args$E)) stop("`E` must be a matrix, but is a ", class(args$E)[1])
    if (!is.matrix(args$V)) stop("`V` must be a matrix, but is a ", class(args$V)[1])

    # Get the dimensions
    B <- nrow(args$X) # Ballot boxes
    C <- ncol(args$X) # Candidates
    G <- ncol(args$W) # Groups
    D <- ncol(args$E) # Districts
    A <- ncol(args$V) # Atributes

    # Edge cases
    if (C < 1) stop("`X` must have at least one column (candidate).")
    if (G < 1) stop("`W` must have at least one column (group).")
    if (D < 1) stop("`E` must have at least one column (district).")
    if (A < 1) stop("`V` must have at least one column (attribute).")

    # Check dimentional coherence
    if (nrow(args$W) != B) {
        stop(
            "`W` must have ", B, " rows (same as X), but has ", nrow(args$W)
        )
    }
    if (nrow(args$E) != B) {
        stop(
            "`E` must have ", B, " rows (same as X), but has ", nrow(args$E)
        )
    }
    if (nrow(args$V) != D) {
        stop(
            "`V` must have ", D, " rows (same as columns of E), but has ", nrow(args$V)
        )
    }

    invisible(TRUE)
}
