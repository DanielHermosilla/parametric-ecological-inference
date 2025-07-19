#' Simulate an election
#'
#' @param num_voters amount of voters per table
#'
#' @param num_districts number of districts
#'
#' @param num_ballots number of tables
#' @param num_groups number of groups
#'
#' @param num_candidates number of candidates
#'
#' @param num_attributes number of attributes
#'
#' @param lambda Shuffling fraction of voters (0=no shuffle, 1=full shuffle)
#'
#' @param seed RNG seed
#'
#' @export
simulate_election <- function(
    num_voters, num_districts, num_ballots,
    num_groups, num_candidates, num_attributes,
    lambda = 0.5, seed = 0) {
    # Seed
    if (!is.null(seed)) set.seed(seed)

    # Coefficients
    alpha <- matrix(rnorm((num_candidates - 1) * num_attributes),
        nrow = num_candidates - 1, ncol = num_attributes
    )
    beta <- matrix(rnorm(num_groups * (num_candidates - 1)),
        nrow = num_groups, ncol = num_candidates - 1
    )

    # District assignment of ballot-boxes
    random_one_in_each_row <- function(n, m) {
        mat <- matrix(0L, nrow = n, ncol = m)
        idx <- sample.int(m, n, replace = TRUE)
        mat[cbind(seq_len(n), idx)] <- 1L
        d <- min(n, m)
        mat[seq_len(d), seq_len(d)] <- diag(1L, d)
        mat
    }
    e_bd <- random_one_in_each_row(num_ballots, num_districts)

    # District & ballot-box attributes (if needed)
    v_da <- matrix(runif(num_districts * num_attributes, min = 0, max = 1),
        nrow = num_districts, ncol = num_attributes
    )
    # v_da <- matrix(rnorm(num_districts * num_attributes), nrow = num_districts, ncol = num_attributes)

    v_ba <- e_bd %*% v_da

    # Compute candidate choice probs
    p_dgc <- array(0, dim = c(num_districts, num_groups, num_candidates))
    for (d in seq_len(num_districts)) {
        for (g in seq_len(num_groups)) {
            u <- alpha %*% v_da[d, ] + beta[g, ]
            u <- c(u, 0)
            p_dgc[d, g, ] <- exp(u) / sum(exp(u))
        }
    }

    # Create output matrices
    W <- matrix(0L, nrow = num_ballots, ncol = num_groups)
    X <- matrix(0L, nrow = num_ballots, ncol = num_candidates)

    # Shuffle and simmulate district-wise
    for (d in seq_len(num_districts)) {
        boxes <- which(e_bd[, d] == 1L)
        B_d <- length(boxes)
        I_d <- num_voters * B_d

        # Sequential assignment per group proportions
        # counts per group
        group_proportions <- rep(1 / num_groups, num_groups)
        base_counts <- floor(I_d * group_proportions)
        rem <- I_d - sum(base_counts)
        if (rem > 0) {
            frac <- I_d * group_proportions - base_counts
            idx <- order(frac, decreasing = TRUE)[seq_len(rem)]
            base_counts[idx] <- base_counts[idx] + 1L
        }
        omega0 <- rep(seq_len(num_groups), times = base_counts)
        # Sanity check: ensure length
        if (length(omega0) > I_d) omega0 <- omega0[seq_len(I_d)]
        if (length(omega0) < I_d) omega0 <- c(omega0, sample(seq_len(num_groups), I_d - length(omega0), replace = TRUE))
        omega <- omega0

        # Shuffle a lambda fraction within district
        n_mix <- round(lambda * I_d)
        # if (n_mix > 0) {
        #     idx <- sample.int(I_d, n_mix)
        #     idx_sort <- sort(idx)
        # Reassign in shuffled positions
        #    omega[idx] <- omega0[idx_sort]
        # }
        if (n_mix > 0) {
            idx <- sample.int(I_d, n_mix)
            # grab the values to be mixed
            vals_to_mix <- omega0[idx]
            # randomly permute those values
            omega[idx] <- sample(vals_to_mix, length(vals_to_mix))
        }

        # Partition into ballot-boxes and tally
        for (j in seq_along(boxes)) {
            b <- boxes[j]
            start <- (j - 1) * num_voters + 1
            end <- j * num_voters
            G_b <- omega[start:end]
            # sample votes by group
            votes_b <- vapply(G_b, function(g) {
                sample.int(num_candidates, 1, prob = p_dgc[d, g, ])
            }, integer(1))
            # fill outputs
            W[b, ] <- tabulate(G_b, nbins = num_groups)
            X[b, ] <- tabulate(votes_b, nbins = num_candidates)
        }
    }

    # Go from district-wise probabilities to ballot-box-wise
    p_bgc <- vector("list", num_ballots)
    for (b in seq_len(num_ballots)) {
        d <- which(e_bd[b, ] == 1L)
        # p_dgc[d,,] is a G×C matrix
        p_bgc[[b]] <- p_dgc[d, , ]
    }

    result <- list(
        W = W,
        X = X,
        V = v_ba,
        prob = p_bgc,
        alpha = alpha,
        beta = beta
    )
    class(result) <- "eip"
    result
}
