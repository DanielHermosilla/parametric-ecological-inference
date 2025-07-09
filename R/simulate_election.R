#' Simulate an election
#'
#' @param num_voters amount of voters per table
#'
#' @param num_districts number of districts
#'
#' @param num_ballots number of tables
#'
#' @param num_groups number of groups
#'
#' @param num_candidates number of candidates
#'
#' @param num_attributes number of attributes
#'
#' @param lambda Shuffling fraction of voters
#'
#' @param seed
#'
#' @export
simulate_election <- function(num_voters, num_districts, num_ballots, num_groups, num_candidates, num_attributes, lambda, seed = 0, verbose = FALSE) {
    set.seed(seed)

    # Define the alpha and beta matrix
    alpha <- matrix(rnorm((num_candidates - 1) * num_attributes), nrow = num_candidates - 1, ncol = num_attributes)
    beta <- matrix(rnorm(num_groups * (num_candidates - 1)), nrow = num_groups, ncol = num_candidates - 1)

    # Returns an identity matrix in the first 'num_districts' rows, and random 1's in subsequent rows
    # This is made for guaranteeing that each district has at least one table assigned to it.
    random_one_in_each_row <- function(n, m) {
        mat <- matrix(0L, nrow = n, ncol = m)
        idx <- sample.int(m, n, replace = TRUE)
        mat[cbind(seq_len(n), idx)] <- 1L
        d <- min(n, m)
        mat[seq_len(d), seq_len(d)] <- diag(1L, d)
        mat
    }

    # Generate e_bd (num_ballots × num_districts) matrix with random assignment of tables to districts
    e_bd <- random_one_in_each_row(num_ballots, num_districts)
    # Generate v_da (num_districts × num_attributes) matrix with random attributes for each district
    v_da <- matrix(rnorm(num_districts * num_attributes), nrow = num_districts, ncol = num_attributes)

    # Obtain the probability tensor with num_attributes softmax function
    p_dgc <- array(0, dim = c(num_districts, num_groups, num_candidates))
    for (d in seq_len(num_districts)) {
        for (g in seq_len(num_groups)) {
            # Obtain the parametric utility for each candidate
            u <- alpha %*% v_da[d, ] + beta[g, ]
            # The first candidate, as baseline, has num_attributes utility of 0
            u <- c(u, 0)
            # Obtain this probabilities using the softmax function
            p_dgc[d, g, ] <- exp(u) / sum(exp(u))
        }
    }

    # Initialize the group and candidate matrices
    w_bg <- matrix(0L, nrow = num_ballots, ncol = num_groups)
    x_bc <- matrix(0L, nrow = num_ballots, ncol = num_candidates)

    # Simulate the election, per each ballot box
    for (d in seq_len(num_districts)) {
        ballot_boxes_d <- which(e_bd[, d] == 1L)
        B_d <- length(ballot_boxes_d)
        I_d <- num_voters * B_d

        # Assign voters in an uniform way
        G_population <- rep(seq_len(num_groups), length.out = I_d)
        votes_population <- integer(I_d)

        # For each voter, sample their vote based on the given probabilities
        for (i in seq_len(I_d)) {
            g_i <- G_population[i]
            votes_population[i] <- sample.int(num_candidates, 1, prob = p_dgc[d, g_i, ])
        }

        # Shuffle num_attributes fractión lambda * I * num_attributes voters
        idx_mix <- sample(seq_len(I_d), size = round(lambda * I_d), replace = FALSE)
        vals_g <- G_population[idx_mix]
        vals_v <- votes_population[idx_mix]
        idx_mix <- sort(idx_mix)
        G_population[idx_mix] <- vals_g
        votes_population[idx_mix] <- vals_v

        # Assign them on its matrices
        G_mat <- matrix(G_population, nrow = B_d, ncol = num_voters, byrow = TRUE)
        V_mat <- matrix(votes_population, nrow = B_d, ncol = num_voters, byrow = TRUE)

        # Count per table
        for (j in seq_len(B_d)) {
            b <- ballot_boxes_d[j]
            for (g in seq_len(num_groups)) {
                w_bg[b, g] <- sum(G_mat[j, ] == g)
            }
            for (c in seq_len(num_candidates)) {
                x_bc[b, c] <- sum(V_mat[j, ] == c)
            }
        }
    }

	# Return results as a list
    to_return <- list(
        X = x_bc, # Votes per table × candidate
        W = w_bg, # Votes per ballot box x group 
        V = v_da, # District attributes
        E = e_bd, # Assignation ballot box -> district 
		P = p_dgc # Probability tensor district x group x candidate
        beta = beta, # Bias matrix group x candidate
        alpha = alpha # Coefficientes of covariates per candidate
    )
	class(to_return) <- "eip"
	return(to_return)
}
