# method 2

simulate_m <- function(initial_mileage, initial_y, K, T, beta) {
    num_row <- length(initial_mileage)
    # store the intermediate results
    s_array <- array(0, dim = c(num_row, T, K))
    y_array <- array(0, dim = c(num_row, T, K))
    logp_array <- array(0, dim = c(num_row, T, K))
    # store the final results
    x_i1_matrix <- matrix(0, nrow = num_row, ncol = K)
    x_i2_matrix <- matrix(0, nrow = num_row, ncol = K)
    x_i3_matrix <- matrix(0, nrow = num_row, ncol = K)
    # create the epsilon matrix
    epsilon_array <- array(rnorm(num_row * T * K, rho, sqrt(sigma_rho)), dim = c(num_row, T, K))
    # fill in the initial values
    y_array[, 1, ] <- outer(rep(initial_y, num_row), rep(1, K))
    s_array[, 1, ] <- outer(initial_mileage, rep(1, K))
    logp_array[, 1, ] <- outer(rep(0, num_row), rep(1, K))

    for (k in 1:K) {
        mileage <- initial_mileage
        for (t in 2:T) {
            mileage <- (1 - y_array[, t - 1, k]) * (mileage) + epsilon_array[, t, k]
            s_array[, t, k] <- mileage
            y_hat <- predict(ccp_probit1, data.frame(mileage = mileage), type = "response")
            y_array[, t, k] <- rbinom(num_row, 1, y_hat)
            logp_array[, t, k] <- log(y_hat * y_array[, t, k] + (1 - y_hat) * (1 - y_array[, t, k]))
        }
        beta_vec <- beta^(0:(T - 1))
        x_i1_matrix[, k] <- (y_array[, , k] %*% beta_vec)
        x_i2_matrix[, k] <- (y_array[, , k] * s_array[, , k]) %*% beta_vec
        x_i3_matrix[, k] <- (euler - logp_array[, , k]) %*% beta_vec
    }
    x_i1 <- rowMeans(x_i1_matrix)
    x_i2 <- rowMeans(x_i2_matrix)
    x_i3 <- rowMeans(x_i3_matrix)
    return(cbind(x_i1, x_i2, x_i3))
}

v_1_matrix_m <- simulate_m(dt$mileage, 1, K = 10, T = 100, beta = 0.99)
v_0_matrix_m <- simulate_m(dt$mileage, 0, K = 10, T = 100, beta = 0.99)

v_diff_matrix_m <- v_1_matrix_m - v_0_matrix_m
dt2_m <- cbind(dt, v_diff_matrix_m)
dt2_m$choice <- as.factor(dt2_m$choice)

logit_m <- glm(choice ~ x_i1 + x_i2 - 1, offset = x_i3, data = dt2_m, family = binomial(link = "logit"))
summary(logit_m)
