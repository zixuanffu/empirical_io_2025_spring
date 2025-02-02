pacman::p_load(data.table)

# clean the data
dt <- fread("Data/RUST.DAT")
setnames(dt, new = c("mileage", "state"))
num_bus <- dt[state == 2, .N] # 58 buses
num_obs <- dt[, .N] # 7308 observations
num_period <- num_obs / num_bus # 126 observations per bus
dt[, id := rep(seq(1, num_bus), each = num_period)]
dt[, time := rep(1:num_period, num_bus)] # time stamp
dt[, choice := shift(state, n = -1), by = id] # one should drop the last period for each bus
dt[, mileage := mileage / 10000]
saveRDS(dt, "Data/RUST.RDS")
# q1: estimate conditional choice probability
dt <- readRDS("Data/RUST.RDS")
# probit1
ccp_probit1 <- glm(choice ~ mileage, data = dt, family = binomial(link = "probit"))
summary(ccp_probit1)
# probit2
ccp_probit2 <- glm(choice ~ mileage + I(mileage^2), data = dt, family = binomial(link = "probit"))
summary(ccp_probit2)
# probit3
ccp_probit3 <- glm(choice ~ mileage + I(mileage^2) + I(mileage^3), data = dt, family = binomial(link = "probit"))
summary(ccp_probit3)
# probitlog
ccp_probitlog <- glm(choice ~ log(mileage), data = dt, family = binomial(link = "probit"))
summary(ccp_probitlog)

# graph the probability of investment as a function of mileage
library(ggplot2)

# calculate the predicted probability
new_data_plot <- data.table(mileage = seq(0, 40, 0.1))
new_data_plot[, prob1 := predict(ccp_probit1, new_data_plot, type = "response")]
new_data_plot[, prob2 := predict(ccp_probit2, new_data_plot, type = "response")]
new_data_plot[, prob3 := predict(ccp_probit3, new_data_plot, type = "response")]
new_data_plot[, problog := predict(ccp_probitlog, new_data_plot, type = "response")]

# plot the graph
graph <- ggplot(new_data_plot, aes(x = mileage)) +
    geom_line(aes(y = prob1), color = "#FF0000") +
    geom_line(aes(y = prob2), color = "blue") +
    geom_line(aes(y = prob3), color = "green") +
    geom_line(aes(y = problog), color = "black") +
    labs(title = "(Probit) Reduced Form of Conditional Choice Probability", x = "Mileage", y = "Probability") +
    theme_minimal()
graph <- graph + theme(
    plot.title = element_text(size = 20, hjust = 0.5),
    axis.title = element_text(size = 15),
    axis.text = element_text(size = 12)
)
ggsave("Figures/ccp_probit.png", graph)


# q2: estimate milage transition probability parameters
dt <- fread("Data/RUST.RDS")

# we take the difference between two periods x_{t}-x_{t-1} when x_{t-1}'s choice is not 1

dt[, mileage_diff := mileage - shift(mileage, n = 1), by = id]
rho <- dt[state == 0, mean(mileage_diff)]
sigma_rho <- dt[state == 0, var(mileage_diff)]

# q3: estimate continuation value via simulation

# for every mileage $s_{it}$, I calculate two continuation values $V_{it}(1)$ and $V_{it}(0)$

num_bus <- dt[state == 2, .N] # 58 buses
num_obs <- dt[, .N] # 7308 observations
num_period <- num_obs / num_bus # 126 observations per bus

K <- 10
T <- 100
beta <- 0.99
euler <- 0.577215664901532

# initialize the matrix
v1_list <- matrix(0, nrow = 1, ncol = 3)
v0_list <- matrix(0, nrow = num_obs, ncol = 3)

simulate_case <- function(initial_mileage, initial_y) {
    xi1_list <- numeric(K)
    xi2_list <- numeric(K)
    xi3_list <- numeric(K)

    for (k in 1:K) {
        mileage <- initial_mileage
        y <- numeric(T)
        s <- numeric(T)
        logp <- numeric(T)

        y[1] <- initial_y
        s[1] <- initial_mileage
        logp[1] <- 0

        for (t in 2:T) {
            mileage <- (1 - y[t - 1]) * mileage + rnorm(1, rho, sqrt(sigma_rho))
            s[t] <- mileage
            yhat_prob <- predict(ccp_probit1, data.frame(mileage = mileage), type = "response")
            logp[t] <- log(yhat_prob)
            y[t] <- rbinom(1, 1, yhat_prob)
        }

        beta_list <- 0.99^(0:(T - 1))
        xi1_list[k] <- sum(-y * beta_list)
        xi2_list[k] <- sum(-((1 - y) * s * beta_list))
        xi3_list[k] <- sum(-(euler - logp) * beta_list)
    }

    return(c(mean(xi1_list), mean(xi2_list), mean(xi3_list)))
}

# Case 1: Replace at the current period (y_t = 1)
v1_list <- simulate_case(0, 1)
for (it in 1:num_obs) {
    # Case 2: No replace in the current period (y_t = 0)
    v0_list[it, ] <- simulate_case(dt[it, mileage], 0)
}

# original code
for (it in 1:2) {
    # case 1: replace at the current period y_{t} = 1
    xi1_list <- rep(0, K)
    xi2_list <- rep(0, K)
    xi3_list <- rep(0, K)
    for (k in 1:K) {
        new_data <- data.frame(mileage = 0)

        y <- rep(0, T)
        s <- rep(0, T)
        logp <- rep(0, T)

        y[1] <- 1
        s[1] <- 0
        logp[1] <- 0

        for (t in 2:T) {
            new_data <- data.frame(mileage = (1 - y[t - 1]) * (new_data$mileage) + rnorm(1, rho, sqrt(sigma_rho)))
            s[t] <- new_data$mileage
            yhat_prob <- predict(ccp_probit1, new_data, type = "response")
            logp[t] <- log(yhat_prob)
            yhat <- rbinom(1, 1, yhat_prob)
            y[t] <- yhat
        }
        beta_list <- c(0.99)^seq(0, T - 1)
        xi1_list[k] <- sum(-y * beta_list)
        xi2_list[k] <- sum(-(1 - y) * s * beta_list)
        xi3_list[k] <- sum(-(euler - logp) * beta_list)
    }
    x_i1 <- mean(xi1_list)
    x_i2 <- mean(xi2_list)
    x_i3 <- mean(xi3_list)
    v1_list[it, ] <- c(x_i1, x_i2, x_i3)

    # case 2: no replace in the current period y_{t} = 0

    xi1_list <- rep(0, K)
    x_i2_list <- rep(0, K)
    x_i3_list <- rep(0, K)
    for (k in 1:K) {
        new_data <- data.frame(mileage = dt[it, mileage])

        y <- rep(0, T)
        s <- rep(0, T)
        logp <- rep(0, T)

        y[1] <- 0
        s[1] <- dt[it, mileage]
        logp[1] <- 0

        for (t in 2:T) {
            new_data <- data.frame(mileage = (1 - y[t - 1]) * (new_data$mileage) + rnorm(1, rho, sqrt(sigma_rho)))
            s[t] <- new_data$mileage
            yhat_prob <- predict(ccp_probit1, new_data, type = "response")
            logp[t] <- log(yhat_prob)
            yhat <- rbinom(1, 1, yhat_prob)
            y[t] <- yhat
        }
        beta_list <- c(0.99)^seq(0, T - 1)
        xi1_list[k] <- sum(-y * beta_list)
        xi2_list[k] <- sum(-((1 - y) * s * beta_list))
        xi3_list[k] <- sum(-(euler - logp) * beta_list)
    }
    x_i1 <- mean(xi1_list)
    x_i2 <- mean(xi2_list)
    x_i3 <- mean(xi3_list)
    v0_list[it, ] <- c(x_i1, x_i2, x_i3)
}

# q4: estimate the parameters of the model
v_list <- v1_list - v0_list
v_list <- data.table(v_list)
setnames(v_list, new = c("x_i1", "x_i2", "x_i3"))
dt2 <- cbind(dt, v_list)
logit <- glm(choice ~ x_i1 + x_i2 + x_i3 - 1, data = dt2, family = binomial(link = "logit"))
summary(logit)
dt2[, prob := predict(logit, dt2, type = "response")]
saveRDS(dt2, "Data/Out/data_q3q4.RDS")

# q5: graph the probability of investment as a function of mileage
new_data_plot <- data.table(mileage = seq(0, 40, 0.1))
# for each mileage, we need to calculate the probability of investment
# calculating the predicted probability requires continuation value for v1 and v0
# v1 and v0 are calculated by simulation

# Case 1: Replace at the current period (y_t = 1)
v1_list_plot <- simulate_case(0, 1)
# Case 2: No replace in the current period (y_t = 0)
v0_list_plot <- matrix(0, nrow = nrow(new_data_plot), ncol = 3)
for (it in 1:nrow(new_data_plot)) {
    # Case 2: No replace in the current period (y_t = 0)
    v0_list_plot[it, ] <- simulate_case(new_data_plot[it, mileage], 0)
}

v_list_plot <- v1_list_plot - v0_list_plot
v_list_plot <- data.table(v_list_plot)
setnames(v_list_plot, new = c("x_i1", "x_i2", "x_i3"))
new_data_plot <- cbind(new_data_plot, v_list_plot)
new_data_plot[, prob := predict(logit, new_data_plot, type = "response")]
saveRDS(new_data_plot, "Data/Out/data_q5.RDS")

graph2 <- ggplot(new_data_plot, aes(x = mileage)) +
    geom_line(aes(y = prob), color = "#FF0000") +
    labs(title = "Conditional Choice Probability", x = "Mileage", y = "Probability") +
    theme_minimal()

graph2 <- graph2 + theme(
    plot.title = element_text(size = 20, hjust = 0.5),
    axis.title = element_text(size = 15),
    axis.text = element_text(size = 12)
)

plot(new_data_plot$mileage, new_data_plot$prob)
ggsave("Figures/ccp_dynamic.png", graph2)
ggsave("Figures/ccp_dynamic.pdf", graph2)

# review
dt2 <- readRDS("Data/Out/data_q3q4.RDS")
new_data_plot <- readRDS("Data/Out/data_q5.RDS")
