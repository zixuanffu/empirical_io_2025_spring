rm(list = ls())
pacman::p_load(data.table)

# ---- clean the data
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

# ---- q1: estimate conditional choice probability
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

# convert the data to long format for easier plotting with a legend
new_data_plot_long <- melt(new_data_plot,
    id.vars = "mileage",
    measure.vars = c("prob1", "prob2", "prob3", "problog"),
    variable.name = "Model", value.name = "Probability"
)

# define custom colors for the legend
color_map <- c("prob1" = "red", "prob2" = "blue", "prob3" = "green", "problog" = "black")

# plot with a legend
graph <- ggplot(new_data_plot_long, aes(x = mileage, y = Probability, color = Model)) +
    geom_line() +
    scale_color_manual(values = color_map) +
    labs(
        title = "(Probit) Reduced Form of Conditional Choice Probability",
        x = "Mileage",
        y = "Probability",
        color = "Model"
    ) +
    # theme_minimal() +
    theme(
        plot.title = element_text(size = 22, hjust = 0.5),
        axis.title = element_text(size = 20),
        axis.text = element_text(size = 18),
        legend.title = element_text(size = 18),
        legend.text = element_text(size = 18)
    )

ggsave("Figures/ccp_probit.png", graph)
ggsave("Figures/ccp_probit.pdf", graph)

# ---- q2: estimate milage transition probability parameters
dt <- readRDS("Data/RUST.RDS")

# we take the difference between two periods x_{t}-x_{t-1} when x_{t-1}'s choice is not 1

dt[, mileage_diff := mileage - shift(mileage, n = 1), by = id]
rho <- dt[state == 0, mean(mileage_diff)]
sigma_rho <- dt[state == 0, var(mileage_diff)]

# q3: estimate continuation value via simulation

# for every mileage $s_{it}$, I calculate two continuation values $V_{it}(1)$ and $V_{it}(0)$

num_bus <- dt[state == 2, .N] # 58 buses
num_obs <- dt[, .N] # 7308 observations
num_period <- num_obs / num_bus # 126 observations per bus

# simulation parameters
K <- 10
T <- 100
beta <- 0.99

# euler mascheroni constant
euler <- 0.577215664901532

# define the simulation function

simulate <- function(initial_mileage, initial_y, K, T, beta) {
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
        # y_array[, 1, k] <- rep(initial_y, num_row)
        # s_array[, 1, k] <- initial_mileage
        # logp_array[, 1, k] <- rep(0, num_row)
        for (t in 2:T) {
            mileage <- (1 - y_array[, t - 1, k]) * (mileage + epsilon_array[, t, k])
            s_array[, t, k] <- mileage
            y_hat <- predict(ccp_probit1, data.frame(mileage = mileage), type = "response")
            logp_array[, t, k] <- log(y_hat)
            y_array[, t, k] <- rbinom(num_row, 1, y_hat)
        }
        beta_vec <- beta^(0:(T - 1))
        x_i1_matrix[, k] <- (-(y_array[, , k] %*% beta_vec))
        x_i2_matrix[, k] <- (-(1 - y_array[, , k]) * s_array[, , k]) %*% beta_vec
        x_i3_matrix[, k] <- (euler - logp_array[, , k]) %*% beta_vec
    }
    x_i1 <- rowMeans(x_i1_matrix)
    x_i2 <- rowMeans(x_i2_matrix)
    x_i3 <- rowMeans(x_i3_matrix)
    return(list(x = cbind(x_i1, x_i2, x_i3), y = y_array, s = s_array, logp = logp_array))
}

# store the simulation results
v_1_res <- simulate(dt$mileage, 1, K = 10, T = 100, beta = 0.99)
v_0_res <- simulate(dt$mileage, 0, K = 10, T = 100, beta = 0.99)
v_1_matrix <- v_1_res$x
v_0_matrix <- v_0_res$x

# tabulate the average of six variables x_i1(1),x_i2(1),x_i3(1),x_i1(0),x_i2(0),x_i3(0)
simulation_average <- data.table(colMeans(v_1_matrix), colMeans(v_0_matrix))
colnames(simulation_average) <- c("Replace", "Non Replace")
rownames(simulation_average) <- c("x_i1", "x_i2", "x_i3")
# print the table
library(xtable)
print(xtable(simulation_average), floating = FALSE, type = "latex", file = "Tables/simulation_average.tex")

# ---- q4: estimate the parameters of the model
v_diff_matrix <- v_1_matrix - v_0_matrix
dt2 <- cbind(dt, v_diff_matrix)
dt2$choice <- as.factor(dt2$choice)
logit <- glm(choice ~ x_i1 + x_i2 - 1, offset = x_i3, data = dt2, family = binomial(link = "logit"))
summary(logit)
dt2[, prob := predict(logit, dt2, type = "response")]
saveRDS(dt2, "Data/Out/data_q3q4.RDS")


# ---- q5: graph the probability of investment as a function of mileage
new_data_plot <- data.table(mileage = seq(0, 40, 0.1))

# for each mileage, we need to calculate the probability of investment
# calculating the predicted probability requires continuation value for v1 and v0
# v1 and v0 are calculated by simulation

v_1_matrix_plot <- simulate(new_data_plot$mileage, 1, K = 10, T = 100, beta = 0.99)$x
v_0_matrix_plot <- simulate(new_data_plot$mileage, 0, K = 10, T = 100, beta = 0.99)$x
v_matrix_plot <- v_1_matrix_plot - v_0_matrix_plot
new_data_plot <- cbind(new_data_plot, v_matrix_plot)
new_data_plot[, prob := predict(logit, new_data_plot, type = "response")]
saveRDS(new_data_plot, "Data/Out/data_q5.RDS")

graph2 <- ggplot(new_data_plot, aes(x = mileage)) +
    geom_point(aes(y = prob), color = "#FF0000") +
    labs(title = "Conditional Choice Probability", x = "Mileage", y = "Probability") +
    theme_minimal()

graph2 <- graph2 + theme(
    plot.title = element_text(size = 22, hjust = 0.5),
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 18)
)

ggsave("Figures/ccp_dynamic.png", graph2)
ggsave("Figures/ccp_dynamic.pdf", graph2)
