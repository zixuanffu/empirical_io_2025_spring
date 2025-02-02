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

# q2: estimate milage transition probability parameters
dt <- fread("Data/RUST.RDS")

# we take the difference between two periods x_{t}-x_{t-1} when x_{t-1}'s choice is not 1

dt[, mileage_diff := mileage - shift(mileage, n = 1), by = id]
rho <- dt[state == 0, mean(mileage_diff)]
sigma_rho <- dt[state == 0, var(mileage_diff)]

# q3: estimate continuation value via simulation

# for every mileage $s_{it}$, I calculate two continuation values $V_{it}(1)$ and $V_{it}(0)$

K <- 10
T <- 100
beta <- 0.99
euler <- 0.577215664901532

# case 1: replace at the current period y_{t} = 1
v1_list <- matrix(0, nrow = num_obs, ncol = 3)
v0_list <- matrix(0, nrow = num_obs, ncol = 3)
for (it in num_obs){
    
    
}

x_i1 <- mean(xi1_list)
x_i2 <- mean(xi2_list)
x_i3 <- mean(xi3_list)

# case 2: no replace in the current period y_{t} = 0
new_data <- data.frame(mileage = dt[time = 2, mileage])
xi1_list <- rep(0, K)
x_i2_list <- rep(0, K)
x_i3_list <- rep(0, K)
for (k in 1:K) {
    y <- rep(0, T)
    s <- rep(0, T)
    logp <- rep(0, T)
    y[1] <- 1
    s[1] <- 0
    logp[1] <- 0
    for (t in 2:T) {
        yhat_prob <- predict(ccp_probit1, new_data, type = "response")
        yhat <- rbinom(1, 1, yhat_prob)
        logp[t] <- log(yhat_prob)
        y[t] <- yhat
        new_data <- data.frame(mileage = y * (new_data$mileage + rnorm(1, rho, sqrt(sigma_rho))))
        s[t] <- new_data$mileage
    }
    beta_list <- c(0.99)^seq(0, T - 1)
    xi1_list[k] <- sum(-y * beta_list)
    xi2_list[k] <- sum(-(1 - y) * s * beta_list)
    xi3_list[k] <- sum(-(euler - logp) * beta_list)
}
x_i1 <- mean(xi1_list)
x_i2 <- mean(xi2_list)
x_i3 <- mean(xi3_list)