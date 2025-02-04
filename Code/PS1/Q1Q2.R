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

# compare
anova(ccp_probit1, ccp_probit2, test = "Chisq")
anova(ccp_probit1, ccp_probit3, test = "Chisq")
anova(ccp_probit2, ccp_probit3, test = "Chisq")

# graph the probability of investment as a function of mileage
pacman::p_load(ggplot2, extrafont)
font_import(pattern = "times") # Only import Times New Roman
loadfonts(device = "pdf") # Load fonts for PDF output

# set the global theme to use Times New Roman
theme_set(
    theme_minimal(base_family = "Times New Roman") +
        theme(
            plot.title = element_text(size = 22, hjust = 0.5),
            axis.title = element_text(size = 18),
            axis.text = element_text(size = 18)
        )
)

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
        title = "Reduced Form of Conditional Choice Probability",
        x = "Mileage",
        y = "Probability",
        color = "Model"
    )

ggsave("Figures/ccp_probit.png", graph)
ggsave("Figures/ccp_probit.pdf", graph)

# ---- q2: estimate milage transition probability parameters
dt <- readRDS("Data/RUST.RDS")

# we take the difference between two periods x_{t}-x_{t-1} when x_{t-1}'s choice is not 1

dt[, mileage_diff := mileage - shift(mileage, n = 1), by = id]
rho <- dt[state == 0, mean(mileage_diff)]
sigma_rho <- dt[state == 0, var(mileage_diff)]
save.image("Data/Out/q1q2.RData")
