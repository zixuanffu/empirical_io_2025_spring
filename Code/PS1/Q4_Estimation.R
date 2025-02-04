rm(list = ls())
pacman::p_load(data.table, stargazer)
# ---- q4: estimate the parameters of the model
load("Data/Out/q3.RData")
v_diff_matrix <- v_1_matrix - v_0_matrix
dt2 <- cbind(dt, v_diff_matrix)
dt2$choice <- as.factor(dt2$choice)
logit <- glm(choice ~ x_i1 + x_i2 - 1, offset = x_i3, data = dt2, family = binomial(link = "logit"))
summary(logit)
dt2[, prob := predict(logit, dt2, type = "response")]
saveRDS(dt2, "Data/Out/data_q3q4.RDS")
stargazer::stargazer(logit, type = "latex", float = FALSE, covariate.labels = c("RC", "$\\mu$"), out = "Tables/q4_est_results.tex")
