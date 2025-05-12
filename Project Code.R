install.packages("AER")
library(survival)
library(sandwich)
library(AER)

data("DoctorVisits")
head(DoctorVisits)
names(DoctorVisits)
str(DoctorVisits)

# No Need
DoctorVisits$freepoor  <- as.numeric(as.factor(DoctorVisits$freepoor))  # "no" -> 1, "yes" -> 2
DoctorVisits$freerepat <- as.numeric(as.factor(DoctorVisits$freerepat)) # "no" -> 1, "yes" -> 2
DoctorVisits$private   <- as.numeric(as.factor(DoctorVisits$private))   # "no" -> 1, "yes" -> 2
DoctorVisits$gender    <- as.numeric(as.factor(DoctorVisits$gender))    # "male" -> 1, "female" -> 2
DoctorVisits$nchronic <- as.numeric(as.factor(DoctorVisits$nchronic))   # "no" -> 1, "yes" -> 2
DoctorVisits$lchronic <- as.numeric(as.factor(DoctorVisits$lchronic))   # "no" -> 1, "yes" -> 2

head(DoctorVisits)


##split the dataset 80% and 20%
install.packages("caret")
library(ggplot2)
library(lattice)
library(caret)

set.seed(123)

n <- nrow(DoctorVisits)
train_indices <- sample(1:n, size = 0.8 * n)

train_data <- DoctorVisits[train_indices, ]
test_data  <- DoctorVisits[-train_indices, ]

dim(train_data)
dim(test_data)



dv1 <- glm(visits ~ age + gender + income + illness + reduced + health + private + freepoor + freerepat 
           + nchronic + lchronic, data = train_data, family = poisson(link = "log"))
summary(dv1)
BIC1 <- BIC(dv1)
print(BIC1)


dv2 <- glm(visits ~ gender + illness + reduced + health + freepoor + lchronic,
           data = train_data, family = poisson(link = "log"))
summary (dv2)
BIC2 <- BIC(dv2)
print(BIC2)

dv3 <- glm(visits ~ age * gender + income*private + illness + reduced + health + freepoor 
           +nchronic + lchronic, data = train_data, family = poisson(link = "log"))
summary(dv3)
BIC3 <- BIC(dv3)
print(BIC3)



library(MASS)  

# Fit the null model (no predictors)
null_model <- glm(visits ~ 1, family = poisson(link = "log"), data = train_data)

dv4 <- step(null_model, scope = list(lower = null_model, upper = dv1), 
                      direction = "forward")
summary(dv4)

# Calculate BIC
forward_model <- step(null_model, scope = list(lower = null_model, upper = dv1), 
                      direction = "forward", k = log(nrow(DoctorVisits)))
summary(forward_model)

#Backward
dv5 <- step(dv1, direction = "backward")
summary(dv5)

#Calculate BIC
backward_model <- step(dv1, direction = "backward", k = log(nrow(DoctorVisits)))
summary(backward_model)

#Step-wise
dv6 <- step(dv1, direction = "both")
summary(dv6)

#Calculate BIC
stepwise_model <- step(dv1, direction = "both", k = log(nrow(DoctorVisits)))
summary(stepwise_model)

dv7 <- glm(visits ~ age^2 + gender + income^2 * private^3 + illness + reduced + health + freepoor 
           +nchronic^2 + lchronic, data = train_data, family = poisson(link = "log"))
summary(dv7)
BIC7 <- BIC(dv7)
print(BIC7)


pq1 <- sum(residuals(dv1, type = "pearson")^2)
pq1

pq2 <- sum(residuals(dv2, type = "pearson")^2)
pq2

pq3 <- sum(residuals(dv3, type = "pearson")^2)
pq3

pq4 <- sum(residuals(dv4, type = "pearson")^2)
pq4

pq5 <- sum(residuals(dv5, type = "pearson")^2)
pq5

pq6 <- sum(residuals(dv6, type = "pearson")^2)
pq6

pq7 <- sum(residuals(dv7, type = "pearson")^2)
pq7







dp3 <- sum(residuals(dv3, type = "pearson")^2) / dv3$df.residual
dp3

p_value3 <- pchisq(deviance(dv3), df.residual(dv3), lower.tail = FALSE)
print(p_value3)

dp5 <- sum(residuals(dv5, type = "pearson")^2) / dv5$df.residual
dp5

p_value5 <- pchisq(deviance(dv5), df.residual(dv5), lower.tail = FALSE)
print(p_value5)

dp6 <- sum(residuals(dv6, type = "pearson")^2) / dv6$df.residual
dp6

p_value6 <- pchisq(deviance(dv6), df.residual(dv6), lower.tail = FALSE)
print(p_value6)



## No Test
lrt_statistic <- 2 * (logLik(dv1) - logLik(dv3))
df_difference <- attr(logLik(dv1), "df") - attr(logLik(dv3), "df")
p_value <- pchisq(lrt_statistic, df_difference, lower.tail = FALSE)
p_value

##
lrt_result <- anova(dv1, dv2, dv3, dv4, dv5, dv6, dv7, test = "Chisq")
print(lrt_result)


## Negative - Binomial
install.packages("MASS")
library(MASS)

# Fit Negative Binomial Model
dv8 <- glm.nb(visits ~ age + gender + income + illness + reduced + health + private + freepoor + freerepat 
              + nchronic + lchronic, data = train_data)
summary (dv8)

dv9 <- glm.nb(visits ~ age * gender + income*private + illness + reduced + health + freepoor 
              +nchronic + lchronic, data = train_data)
summary(dv9)

deviance_nb <- sum(residuals(dv9, type = "pearson")^2)
df_nb <- dv9$df.residual
dispersion_nb <- deviance_nb / df_nb
dispersion_nb

p_value <- pchisq(deviance(dv9), df.residual(dv9), lower.tail = FALSE)
print(p_value)

install.packages("pscl")
library(pscl)
mcfadden_r2 <- pR2(dv9)["McFadden"]
print(mcfadden_r2)

# Predict on Train data
train_data$predicted_nb <- predict(dv9, newdata = train_data, type = "response")

# Predict on Test data
test_data$predicted_nb <- predict(dv9, newdata = test_data, type = "response")

install.packages("Metrics")
library(Metrics)

rmse(test_data$visits, test_data$predicted_nb)
rmse(train_data$visits, train_data$predicted_nb)

mae(test_data$visits, test_data$predicted_nb)
mae(train_data$visits, train_data$predicted_nb)


#Random Forest Model
install.packages("randomForest")
library(randomForest)

# Train Random Forest model
dv10 <- randomForest(visits ~ age * gender + income*private + illness + reduced + health + freepoor 
                    +nchronic + lchronic, data = train_data, ntree = 500, mtry = 3)
summary(dv10)

# Predict on test data
test_data$predicted_rf <- predict(dv10, newdata = test_data)

# Predict on Train data
train_data$predicted_rf <- predict(dv10, newdata = train_data)

# Evaluate RMSE
rmse(test_data$visits, test_data$predicted_rf)
rmse(train_data$visits, train_data$predicted_rf)

mae(test_data$visits, test_data$predicted_rf)
mae(train_data$visits, train_data$predicted_rf)


#XGBoost
# Install and load necessary packages
install.packages("xgboost")
library(xgboost)

# Convert dataset to matrix format (XGBoost requires numeric input)
train_matrix <- model.matrix(visits ~ . -1, data = train_data)
test_matrix  <- model.matrix(~ . - 1, data = test_data[, colnames(train_data)])
# Convert target variable to a matrix
train_label <- train_data$visits
test_label  <- test_data$visits

# Convert data into DMatrix format for XGBoost
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest  <- xgb.DMatrix(data = test_matrix, label = test_label)

# Train XGBoost model
xgb_model <- xgboost(data = dtrain, nrounds = 100, objective = "reg:squarederror")

# Predict on test data
test_data$predicted_xgb <- predict(xgb_model, newdata = dtrain)

# Evaluate RMSE
rmse(test_data$visits, test_data$predicted_xgb)




