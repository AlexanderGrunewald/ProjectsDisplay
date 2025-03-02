library(dplyr)
library(tidyverse)
library(VIM)
library(faraway)
library(ggplot2)
library(naniar)
library(GGally)
library(caret)
library(fastDummies)
library(leaps)
library(recipes)
library(rstan)
library(tree)
library(bayesplot)
library(mvtnorm)



diabetes$diabetesT <- as.factor(ifelse(diabetes$glyhb >= 7, 1, 0))
ggplot(data = diabetes, mapping = aes(x= diabetesT)) + 
  geom_bar()

stab.gender.gplot <- ggplot(data= diabetes, mapping = aes(x=gender, y= stab.glu)) + geom_boxplot() + facet_wrap(~location) + labs(
  x = "Gender",
  y = "Stable Glucose Level",
  title = "Glucose Level by Gender and Location"
  
)
ggsave("StabGlucoseLevelGenderLocation.png", plot = stab.gender.gplot, width = 8, height = 6)

plot <- ggplot(data = diabetes, mapping = aes(x = gender, y = glyhb)) +
  geom_boxplot() +
  facet_wrap(~location) +
  labs(
    x = "Gender",
    y = "GLYHB Level",
    title = "GLYHB Level by Gender and Location"
  )

ggsave("GLYHBLevelGenderLocation.png", plot = plot, width = 8, height = 6)

par(mfrow=c(1,2)) 
split_data <- split(log(diabetes$glyhb), diabetes$diabetesT)
hist(split_data[[1]], main="Log(glyhb) for diabetesT=0", xlab="log(glyhb)")
hist(split_data[[2]], main="Log(glyhb) for diabetesT=1", xlab="log(glyhb)")

ggplot(data = diabetes, mapping = aes(x=stab.glu, y=glyhb, alpha = 0.5)) + geom_point(aes(size = age, color = gender))


glyhb.stab.scatter <- ggplot(data = diabetes, mapping = aes(x= age, y = log(glyhb))) + geom_point(aes(size = as.numeric(ratio), colour = as.numeric(stab.glu),alpha = 0.5)) + scale_colour_binned() +
  facet_wrap(~gender) + labs(x="Age", y="GLYHB", title = "GLYHB age comparison", caption = "Those who are older, tend to have a higher glyhb level with a combination of stab glucose and high ratio") 

ggsave("GLYHBLevelGenderAgeStabGlu.png", plot = glyhb.stab.scatter, width = 8, height = 6)

scatter.glyhb <- ggplot(data = diabetes, mapping = aes(x=waist, y=glyhb, size = as.numeric(stab.glu), colour = location, alpha = 0.5)) + geom_point() + facet_wrap(~gender)
ggsave("GLYHBLevelGenderLocationScatter.png", plot = scatter.glyhb, width = 8, height = 6)

diabetes_clean <- diabetes %>% 
  dplyr::select(-c(id, location, gender, frame, diabetesT)) %>% 
  filter(complete.cases(.))

cor_matrix <- cor(diabetes_clean)
diag(cor_matrix) <- 0 
cor_data <- as.data.frame(as.table(cor_matrix))

ggplot(cor_data, aes(Var1, Var2, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", 
                       name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1)) +
  coord_fixed()

diabetes_clean_T <- diabetes %>% 
  filter(diabetesT == 1) %>%
  dplyr::select(-c(id, location, gender, frame, diabetesT)) %>% 
  filter(complete.cases(.))

cor_matrix_T <- cor(diabetes_clean_T)
diag(cor_matrix_T) <- 0 
cor_data_T <- as.data.frame(as.table(cor_matrix_T))

diabetes_clean_F <- diabetes %>% 
  filter(diabetesT == 0) %>%
  dplyr::select(-c(id, location, gender, frame, diabetesT)) %>% 
  filter(complete.cases(.))

cor_matrix_F <- cor(diabetes_clean_F)
diag(cor_matrix_F) <- 0 
cor_data_F <- as.data.frame(as.table(cor_matrix_F))

ggplot(cor_data_F, aes(Var1, Var2, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", 
                       name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1)) +
  coord_fixed() +
  ggtitle("Correlation of Non-Diabetes Patients")

par(mfrow=c(1,2))

ggplot(cor_data_T, aes(Var1, Var2, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", 
                       name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1)) +
  coord_fixed() +
  ggtitle("Correlation of Diabetes Patients")

# Preprocessing ---------------------
diabetes.drop <- diabetes %>% filter(!is.na(glyhb)) %>% dplyr::select(-c(diabetesT, id))

X <- subset(diabetes.drop, select = -c(glyhb)) #%>% dummy_columns(remove_first_dummy = T, remove_selected_columns = T, ignore_na = T)
y <- log(diabetes.drop$glyhb)

recipe <- recipe(~ ., data = X) %>%
  step_impute_knn(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())  %>%
  step_dummy(all_factor_predictors())

prepared_recipe <- prep(recipe)
diabetes.clean <- bake(prepared_recipe, new_data = NULL)


set.seed(42)
train.indx <- createDataPartition(y = y, p = .8, list = FALSE)

X.train <-diabetes.clean[train.indx,]
X.test <- diabetes.clean[-train.indx,]

y.train <- y[train.indx] 
y.test <- y[-train.indx]

Xy.train <- cbind(X.train, y.train)
Xy.test <- cbind(X.test, y.test)


# Linear Modeling ------
linear.model <- lm(y.train~., data = Xy.train)

y.hat.lin <- predict(linear.model, X.test)
exp(sqrt(mean((y.hat.lin-y.test)^2)))

linear.model.interactions <- lm(y.train~stab.glu + age + time.ppn + bp.1d*chol+ gender_female, data = Xy.train)

# Subsets ------
interactions <- model.matrix(~(.)^2-1, data = X.train)

subset.model <- summary(regsubsets(x = interactions, y= y.train, method = "backward"))


best.model.index <- which.max(subset.model$adjr2)



best.model.variables <- colnames(interactions)[subset.model$which[best.model.index,]]


print(best.model.variables)

selected.variables <- best.model.variables

X.train.selected <- interactions[, c(selected.variables), drop = FALSE]#,"stab.glu", "gender_female", "location_Louisa", "waist", "bp.1s", "frame_medium"), drop = FALSE]

final.model <- lm(y.train ~ . ,data = as.data.frame(X.train.selected))

summary(final.model)

interactions.test <- model.matrix(~(.)^2-1, data = X.test)

X.test.selected <- as.data.frame(interactions.test[, c(selected.variables,"stab.glu", "gender_female", "location_Louisa", "waist", "bp.1s", "frame_medium"), drop = FALSE])
y.hat <- predict(final.model, X.test.selected)
exp(sqrt(mean((y.hat - y.test)^2)))


# Bayes Modeling -----------

X.train <- as.matrix(X.train); y.train <- as.matrix(y.train); X.test <- as.matrix(X.test)
s20 <- mean((y.test - X.test %*% solve(t(X.train) %*% X.train) %*% t(X.train) %*% y.train)^2)
n <- g <- length(y.train); p <- ncol(X.train); nu0 <- 1; S <- 1000

Hg <- g/(g+1) * X.train%*%solve(t(X.train)%*%X.train) %*% t(X.train) 
SSRg <- t(y.train) %*% (diag(1,nrow = n) - Hg) %*% y.train

s2 <- 1/rgamma(S, (nu0+n)/2, (nu0*s20+SSRg)/2)

Vb <- g*solve(t(X.train)%*%X.train)/(g+1)
Eb <- Vb%*%t(X.train)%*%y.train

E <- matrix(rnorm(S*p, 0, sqrt(s2)), S, p)
beta <- t( t(E%*%chol(Vb)) + c(Eb))

beta.estimates <- as.data.frame(beta) %>%
  summarise(across(
    everything(),
    list(
      mean = mean,
      `2.5%` = ~ quantile(.x, 0.025),
      `97.5%` = ~ quantile(.x, 0.975)
    ),
    .names = "{.col}_{.fn}"
  )) %>%
  pivot_longer(
    cols = everything(),
    names_to = "name",
    values_to = "value"
  ) %>%
  separate(
    col = name,
    into = c("beta", "stat"),
    sep = "_(?=[^_]+$)"
  ) %>%
  pivot_wider(
    names_from = stat,
    values_from = value
  ) %>%
  column_to_rownames(var = "beta")


beta.estimates <- beta.estimates %>%
  rownames_to_column(var = "beta") %>%
  mutate(significant = ifelse(`2.5%` > 0 | `97.5%` < 0, "Significant", "Not Significant"))


beta.est.plot <- ggplot(beta.estimates, aes(x = beta, y = mean, ymin = `2.5%`, ymax = `97.5%`, color = significant)) +
  geom_pointrange() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Beta Estimates with Confidence Intervals",
       x = "Beta",
       y = "Estimate") +
  theme_minimal() +
  scale_color_manual(values = c("Significant" = "red", "Not Significant" = "blue")) + coord_flip()
beta.est.plot

ggsave("betaEstimatesGPrior.png", plot = beta.est.plot)
B <- as.matrix(beta.estimates$mean, nrow=nrow(beta.estimates), ncol=1)
y.hat.bayes <- X.test %*% B
exp(mean((y.hat.bayes - y.test)^2))



# Model Sleection --=-------
# RELEVANT PIECE
X.train <- as.matrix(X.train)

###### Bayesian model averaging (slide 60 -62)
## the sampling part for z is the same as before
source("regression_gprior.R") # Inclass code
S <- 1000
BETA <- Z <- matrix(NA, S, dim(X.train)[2]) ## store the MCMC samples
z <- rep(1, dim(X.train)[2])  
lpy.c <- lpy.X(y.train, X.train[, z == 1, drop = FALSE])
print("Entering loop")
for(s in 1 : S){
  print(cat("Iteration:", s, "\n"))
  for(j in sample(1 : dim(X.train)[2])){
    zp <- z; zp[j] <- 1 - zp[j];
    lpy.p <- lpy.X(y.train, X.train[, zp == 1, drop = FALSE])
    r <- (lpy.p - lpy.c) * (-1) ^ (zp[j] == 0)
    z[j] <- rbinom(1, 1, 1 / (1 + exp(-r)))  
    if(z[j] == zp[j]) {lpy.c <- lpy.p}
  }
  beta <- z
  ## the function lm.gprior generates samples for beta and sigma^2
  if(sum(z) > 0) {beta[z == 1] <- lm.gprior(y.train, X.train[, z == 1, drop = FALSE], S = 1)$beta}
  
  Z[s, ] <- z
  BETA[s, ] <- beta
}

betas <- as.matrix(apply(BETA, MARGIN = 2, FUN = mean))
inclusion <- as.vecotr(apply(Z, MARGIN = 2, FUN = mean))

x.selected <- as.matrix(X.test)[,inclusion > 0.5]

subset(X.train, select = -which(inclusion > 0.5))
bayesplot::mcmc_intervals(as.data.frame(BETA))

# HEREEEEEEEEEEEEEEEEE
iterations <- 1:nrow(params$beta)

# Prepare data for beta convergence plot
beta_data <- data.frame(
  Iteration = rep(iterations, times = ncol(params$beta)),
  Parameter = rep(paste0("beta[", 1:ncol(params$beta), "]"), each = length(iterations)),
  Value = as.vector(params$beta)
)

beta_plot <- ggplot(beta_data, aes(x = Iteration, y = Value, color = Parameter)) +
  geom_line(alpha = 0.7) +
  labs(
    title = "Convergence of Beta Coefficients",
    x = "Iteration",
    y = "Value",
    color = "Parameter"
  ) +
  theme_minimal()

# Print and save the plot
print(beta_plot)
ggsave("convergence_beta_plot.png", plot = beta_plot, width = 8, height = 6)

## Convergence Plot for Inclusion Probabilities
# Prepare data for inclusion_probs convergence plot
inclusion_data <- data.frame(
  Iteration = rep(iterations, times = ncol(params$inclusion_probs)),
  Parameter = rep(paste0("inclusion_probs[", 1:ncol(params$inclusion_probs), "]"), each = length(iterations)),
  Value = as.vector(params$inclusion_probs)
)

# Plot convergence for inclusion probabilities
inclusion_plot <- ggplot(inclusion_data, aes(x = Iteration, y = Value, color = Parameter)) +
  geom_line(alpha = 0.7) +
  labs(
    title = "Convergence of Inclusion Probabilities",
    x = "Iteration",
    y = "Value",
    color = "Parameter"
  ) +
  theme_minimal()

# Print and save the plot
print(inclusion_plot)
ggsave("convergence_inclusion_plot.png", plot = inclusion_plot, width = 8, height = 6)

# Assuming beta.estimates contains the mean, 2.5%, and 97.5% CI values
# Calculate whether the estimate is significant

# Prepare beta.estimates data frame
beta.estimates <- data.frame(
  beta = colnames(X.train)[1:5],  
  mean = colMeans(params$beta),
  X2.5. = apply(params$beta, 2, quantile, 0.025),
  X97.5. = apply(params$beta, 2, quantile, 0.975)
)


beta.estimates <- beta.estimates %>%
  mutate(significant = ifelse(X2.5. > 0 | X97.5. < 0, "Significant", "Not Significant"))

colnames(beta.estimates)[colnames(beta.estimates) == "X2.5."] <- "lower"
colnames(beta.estimates)[colnames(beta.estimates) == "X97.5."] <- "upper"


beta.est.plot <- ggplot(beta.estimates, aes(x = beta, y = mean, ymin = lower, ymax = upper, color = significant)) +
  geom_pointrange() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title = "Beta Estimates with Confidence Intervals",
    x = "Beta",
    y = "Estimate"
  ) +
  theme_minimal() +
  scale_color_manual(values = c("Significant" = "red", "Not Significant" = "blue")) +
  coord_flip()

print(beta.est.plot)

ggsave("beta_estimates_plot.png", plot = beta.est.plot, width = 8, height = 6)
