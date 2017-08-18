set.seed(3719)
n <- 2000
# Generate variables x1, ... x10
X <- matrix(rnorm(10*n), n, 10)
# y = +1 if sum_i x_{ij}^2 > chisq median on 10 df
y <- rep(-1, n)
y[apply(X*X, 1, sum) > qchisq(.5, 10)] <- 1
# Assign names to the columns of X:
dimnames(X)[[2]] <- c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10")
# Convert to data.frame
train.data <- as.data.frame(X)
# Add y
train.data$y <- y
# Now repeat for 10000 test data
n <- 10000
X <- matrix(rnorm(10*n), n, 10)
y <- rep(-1, n)
y[apply(X*X, 1, sum) > qchisq(.5, 10)] <- 1
dimnames(X)[[2]] <- c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10")
test.data <- as.data.frame(X)
test.data$y <- y
# Need to put training and test data together for gbm below and convert
# to 0-1 data
train.data2 <- train.data
train.data2$y[train.data2$y < 0] <- 0
test.data2 <- test.data
test.data2$y[test.data2$y < 0] <- 0
all.data2 <- rbind(train.data2, test.data2)
# First try a classification tree
library(rpart)
fit.rp <- rpart(as.factor(y) ~ ., train.data, control=rpart.control(cp=.001))
plot(fit.rp)
text(fit.rp)
plotcp(fit.rp)
# CV choice: somewhere around .005
fit.rp2 <- rpart(as.factor(y) ~ ., train.data, control=(cp=.005))
plot(fit.rp2)
text(fit.rp2)
confusion <- function(a, b){
  tbl <- table(a, b)
  mis <- 1 - sum(diag(tbl))/sum(tbl)
  list(table = tbl, misclass.prob = mis)
}
confusion(predict(fit.rp2, test.data, type="class"), as.factor(test.data$y))
# Pretty bad predictor
###############################################################################
# Boosting program
###############################################################################
library(gbm)
# Start with stump classifier
# Note that I've put the test data with the training data. The
# train.fraction is 2000/12000. For now, we take shrinkage = 1.
fit.gbm1 <- gbm(y ~ ., data=all.data2, dist="adaboost", n.tree = 400,
                shrinkage = 1, train.fraction = (2/12))
# Set to hit return for next plot
par(ask=T)
# gbm.perf returns estimated best number of trees. This is a bit misleading
# since we're using the test sample both to validate and to test.
gbm.perf(fit.gbm1)

# Or maybe 200 from plot
# Here is the best predictor:
confusion(predict(fit.gbm1, test.data2, n.trees = 69) > 0, test.data2$y > 0)

confusion(predict(fit.gbm1, test.data2, n.trees = 200) > 0, test.data2$y > 0)
# Even with 400 trees, not seriously overfit
confusion(predict(fit.gbm1, test.data2) > 0, test.data2$y > 0)
# Note that we have almost perfect classification on training sample
confusion(predict(fit.gbm1, train.data2) > 0, train.data2$y > 0)
# Try again with depth 10 in the trees. Note that this goes bad quickly.
fit.gbm2 <- gbm(y ~ ., data=all.data2, dist="adaboost", n.tree = 50,
                shrinkage = 1, train.fraction = (2/12), interaction.depth = 10)
gbm.perf(fit.gbm2, method="test")

confusion(predict(fit.gbm2, test.data2, n.trees = 13) > 0, test.data2$y > 0)
confusion(predict(fit.gbm2, train.data2, n.trees = 13) > 0, train.data2$y > 0)
# By construction, this is a special case where fitting an additive model
# is great.
###############################################################################
# Now rework with binomial loss
fit.gbm3 <- gbm(y ~ ., data=all.data2, dist="bernoulli", n.tree = 400,
                shrinkage = 1, train.fraction = (2/12))
gbm.perf(fit.gbm3, method="test")
confusion(predict(fit.gbm3, test.data2, n.trees = 255) > 0, test.data2$y > 0)
