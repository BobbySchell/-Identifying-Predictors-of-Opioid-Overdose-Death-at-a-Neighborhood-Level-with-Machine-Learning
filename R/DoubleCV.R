# --------------------------------------
# Main Analysis Script for "Identifying Predictors of Opioid Overdose Death with Machine Learning"
# Double Cross-Validation and Calibration Plots
# Excludes Elastic Net Sensitivity Analysis and Map Codes
# --------------------------------------


# Setting Seed to ensure folds split in same way when rerun
set.seed(2021)

# First randomizing order of rows before creating folds (in case ID provides meaningful info)
randomrowdf <- df[sample(nrow(df), replace=F),]


# break data frame into 4 outer folds by indexing data from 1 to 4
outerfoldindex <- cut(seq(1,nrow(randomrowdf)), breaks=4, labels=F) # Creates the indices used in for loop below 


# Empty data frames to store predictive performance of LASSO model only on each of the four outer folds
rsqLASSO <- data.frame(matrix(NA,nrow = 4, ncol = 2))
MSELASSO <- data.frame(matrix(NA,nrow = 4, ncol = 2))


# Empty data frames to store predictive performance of entire model on each of the four outer folds
MSE <- data.frame(matrix(NA,nrow = 4, ncol = 2))
rsq <- data.frame(matrix(NA,nrow = 4, ncol = 2))

# Object to store variable importance on each of the outer folds
outerimp <- list()

# Object to store outer prediction results to be used in calibration plot
outerpredict <- list()

# Object to store coefficients that remain after regularization for each outer fold
thenonzerocoef <- list()

# Object to store y values from testing set for each outer fold
yactual <- list()

# Object to store lambda that minimizes cross-validated MSE 
MinLambda <- data.frame(matrix(NA, nrow = 4, ncol = 1))

# Create object to store results of Random Forest hyperparameter tuning function w/ appropriate column names
rfparams <- data.frame(matrix(NA,nrow = 8, ncol = 3))
colnames(rfparams) <- c("MSE","mtry","ntree")

# Create object to store parameters from RF w/ lowest MSE from each of the 10 inner folds
# Will select mode mtry and ntree FROM THIS
rfparamssaved <- data.frame(matrix(NA, nrow = 10, ncol = 2))



system.time(
# Creating nested for loop to run double cross-validation process
for (i in 1:4) {
  
  # select the training and test data for this fold based on outer fold index
  testindex <- which(outerfoldindex==i, arr.ind=T)
  
  train <- randomrowdf[-testindex,]
  test <- randomrowdf[testindex,]
  
  # Create the 10 inner folds (within each outer fold)
  foldstrain <- cut(seq(1,nrow(train)), breaks=10, labels=F)
  
  
  # separating dependent and independent variables (for glmnet) - training set
  xtrain <- as.matrix(train[,!colnames(train) %in% "OpioidRate"])
  ytrain <- as.matrix(train[,"OpioidRate"])
  
  # separating dependent and independent variables - testing set
  xtest <- as.matrix(test[,!colnames(test) %in% "OpioidRate"])
  ytest <- as.matrix(test[,"OpioidRate"])
  
  # Training LASSO Regression on training sets created by inner folds and evaluating on inner fold testing sets
  # Unlike RF (which is implemented manually via function) glmnet has built-in CV functionality
  cv.fit <- glmnet::cv.glmnet(x=xtrain,y=ytrain, alpha=1, type.measure="mse", foldid=foldstrain, family="gaussian")

  MinLambda[i,1] <- print(cv.fit$lambda.min)
  # Saving value of lambda that minimizes cross-validated MSE from inner folds
  
  # Predictive performance of LASSO Regression in outer fold test set
  predsLASSO <- predict(cv.fit, xtest, type = "response")
  
  # Saving performance metrics of LASSO on outer fold test set
  rsqLASSO[i, 1] <- sum((predsLASSO-mean(ytest))^2)/sum((ytest-mean(ytest))^2)
  MSELASSO[i, 1] <- mean((predsLASSO-ytest)^2)
  
  # Show list of coefficients that do/don't shrink to zero at optimal cross-validated lambda
  cfs <- coef(cv.fit, s="lambda.min")
  print(cfs)
  
  # select coefficients that are nonzero after shrinkage
  nonzero <- (rownames(cfs)[abs(cfs[,1])>0])
  
  # Saving list of nonzero coefficients for all outer folds
  thenonzerocoef[[i]] <- nonzero
  
  # Adding outcome to variable list
  nonzero <- c(nonzero, "OpioidRate")
  

  # setting up training and testing sets for outer folds EXCLUDING coeffs shrunk by LASSO
  train0 <- train[,colnames(train) %in% nonzero]
  test0 <- test[,colnames(test) %in% nonzero]
  
  xtrain0 <- as.matrix(train0[,!colnames(train0) %in% "OpioidRate"])
  xtest0 <- as.matrix(test0[,!colnames(test0) %in% "OpioidRate"])
  

  # Initializing the inner loop - repeat 10 times for EVERY outer fold
  # (Did so for LASSO using glmnet built-in funct above)
  # NOW inner fold hyperparameter tuning for Random Forest algorithm using own function
  
  for (j in 1:10) {
    
    # select the training and test data for this fold
    innerindex <- which(foldstrain==j, arr.ind=T)
    
    innertrain <- train0[-innerindex, ]
    innertest <- train0[innerindex, ]
    
    
    # separating dependent and independent vars - training set
    innerxtrain <- as.matrix(innertrain[,!colnames(innertrain) %in% "OpioidRate"])
    innerytrain <- as.matrix(innertrain[,"OpioidRate"])
    
    # separating dependent and  independent vars - testing set
    innerxtest <- as.matrix(innertest[,!colnames(innertest) %in% "OpioidRate"])
    #create vector of outcome values 
    innerytest <- as.matrix(innertest[,"OpioidRate"])
    
    
    # Tuning Random Forest
    # fit random forest
    # ntree is the number of trees grown (generally, larger is better w/ computation tradeoff...)
    # mtry is the number of covariates selected as candidates for each split
    rfs <- create_rf_alg_list(x = innerxtrain, y = innerytrain, mtry = c(3,6,9,12), ntree = c(1000,5000), ytest = innerytest, xtest = innerxtest)
    

    # Selecting optimal hyperparameters based on inner fold cross-validation!
    # PREDICTION RESULTS FOR EACH HYPERPARAM COMBO FROM RF.FITTRAIN... store then select based on this...



    # saving ideal hyperparameters (that minimize CV MSE) from each inner fold and saving
    rfparamssaved[j, 1] <- rfs$mtry
    rfparamssaved[j, 2] <- rfs$ntree
    

  }
  
  # Take mode to find optimal configuration of hyperparams for each ind outerfold (rfparammssaved is 10x2 df)
  mtry <- Mode(rfparamssaved[ , 1])
  ntree <- Mode(rfparamssaved[ , 2])
  
  # Save the optimal hyperparameters for EACH outer fold
  print(mtry)
  print(ntree)
  

  # Rerun Random Forest evaluated on outer fold for performance metrics
  rf.outer <- randomForest(x=xtrain0, y=ytrain, ntree=ntree, mtry=mtry, keep.forest=T,importance=T)

  # Variable importance results over each outer fold
  outerimp[[i]] <-importance(rf.outer)
  
  # Predictions based on test set
  outerpred <- predict(rf.outer, xtest0, type = "response")

  
  # R-squared and MSE from outer loop then averaged...
  MSE[i, 1] <- mean((outerpred-ytest)^2)
  rsq[i, 1] <- sum((outerpred-mean(ytest))^2)/sum((ytest-mean(ytest))^2)
  
  # Saving results for calibration plot
  outerpredict[[i]] <- predict(rf.outer, xtest0, type = "response")
  yactual[[i]] <- ytest

}
)

# ----------------
# Calibration Plot
# ---------------
yactual1 <- as.data.frame(yactual[1])
yactual2 <- as.data.frame(yactual[2])
yactual3 <- as.data.frame(yactual[3])
yactual4 <- as.data.frame(yactual[4])

# Changing colnames
colnames(yactual1) <- "Actual"
colnames(yactual2) <- "Actual"
colnames(yactual3) <- "Actual"
colnames(yactual4) <- "Actual"


# Combining w rbind
yactualfull <- rbind(yactual1, yactual2, yactual3, yactual4)


outerpredict1 <- as.data.frame(outerpredict[1])
outerpredict2 <- as.data.frame(outerpredict[2])
outerpredict3 <- as.data.frame(outerpredict[3])
outerpredict4 <- as.data.frame(outerpredict[4])

# Changing colnames
colnames(outerpredict1) <- "Pred"
colnames(outerpredict2) <- "Pred"
colnames(outerpredict3) <- "Pred"
colnames(outerpredict4) <- "Pred"


# Combining w rbind
outerpredfull <- rbind(outerpredict1, outerpredict2, outerpredict3, outerpredict4)

# Dataset combining all of this
fullcalibdata <- cbind(yactualfull, outerpredfull)

# predicted and testedy above
lm(Actual ~ Pred, data = fullcalibdata)
# Intercept = 0.35 and Slope = 0.73

y <- c(0.35,1.08,1.81)
x <- c(0,1,2)

df <- cbind.data.frame(y,x)

lm(y~x, data = df)
eq <- function(x,y) {
  m <- lm(y~x, data = df)
  as.character(
    as.expression(
      substitute(italic(y) == a+b %.% italic(x),
                 list(a = format(coef(m)[1], digits = 2),
                      b = format(coef(m)[2], digits = 2)))
    )
  )
}

# Creating eqtn to display on calibration plot
equation <- as.character(as.expression(substitute(italic(y) == a+b %.% italic(x),
                       list(a = 0.35,
                            b = 0.71))))

as.character(as.expression(equation))

calibplot <- ggplot(data = fullcalibdata, aes(x = Pred, y = Actual)) + geom_point() + 
  labs(title = "Calibration Plot", x = "Predicted Opioid Overdose Death Rate", y = "Actual Opioid Overdose Death Rate") + theme_classic() + geom_abline(intercept = 0.35, slope = 0.73)

calibplot + scale_y_continuous(limits = c(0,5), expand = c(0,0)) + scale_x_continuous(limits = c(0,3), expand = c(0,0)) + theme(axis.title = element_text(size = 12), axis.text = element_text(size = 12, colour = "Black"), title = element_text(size = 14)) 




