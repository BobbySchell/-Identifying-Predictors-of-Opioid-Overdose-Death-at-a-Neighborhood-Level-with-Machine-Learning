# ---------------
# Functions to insert RF into double CV design
# ---------------

# Specifying initial Random Forest function
RFFit <- function(mtry, ntree, x, y){
  randomForest(x=x, y=y, mtry = mtry, ntree = ntree)

  }

# Applying hyperparameter range to RF
RFModels <- mapply(RFFit, mtry = c(3,6,9,12), ntree = c(1000,5000), x = xtrain, y = ytrain)

# Then predict overdose deaths with these hyperparameters
predictions <- sapply(RFModels, predict, newdata = xtest)

# Setting column names for rfparam object
colnames(rfparam) <- c("MSE","mtry","ntree")

# Saving MSE/mtry/ntree for each algorithm
for(i in 1:length(predictions)){
  
  # Store MSE results and hyperparameters in data frame together
  rfparam[i,1] <- mean((predictions[i] - ytest))^2
  rfparam[i,2] <- RFModels[i]$mtry
  rfparam[i,3] <- RFModels[i]$ntree
  
  rfparambest <- subset(rfparam, MSE == min(MSE))
  

  return(rfparambest)
}


# Function to get the mode (optimal parameter tuning for RF)
Mode <- function(x) {
  keys <- unique(x)
  keys[which.max(tabulate(match(x, keys)))]
}
