#import require library
library(AUC)
library(onehot)
library(xgboost)

#normalize Function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

#read file
X_train <- read.csv(sprintf("hw08_training_data.csv"), header = TRUE)
X_train_label <- read.csv(sprintf("hw08_training_label.csv"), header = TRUE)
X_test  <- read.csv(sprintf("hw08_test_data.csv"), header = TRUE)

#removes the first variable(id) from the data set.
X_train <- X_train[,-1]

encoder <- onehot(X_train, addNA = TRUE, max_levels = Inf)

#X_train_prediction generate
X_train_prediction = predict(encoder,X_train)

#X_test_prediction generate
X_test_prediction = predict(encoder,X_test[,-1])

#get row number
X_test_nrow <- nrow(X_test_prediction)

#get col number
X_train_label_col <- ncol(X_train_label)

#create test_predict matrix[,]
test_predict <- matrix(0, nrow = X_test_nrow, ncol = X_train_label_col)
colnames(test_predict) <- colnames(X_train_label)
test_predict[,1] <- X_test[, 1]

for (outcome in 1:6) {
    valid_customers <- which(is.na(X_train_label[,outcome + 1]) == FALSE)
    
    xgboost_model  <- xgboost(
                        data = X_train_prediction[valid_customers, -1], 
                        label = X_train_label[valid_customers, outcome + 1], 
                        nrounds = 10, 
                        objective= "binary:logitraw"
                      )
    
    #Training Score Calculator
    training_scores <- predict(xgboost_model, X_train_prediction[valid_customers, -1])
    
    # AUC score for training data
    print(auc(roc(predictions = training_scores, labels = as.factor(X_train_label[valid_customers, outcome + 1]))))
    
    #Test scoring calculator
    test_scores <- predict(xgboost_model, X_test_prediction[,-1])
    
    #Test scoring convert normalize
    test_scores_normalize <- normalize(test_scores)
    
    test_predict[, outcome + 1] <- test_scores_normalize
}

write.table(test_predict, file = "hw08_test_predictions.csv", row.names = FALSE, sep = ",")


