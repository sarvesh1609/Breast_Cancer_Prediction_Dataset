# Breast Cancer Prediction Dataset

# 1. First Stage :- All the necessary pacakages installations

install.packages("jsonlite")
install.packages("corrplot")
install.packages("caret")
install.packages("e1071")
install.packages("rpart")
install.packages("nnet")
install.packages( "yardstick")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("caTools")
install.packages("class")
install.packages("randomForest")
install.packages("GGally")
install.packages("pheatmap")


############################## First Stage End ##################################################

#################################################################################################

# 2. Second stage :- loading the Libraries 

library(jsonlite)
library(corrplot)
library(caret)
library(e1071)
library(rpart)
library(nnet)
library(yardstick)
library(dplyr)
library(ggplot2)
library(caTools)
library(class)
library(randomForest)
library(GGally)

############################## Secound Stage End ##################################################

#################################################################################################

# 3. Third stage :- Load necessary data sets 

#we have two dataset. 
#1. CSV File :- Breast cancer dataset comprises six columns, including 'diagnosis', 
#   where a value of 0 indicates the absence of cancer, while a value of 1 signifies the presence of cancer.

# Song data - First file about the track along with the genre
Breast_cancer_dataset <- read.csv("/Users/Breast_cancer_data.csv")


############################## Third Stage End ##################################################

#################################################################################################

# 4. Fourth stage :- Data Exploration

# we have total 569 rows 
total_row_count <- nrow(Breast_cancer_dataset)
print(total_row_count)

# we have total 569 distinct rows 
distinct_rows <- unique(Breast_cancer_dataset)
count_distinct_rows <- nrow(distinct_rows)
print(count_distinct_rows)

str(Breast_cancer_dataset)

summary(Breast_cancer_dataset)

# lets check if the data is balanced or not 

# Create a color palette for 0 and 1
colors <- c("0" = "skyblue", "1" = "salmon")

# Plot
ggplot(Breast_cancer_dataset, aes(x = factor(diagnosis), fill = factor(diagnosis))) +
  geom_bar() +
  scale_fill_manual(values = colors) +  # Set bar colors
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5, size = 4) +  # Display counts on bars
  labs(title = "Count Plot of Diagnosis", x = "Diagnosis", y = "Count") +
  theme_minimal()  # Optional: Choose a theme

#cancer = 0 = 212
#cancer = 1 = 357

# Balancing looks fine. 

# checking the values of dataset

# Assuming your dataset is named 'Breast_cancer_dataset'
# and 'diagnosis' is a column in the dataset


# Create a pair plot
ggpairs(Breast_cancer_dataset, aes(color = as.factor(diagnosis)))

# From the graph it is observed that, there might be some error in diagnosis values. As from the graph
# we can see that cancer = 0 has high value of mean-radius, mean-texture, mean-perimeter and mean-smoothness values.
# this should be opposite, cancer=1 should have higher mean-radius, mean-texture, mean-perimeter and mean-smoothness values
# Lets fix this by interchanging the diagnosis values. 


############################## Fourth Stage End ##################################################

#################################################################################################

# 5. Reverse the values :- Reverting Diagnosis column.

# Reverting Diagnosis column
outputcolumn <- Breast_cancer_dataset$diagnosis
finaloutputcolumn <- numeric(length(outputcolumn))

for (i in seq_along(outputcolumn)) {
  if (outputcolumn[i] == 0) {
    finaloutputcolumn[i] <- 1
  } else if (outputcolumn[i] == 1) {
    finaloutputcolumn[i] <- 0
  }
}

Breast_cancer_dataset$diagnosis <- finaloutputcolumn

# Create a color palette for 0 and 1
colors <- c("0" = "skyblue", "1" = "salmon")

# Plot
ggplot(Breast_cancer_dataset, aes(x = factor(diagnosis), fill = factor(diagnosis))) +
  geom_bar() +
  scale_fill_manual(values = colors) +  # Set bar colors
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5, size = 4) +  # Display counts on bars
  labs(title = "Count Plot of Diagnosis", x = "Diagnosis", y = "Count") +
  theme_minimal()  # Optional: Choose a theme

#cancer = 0 = 357
#cancer = 1 = 212


# Create a pair plot
ggpairs(Breast_cancer_dataset, aes(color = as.factor(diagnosis)))

# Now looks better 

############################## Fifth Stage End ##################################################

#################################################################################################

# 6. Sixth stage :- correlation of Breast_cancer_dataset.

# 1. Find the correlation amoung the varaibles. The highely correlated variables should 
#    be removed and reduce the varaibles if possible. 


# Calculate the correlation matrix
numeric_data <- Breast_cancer_dataset[, !colnames(Breast_cancer_dataset) %in% c("diagnosis")]

# Calculate the correlation matrix
correlation_matrix <- cor(numeric_data)

# Print the correlation matrix
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap
corrplot(
  correlation_matrix,
  method = "color",        # Use color to represent correlation
  #type = "upper",          # Show upper triangle of the matrix
  order = "hclust",        # Use hierarchical clustering to reorder variables
  tl.col = "black",        # Set text label color to black
  tl.srt = 45,             # Rotate text labels for better visibility
  addCoef.col = "black"     # Add numeric values to the plot with black color
)

#Variables such as mean area, mean radius and mean perimeter are highly correlated to each other but as we 
# have very few features or varaibles so we cant remove any further, it will create problem in modelling

############################## Sixth Stage End ##################################################

#################################################################################################

# 7. Seventh stage :- Normalise merge data set.

# The next step is to normalize the data. our data can have various scale values present, 
# normalising the scale values helps preventing certain features from dominating others. 
# We will be using r scale function which uses Standardize features to normalise data. 

# First step would be creating two dataframes. 
# 1. With only numeric varaibles (8 varaibles) called as 'Predictors'.
# 2. With geners values (column name genre_top) called as 'Class'.

# Drop 'genre_top' and 'track_id' columns to create Predictors
Predictors <- Breast_cancer_dataset[, !(names(Breast_cancer_dataset) %in% c('diagnosis'))]

# Create Class using the 'genre_top' column
Class <- Breast_cancer_dataset$diagnosis



# Secound step would be Standardize the features using the scale function.

# Standardize the features using the scale function
Scaled_train_Predictors <- scale(Predictors)

# Display the first few rows of the scaled features

head(Scaled_train_Predictors)

# Convert the scaled matrix to a data frame
Scaled_train_Predictors <- as.data.frame(Scaled_train_Predictors)


############################## Seventh Stage End ##################################################

#################################################################################################

# 8. Eight stage :- Split the data named Train_Predictors, Test_Predictors, Train_Class, and Test_Class.



# Set the seed for reproducibility
set.seed(10)

# Split the data
split <- sample.split(Class, SplitRatio = 0.7)
Train_Predictors <- Scaled_train_Predictors[split, ]
Test_Predictors <- Scaled_train_Predictors[!split, ]
Train_Class <- Class[split]
Test_Class <- Class[!split]




###########################################################################


############################## Eight Stage End ##################################################

#################################################################################################

# 9. Nineth stage :- Modelling.

# Here we have seven models.
# A. Decision Tree
# B. Logistic Regression
# C. KNeighborsClassifier
# D. SVC method of svm class to use Support Vector Machine Algorithm
# E. SVC method of svm class to use Kernel SVM Algorithm
# F. GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm
# G. RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm




# A. Decision Tree

# Train the decision tree
Decision_Tree <- rpart(Train_Class ~ ., data = as.data.frame(Train_Predictors), method = "class", control = rpart.control(seed = 10))

# Predict the labels for the test data
Predict_labels_for_Decision_Tree <- predict(Decision_Tree, as.data.frame(Test_Predictors), type = "class")


# Create Confusion Matrix and Statistics for desicion tree model


# Manually set the levels based on expected classes
expected_levels <- c(1, 0)  

# Set levels for both vectors
Predict_labels_for_Decision_Tree <- factor(Predict_labels_for_Decision_Tree, levels = expected_levels)
Test_Class <- factor(Test_Class, levels = expected_levels)

# Now create the confusion matrix
Confusion_Matrix_tree <- confusionMatrix(Test_Class, Predict_labels_for_Decision_Tree)

# Print the results

print("Decision Tree: \n")
print(Confusion_Matrix_tree)


# Extract Accuracy, precision, recall, F1 score,Sensitivity,Specificity and support
Cancer_1_Pridiction_Accuracy     <- Confusion_Matrix_tree$byClass['Pos Pred Value']
Cancer_0_Pridiction_Accuracy     <- Confusion_Matrix_tree$byClass['Neg Pred Value']
DecisionTree_Avg_Accuracy        <- Confusion_Matrix_tree$byClass['Balanced Accuracy']
DecisionTree_precision           <- Confusion_Matrix_tree$byClass['Precision']
DecisionTree_recall              <- Confusion_Matrix_tree$byClass['Recall']
f1_score                         <- Confusion_Matrix_tree$byClass['F1']
support                          <- Confusion_Matrix_tree$byClass['Support']
Sensitivity                      <- Confusion_Matrix_tree$byClass['Sensitivity']
Specificity                      <- Confusion_Matrix_tree$byClass['Specificity']

# Print the results

cat(" Decision-Tree Cancer_1 Accuracy:", Cancer_1_Pridiction_Accuracy, "\n",
    "Decision-Tree Cancer_0 Accuracy:", Cancer_0_Pridiction_Accuracy, "\n",
    "Decision-Tree Avg_Accuracy:", DecisionTree_Avg_Accuracy, "\n",
    "Decision-Tree Precision:", DecisionTree_precision, "\n",
    "Decision-Tree Recall:", DecisionTree_recall, "\n",
    "Decision-Tree F1 Score:", f1_score, "\n",
    "Decision-Tree Sensitivity:", Sensitivity, "\n",
    "Decision-Tree Specificity:", Specificity, "\n")



# B. Logistic Regression


# Convert class labels to factors
Train_Class <- as.factor(Train_Class)
Test_Class <- as.factor(Test_Class)

# Train Multinomial Logistic Regression
multinom_model <- multinom(Train_Class ~ ., data = as.data.frame(Train_Predictors))

# Predict with Multinomial Logistic Regression
Predict_labels_for_Logistic_Regression <- predict(multinom_model, newdata = as.data.frame(Test_Predictors), type = "class")


# Create confusion matrix

Confusion_Matrix_Logistic_Regression <- confusionMatrix(Test_Class,Predict_labels_for_Logistic_Regression)

# Print the results

print("Logistic Regression: \n")
print(Confusion_Matrix_Logistic_Regression)


# Extract Accuracy, precision, recall, F1 score,Sensitivity,Specificity and support
Cancer_0_Pridiction_Accuracy      <- Confusion_Matrix_Logistic_Regression$byClass['Pos Pred Value']
Cancer_1_Pridiction_Accuracy      <- Confusion_Matrix_Logistic_Regression$byClass['Neg Pred Value']
Logistic_Regression_Avg_Accuracy  <- Confusion_Matrix_Logistic_Regression$byClass['Balanced Accuracy']
Logistic_Regression_precision     <- Confusion_Matrix_Logistic_Regression$byClass['Precision']
Logistic_Regression_recall        <- Confusion_Matrix_Logistic_Regression$byClass['Recall']
f1_score                          <- Confusion_Matrix_Logistic_Regression$byClass['F1']
support                           <- Confusion_Matrix_Logistic_Regression$byClass['Support']
Sensitivity                       <- Confusion_Matrix_Logistic_Regression$byClass['Sensitivity']
Specificity                       <- Confusion_Matrix_Logistic_Regression$byClass['Specificity']

# Print the results
cat(" Logistic_Regression Cancer_0 Accuracy:", Cancer_0_Pridiction_Accuracy, "\n",
    "Logistic_Regression Cancer_1 Accuracy:", Cancer_1_Pridiction_Accuracy, "\n",
    "Logistic_Regression Avg_Accuracy:", Logistic_Regression_Avg_Accuracy, "\n",
    "Logistic_Regression Precision:", Logistic_Regression_precision, "\n",
    "Logistic_Regression Recall:", Logistic_Regression_recall, "\n",
    "Logistic_Regression F1 Score:", f1_score, "\n",
    "Logistic_Regression Support:", support, "\n",
    "Logistic_Regression F1 Score:", Sensitivity, "\n",
    "Logistic_Regression Support:", Specificity, "\n")




# C. KNeighborsClassifier

# Training the k-Nearest Neighbors (KNN) Classifier
k_neighbors <- 5
KNeighborsClassifier <- knn(train = Train_Predictors, test = Test_Predictors, cl = Train_Class, k = k_neighbors)


# C Predictions using the trained KNN classifier
KNeighborsClassifier_pred <- as.factor(KNeighborsClassifier)


# Create confusion matrix

Confusion_Matrix_KNeighborsClassifier <- confusionMatrix(Test_Class,KNeighborsClassifier_pred)

# Print the results

print("KNeighborsClassifier: \n")
print(Confusion_Matrix_KNeighborsClassifier)


# Extract Accuracy, precision, recall, F1 score,Sensitivity,Specificity and support
Cancer_0_Pridiction_Accuracy            <- Confusion_Matrix_KNeighborsClassifier$byClass['Pos Pred Value']
Cancer_1_Pridiction_Accuracy            <- Confusion_Matrix_KNeighborsClassifier$byClass['Neg Pred Value']
KNeighborsClassifier_Avg_Accuracy       <- Confusion_Matrix_KNeighborsClassifier$byClass['Balanced Accuracy']
KNeighborsClassifier_precision          <- Confusion_Matrix_KNeighborsClassifier$byClass['Precision']
KNeighborsClassifier_recall             <- Confusion_Matrix_KNeighborsClassifier$byClass['Recall']
f1_score                                <- Confusion_Matrix_KNeighborsClassifier$byClass['F1']
support                                 <- Confusion_Matrix_KNeighborsClassifier$byClass['Support']
Sensitivity                             <- Confusion_Matrix_KNeighborsClassifier$byClass['Sensitivity']
Specificity                             <- Confusion_Matrix_KNeighborsClassifier$byClass['Specificity']

# Print the results
cat(" KNeighborsClassifier Cancer_0 Accuracy:", Cancer_0_Pridiction_Accuracy, "\n",
    "KNeighborsClassifier Cancer_1 Accuracy:", Cancer_1_Pridiction_Accuracy, "\n",
    "KNeighborsClassifier Avg_Accuracy:", KNeighborsClassifier_Avg_Accuracy, "\n",
    "KNeighborsClassifier Precision:", KNeighborsClassifier_precision, "\n",
    "KNeighborsClassifier Recall:", KNeighborsClassifier_recall, "\n",
    "KNeighborsClassifier F1 Score:", f1_score, "\n",
    "KNeighborsClassifier Support:", support, "\n",
    "KNeighborsClassifier F1 Score:", Sensitivity, "\n",
    "KNeighborsClassifier Support:", Specificity, "\n")


# D. SVC method of svm class to use Support Vector Machine Algorithm

# Training the Support Vector Machine (SVM) Classifier with a linear kernel
SVMClassifier <- svm(Train_Class ~ ., data = cbind(Train_Class, Train_Predictors), kernel = "linear", scale = TRUE)


# D Prediction using Support Vector Machine (SVM) Classifier
SVMClassifier_pred <- predict(SVMClassifier, newdata = as.data.frame(Test_Predictors))

# Create confusion matrix
Confusion_Matrix_SVM <- confusionMatrix(Test_Class,SVMClassifier_pred)


# Print the results

print("SVM: \n")
print(Confusion_Matrix_SVM)

# Extract Accuracy, precision, recall, F1 score,Sensitivity,Specificity and support
Cancer_0_Pridiction_Accuracy    <- Confusion_Matrix_SVM$byClass['Pos Pred Value']
Cancer_1_Pridiction_Accuracy    <- Confusion_Matrix_SVM$byClass['Neg Pred Value']
SVM_Avg_Accuracy                <- Confusion_Matrix_SVM$byClass['Balanced Accuracy']
SVM_precision                   <- Confusion_Matrix_SVM$byClass['Precision']
SVM_recall                      <- Confusion_Matrix_SVM$byClass['Recall']
f1_score                        <- Confusion_Matrix_SVM$byClass['F1']
support                         <- Confusion_Matrix_SVM$byClass['Support']
Sensitivity                     <- Confusion_Matrix_SVM$byClass['Sensitivity']
Specificity                     <- Confusion_Matrix_SVM$byClass['Specificity']

# Print the results
cat(" SVM Cancer_0 Accuracy:", Cancer_0_Pridiction_Accuracy, "\n",
    "SVM Cancer_1 Accuracy:", Cancer_1_Pridiction_Accuracy, "\n",
    "SVM Avg_Accuracy:", SVM_Avg_Accuracy, "\n",
    "SVM Precision:", SVM_precision, "\n",
    "SVM Recall:", SVM_recall, "\n",
    "SVM F1 Score:", f1_score, "\n",
    "SVM Support:", support, "\n",
    "SVM F1 Score:", Sensitivity, "\n",
    "SVM Support:", Specificity, "\n")




# E. SVC method of svm class to use Kernel SVM Algorithm

# Training the Support Vector Machine (SVM) Classifier with an RBF kernel
KernelSVMClassifier <- svm(Train_Class ~ ., data = cbind(Train_Class, Train_Predictors), kernel = "radial", scale = TRUE)

# E Prediction using Kernel SVM Classifier
KernelSVMClassifier_pred <- predict(KernelSVMClassifier, newdata = as.data.frame(Test_Predictors))


# Create confusion matrix
Confusion_Matrix_KernelSVMClassifier <- confusionMatrix(Test_Class,KernelSVMClassifier_pred)

# Print the results

print("KernelSVMClassifier: \n")
print(Confusion_Matrix_KernelSVMClassifier)

# Extract Accuracy, precision, recall, F1 score,Sensitivity,Specificity and support
Cancer_0_Pridiction_Accuracy       <- Confusion_Matrix_KernelSVMClassifier$byClass['Pos Pred Value']
Cancer_1_Pridiction_Accuracy       <- Confusion_Matrix_KernelSVMClassifier$byClass['Neg Pred Value']
KernelSVMClassifier_Avg_Accuracy   <- Confusion_Matrix_KernelSVMClassifier$byClass['Balanced Accuracy']
KernelSVMClassifier_precision      <- Confusion_Matrix_KernelSVMClassifier$byClass['Precision']
KernelSVMClassifier_recall         <- Confusion_Matrix_KernelSVMClassifier$byClass['Recall']
f1_score                           <- Confusion_Matrix_KernelSVMClassifier$byClass['F1']
support                            <- Confusion_Matrix_KernelSVMClassifier$byClass['Support']
Sensitivity                        <- Confusion_Matrix_KernelSVMClassifier$byClass['Sensitivity']
Specificity                        <- Confusion_Matrix_KernelSVMClassifier$byClass['Specificity']

# Print the results
cat(" KernelSVMClassifier Cancer_0 Accuracy:", Cancer_0_Pridiction_Accuracy, "\n",
    "KernelSVMClassifier Cancer_1 Accuracy:", Cancer_1_Pridiction_Accuracy, "\n",
    "KernelSVMClassifier Avg_Accuracy:", KernelSVMClassifier_Avg_Accuracy, "\n",
    "KernelSVMClassifier Precision:", KernelSVMClassifier_precision, "\n",
    "KernelSVMClassifier Recall:", KernelSVMClassifier_recall, "\n",
    "KernelSVMClassifier F1 Score:", f1_score, "\n",
    "KernelSVMClassifier Support:", support, "\n",
    "KernelSVMClassifier F1 Score:", Sensitivity, "\n",
    "KernelSVMClassifier Support:", Specificity, "\n")



# F. GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm

# Training the Gaussian Naive Bayes Classifier
NaiveBayesClassifier <- naiveBayes(Train_Class ~ ., data = cbind(Train_Class, Train_Predictors))

# F Prediction using Naive Bayes Classifier
NaiveBayesClassifier_pred <- predict(NaiveBayesClassifier, newdata = as.data.frame(Test_Predictors))


# Create confusion matrix
Confusion_Matrix_NaiveBayesClassifier <- confusionMatrix(Test_Class,NaiveBayesClassifier_pred)

# Print the results

print("NaiveBayesClassifier: \n")
print(Confusion_Matrix_NaiveBayesClassifier)

# Extract Accuracy, precision, recall, F1 score,Sensitivity,Specificity and support
Cancer_0_Pridiction_Accuracy        <- Confusion_Matrix_NaiveBayesClassifier$byClass['Pos Pred Value']
Cancer_1_Pridiction_Accuracy        <- Confusion_Matrix_NaiveBayesClassifier$byClass['Neg Pred Value']
NaiveBayesClassifier_Avg_Accuracy   <- Confusion_Matrix_NaiveBayesClassifier$byClass['Balanced Accuracy']
NaiveBayesClassifier_precision      <- Confusion_Matrix_NaiveBayesClassifier$byClass['Precision']
NaiveBayesClassifier_recall         <- Confusion_Matrix_NaiveBayesClassifier$byClass['Recall']
f1_score                            <- Confusion_Matrix_NaiveBayesClassifier$byClass['F1']
support                             <- Confusion_Matrix_NaiveBayesClassifier$byClass['Support']
Sensitivity                         <- Confusion_Matrix_NaiveBayesClassifier$byClass['Sensitivity']
Specificity                         <- Confusion_Matrix_NaiveBayesClassifier$byClass['Specificity']

# Print the results
cat(" NaiveBayesClassifier Cancer_0 Accuracy:", Cancer_0_Pridiction_Accuracy, "\n",
    "NaiveBayesClassifier Cancer_1 Accuracy:", Cancer_1_Pridiction_Accuracy, "\n",
    "NaiveBayesClassifier Avg_Accuracy:", NaiveBayesClassifier_Avg_Accuracy, "\n",
    "NaiveBayesClassifier Precision:", NaiveBayesClassifier_precision, "\n",
    "NaiveBayesClassifier Recall:", NaiveBayesClassifier_recall, "\n",
    "NaiveBayesClassifier F1 Score:", f1_score, "\n",
    "NaiveBayesClassifier Support:", support, "\n",
    "NaiveBayesClassifier F1 Score:", Sensitivity, "\n",
    "NaiveBayesClassifier Support:", Specificity, "\n")




# G. RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm

# Training the Random Forest Classifier
RandomForestClassifier <- randomForest(Train_Class ~ ., data = cbind(Train_Class, Train_Predictors), ntree = 10, mtry = ncol(Train_Predictors), nodesize = 1)

# G Prediction using Random Forest Classifier
RandomForestClassifier_pred <- predict(RandomForestClassifier, newdata = as.data.frame(Test_Predictors))


# Create confusion matrix
Confusion_Matrix_RandomForestClassifier <- confusionMatrix(Test_Class,RandomForestClassifier_pred)

# Print the results

print("RandomForestClassifier: \n")
print(Confusion_Matrix_RandomForestClassifier)

# Extract Accuracy, precision, recall, F1 score,Sensitivity,Specificity and support
Cancer_0_Pridiction_Accuracy           <- Confusion_Matrix_RandomForestClassifier$byClass['Pos Pred Value']
Cancer_1_Pridiction_Accuracy           <- Confusion_Matrix_RandomForestClassifier$byClass['Neg Pred Value']
RandomForestClassifier_Avg_Accuracy    <- Confusion_Matrix_RandomForestClassifier$byClass['Balanced Accuracy']
RandomForestClassifier_precision       <- Confusion_Matrix_RandomForestClassifier$byClass['Precision']
RandomForestClassifier_recall          <- Confusion_Matrix_RandomForestClassifier$byClass['Recall']
f1_score                               <- Confusion_Matrix_RandomForestClassifier$byClass['F1']
support                                <- Confusion_Matrix_RandomForestClassifier$byClass['Support']
Sensitivity                            <- Confusion_Matrix_RandomForestClassifier$byClass['Sensitivity']
Specificity                            <- Confusion_Matrix_RandomForestClassifier$byClass['Specificity']


# Print the results
cat(" RandomForestClassifier Cancer_0 Accuracy:", Cancer_0_Pridiction_Accuracy, "\n",
    "RandomForestClassifier Cancer_1 Accuracy:", Cancer_1_Pridiction_Accuracy, "\n",
    "RandomForestClassifier Avg_Accuracy:", RandomForestClassifier_Avg_Accuracy, "\n",
    "RandomForestClassifier Precision:", RandomForestClassifier_precision, "\n",
    "RandomForestClassifier Recall:", RandomForestClassifier_recall, "\n",
    "RandomForestClassifier F1 Score:", f1_score, "\n",
    "RandomForestClassifier Support:", support, "\n",
    "RandomForestClassifier F1 Score:", Sensitivity, "\n",
    "RandomForestClassifier Support:", Specificity, "\n")



############################## Nineth Stage End ##################################################

#################################################################################################

# 10. Tenth stage :- Model validation.

# Print all model results
cat(" Decision-Tree Avg_Accuracy:", DecisionTree_Avg_Accuracy, "\n",
    "Logistic_Regression Avg_Accuracy:", Logistic_Regression_Avg_Accuracy, "\n",
    "KNeighborsClassifier Avg_Accuracy:", KNeighborsClassifier_Avg_Accuracy, "\n",
    "SVM Avg_Accuracy:", SVM_Avg_Accuracy, "\n",
    "KernelSVMClassifier Avg_Accuracy:", KernelSVMClassifier_Avg_Accuracy, "\n",
    "NaiveBayesClassifier Avg_Accuracy:", NaiveBayesClassifier_Avg_Accuracy, "\n",
    "RandomForestClassifier Avg_Accuracy:", RandomForestClassifier_Avg_Accuracy, "\n")


# A. Plot Graph for Accuracy

# Original accuracy values
accuracies <- c(DecisionTree_Avg_Accuracy
                , Logistic_Regression_Avg_Accuracy
                , KNeighborsClassifier_Avg_Accuracy
                , SVM_Avg_Accuracy
                , KernelSVMClassifier_Avg_Accuracy
                , NaiveBayesClassifier_Avg_Accuracy
                , RandomForestClassifier_Avg_Accuracy)

# Convert to percentage
accuracies_percentage <- accuracies * 100

# Models names
models <- c("Decision-Tree", "Logistic Regression", "KNeighbors", "SVM", "Kernel SVM", "Naive Bayes", "Random Forest")

# Create a data frame
data <- data.frame(Model = models, Accuracy = accuracies_percentage)

# Plotting
library(ggplot2)
ggplot(data, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = sprintf("%.2f%%", Accuracy)),
            position = position_stack(vjust = 0.5),
            size = 3,
            color = "white") +  # Adjust label appearance
  labs(title = "Model Accuracy Comparison",
       x = "Models",
       y = "Accuracy (%)") +
  theme_minimal()

# From the above model stats and graph it is observe that Logistic Regression possess higest accuracy value.

# We also need to check the false negative values that is Predicting someone does not have cancer while he did.
# Also need to check precision values.



# B. plot Graph for False negative values 

# Original False negative values 
False_Negative_Values  <- c(4
                            , 4
                            , 7
                            , 4
                            , 5
                            , 4
                            , 7)



# Models names
models <- c("Decision-Tree", "Logistic Regression", "KNeighbors", "SVM", "Kernel SVM", "Naive Bayes", "Random Forest")

# Create a data frame
data <- data.frame(Model = models, False_Negative = False_Negative_Values)

# Plotting
library(ggplot2)
ggplot(data, aes(x = Model, y = False_Negative, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = False_Negative),
            position = position_stack(vjust = 0.5),
            size = 3,
            color = "white") +  # Adjust label appearance
  labs(title = "False Negative Comparison",
       x = "Models",
       y = "False Negative") +
  theme_minimal()




# C. Plot Graph for recall

# Original recall values
recall <- c(DecisionTree_recall
            , Logistic_Regression_recall
            , KNeighborsClassifier_recall
            , SVM_recall
            , KernelSVMClassifier_recall
            , NaiveBayesClassifier_recall
            , RandomForestClassifier_recall)


# Convert to percentage
recall_percentage <- recall * 100

# Models names
models <- c("Decision-Tree", "Logistic Regression", "KNeighbors", "SVM", "Kernel SVM", "Naive Bayes", "Random Forest")

# Create a data frame
data <- data.frame(Model = models, recall = recall_percentage)

# Plotting
library(ggplot2)
ggplot(data, aes(x = Model, y = recall, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = sprintf("%.2f%%", recall)),
            position = position_stack(vjust = 0.5),
            size = 3,
            color = "white") +  # Adjust label appearance
  labs(title = "Model recall Comparison",
       x = "Models",
       y = "recall (%)") +
  theme_minimal()


# Considering all necessary metrics Logistic Regression is the best bet for this project.
