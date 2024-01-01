#!/usr/bin/env python
# coding: utf-8


# Breast Cancer Prediction Dataset

# 1. First Stage: All the necessary packages installations


# 1. First Stage: All the necessary packages installations

# Assuming you are using Jupyter Notebook or any Python environment with pip installed
get_ipython().system('pip install pandas')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install matplotlib')




#################################################################################################

# 2. Second stage: Loading the Libraries

import json
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# Other Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

############################## Second Stage End ##################################################




# 3. Third stage: Load necessary datasets

# Assuming you have the breast cancer dataset CSV file in the specified path
csv_file_path = r"C:\Users\Breast_cancer_data.csv"

# Load the breast cancer dataset using pandas in Python
breast_cancer_dataset = pd.read_csv(csv_file_path)

# Display the first few rows of the dataset
print(breast_cancer_dataset.head())

############################## Third Stage End ##################################################



# 4. Fourth stage: Data Exploration


# Display total number of rows
total_row_count = len(breast_cancer_dataset)
print(total_row_count)

# Display total number of distinct rows
count_distinct_rows = len(breast_cancer_dataset.drop_duplicates())
print(count_distinct_rows)

# Display data structure information
print(breast_cancer_dataset.info())

# Display summary statistics
print(breast_cancer_dataset.describe())

# Check if the data is balanced or not

# Create a color palette for 0 and 1
colors = {0: "skyblue", 1: "salmon"}

# Plot using seaborn in Python
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='diagnosis', data=breast_cancer_dataset, palette=colors)
ax.set(title='Count Plot of Diagnosis', xlabel='Diagnosis', ylabel='Count')

# Display counts on bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                xytext=(0, 10), textcoords='offset points', size=12)

plt.show()


# Balancing looks fine. 

# checking the values of dataset

# Create a pair plot
# Using Seaborn for pair plot
sns.pairplot(breast_cancer_dataset, hue='diagnosis', palette={0: "skyblue", 1: "salmon"})
plt.show()




############################## Fourth Stage End ##################################################


# ### From the graph it is observed that, there might be some error in diagnosis values. As from the graph
# ### we can see that cancer = 0 has high value of mean-radius, mean-texture, mean-perimeter and mean-smoothness values.
# ### this should be opposite, cancer=1 should have higher mean-radius, mean-texture, mean-perimeter and mean-smoothness values
# ### Lets fix this by interchanging the diagnosis values. 


# 5. Reverse the values: Reverting Diagnosis column

# Reverting Diagnosis column
breast_cancer_dataset['diagnosis'] = 1 - breast_cancer_dataset['diagnosis']

# Create a color palette for 0 and 1
colors = {0: "skyblue", 1: "salmon"}

# Plot using seaborn in Python
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='diagnosis', data=breast_cancer_dataset, palette=colors)
ax.set(title='Count Plot of Diagnosis', xlabel='Diagnosis', ylabel='Count')

# Display counts on bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                xytext=(0, 10), textcoords='offset points', size=12)

plt.show()

# Create a pair plot
# Using Seaborn for pair plot
sns.pairplot(breast_cancer_dataset, hue='diagnosis', palette={0: "skyblue", 1: "salmon"})
plt.show()


# Now looks better 

############################## Fifth Stage End ##################################################


# 6. Sixth stage: Correlation of Breast_cancer_dataset

# Calculate the correlation matrix
numeric_data = breast_cancer_dataset.drop(columns=['diagnosis'])
correlation_matrix = numeric_data.corr()

# Print the correlation matrix
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap
sns.set(style='white')
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

#Variables such as mean area, mean radius and mean perimeter are highly correlated to each other but as we
# have very few features or varaibles so we cant remove any further, it will create problem in modelling 

############################## Sixth Stage End ##################################################



# 7. Seventh stage: Normalize merged dataset

# The next step is to normalize the data. 
# Our data can have various scale values present. 
# Normalizing the scale values helps prevent certain features from dominating others.



# Create two dataframes
# 1. With only numeric variables (8 variables) called 'Predictors'.
# 2. With genre values (column name 'diagnosis') called 'Class'.

# Drop 'diagnosis' column to create Predictors
Predictors = breast_cancer_dataset.drop(columns=['diagnosis'])

# Create Class using the 'diagnosis' column
Class = breast_cancer_dataset['diagnosis']

# Standardize the features using the StandardScaler
scaler = StandardScaler()
scaled_train_predictors = scaler.fit_transform(Predictors)

# Display the first few rows of the scaled features
print(scaled_train_predictors[:5, :])

# Convert the scaled matrix to a data frame
scaled_train_predictors_df = pd.DataFrame(scaled_train_predictors, columns=Predictors.columns)



############################## Seventh Stage End ##################################################


# 8. Eighth stage: Split the data into Train_Predictors, Test_Predictors, Train_Class, and Test_Class.



# Set the seed for reproducibility
np.random.seed(10)

# Split the data
train_predictors, test_predictors, train_class, test_class = train_test_split(
    scaled_train_predictors_df, Class, test_size=0.3, random_state=10
)

# Display the shape of the resulting sets
print("Train_Predictors shape:", train_predictors.shape)
print("Test_Predictors shape:", test_predictors.shape)
print("Train_Class shape:", train_class.shape)
print("Test_Class shape:", test_class.shape)




############################## Eight Stage End ##################################################


# 9. Ninth stage: Modeling

# A. Decision Tree


# Train the decision tree
decision_tree = DecisionTreeClassifier(random_state=10)
decision_tree.fit(train_predictors, train_class)

# Predict the labels for the test data
predict_labels_for_decision_tree = decision_tree.predict(test_predictors)

# Create Confusion Matrix and Statistics for decision tree model
confusion_matrix_tree = confusion_matrix(test_class, predict_labels_for_decision_tree)

# Print the confusion matrix
print("Decision Tree: \n")
print(confusion_matrix_tree)

# Extract Accuracy, precision, recall, F1 score, Sensitivity, Specificity, and support
decision_tree_accuracy = decision_tree.score(test_predictors, test_class)
decision_tree_precision = confusion_matrix_tree[1, 1] / (confusion_matrix_tree[1, 1] + confusion_matrix_tree[0, 1])
decision_tree_recall = confusion_matrix_tree[1, 1] / (confusion_matrix_tree[1, 1] + confusion_matrix_tree[1, 0])
f1_score_tree = 2 * (decision_tree_precision * decision_tree_recall) / (decision_tree_precision + decision_tree_recall)
support_tree = confusion_matrix_tree.sum(axis=1)
sensitivity_tree = confusion_matrix_tree[1, 1] / support_tree[1]
specificity_tree = confusion_matrix_tree[0, 0] / support_tree[0]

# Print the results
print(f"Decision-Tree Accuracy: {decision_tree_accuracy}\n"
      f"Decision-Tree Precision: {decision_tree_precision}\n"
      f"Decision-Tree Recall: {decision_tree_recall}\n"
      f"Decision-Tree F1 Score: {f1_score_tree}\n"
      f"Decision-Tree Sensitivity: {sensitivity_tree}\n"
      f"Decision-Tree Specificity: {specificity_tree}\n")





# B. Logistic Regression


# Convert class labels to factors
train_class = train_class.astype('category')
test_class = test_class.astype('category')

# Train Logistic Regression
logistic_regression_model = LogisticRegression(max_iter=1000, random_state=10)
logistic_regression_model.fit(train_predictors, train_class)

# Predict with Logistic Regression
predict_labels_for_logistic_regression = logistic_regression_model.predict(test_predictors)

# Create confusion matrix
confusion_matrix_logistic_regression = confusion_matrix(test_class, predict_labels_for_logistic_regression)

# Print the confusion matrix
print("Logistic Regression: \n")
print(confusion_matrix_logistic_regression)

# Extract Accuracy, precision, recall, F1 score, Sensitivity, Specificity, and support
logistic_regression_accuracy = logistic_regression_model.score(test_predictors, test_class)
logistic_regression_precision = confusion_matrix_logistic_regression[1, 1] / (confusion_matrix_logistic_regression[1, 1] + confusion_matrix_logistic_regression[0, 1])
logistic_regression_recall = confusion_matrix_logistic_regression[1, 1] / (confusion_matrix_logistic_regression[1, 1] + confusion_matrix_logistic_regression[1, 0])
f1_score_logistic_regression = 2 * (logistic_regression_precision * logistic_regression_recall) / (logistic_regression_precision + logistic_regression_recall)
support_logistic_regression = confusion_matrix_logistic_regression.sum(axis=1)
sensitivity_logistic_regression = confusion_matrix_logistic_regression[1, 1] / support_logistic_regression[1]
specificity_logistic_regression = confusion_matrix_logistic_regression[0, 0] / support_logistic_regression[0]

# Print the results
print(f"Logistic_Regression Accuracy: {logistic_regression_accuracy}\n"
      f"Logistic_Regression Precision: {logistic_regression_precision}\n"
      f"Logistic_Regression Recall: {logistic_regression_recall}\n"
      f"Logistic_Regression F1 Score: {f1_score_logistic_regression}\n"
      f"Logistic_Regression Sensitivity: {sensitivity_logistic_regression}\n"
      f"Logistic_Regression Specificity: {specificity_logistic_regression}\n")



# C. KNeighborsClassifier


# Training the k-Nearest Neighbors (KNN) Classifier
k_neighbors = 5
k_neighbors_classifier = KNeighborsClassifier(n_neighbors=k_neighbors)
k_neighbors_classifier.fit(train_predictors, train_class)

# C Predictions using the trained KNN classifier
k_neighbors_classifier_pred = k_neighbors_classifier.predict(test_predictors)

# Create confusion matrix
confusion_matrix_k_neighbors_classifier = confusion_matrix(test_class, k_neighbors_classifier_pred)

# Print the confusion matrix
print("KNeighborsClassifier: \n")
print(confusion_matrix_k_neighbors_classifier)

# Extract Accuracy, precision, recall, F1 score, Sensitivity, Specificity, and support
k_neighbors_classifier_accuracy = k_neighbors_classifier.score(test_predictors, test_class)
k_neighbors_classifier_precision = confusion_matrix_k_neighbors_classifier[1, 1] / (
    confusion_matrix_k_neighbors_classifier[1, 1] + confusion_matrix_k_neighbors_classifier[0, 1]
)
k_neighbors_classifier_recall = confusion_matrix_k_neighbors_classifier[1, 1] / (
    confusion_matrix_k_neighbors_classifier[1, 1] + confusion_matrix_k_neighbors_classifier[1, 0]
)
f1_score_k_neighbors_classifier = 2 * (
    k_neighbors_classifier_precision * k_neighbors_classifier_recall
) / (k_neighbors_classifier_precision + k_neighbors_classifier_recall)
support_k_neighbors_classifier = confusion_matrix_k_neighbors_classifier.sum(axis=1)
sensitivity_k_neighbors_classifier = (
    confusion_matrix_k_neighbors_classifier[1, 1] / support_k_neighbors_classifier[1]
)
specificity_k_neighbors_classifier = (
    confusion_matrix_k_neighbors_classifier[0, 0] / support_k_neighbors_classifier[0]
)

# Print the results
print(f"KNeighborsClassifier Accuracy: {k_neighbors_classifier_accuracy}\n"
      f"KNeighborsClassifier Precision: {k_neighbors_classifier_precision}\n"
      f"KNeighborsClassifier Recall: {k_neighbors_classifier_recall}\n"
      f"KNeighborsClassifier F1 Score: {f1_score_k_neighbors_classifier}\n"
      f"KNeighborsClassifier Sensitivity: {sensitivity_k_neighbors_classifier}\n"
      f"KNeighborsClassifier Specificity: {specificity_k_neighbors_classifier}\n")






# D. Support Vector Machine (SVM) Classifier


# Training the Support Vector Machine (SVM) Classifier with a linear kernel
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(train_predictors, train_class)

# D. Predictions using Support Vector Machine (SVM) Classifier
svm_classifier_pred = svm_classifier.predict(test_predictors)

# Create confusion matrix
confusion_matrix_svm = confusion_matrix(test_class, svm_classifier_pred)

# Print the confusion matrix
print("SVM: \n")
print(confusion_matrix_svm)

# Extract Accuracy, precision, recall, F1 score, Sensitivity, Specificity, and support
svm_accuracy = svm_classifier.score(test_predictors, test_class)
svm_precision = confusion_matrix_svm[1, 1] / (confusion_matrix_svm[1, 1] + confusion_matrix_svm[0, 1])
svm_recall = confusion_matrix_svm[1, 1] / (confusion_matrix_svm[1, 1] + confusion_matrix_svm[1, 0])
f1_score_svm = 2 * (svm_precision * svm_recall) / (svm_precision + svm_recall)
support_svm = confusion_matrix_svm.sum(axis=1)
sensitivity_svm = confusion_matrix_svm[1, 1] / support_svm[1]
specificity_svm = confusion_matrix_svm[0, 0] / support_svm[0]

# Print the results
print(f"SVM Accuracy: {svm_accuracy}\n"
      f"SVM Precision: {svm_precision}\n"
      f"SVM Recall: {svm_recall}\n"
      f"SVM F1 Score: {f1_score_svm}\n"
      f"SVM Sensitivity: {sensitivity_svm}\n"
      f"SVM Specificity: {specificity_svm}\n")



# E. Kernel SVM Classifier

# Training the Support Vector Machine (SVM) Classifier with an RBF kernel
kernel_svm_classifier = SVC(kernel='rbf', C=1.0)
kernel_svm_classifier.fit(train_predictors, train_class)

# E. Predictions using Kernel SVM Classifier
kernel_svm_classifier_pred = kernel_svm_classifier.predict(test_predictors)

# Create confusion matrix
confusion_matrix_kernel_svm_classifier = confusion_matrix(test_class, kernel_svm_classifier_pred)

# Print the confusion matrix
print("KernelSVMClassifier: \n")
print(confusion_matrix_kernel_svm_classifier)

# Extract Accuracy, precision, recall, F1 score, Sensitivity, Specificity, and support
kernel_svm_classifier_accuracy = kernel_svm_classifier.score(test_predictors, test_class)
kernel_svm_classifier_precision = confusion_matrix_kernel_svm_classifier[1, 1] / (
    confusion_matrix_kernel_svm_classifier[1, 1] + confusion_matrix_kernel_svm_classifier[0, 1]
)
kernel_svm_classifier_recall = confusion_matrix_kernel_svm_classifier[1, 1] / (
    confusion_matrix_kernel_svm_classifier[1, 1] + confusion_matrix_kernel_svm_classifier[1, 0]
)
f1_score_kernel_svm_classifier = 2 * (
    kernel_svm_classifier_precision * kernel_svm_classifier_recall
) / (kernel_svm_classifier_precision + kernel_svm_classifier_recall)
support_kernel_svm_classifier = confusion_matrix_kernel_svm_classifier.sum(axis=1)
sensitivity_kernel_svm_classifier = (
    confusion_matrix_kernel_svm_classifier[1, 1] / support_kernel_svm_classifier[1]
)
specificity_kernel_svm_classifier = (
    confusion_matrix_kernel_svm_classifier[0, 0] / support_kernel_svm_classifier[0]
)

# Print the results
print(f"KernelSVMClassifier Accuracy: {kernel_svm_classifier_accuracy}\n"
      f"KernelSVMClassifier Precision: {kernel_svm_classifier_precision}\n"
      f"KernelSVMClassifier Recall: {kernel_svm_classifier_recall}\n"
      f"KernelSVMClassifier F1 Score: {f1_score_kernel_svm_classifier}\n"
      f"KernelSVMClassifier Sensitivity: {sensitivity_kernel_svm_classifier}\n"
      f"KernelSVMClassifier Specificity: {specificity_kernel_svm_classifier}\n")



# F. Naive Bayes Classifier


# Training the Gaussian Naive Bayes Classifier
naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(train_predictors, train_class)

# F. Predictions using Naive Bayes Classifier
naive_bayes_classifier_pred = naive_bayes_classifier.predict(test_predictors)

# Create confusion matrix
confusion_matrix_naive_bayes_classifier = confusion_matrix(test_class, naive_bayes_classifier_pred)

# Print the confusion matrix
print("NaiveBayesClassifier: \n")
print(confusion_matrix_naive_bayes_classifier)

# Extract Accuracy, precision, recall, F1 score, Sensitivity, Specificity, and support
naive_bayes_classifier_accuracy = (
    (confusion_matrix_naive_bayes_classifier[0, 0] + confusion_matrix_naive_bayes_classifier[1, 1])
    / confusion_matrix_naive_bayes_classifier.sum()
)
naive_bayes_classifier_precision = (
    confusion_matrix_naive_bayes_classifier[1, 1]
    / (confusion_matrix_naive_bayes_classifier[1, 1] + confusion_matrix_naive_bayes_classifier[0, 1])
)
naive_bayes_classifier_recall = (
    confusion_matrix_naive_bayes_classifier[1, 1]
    / (confusion_matrix_naive_bayes_classifier[1, 1] + confusion_matrix_naive_bayes_classifier[1, 0])
)
f1_score_naive_bayes_classifier = 2 * (
    naive_bayes_classifier_precision * naive_bayes_classifier_recall
) / (naive_bayes_classifier_precision + naive_bayes_classifier_recall)
support_naive_bayes_classifier = confusion_matrix_naive_bayes_classifier.sum(axis=1)
sensitivity_naive_bayes_classifier = (
    confusion_matrix_naive_bayes_classifier[1, 1] / support_naive_bayes_classifier[1]
)
specificity_naive_bayes_classifier = (
    confusion_matrix_naive_bayes_classifier[0, 0] / support_naive_bayes_classifier[0]
)

# Print the results
print(f"NaiveBayesClassifier Accuracy: {naive_bayes_classifier_accuracy}\n"
      f"NaiveBayesClassifier Precision: {naive_bayes_classifier_precision}\n"
      f"NaiveBayesClassifier Recall: {naive_bayes_classifier_recall}\n"
      f"NaiveBayesClassifier F1 Score: {f1_score_naive_bayes_classifier}\n"
      f"NaiveBayesClassifier Sensitivity: {sensitivity_naive_bayes_classifier}\n"
      f"NaiveBayesClassifier Specificity: {specificity_naive_bayes_classifier}\n")



# G. RandomForestClassifier


# Training the Random Forest Classifier
random_forest_classifier = RandomForestClassifier(n_estimators=10, max_features='sqrt', random_state=10)
random_forest_classifier.fit(train_predictors, train_class)

# G. Predictions using Random Forest Classifier
random_forest_classifier_pred = random_forest_classifier.predict(test_predictors)

# Create confusion matrix
confusion_matrix_random_forest_classifier = confusion_matrix(test_class, random_forest_classifier_pred)

# Print the confusion matrix
print("RandomForestClassifier: \n")
print(confusion_matrix_random_forest_classifier)

# Extract Accuracy, precision, recall, F1 score, Sensitivity, Specificity, and support
random_forest_classifier_accuracy = (
    (confusion_matrix_random_forest_classifier[0, 0] + confusion_matrix_random_forest_classifier[1, 1])
    / confusion_matrix_random_forest_classifier.sum()
)
random_forest_classifier_precision = (
    confusion_matrix_random_forest_classifier[1, 1]
    / (confusion_matrix_random_forest_classifier[1, 1] + confusion_matrix_random_forest_classifier[0, 1])
)
random_forest_classifier_recall = (
    confusion_matrix_random_forest_classifier[1, 1]
    / (confusion_matrix_random_forest_classifier[1, 1] + confusion_matrix_random_forest_classifier[1, 0])
)
f1_score_random_forest_classifier = 2 * (
    random_forest_classifier_precision * random_forest_classifier_recall
) / (random_forest_classifier_precision + random_forest_classifier_recall)
support_random_forest_classifier = confusion_matrix_random_forest_classifier.sum(axis=1)
sensitivity_random_forest_classifier = (
    confusion_matrix_random_forest_classifier[1, 1] / support_random_forest_classifier[1]
)
specificity_random_forest_classifier = (
    confusion_matrix_random_forest_classifier[0, 0] / support_random_forest_classifier[0]
)

# Print the results
print(f"RandomForestClassifier Accuracy: {random_forest_classifier_accuracy}\n"
      f"RandomForestClassifier Precision: {random_forest_classifier_precision}\n"
      f"RandomForestClassifier Recall: {random_forest_classifier_recall}\n"
      f"RandomForestClassifier F1 Score: {f1_score_random_forest_classifier}\n"
      f"RandomForestClassifier Sensitivity: {sensitivity_random_forest_classifier}\n"
      f"RandomForestClassifier Specificity: {specificity_random_forest_classifier}\n")


############################## Nineth Stage End ##################################################


#################################################################################################

# 10. Tenth stage :- Model validation.


# Model names
models = ["Decision-Tree", "Logistic Regression", "KNeighbors", "SVM", "Kernel SVM", "Naive Bayes", "Random Forest"]

# Plotting all model accuracy

# Original accuracy values
accuracies = [decision_tree_accuracy,
               logistic_regression_accuracy,
               k_neighbors_classifier_accuracy,
               svm_accuracy,
               kernel_svm_classifier_accuracy,
               naive_bayes_classifier_accuracy,
               random_forest_classifier_accuracy]

# Convert to percentage
accuracies_percentage = np.array(accuracies) * 100

# Plotting with rotated x-axis labels
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies_percentage, color='skyblue')
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')

# Add text labels
for bar, acc in zip(bars, accuracies_percentage):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{acc:.2f}%",
             ha='center', va='bottom', color='black', fontsize=10)

plt.tight_layout()
plt.show()




# Plotting all model False negative values

# False negative values
false_negative_values = [2, 6, 6, 6, 6, 5, 4]

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.bar(models, false_negative_values, color='salmon')
plt.title("False Negative Comparison")
plt.xlabel("Models")
plt.ylabel("False Negative")

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')

# Add text labels
for bar, value in zip(bars, false_negative_values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(value),
             ha='center', va='bottom', color='black', fontsize=10)

plt.show()





# Plotting all model Recall values

# Recall values

recall = [decision_tree_recall, logistic_regression_recall, k_neighbors_classifier_recall,
          svm_recall, kernel_svm_classifier_recall, naive_bayes_classifier_recall, random_forest_classifier_recall]

# Convert to percentage
recall_percentage = [value * 100 for value in recall]

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.bar(models, recall_percentage, color='lightgreen')
plt.title("Model Recall Comparison")
plt.xlabel("Models")
plt.ylabel("Recall (%)")

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')

# Add text labels
for bar, value in zip(bars, recall_percentage):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{value:.2f}%",
             ha='center', va='bottom', color='black', fontsize=10)

plt.show()



# Considering all necessary metrics Logistic Regression and SVM is the best bet for this project.

