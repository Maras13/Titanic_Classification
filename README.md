# Titanic Classification Project Documentation

## Project Overview

The Titanic Classification Project aims to predict the survival of passengers aboard the RMS Titanic based on various demographic and travel-related features. This classification problem serves as a practical exercise in applying machine learning techniques to real-world data, with the goal of accurately predicting binary outcomes (survived or did not survive).

## 1. Introduction

The Titanic sank on April 15, 1912, during its maiden voyage, resulting in the deaths of over 1,500 passengers and crew. The goal of this project is to develop a predictive model that can classify passengers as either "survived" or "not survived" based on available data. The dataset used for this project is sourced from Kaggle and includes information such as age, sex, passenger class, and fare.

## 2. Data Description

The dataset consists of the following features:

- **PassengerId**: Unique identifier for each passenger.
- **Survived**: Survival status (0 = No, 1 = Yes).
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
- **Name**: Name of the passenger.
- **Sex**: Gender of the passenger.
- **Age**: Age of the passenger in years.
- **SibSp**: Number of siblings/spouses aboard.
- **Parch**: Number of parents/children aboard.
- **Ticket**: Ticket number.
- **Fare**: Fare paid for the ticket.
- **Cabin**: Cabin number.
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## 3. Exploratory Data Analysis (EDA)

In this section, we perform various analyses to understand the data better:

- **Distribution of Survived vs. Not Survived**: Visualizing the survival rate using bar plots.
- **Correlation Analysis**: Checking correlations between numerical features and survival.
- **Categorical Analysis**: Analyzing the impact of features like sex and class on survival rates.
- **Missing Values**: Identifying and visualizing missing values in the dataset.

## 4. Data Preprocessing

Before training the model, the data needs to be preprocessed:

- **Handling Missing Values**: Impute or remove missing values (e.g., filling missing ages with the median age).
- **Encoding Categorical Variables**: Convert categorical features into numerical formats using techniques such as one-hot encoding or label encoding.
- **Feature Scaling**: Normalize or standardize features to ensure that the model converges effectively.

## 5. Model Building

We explore several classification algorithms to determine the best model for our dataset:

- **Logistic Regression**: A simple yet effective model for binary classification.
- **Decision Trees**: A non-linear model that can capture complex patterns.
- **Random Forest**: An ensemble method that improves prediction accuracy.
- **Support Vector Machine (SVM)**: A powerful algorithm for high-dimensional spaces.
- **XGBoost**: An efficient and scalable implementation of gradient boosting framework for supervised learning problems.
- **K-Nearest Neighbors (KNN)**: A simple and effective algorithm that classifies instances based on the closest training examples in the feature space.

## 6. Model Evaluation

Model performance is evaluated using:

- **Confusion Matrix**: To visualize true positives, true negatives, false positives, and false negatives.
- **Accuracy Score**: The ratio of correctly predicted instances to the total instances.
- **Precision, Recall, F1-Score**: Additional metrics to assess the model’s performance on imbalanced datasets.
- **ROC-AUC Curve**: To evaluate the trade-off between true positive and false positive rates.

## 7. Conclusion

The Titanic Classification Project successfully demonstrates the application of machine learning techniques in a real-world scenario. 
## 8. Usage

To use the project:

1. Clone the repository: 
   ```bash
   git clone https://github.com/Maras13/Titanic_Classification.git
