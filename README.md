# Mashrooms-are-Edible-or-not-

Mushroom Edibility Prediction using Random Forest

Project Overview

This project aims to classify mushrooms as edible or poisonous based on various features using the Random Forest Classifier. The dataset contains categorical attributes that describe the physical characteristics of mushrooms.

Dataset Information

The dataset consists of multiple categorical attributes related to mushrooms.

The target variable is class, which indicates whether a mushroom is edible or poisonous.

Features in the Dataset

Cap Shape

Cap Surface

Cap Color

Bruises

Odor

Gill Attachment

Gill Spacing

Gill Size

Gill Color

Stalk Shape

Stalk Root

Stalk Surface Above Ring

Stalk Surface Below Ring

Stalk Color Above Ring

Stalk Color Below Ring

Veil Type

Veil Color

Ring Number

Ring Type

Spore Print Color

Population

Habitat

Class (Target Variable: Edible or Poisonous)

Steps in the Project

1. Data Preprocessing

Load the dataset using Pandas.

Perform exploratory data analysis (EDA) with df.info() and df.describe().

Check for missing values and handle them if any exist.

Encode categorical variables using LabelEncoder.

Split the dataset into features (X) and target variable (y).

2. Data Encoding

Convert categorical features into numerical values using LabelEncoder.

Save the encoded dataset for further use.

3. Splitting Data

Split the dataset into training and testing sets (80%-20% split).

4. Feature Scaling

Standardize the features using StandardScaler to improve model performance.

5. Model Training

Train a Random Forest Classifier with entropy criterion.

Use n_estimators=50 and random_state=42 for reproducibility.

6. Model Evaluation

Make predictions on the test dataset.

Compute the accuracy score.

Generate and visualize the confusion matrix using Seaborn.

Print the classification report including precision, recall, and F1-score.

7. Prediction on New Data

Perform predictions on new mushroom feature inputs using the trained model.

Dependencies

Python

Pandas

NumPy

Seaborn

Matplotlib

Scikit-learn
Results

The classifier effectively distinguishes between edible and poisonous mushrooms based on input features.
Conclusion

This project successfully demonstrates the application of machine learning for mushroom edibility classification. The Random Forest Classifier provides high accuracy and reliable results for distinguishing between edible and poisonous mushrooms.
