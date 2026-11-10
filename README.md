# Titanic Survival Prediction – Logistic Regression

## Project Overview
This project uses the Titanic dataset to predict passenger survival using **Logistic Regression**, a supervised machine learning classification algorithm.

The objective is to identify patterns in passenger characteristics and understand which factors influenced the likelihood of survival.

## Dataset
The project uses two datasets:
- `Titanic - train.csv` – Training dataset containing passenger information and survival outcomes.
- `Titanic - test.csv` – Test dataset used to generate survival predictions.

## Tools & Libraries
- Python
- Pandas & NumPy – Data manipulation and preprocessing
- Matplotlib & Seaborn – Data visualization
- Scikit-learn – Logistic Regression and feature selection

## Methodology

1. **Data Exploration:** Examined the dataset, columns, missing values, and passenger characteristics.
2. **Exploratory Data Analysis (EDA):** Used count plots and histograms to explore survival patterns based on gender, passenger class, and age.
3. **Feature Engineering:** Converted categorical variables such as Sex, Pclass, and Embarked into dummy variables using one-hot encoding.
4. **Feature Selection:** Applied the Chi-Square test to examine the relationship between categorical features and survival.
5. **Model Building:** Developed Logistic Regression models using selected features.
6. **Prediction:** Generated survival predictions for the test dataset.

## Key Observations
- Female passengers had a considerably higher survival rate than male passengers.
- Passenger class played an important role in survival outcomes.
- Most passengers were between 20 and 40 years old.
- Gender and passenger class were retained as the primary predictors in the final model.

## Models
Two Logistic Regression models were developed:
- Model 1: Included gender, passenger class, and embarkation-related features.
- Model 2: Used gender and passenger class after feature selection.

## Future Improvements
- Handle missing values in Age and other relevant variables.
- Evaluate model performance using a train-validation split.
- Explore additional features such as Fare, SibSp, and Parch.
- Compare Logistic Regression with other classification algorithms.

## Disclaimer
This project is an introductory machine learning exercise focused on data exploration, feature engineering, and binary classification.
