# ================================================================
# PROJECT 1: MACHINE LEARNING PIPELINE
#Leon Olejarski

# IMPORT REQUIRED LIBRARIES
# ----------------------------
import pandas as pd
import pathlib as path
import numpy as np
import math
import sympy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint
import joblib

# STEP 1: LOAD AND INSPECT DATA
# ----------------------------

# Read dataset (CSV file should be in same folder as script)
data_file = pd.read_csv("Project 1 Data.csv")

# Display first few rows to verify successful import
print(data_file.head())

# Display column names and feature overview
print(data_file.columns)

# Print each variable for a quick check
print(data_file.X)
print(data_file.Y)
print(data_file.Z)
print(data_file.Step)

# STEP 2: STRATIFIED DATA SPLIT
# ----------------------------
# We split data so that the distribution of "Step" remains consistent
# in both training and testing sets.

from sklearn.model_selection import StratifiedShuffleSplit

# Create a stratified splitter (80% train, 20% test)
split_data = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Apply stratified sampling
for tr_idx, ts_idx in split_data.split(data_file, data_file["Step"]):
    data_train = data_file.iloc[tr_idx].reset_index(drop=True)
    data_test = data_file.iloc[ts_idx].reset_index(drop=True)

# Display dataset sizes and class distributions
print("\nStratified sampling complete.")
print(f"Training set size: {data_train.shape}")
print(f"Testing set size:  {data_test.shape}")
print("\nTraining class distribution:")
print(data_train['Step'].value_counts(normalize=True))
print("\nTesting class distribution:")
print(data_test['Step'].value_counts(normalize=True))

# STEP 3: PREPARE TRAIN & TEST DATA
# ----------------------------

# Define target variable (y) and predictors (X)
ytrain = data_train['Step']
ytest = data_test['Step']
Xtrain = data_train.drop(columns='Step')
Xtest = data_test.drop(columns='Step')


# STEP 4: CORRELATION ANALYSIS
# ----------------------------
# This step checks how each coordinate correlates with 'Step'
# using Pearson correlation to identify potential redundancy.

corr_table = data_train.corr()

# Print correlation values with respect to 'Step'
print("\nFeature correlations with Step:")
print("X Step:", abs(ytrain.corr(Xtrain['X'])))
print("Y Step:", abs(ytrain.corr(Xtrain['Y'])))
print("Z Step:", abs(ytrain.corr(Xtrain['Z'])))

# Plot the absolute correlation matrix
plt.figure(figsize=(6, 4))
sns.heatmap(abs(corr_table))
plt.title("Correlation Matrix (Absolute Pearson Values)")
plt.show()

# Individual correlations (rounded for clarity)
corr_X = abs(ytrain.corr(Xtrain['X']))
corr_Y = abs(ytrain.corr(Xtrain['Y']))
corr_Z = abs(ytrain.corr(Xtrain['Z']))

print("\nCorrelation of each feature with 'Step':")
print(f"  | X vs Step: {corr_X:.4f}")
print(f"  | Y vs Step: {corr_Y:.4f}")
print(f"  | Z vs Step: {corr_Z:.4f}")

# STEP 5: MODEL IMPORTS & PIPELINE SETUP
# ----------------------------
# Create machine learning pipelines for multiple classifiers.
# Each pipeline ensures consistent scaling and preprocessing.

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as pln
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.tree import DecisionTreeClassifier as TreeClass
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV as RandSearch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Logistic Regression pipeline
pipe_LogReg = pln([
    ('scale', StandardScaler()),
    ('model', LogReg(max_iter=1000, random_state=42))
])

# Decision Tree pipeline
pipe_DecTree = pln([
    ('model', TreeClass(random_state=42))
])

# Support Vector Machine pipeline
pipe_SVM = pln([
    ('model', SVC(random_state=42))
])

# Random Forest with randomized hyperparameter search
rnd_forest = RandomForestClassifier(random_state=42, n_jobs=-1)

# Define parameter grid for tuning
forest_grid = {
    'n_estimators': randint(60, 220),
    'max_depth': randint(3, 12),
    'min_samples_split': randint(2, 9),
    'min_samples_leaf': randint(1, 5)
}

# Create RandomizedSearchCV pipeline
pipe_RFSearch = pln([
    ('model', RandSearch(
        estimator=rnd_forest,
        param_distributions=forest_grid,
        n_iter=6,
        cv=4,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    ))
])


# STEP 6: STACKING CLASSIFIERS
# ----------------------------
# Combine different models using stacking to improve performance.

from sklearn.ensemble import StackingClassifier


combo6 = [('Decision Tree', pipe_DecTree), ('Random Forest', pipe_RFSearch)]

# Create stacking classifiers
stacked_6 = StackingClassifier(estimators=combo6, cv=4, n_jobs=-1, passthrough=False)

# STEP 7: MODEL TRAINING & EVALUATION
# ----------------------------
# Train all models, evaluate accuracy, and visualize confusion matrices.

# Dictionaries for storing results
y_pred_dict = {}
conf_mats = {}
score_dict = {}
prec_dict = {}
rec_dict = {}
f1_dict = {}

# Model list to iterate through
model_collection = [
    ('Logistic Regression', pipe_LogReg),
    ('Decision Tree', pipe_DecTree),
    ('SVM', pipe_SVM),
    ('Random Forest', pipe_RFSearch),

    ('Stacked Classifier 6', stacked_6)
]

# Loop through each model for training and performance evaluation
for name, mdl in model_collection:
    # Fit model on training data
    mdl.fit(Xtrain, ytrain)

    # Calculate training and testing accuracy
    train_acc = mdl.score(Xtrain, ytrain)
    test_acc = mdl.score(Xtest, ytest)
    print(f"\n{name}")
    print(f"Training Accuracy: {train_acc:.3f}")
    print(f"Testing Accuracy:  {test_acc:.3f}")

    # Generate predictions and confusion matrix
    y_pred_dict[name] = mdl.predict(Xtest)
    conf_mats[name] = confusion_matrix(ytest, y_pred_dict[name])

    # Plot confusion matrix for each model
    plt.figure(figsize=(4, 3))
    sns.heatmap(conf_mats[name], cmap='viridis', annot=True, fmt='d')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Compute precision, recall, and F1 scores
    prec_dict[name] = precision_score(ytest, y_pred_dict[name], average='weighted', zero_division=0)
    rec_dict[name] = recall_score(ytest, y_pred_dict[name], average='weighted', zero_division=0)
    f1_dict[name] = f1_score(ytest, y_pred_dict[name], average='weighted', zero_division=0)

    # Display evaluation metrics
    print(f"Precision: {prec_dict[name]:.3f}")
    print(f"Recall:    {rec_dict[name]:.3f}")
    print(f"F1 Score:  {f1_dict[name]:.3f}")

# STEP 8: SAVE FINAL MODEL
# ----------------------------
# Save the trained model to disk for future use (deployment or testing).

joblib.dump(stacked_6, 'Projectleon.joblib')
print("\nModel saved successfully as 'Projectleon.joblib'")
