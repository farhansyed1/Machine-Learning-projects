"""
Creating a stacking classifier with XGBoost, Random forest and Naive Bayes
Farhan Syed, 2024

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from scipy.stats import randint

# Preprocessing - training data
df = pd.read_csv('TrainOnMe_orig.csv').drop(['x12', 'Unnamed: 0'], axis=1) # x12 consists of true values for all rows   
df = pd.get_dummies(df, columns=['x7']).astype({'x7_Beach': 'uint8', 'x7_Casahouse': 'uint8', 'x7_Horses': 'uint8', 'x7_Mojodojo': 'uint8', 'x7_Sublime': 'uint8'})

le = LabelEncoder()
df['y'] = le.fit_transform(df['y']) # Convert target variable to numbers

scaler = StandardScaler()
features = df.drop('y', axis=1).columns
df[features] = scaler.fit_transform(df[features])

# Preprocessing - evaluation data 
evaluation_df = pd.read_csv('EvaluateOnMe.csv').drop(['x12', 'Unnamed: 0'], axis=1)
evaluation_df = pd.get_dummies(evaluation_df, columns=['x7']).astype({'x7_Beach': 'uint8', 'x7_Casahouse': 'uint8', 'x7_Horses': 'uint8', 'x7_Mojodojo': 'uint8', 'x7_Sublime': 'uint8'})
evaluation_df[features] = scaler.transform(evaluation_df[features])

# Creating training and testing sets
X, Y = df.drop('y', axis=1), df['y']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)

# Function to train and evaluate a model
def train_evaluate_model(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print(f"Accuracy on test set: {accuracy_score(Y_test, Y_pred):.4f}")

# Hyperparameter grids for xgBoost and random forest
xgb_grid = {
    'learning_rate': np.linspace(0.01, 0.5, 10),
    'n_estimators': [100, 200, 300],
    'max_depth': randint(10, 30),
    'subsample': np.linspace(0.5, 1.0, 5),
    'colsample_bytree': np.linspace(0.5, 1.0, 5),
}

forest_grid = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': randint(10, 70),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
}

models = {
    "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_grid),
    "Random Forest": (RandomForestClassifier(criterion='entropy', class_weight='balanced'), forest_grid),
}

bestModels = {}

# Finding best models
for name, (model, params) in models.items():
    print(f"Finding best parameters of {name}")
    s = RandomizedSearchCV(model, param_distributions=params, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1)
    s.fit(X_train, Y_train)
    bestModels[name] = s.best_estimator_
    train_evaluate_model(s.best_estimator_, X_train, Y_train, X_test, Y_test)

bestModels["Gaussian Naive Bayes"] = GaussianNB() # Naive bayes also used for stacking classifier 

# Create stacking classifier
estimators = [(name.lower().replace(" ", "_"), model) for name, model in bestModels.items()]
stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), n_jobs=-1)
print("Stacking Classifier: ")
train_evaluate_model(stacking_classifier, X_train, Y_train, X_test, Y_test)

# Make predictions with the stacking classifier
y_eval_pred = stacking_classifier.predict(evaluation_df[features])  
y_eval_pred_labels = le.inverse_transform(y_eval_pred)
prediction_df = pd.DataFrame(y_eval_pred_labels, columns=['Predicted'])
prediction_df.to_csv('Prediction.txt', header=False, index=False)