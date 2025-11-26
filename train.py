import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier


df = pd.read_csv("data.csv")

X = df.drop(columns=["id", "name", "mood"])
y = df["mood"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scalers = [StandardScaler(), MinMaxScaler()]

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", KNeighborsClassifier(n_neighbors=1))
])

param_grid = [
    # KNN
    {
        "scale": [StandardScaler(), MinMaxScaler()],
        "model": [KNeighborsClassifier()],
        "model__n_neighbors": list(range(1, 11)),
    },

    # SVC
    {
        "scale": [StandardScaler(), MinMaxScaler()],
        "model": [SVC(probability=True)],
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"],
    },

    # Logistic Regression
    {
        "scale": [StandardScaler(), MinMaxScaler()],
        "model": [LogisticRegression(max_iter=1000)],
        "model__C": [0.1, 1, 10],
    },

    # Random Forest
    {
        # "scale": [StandardScaler(), MinMaxScaler()], # Unnecesary for random forest, i think
        "model": [RandomForestClassifier(random_state=67)],
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4]
    }
]

knn = GridSearchCV(pipe, param_grid[0], cv=3).fit(X_train, y_train)
svc = GridSearchCV(pipe, param_grid[1], cv=3).fit(X_train, y_train)
logreg = GridSearchCV(pipe, param_grid[2], cv=3).fit(X_train, y_train)
rndforest = GridSearchCV(pipe, param_grid[3], cv=3).fit(X_train, y_train)

#mod = GridSearchCV(pipe, param_grid, cv=3)
#mod.fit(X_train, y_train)

#print(mod.best_estimator_)
#print(mod.best_params_)
#print(mod.best_score_)

models = [
    ('knn', knn.best_estimator_),
    ('svc', svc.best_estimator_),
    ('logreg', logreg.best_estimator_),
    ('rf', rndforest.best_estimator_)
]

# Define the meta-model
meta = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')

# Stacking classifier
stacker = StackingClassifier(
    estimators = models,
    final_estimator = meta,
    cv=3,
    passthrough=True
)
stacker.fit(X_train, y_train)

voter = VotingClassifier(
    estimators=models,
    voting='soft'  # 'soft' averages predicted probabilities; 'hard' uses majority vote
)
voter.fit(X_train, y_train)

pred_knn = knn.predict(X_test)
pred_svc = svc.predict(X_test)
pred_logreg = logreg.predict(X_test)
pred_rndforest = rndforest.predict(X_test)
pred_voter = voter.predict(X_test)
pred_stacker = stacker.predict(X_test)

acc_knn = accuracy_score(y_test, pred_knn)
acc_svc = accuracy_score(y_test, pred_svc)
acc_logreg = accuracy_score(y_test, pred_logreg)
acc_rndforest = accuracy_score(y_test, pred_rndforest)
acc_voter = accuracy_score(y_test, pred_voter)
acc_stacker = accuracy_score(y_test, pred_stacker)

print("KNN Accuracy:", f"{acc_knn*100:.2f}%")
print("SVC Accuracy:", f"{acc_svc*100:.2f}%")
print("LogReg Accuracy:", f"{acc_logreg*100:.2f}%")
print("Random Forest Accuracy:", f"{acc_rndforest*100:.2f}%")
print("Voting Accuracy:", f"{acc_voter*100:.2f}%")
print("Stacker Accuracy:", f"{acc_stacker*100:.2f}%")

joblib.dump(voter, "voting_model.pkl")
joblib.dump(stacker, "stacking_model.pkl")