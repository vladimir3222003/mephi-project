import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# загрузка данных

df = pd.read_excel(
    "Данные_для_курсовои_Классическое_МО.xlsx"
)

# создаем целевую переменную

median_si = df["SI"].median()

y = (
    df["SI"] > median_si
).astype(int)

print(y.value_counts())

# признаки

X = df.drop(
    columns=[
        "IC50, mM",
        "CC50, mM",
        "SI"
    ]
)

X = X.select_dtypes(include=np.number)

X = X.fillna(X.median())

# делим данные

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# масштабирование

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# логистическая регрессия

logreg = LogisticRegression(max_iter=1000)

logreg.fit(X_train_scaled, y_train)

pred_logreg = logreg.predict(X_test_scaled)

print("\nLogistic Regression")
print("Accuracy:", round(accuracy_score(y_test, pred_logreg), 4))
print("Precision:", round(precision_score(y_test, pred_logreg), 4))
print("Recall:", round(recall_score(y_test, pred_logreg), 4))
print("F1:", round(f1_score(y_test, pred_logreg), 4))

# дерево решений

tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

tree.fit(X_train, y_train)

pred_tree = tree.predict(X_test)

print("\nDecision Tree")
print("Accuracy:", round(accuracy_score(y_test, pred_tree), 4))
print("Precision:", round(precision_score(y_test, pred_tree), 4))
print("Recall:", round(recall_score(y_test, pred_tree), 4))
print("F1:", round(f1_score(y_test, pred_tree), 4))

# случайный лес

forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

forest.fit(X_train, y_train)

pred_forest = forest.predict(X_test)

print("\nRandom Forest")
print("Accuracy:", round(accuracy_score(y_test, pred_forest), 4))
print("Precision:", round(precision_score(y_test, pred_forest), 4))
print("Recall:", round(recall_score(y_test, pred_forest), 4))
print("F1:", round(f1_score(y_test, pred_forest), 4))
