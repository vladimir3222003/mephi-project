import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    LinearRegression,
    Ridge
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# загрузка данных

df = pd.read_excel(
    "Данные_для_курсовои_Классическое_МО.xlsx"
)

# для SI не используем IC50 и CC50,
# чтобы избежать утечки данных

X = df.drop(
    columns=[
        "IC50, mM",
        "CC50, mM",
        "SI"
    ]
)

# целевая переменная

y = df["SI"]

# оставляем только числовые признаки

X = X.select_dtypes(include=np.number)

# заполняем пропуски

X = X.fillna(X.median())

# делим выборку

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# масштабирование для линейных моделей

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# линейная регрессия

linear = LinearRegression()

linear.fit(X_train_scaled, y_train)

pred_linear = linear.predict(X_test_scaled)

print("\nLinear Regression")
print("R2:", round(r2_score(y_test, pred_linear), 4))
print("MAE:", round(mean_absolute_error(y_test, pred_linear), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred_linear)), 4))

# L2 регуляризация

ridge = Ridge(alpha=1)

ridge.fit(X_train_scaled, y_train)

pred_ridge = ridge.predict(X_test_scaled)

print("\nRidge")
print("R2:", round(r2_score(y_test, pred_ridge), 4))
print("MAE:", round(mean_absolute_error(y_test, pred_ridge), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred_ridge)), 4))

# дерево решений

tree = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

tree.fit(X_train, y_train)

pred_tree = tree.predict(X_test)

print("\nDecision Tree")
print("R2:", round(r2_score(y_test, pred_tree), 4))
print("MAE:", round(mean_absolute_error(y_test, pred_tree), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred_tree)), 4))

# случайный лес

forest = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

forest.fit(X_train, y_train)

pred_forest = forest.predict(X_test)

print("\nRandom Forest")
print("R2:", round(r2_score(y_test, pred_forest), 4))
print("MAE:", round(mean_absolute_error(y_test, pred_forest), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred_forest)), 4))
