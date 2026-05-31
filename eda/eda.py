import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# загрузка данных

df = pd.read_excel(
    "Данные_для_курсовои_Классическое_МО.xlsx"
)

# просмотр данных

print(df.head())

print("\nИнформация о датасете:")
print(df.info())

print("\nОсновные статистики:")
print(df.describe())

# проверка пропусков

print("\nКоличество пропусков:")

missing = df.isna().sum()

print(
    missing[missing > 0]
    .sort_values(ascending=False)
)

targets = [
    "IC50, mM",
    "CC50, mM",
    "SI"
]

for column in targets:

    print("\nПризнак:", column)

    value = skew(df[column].dropna())

    print("Коэффициент асимметрии:", value)

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[column],
        kde=True
    )

    plt.title(column)

    plt.show()

for column in targets:

    plt.figure(figsize=(8, 3))

    sns.boxplot(
        x=df[column]
    )

    plt.title(
        f"Boxplot: {column}"
    )

    plt.show()

corr_matrix = df.corr(
    numeric_only=True
)

plt.figure(figsize=(14, 10))

sns.heatmap(
    corr_matrix,
    cmap="coolwarm"
)

plt.title("Correlation matrix")

plt.show()

for target in targets:

    print("\nКорреляции для", target)

    print(
        corr_matrix[target]
        .sort_values(ascending=False)
        .head(10)
    )

print("\nПроверка выбросов")

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[
        (df[column] < lower)
        | (df[column] > upper)
    ]

    print(
        column,
        "-",
        len(outliers)
    )

X = df.drop(
    columns=targets
)

X = X.select_dtypes(
    include=np.number
)

X = X.fillna(
    X.median()
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

pca = PCA(
    n_components=2
)

components = pca.fit_transform(
    X_scaled
)

print("\nОбъяснённая дисперсия:")
print(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 6))

plt.scatter(
    components[:, 0],
    components[:, 1]
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title("PCA")

plt.show()
