from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Load the dataset and clean up inconsistent headers/rows.
data_path = Path(__file__).with_name("test _main.csv")
if not data_path.exists():
    data_path = Path(__file__).with_name("main.csv")

df = pd.read_csv(data_path, skipinitialspace=True)
df.columns = df.columns.str.strip()

if "loction" in df.columns and "location" not in df.columns:
    df = df.rename(columns={"loction": "location"})

df = df[df["price"].astype(str).str.strip() != "price"].copy()

numeric_columns = ["area", "bedrooms", "year_build", "price"]
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df["location"] = df["location"].astype(str).str.strip()
df = df.dropna(subset=["area", "bedrooms", "location", "year_build", "price"])

print("Dataset preview:")
print(df.head())

# Features and target.
x = df[["area", "bedrooms", "location", "year_build"]]
y = df["price"]

# Preprocess the categorical column and pass numeric columns through.
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), ["location"])],
    remainder="passthrough",
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model.fit(x_train, y_train)

# Predict on the test set.
y_pred = model.predict(x_test)

print("\nModel evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 score:", r2_score(y_test, y_pred))

# Predict the price for a new house.
new_house = pd.DataFrame({
    "area": [1600],
    "bedrooms": [3],
    "location": ["dwarka"],
    "year_build": [4],
})

predicted_price = model.predict(new_house)
print("\nPredicted house price:", predicted_price[0])
