import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
# 1️⃣ Load dataset
data = pd.read_csv("train.csv")

# 2️⃣ Separate features and target
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]

# 3️⃣ Identify numerical and categorical columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# 4️⃣ Preprocessing
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean"))
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# 5️⃣ Full pipeline with strong model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(n_estimators=200, random_state=42))
])

# 6️⃣ Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7️⃣ Train model
model.fit(X_train, y_train)

# 8️⃣ Evaluate
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

print("Improved Model Accuracy:", accuracy)

# 9️⃣ Save
with open("advanced_model.pkl", "wb") as f:
    pickle.dump((model, accuracy), f)