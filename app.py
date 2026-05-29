from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(page_title="Diamond Price Prediction", page_icon="💎", layout="centered")


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "diamonds.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_model(df: pd.DataFrame) -> tuple[Pipeline, float, float]:
    X = df.drop("price", axis=1)
    y = df["price"]

    categorical_cols = ["cut", "color", "clarity"]
    numeric_cols = ["carat", "depth", "table", "x", "y", "z"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return model, mse, r2


df = load_data()
model, mse, r2 = train_model(df)

st.title("💎 Diamond Price Prediction App")
st.write("Enter the diamond details below and get an estimated price.")

col1, col2 = st.columns(2)

with col1:
    carat = st.number_input("Carat", min_value=0.1, max_value=5.0, step=0.01, value=1.0)
    cut = st.selectbox("Cut", sorted(df["cut"].unique().tolist()))
    color = st.selectbox("Color", sorted(df["color"].unique().tolist()))
    clarity = st.selectbox("Clarity", sorted(df["clarity"].unique().tolist()))

with col2:
    depth = st.number_input("Depth %", min_value=40.0, max_value=80.0, step=0.1, value=61.0)
    table = st.number_input("Table %", min_value=40.0, max_value=80.0, step=0.1, value=57.0)
    x = st.number_input("Length (x mm)", min_value=2.0, max_value=10.0, step=0.01, value=5.0)
    y = st.number_input("Width (y mm)", min_value=2.0, max_value=10.0, step=0.01, value=5.0)
    z = st.number_input("Depth (z mm)", min_value=1.0, max_value=10.0, step=0.01, value=3.0)

if st.button("Predict Price"):
    new_diamond = pd.DataFrame(
        [
            {
                "carat": carat,
                "cut": cut,
                "color": color,
                "clarity": clarity,
                "depth": depth,
                "table": table,
                "x": x,
                "y": y,
                "z": z,
            }
        ]
    )
    predicted_price = model.predict(new_diamond)[0]
    st.success(f"Predicted Diamond Price: ${predicted_price:,.2f}")


with st.expander("Model details"):
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R² Score: {r2:.4f}")
    st.caption("Built from diamonds.csv in the same folder.")