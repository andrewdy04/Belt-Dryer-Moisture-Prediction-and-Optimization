import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import differential_evolution

st.set_page_config(page_title="Moisture Prediction Tool", layout="wide")
st.title("\U0001F4C8 Moisture Prediction & Optimization App")

# --- Find available product data files ---
@st.cache_data
def list_excel_files():
    files = [f for f in os.listdir('.') if f.endswith("Moisture Data.xlsx")]
    return files

# --- Load and process dataset ---
@st.cache_data
def load_data(filename):
    df = pd.read_excel(filename)
    df = clean_and_validate_data(df)
    return df


def clean_and_validate_data(df):
    required_cols = ['Final_Moisture', 'Flowrate', 'Tank_Level', 'Low_Side_Vac',
                     '1st_Temp', '2nd_Temp', '3rd_Temp', '4th_Temp']

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df[required_cols].copy()
    df = df.dropna()

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()

    return df


# --- Train model ---
def train_model(df):
    X = df[['Flowrate', 'Tank_Level', 'Low_Side_Vac',
            '1st_Temp', '2nd_Temp', '3rd_Temp', '4th_Temp']]
    y = df['Final_Moisture']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X)
    y_pred_test = model.predict(X_test)

    r2_full = r2_score(y, y_pred)
    rmse_full = np.sqrt(mean_squared_error(y, y_pred))
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    return model, r2_full, rmse_full, r2_test, rmse_test, X.columns, X_train, y_train

# --- Plot feature importance ---
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    fig, ax = plt.subplots()
    sns.barplot(x=importances, y=feature_names, ax=ax)
    ax.set_title("Feature Importance (Higher = More Influence on Prediction)")
    st.pyplot(fig)

# --- UI ---
data_files = list_excel_files()

if not data_files:
    st.warning("No 'Moisture Data.xlsx' files found in this directory.")
else:
    selected_file = st.selectbox("Select Product Dataset:", data_files)
    df = load_data(selected_file)
    model, r2_full, rmse_full, r2_test, rmse_test, feature_names, X_train, y_train = train_model(df)

    st.subheader("Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² (All Data)", f"{r2_full:.3f}")
    col2.metric("RMSE (All Data)", f"{rmse_full:.3f}")
    col3.metric("R² (Test Only)", f"{r2_test:.3f}")
    col4.metric("RMSE (Test Only)", f"{rmse_test:.3f}")

    st.subheader("Feature Importance")
    plot_feature_importance(model, feature_names)

    st.subheader("Predict Final Moisture")
    with st.form("predict_form"):
        flow = st.number_input("Flowrate (gpm)", value=40.0)
        tank = st.number_input("Tank Level", value=45.0)
        vac = st.number_input("Vacuum (inHg)", value=6.0)
        t1 = st.number_input("1st Temp (°F)", value=180.0)
        t2 = st.number_input("2nd Temp (°F)", value=240.0)
        t3 = st.number_input("3rd Temp (°F)", value=260.0)
        t4 = st.number_input("4th Temp (°F)", value=260.0)
        submitted = st.form_submit_button("Predict")

    if submitted:
        st.write("Success")

    input_df = pd.DataFrame([[flow, tank, vac, t1, t2, t3, t4]], columns=feature_names)
    prediction = model.predict(input_df)[0]

    # Compute residuals and CI
    try:
        residuals = model.predict(X_train) - y_train
        std_dev = np.std(residuals)
        margin = 1.645 * std_dev  # 90% confidence
        lower = prediction - margin
        upper = prediction + margin

        st.success(f"Predicted Final Moisture: {prediction:.2f}%")
        st.info(f"With 90% confidence, moisture is between {lower:.2f}% and {upper:.2f}%")

        # Optional: save to file (for testing)
        result_df = input_df.copy()
        result_df["Prediction"] = prediction
        result_df["CI Lower"] = lower
        result_df["CI Upper"] = upper
    
    except Exception as e:
        st.error(f"Failed to compute confidence interval: {e}")

    st.subheader("Optimize to Target Moisture")
    with st.form("optimize_form"):
        target = st.number_input("Target Final Moisture (%)", value=8.0)
        opt_flow = st.text_input("Flowrate (leave blank to optimize)")
        opt_tank = st.text_input("Tank Level (leave blank to optimize)")
        opt_vac = st.text_input("Vacuum (leave blank to optimize)")
        opt_t1 = st.text_input("1st Temp (leave blank to optimize)")
        opt_t2 = st.text_input("2nd Temp (leave blank to optimize)")
        opt_t3 = st.text_input("3rd Temp (leave blank to optimize)")
        opt_t4 = st.text_input("4th Temp (leave blank to optimize)")
        opt_submit = st.form_submit_button("Optimize")

    if opt_submit:
        bounds = {
            'Flowrate': (15, 50),
            'Tank_Level': (20, 60),
            'Low_Side_Vac': (1, 15),
            '1st_Temp': (120, 250),
            '2nd_Temp': (180, 300),
            '3rd_Temp': (180, 320),
            '4th_Temp': (180, 300)
        }

        fixed = {
            'Flowrate': opt_flow,
            'Tank_Level': opt_tank,
            'Low_Side_Vac': opt_vac,
            '1st_Temp': opt_t1,
            '2nd_Temp': opt_t2,
            '3rd_Temp': opt_t3,
            '4th_Temp': opt_t4
        }

        opt_vars = []
        opt_bounds = []
        fixed_values = {}

        for key, val in fixed.items():
            if val.strip() == '':
                opt_vars.append(key)
                opt_bounds.append(bounds[key])
            else:
                fixed_values[key] = float(val)

        if len(opt_vars) == 0:
            st.warning("You must leave at least one variable blank to optimize.")
        else:
            def objective(x):
                inputs = fixed_values.copy()
                for i, var in enumerate(opt_vars):
                    inputs[var] = x[i]
                row = pd.DataFrame([[inputs[col] for col in feature_names]], columns=feature_names)
                pred = model.predict(row)[0]
                return (pred - target) ** 2

            with st.spinner("Optimizing, please wait..."):
                result = differential_evolution(
                    objective,
                    opt_bounds,
                    seed=42,
                    maxiter=1000,
                    tol=1e-5,
                    polish=True
                )

            if result.success or result.fun < 1.0:
                for i, var in enumerate(opt_vars):
                    fixed_values[var] = result.x[i]
                row = pd.DataFrame([[fixed_values[col] for col in feature_names]], columns=feature_names)
                pred = model.predict(row)[0]

                st.success("Optimization Complete!")
                for k, v in fixed_values.items():
                    st.write(f"**{k}**: {v:.2f}")
                st.write(f"**Predicted Moisture**: {pred:.2f}% (Target: {target:.2f}%)")
            else:
                st.warning("Optimization did not fully converge, but here is the best result found:")
                for i, var in enumerate(opt_vars):
                    fixed_values[var] = result.x[i]
                row = pd.DataFrame([[fixed_values[col] for col in feature_names]], columns=feature_names)
                pred = model.predict(row)[0]
                for k, v in fixed_values.items():
                    st.write(f"**{k}**: {v:.2f}")
                st.write(f"**Predicted Moisture**: {pred:.2f}% (Target: {target:.2f}%)")
