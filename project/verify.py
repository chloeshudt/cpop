import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import traceback

# --- Clean and filter data to specific timepoint ---
def clean_and_filter_data(df, timepoint):
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[ /()-]", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )

    df = df[df["timepoint"].str.lower() == timepoint.lower()].copy()

    df["widespread_pain"] = df["widespread_pain"].map({"Yes": 1, "No": 0}) if df["widespread_pain"].dtype == object else df["widespread_pain"]
    df["gender"] = df["gender"].astype("category")
    df["race_ethnicity"] = df["race_ethnicity"].astype("category")

    return df

# --- Run regression and return only significant results with labels ---
def run_regression_model(df, cytokine_variable):
    if df.empty:
        print(f"Skipping {cytokine_variable}: no data available after filtering.")
        return

    predictors = ["widespread_pain", "age"]

    for cat_var in ["gender", "race_ethnicity"]:
        if cat_var in df.columns and df[cat_var].nunique() > 1:
            predictors.append(cat_var)
        else:
            print(f"Note: {cat_var} excluded from model for {cytokine_variable} due to insufficient categories or missing column.")

    formula = f"{cytokine_variable} ~ " + " + ".join(predictors)

    try:
        model = smf.ols(formula=formula, data=df, missing="drop").fit()

        summary_df = pd.DataFrame({
            "Variable": model.params.index,
            "Coefficient": model.params.values,
            "P-value": model.pvalues.values
        })

        significant = summary_df[
            (summary_df["P-value"] < 0.05) & (summary_df["Variable"] != "Intercept")
        ]

        if not significant.empty:
            print(f"\n--- Significant Associations for {cytokine_variable.upper()} ---")
            print(significant[["Variable", "Coefficient", "P-value"]].to_string(index=False))
        else:
            print(f"\nNo significant predictor associations found for {cytokine_variable.upper()}.")

    except Exception as e:
        print(f"Error fitting model for {cytokine_variable}: {e}")
        traceback.print_exc()

# --- Main function to analyze specific timepoint ---
def analyze_timepoint(df, timepoint, cytokines):
    filtered_df = clean_and_filter_data(df, timepoint)

    for cytokine in cytokines:
        if cytokine in filtered_df.columns:
            print(f"\nRunning model for: {cytokine}")
            run_regression_model(filtered_df, cytokine)
        else:
            print(f"{cytokine} not found in dataset.")

# --- Create a simulated dataset for verification ---
np.random.seed(1)

n = 100
test_df = pd.DataFrame({
    "timepoint": ["baseline"] * n,
    "age": np.linspace(20, 60, n),
    "widespread_pain": np.random.choice([0, 1], n),
    "gender": np.random.choice(["Male", "Female"], n),
    "race_ethnicity": np.random.choice(["Black", "Latino"], n),
    "il_17a_pg_ml": 5 + 0.08 * np.linspace(20, 60, n) + np.random.normal(0, 0.5, n),
    "il_10_pg_ml": 3 + 0.01 * np.random.randn(n),
    "il_1b_pg_ml": 6 + 0.03 * np.random.randn(n)
})

# --- Run the verification analysis ---
cytokines_to_test = ["il_10_pg_ml", "il_17a_pg_ml", "il_1b_pg_ml"]
analyze_timepoint(test_df, "baseline", cytokines_to_test)
