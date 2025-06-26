import pandas as pd
import statsmodels.formula.api as smf
import os

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


    df["widespread_pain"] = df["widespread_pain"].map({"Yes": 1, "No": 0})
    df["gender"] = df["gender"].astype("category")
    df["race_ethnicity"] = df["race_ethnicity"].astype("category")

    return df

# --- Run regression and return only significant results ---
def run_regression_model(df, cytokine_variable):
    if df.empty:
        print(f"Skipping {cytokine_variable}: no data available after filtering.")
        return
    
    predictors = ["widespread_pain", "age"]

    for cat_var in ["gender", "race_ethnicity"]:
        if cat_var in df and df[cat_var].nunique() > 1:
            predictors.append(cat_var)
        else:
            print(f"Note: {cat_var} excluded from model for {cytokine_variable} due to insufficient categories.")

    formula = f"{cytokine_variable} ~ " + " + ".join(predictors)

    try:
        model = smf.ols(formula=formula, data=df, missing="drop").fit()

        summary_df = pd.DataFrame({
            "Coefficient": model.params,
            "P-value": model.pvalues
        })

        significant = summary_df[summary_df["P-value"] < 0.05]

        if not significant.empty:
            print(f"\n--- Significant Associations for {cytokine_variable.upper()} ---")
            print(significant)
        else:
            print(f"\nNo significant associations found for {cytokine_variable.upper()}.")

    except Exception as e:
        print(f"Error fitting model for {cytokine_variable}: {e}")

# --- Main function to analyze specific timepoint ---
def analyze_timepoint(df, timepoint, cytokines):
    filtered_df = clean_and_filter_data(df, timepoint)

    for cytokine in cytokines:
        if cytokine in filtered_df.columns:
            print(f"\nRunning model for: {cytokine}")
            run_regression_model(filtered_df, cytokine)
        else:
            print(f"{cytokine} not found in dataset.")

# --- Load and run ---
df = pd.read_excel("50_subjects.xlsx")

cytokines_to_test = ["il_10_pg_ml", "il_17a_pg_ml", "il_1b_pg_ml"]

analyze_timepoint(df, "baseline", cytokines_to_test)
