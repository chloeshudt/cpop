import pandas as pd
import statsmodels.formula.api as smf

# Load the dataset
df = pd.read_excel("practice_data.xlsx")

# Filter to baseline only
baseline_df = df[df["Timepoint"] == "Baseline"].copy()

# Convert and clean variables
baseline_df["Widespread_Pain"] = baseline_df["Widespread Pain"].map({"Yes": 1, "No": 0})
baseline_df["Gender"] = baseline_df["Gender"].astype("category")
baseline_df["Race_Ethnicity"] = baseline_df["Race/Ethnicity"].astype("category")
baseline_df["IL10"] = baseline_df["IL-10 (pg/mL)"]  # Rename for formula compatibility

# Remove non-varying predictors
for col in ["Gender", "Race_Ethnicity"]:
    if baseline_df[col].nunique() <= 1:
        print(f"Dropping column with no variation: {col}")
        baseline_df.drop(columns=[col], inplace=True)

# Check and drop rows with missing values in relevant columns
baseline_df.dropna(subset=["IL10", "Widespread_Pain", "Age"], inplace=True)

# Fit fixed-effects only model (no random effect)
model = smf.ols(
    formula="IL10 ~ Widespread_Pain + Age" + (
        " + Gender" if "Gender" in baseline_df.columns else ""
    ) + (
        " + Race_Ethnicity" if "Race_Ethnicity" in baseline_df.columns else ""
    ),
    data=baseline_df
).fit()

# Output results
print(model.summary())