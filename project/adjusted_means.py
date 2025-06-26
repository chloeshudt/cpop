import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import traceback
from itertools import combinations
from patsy import dmatrix
from pandas.api.types import CategoricalDtype

def clean_and_filter_data(df, timepoint):
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[ /()-]", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )

    df = df[df["timepoint"].str.lower() == timepoint.lower()].copy()

    if "widespread_pain" in df.columns and df["widespread_pain"].dtype == object:
        df["widespread_pain"] = df["widespread_pain"].map({"Yes": 1, "No": 0})

    for cat_var in ["gender", "race_ethnicity", "glycemic_group"]:
        if cat_var in df.columns:
            df[cat_var] = df[cat_var].astype("category")

    return df


from patsy import dmatrix
from pandas.api.types import CategoricalDtype

def get_adjusted_means(model, df, predictor):
    """
    Calculate adjusted means and standard errors for each category of a categorical predictor.
    """
    if predictor not in df.columns:
        return None

    # Check if the variable is categorical or binary numeric
    is_cat = isinstance(df[predictor].dtype, CategoricalDtype) or df[predictor].dtype == object

    try:
        is_binary_numeric = (
            df[predictor].dropna().nunique() == 2 and
            np.issubdtype(np.dtype(df[predictor].dtype), np.number)
        )
    except TypeError:
        is_binary_numeric = False

    if not (is_cat or is_binary_numeric):
        return None

    categories = df[predictor].dropna().unique()
    results = []

    for cat in categories:
        df_copy = df.copy()
        df_copy[predictor] = cat  # Set all rows to the same category level

        try:
            design_info = model.model.data.design_info
            exog = dmatrix(design_info, df_copy, return_type='dataframe')
        except Exception as e:
            print(f"Error building design matrix for predictor {predictor} category {cat}: {e}")
            continue

        preds = np.dot(exog, model.params)
        mean_pred = np.mean(preds)

        cov = model.cov_params()
        ses = []
        for row in exog.values:
            se = np.sqrt(np.dot(np.dot(row, cov), row.T))
            ses.append(se)
        se_pred = np.mean(ses)

        results.append({
            "category": cat,
            "adjusted_mean": mean_pred,
            "std_error": se_pred
        })

    return results



def run_regression_model(df, cytokine_variable):
    if df.empty:
        print(f"Skipping {cytokine_variable}: no data available after filtering.")
        return None

    predictors = ["widespread_pain", "age"]
    for cat_var in ["gender", "race_ethnicity", "glycemic_group"]:
        if cat_var in df.columns and df[cat_var].nunique() > 1:
            predictors.append(cat_var)

    interaction_terms = [f"{a}:{b}" for a, b in combinations(predictors, 2)]
    formula = f"{cytokine_variable} ~ " + " + ".join(predictors + interaction_terms)

    try:
        print(f"\nRunning model for: {cytokine_variable}")
        model = smf.ols(formula=formula, data=df, missing="drop").fit()
        print(model.summary())

        summary_df = pd.DataFrame({
            "Variable": model.params.index,
            "Coefficient": model.params.values,
            "P-value": model.pvalues.values
        })

        significant = summary_df[
            (summary_df["P-value"] < 0.05) & (summary_df["Variable"] != "Intercept")
        ]

        adjusted_means_info = {}
        for var in significant["Variable"]:
            if ":" in var:  # skip interaction terms for now
                continue
            base_var = var.split("[")[0]
            means = get_adjusted_means(model, df, base_var)
            if means:
                adjusted_means_info[base_var] = means

        return {
            "significant": significant,
            "adjusted_means": adjusted_means_info
        }

    except Exception as e:
        print(f"Error fitting model for {cytokine_variable}: {e}")
        traceback.print_exc()
        return None


def run_regression_model(df, cytokine_variable):
    if df.empty:
        print(f"Skipping {cytokine_variable}: no data available after filtering.")
        return None

    predictors = ["widespread_pain", "age"]
    for cat_var in ["gender", "race_ethnicity", "glycemic_group"]:
        if cat_var in df.columns and df[cat_var].nunique() > 1:
            predictors.append(cat_var)

    interaction_terms = [f"{a}:{b}" for a, b in combinations(predictors, 2)]
    formula = f"{cytokine_variable} ~ " + " + ".join(predictors + interaction_terms)

    try:
        model = smf.ols(formula=formula, data=df, missing="drop").fit()

        summary_df = pd.DataFrame({
            "Variable": model.params.index,
            "Coefficient": model.params.values,
            "P-value": model.pvalues.values
        })

        adjusted_means_info = {}
        for predictor in predictors:
            if predictor in df.columns and (
                isinstance(df[predictor].dtype, CategoricalDtype) or 
                df[predictor].dtype == object or 
                (df[predictor].dropna().nunique() == 2 and np.issubdtype(df[predictor].dtype, np.number))
            ):
                means = get_adjusted_means(model, df, predictor)
                if means:
                    adjusted_means_info[predictor] = means

        return {
            "significant": summary_df[
                (summary_df["P-value"] < 0.05) & (summary_df["Variable"] != "Intercept")
            ],
            "adjusted_means": adjusted_means_info
        }

    except Exception as e:
        print(f"Error fitting model for {cytokine_variable}: {e}")
        traceback.print_exc()
        return None


def save_significant_to_pdf(results_dict, pdf_filename):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    elements = []

    # Part 1: Significant associations for each cytokine
    for cytokine, result in results_dict.items():
        elements.append(Paragraph(f"Significant Associations for {cytokine.upper()}", styles['Heading2']))
        elements.append(Spacer(1, 12))

        if result is None or result.get("significant") is None or result["significant"].empty:
            elements.append(Paragraph("No significant predictor associations found.", styles['Normal']))
            elements.append(Spacer(1, 24))
            continue

        sig_df = result["significant"]
        data = [["Variable", "Coefficient", "P-value"]]
        for _, row in sig_df.iterrows():
            data.append([
                row['Variable'],
                f"{row['Coefficient']:.4f}",
                f"{row['P-value']:.4g}"
            ])

        table = Table(data, colWidths=[250, 100, 100])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 24))

    # Part 2: Adjusted Means Tables (by categorical predictor)
    all_predictors = set()
    for result in results_dict.values():
        if result and "adjusted_means" in result:
            all_predictors.update(result["adjusted_means"].keys())

    for predictor in sorted(all_predictors):
        # Determine categories (from any result that includes this predictor)
        categories = []
        for result in results_dict.values():
            if result and predictor in result.get("adjusted_means", {}):
                categories = [entry["category"] for entry in result["adjusted_means"][predictor]]
                break

        if not categories:
            continue

        header = ["Cytokine"] + [str(cat) for cat in categories] + ["Std Error", "P"]
        data = [header]

        for cytokine, result in results_dict.items():
            if not result:
                continue

            means_list = result.get("adjusted_means", {}).get(predictor)
            if not means_list or len(means_list) != len(categories):
                continue

            row = [cytokine.upper()]
            for entry in means_list:
                row.append(f"{entry['adjusted_mean']:.2f} pg/mL")

            std_error = np.mean([entry["std_error"] for entry in means_list])
            row.append(f"± {std_error:.2f} pg/mL")

            sig_df = result.get("significant")
            if sig_df is not None and not sig_df.empty:
                pval_row = sig_df[sig_df["Variable"].str.contains(fr"{predictor}", regex=True)]
                pval = f"{pval_row['P-value'].values[0]:.4g}" if not pval_row.empty else "N/A"
            else:
                pval = "N/A"

            row.append(pval)
            data.append(row)

        if len(data) > 1:
            elements.append(Paragraph(f"Adjusted Means by {predictor.replace('_', ' ').title()}", styles['Heading2']))
            elements.append(Spacer(1, 12))

            table = Table(data, colWidths=[80] + [80]*len(categories) + [80, 60])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 24))

    doc.build(elements)



def analyze_timepoint(df, timepoint, cytokines):
    filtered_df = clean_and_filter_data(df, timepoint)
    all_significant = {}

    for cytokine in cytokines:
        if cytokine in filtered_df.columns:
            sig_df = run_regression_model(filtered_df, cytokine)
            all_significant[cytokine] = sig_df
        else:
            print(f"{cytokine} not found in dataset.")
            all_significant[cytokine] = None

    pdf_path = os.path.join(os.getcwd(), "significant_associations.pdf")
    save_significant_to_pdf(all_significant, pdf_path)
    print(f"\nSignificant associations saved to PDF: {pdf_path}")

if __name__ == "__main__":
    df = pd.read_excel("simulated_50_subjects.xlsx")
    cytokines_to_test = ["il_10_pg_ml", "il_17a_pg_ml", "il_1b_pg_ml"]
    analyze_timepoint(df, "baseline", cytokines_to_test)
