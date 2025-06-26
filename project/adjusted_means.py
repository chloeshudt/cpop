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
from patsy import dmatrix
from pandas.api.types import CategoricalDtype

def get_adjusted_means(model, df, predictor):
    """
    Calculate adjusted means and standard errors for each category of a categorical predictor.
    """
    # Check if the variable is categorical or binary numeric
    # TODO: is categorical specifically 3+ options?
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

def run_regression_model(df, cytokine_variable):

    # define independent vars of interest
    predictors = ["age", "gender", "race_ethnicity", "glycemic_group", "widespread_pain"]

    combos = combinations(predictors, 2)

    interaction_terms = [f"{a}:{b}" for a, b in combos]

    formula = f"{cytokine_variable} ~ " + " + ".join(predictors + interaction_terms)

    # TODO: look up what it means for the model to 'fail'
    try:
        # TODO: CONFIRM MODEL APPROACH
        model = smf.ols(formula=formula, data=df, missing="drop").fit()

        summary_df = pd.DataFrame({
            "Variable": model.params.index,
            "Coefficient": model.params.values,
            "P-value": model.pvalues.values
        })

        adjusted_means_info = {}
        for predictor in predictors:
            if (
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

def analyze_timepoint(df, timepoint, cytokines):
    # filters data to specific timepoint
    filtered_df = df[df["timepoint"].str.lower() == timepoint.lower()].copy()
    
    all_significant = {}

    # run regression model for each cytokine and save significant 
    for cytokine in cytokines:
        sig_df = run_regression_model(filtered_df, cytokine)
        all_significant[cytokine] = sig_df

    pdf_path = os.path.join(os.getcwd(), "significant_associations.pdf")
    
    save_significant_to_pdf(all_significant, pdf_path)
    print(f"\nSignificant associations saved to PDF: {pdf_path}")

def clean_data(df):
    # ensure all columns and data are snakecase
    df.columns = df.columns.str.lower()
    df = df.map(lambda x: x.lower().replace("-", "_") if isinstance(x, str) else x)
    return(df)

if __name__ == "__main__":
    # load and clean patient data
    raw_df = pd.read_excel("data/simulated_50_subjects.xlsx")
    clean_df = clean_data(raw_df)

    # run single-timepoint analyses #####################################################
    # defining our dependent variables of interest
    cytokines_to_test = ["il_10_pg_ml", "il_17a_pg_ml"]
    timepoints_to_test = ["baseline"]

    # run single timepoint analyses
    for timepoint in timepoints_to_test:
        analyze_timepoint(clean_df, timepoint, cytokines_to_test)