import os
import pandas as pd
import statsmodels.formula.api as smf
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import traceback

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


from itertools import combinations

def run_regression_model(df, cytokine_variable):
    if df.empty:
        print(f"Skipping {cytokine_variable}: no data available after filtering.")
        return None

    # Define main predictors
    predictors = ["widespread_pain", "age"]
    for cat_var in ["gender", "race_ethnicity", "glycemic_group"]:
        if cat_var in df.columns and df[cat_var].nunique() > 1:
            predictors.append(cat_var)

    # Generate all two-way interaction terms
    interaction_terms = [f"{a}:{b}" for a, b in combinations(predictors, 2)]

    # Build full regression formula
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

        return significant

    except Exception as e:
        print(f"Error fitting model for {cytokine_variable}: {e}")
        traceback.print_exc()
        return None



def save_significant_to_pdf(significant_results_dict, pdf_filename):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    elements = []

    for cytokine, df_sig in significant_results_dict.items():
        elements.append(Paragraph(f"Significant Associations for {cytokine.upper()}", styles['Heading2']))
        elements.append(Spacer(1, 12))

        if df_sig is None or df_sig.empty:
            elements.append(Paragraph("No significant predictor associations found.", styles['Normal']))
        else:
            data = [list(df_sig.columns)]
            for _, row in df_sig.iterrows():
                data.append([
                    row['Variable'],
                    f"{row['Coefficient']:.4f}",
                    f"{row['P-value']:.4g}"
                ])

            table = Table(data, colWidths=[250, 100, 100])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                ('ALIGN',(1,1),(-1,-1),'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 8),
                ('GRID', (0,0), (-1,-1), 0.5, colors.black),
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
