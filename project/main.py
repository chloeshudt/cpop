import pandas as pd
import statsmodels.formula.api as smf
from itertools import combinations
from reportlab.pdfgen import canvas

def analyze_timepoint(df, timepoint, cytokines):
    print(f"testing analyze timepoint: {timepoint}")
    # TODO: filter df for timepoint
    filtered_df = df[df["timepoint"] == timepoint].copy()

    all_significant = {}
    # TODO: run the model for each cytokine using run_regression_model(). Assign output to dictionary.
    for cytokine in cytokines:
        sig_df = run_regression_model(filtered_df, cytokine)
        all_significant[cytokine] = sig_df
    # TODO: output all_results to PDF called f"{timepoint}_all_results.pdf" 
    pdf_path = f"{timepoint}_all_results.pdf"
    save_significant_to_pdf(all_significant, pdf_path)
    return all_significant

def save_significant_to_pdf(results_dict, pdf_path):
    #save to pdf
    c = canvas.Canvas(pdf_path)
    text = c.beginText(50, 800)
    text.setFont("Helvetica", 10)

    #search cytokines in results dictionary 
    for cytokine, result in results_dict.items():
        text.textLine(f"Results for {cytokine}:")
        #pulls significant results only
        sig_df = result["significant"]
        for _, row in sig_df.iterrows():
            variable = row["Variable"]
            coef = row["Coefficient"]
            pval = row["P-value"]
            text.textLine(f"  {variable}: coef={coef:.3f}, p={pval:.3f}")
        text.textLine("")  # blank line between cytokines

    c.drawText(text)
    c.save()
    print(f"Results saved to PDF at: {pdf_path}")

def run_regression_model(df, cytokine_variable):
    # TODO: use this function to run the model and return the results
    # define independent vars of interest
    predictors = ["age", "gender", "race_ethnicity", "glycemic_group", "widespread_pain"]

    #generate all possible pairs of predictors for building two-way interactions
    combos = combinations(predictors, 2)

    #syntax for interaction terms
    interaction_terms = [f"{a}:{b}" for a, b in combos]

    #formula for statsmodel dependent variable ~ all independent varaibles
    formula = f"{cytokine_variable} ~ " + " + ".join(predictors + interaction_terms)

    #fitting OLS regression, will drop any rows with missing data 
    model = smf.ols(formula=formula, data=df, missing="drop").fit()
    print(model)

    #output summary with variable names, coefficient, p-values
    summary_df = pd.DataFrame({"Variable": model.params.index,"Coefficient": model.params.values,"P-value": model.pvalues.values})
    return {"significant": summary_df[(summary_df["P-value"] < 0.05) & (summary_df["Variable"] != "Intercept")]}

def clean_data(df):
    # ensure all columns and data are snakecase
    df.columns = df.columns.str.lower()
    df = df.map(lambda x: x.lower().replace("-", "_") if isinstance(x, str) else x)
    return(df)

if __name__ == "__main__":
    # load and clean data
    df = pd.read_excel("data/simulated_50_subjects.xlsx")
    clean_df = clean_data(df)

    # define which cytokines and timepoints to analyze
    cytokines_to_test = ["il_10_pg_ml", "il_17a_pg_ml"]

    # run single timepoint analyses
    analyze_timepoint(clean_df, "baseline", cytokines_to_test)

    # output all-analyses report
    # output significant association report
