import pandas as pd

def analyze_timepoint(df, timepoint, cytokines):
    print(f"testing analyze timepoint: {timepoint}")
    # TODO: filter df for timepoint

    all_results = {}
    # TODO: run the model for each cytokine using run_regression_model(). Assign output to dictionary.

    # TODO: output all_results to PDF called f"{timepoint}_all_results.pdf" 
    return 

def run_regression_model():
    # TODO: use this function to run the model and return the results
    return

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
