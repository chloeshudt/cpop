import pandas as pd

def analyze_timepoint(df, timepoint, cytokines):
    print(f"testing analyze timepoint: {timepoint}")
    # run regression model for each cytokine by calling run_regression()

    # TODO research what stats tools to use for your model. 

    return 

def run_regression():
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
    timepoints_to_test = ["baseline", "3mo"]

    # run single timepoint analyses
    for timepoint in timepoints_to_test:
        analyze_timepoint(clean_df, timepoint, cytokines_to_test)

    # output all-analyses report
    # output significant association report
