import pandas as pd

def clean_data(df):
    # ensure all columns and data are snakecase
    df.columns = df.columns.str.lower()
    df = df.applymap(lambda x: x.lower().replace("-", "_") if isinstance(x, str) else x)
    return(df)

if __name__ == "__main__":
    # load raw dataset
    df = pd.read_excel("data/simulated_50_subjects.xlsx")
    clean_df = clean_data(df)
    # clean data set
    # run analysis
    # output all-analyses report
    # output significant association report
