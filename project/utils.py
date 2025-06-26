def clean_data(df):
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[ /()-]", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.strip("_")
    )
    
    if "widespread_pain" in df.columns and df["widespread_pain"].dtype == object:
        df["widespread_pain"] = df["widespread_pain"].map({"Yes": 1, "No": 0})

    for cat_var in ["gender", "race_ethnicity", "glycemic_group"]:
        if cat_var in df.columns:
            df[cat_var] = df[cat_var].astype("category")

    return df
