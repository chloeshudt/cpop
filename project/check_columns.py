import pandas as pd

# Load Excel file
df = pd.read_excel(r"C:\Users\lniu0\duke_data\50_subjects.xlsx")

# Clean column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(r"[ /()\-\[\]]", "_", regex=True)
    .str.replace(r"__+", "_", regex=True)
    .str.strip("_")
)

# Print out cleaned column names
print("🧪 Cleaned column names:\n", df.columns.tolist())
