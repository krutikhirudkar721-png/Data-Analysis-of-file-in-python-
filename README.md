# Data-Analysis-of-file-in-python-


     # =====================================================
    # UNIVERSAL DATA ANALYSIS SCRIPT (ALL PYTHON PLATFORMS)
    # =====================================================

    # ---------- 1. Import Required Libraries ----------
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import sys

    # ---------- 2. Load Dataset ----------
    # For Google Colab: upload the file first
    # For local systems: place CSV in the same folder as this script

    FILE_NAME = "PCA11-A05APPX-0000.csv"

    if not os.path.exists(FILE_NAME):
    print(f"ERROR: '{FILE_NAME}' not found.")
    print("➡️ Make sure the CSV file is in the same directory.")
    sys.exit()

    df = pd.read_csv(FILE_NAME)

    print("\n✅ Dataset loaded successfully!\n")

    # ---------- 3. Basic Dataset Overview ----------
    print("📌 Dataset Shape:", df.shape)
    print("\n📌 Column Names:")
    print(df.columns.tolist())

    print("\n📌 Data Info:")
    df.info()

    print("\n📌 First 5 Rows:")
    print(df.head())

    # ---------- 4. Clean Column Names ----------
    df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    )

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains("unnamed")]

    # ---------- 5. Handle Missing Values ----------
    print("\n📌 Missing Values Before Cleaning:")
    print(df.isnull().sum())

    # Fill numeric NaNs with median
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical NaNs with mode
    for col in df.select_dtypes(include="object").columns:
      df[col].fillna(df[col].mode()[0], inplace=True)

    print("\n📌 Missing Values After Cleaning:")
    print(df.isnull().sum())

    # ---------- 6. Convert Numeric Columns Safely ----------
    for col in df.columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors="ignore")

    # ---------- 7. Statistical Summary ----------
    print("\n📊 Statistical Summary:")
    print(df.describe(include="all"))

    # ---------- 8. Separate Numeric & Categorical Data ----------
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include="object").columns

    print("\n📌 Numeric Columns:", list(numeric_cols))
    print("📌 Categorical Columns:", list(categorical_cols))

    # ---------- 9. Data Visualization ----------
    sns.set(style="whitegrid")

    # Histograms for numeric columns
    for col in numeric_cols:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()

    # Bar plots for categorical columns (top 10)
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
         df[col].value_counts().head(10).plot(kind="bar")
        plt.title(f"Top Categories in {col}")
        plt.tight_layout()
        plt.show()

    # ---------- 10. Correlation Heatmap ----------
    if len(numeric_cols) > 1:
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # ---------- 11. Save Cleaned Dataset ----------
    OUTPUT_FILE = "cleaned_dataset.csv"
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Cleaned dataset saved as '{OUTPUT_FILE}'")
    print("\n🎉 DATA ANALYSIS COMPLETED SUCCESSFULLY 🎉")
