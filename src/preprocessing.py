import pandas as pd
import ast
import re

def load_data(company_file, taxonomy_file):
    df_companies = pd.read_csv(company_file).fillna("")
    df_taxonomy = pd.read_csv(taxonomy_file)
    return df_companies, df_taxonomy

def preprocess_companies(df):
    def combine_features(row):
        try:
            tags = " ".join(ast.literal_eval(row['business_tags'])) if row['business_tags'].startswith("[") else row['business_tags']
        except Exception:
            tags = row['business_tags']
        text = f"{row['description']} {tags} {row['sector']} {row['category']} {row['niche']}"
        return re.sub(r"\s+", " ", text.strip().lower())

    df["combined_text"] = df.apply(combine_features, axis=1)
    return df

def preprocess_taxonomy(df):
    return df["label"].fillna("").str.strip().str.lower().tolist()
