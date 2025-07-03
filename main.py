import os
from src.config import COMPANY_FILE, TAXONOMY_FILE, OUTPUT_FILE, THRESHOLD
from src.preprocessing import load_data, preprocess_companies, preprocess_taxonomy
from src.classifier import classify_companies # type: ignore

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print("🔍 Loading data...")
    df_companies, df_taxonomy = load_data(COMPANY_FILE, TAXONOMY_FILE)

    print(" Preprocessing...")
    df_companies = preprocess_companies(df_companies)
    taxonomy_labels = preprocess_taxonomy(df_taxonomy)

    print(" Classifying...")
    df_annotated = classify_companies(df_companies, taxonomy_labels, threshold=THRESHOLD)

    print(f" Saving results to {OUTPUT_FILE}...")
    df_annotated.to_csv(OUTPUT_FILE, index=False)

    print(" Done.")

if __name__ == "__main__":
    main()
