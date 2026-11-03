
from pathlib import Path
import pandas as pd

# Locate the dataset
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "online_shoppers_intention.csv"
)

def verify_dataset():

    print("\n===== DATASET VERIFICATION =====")

    # Step 1: Check file existence
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    # Step 2: Load dataset
    df = pd.read_csv(DATASET_PATH)

    # Step 3: Check dataset dimensions
    rows, columns = df.shape
    features = columns - 1

    print(f"Total rows: {rows}")
    print(f"Total columns: {columns}")
    print(f"Input features: {features}")

    # Step 4: Verify assignment requirements
    assert rows >= 500, "Insufficient instances"
    assert features >= 12, "Insufficient features"
    assert "Revenue" in df.columns, "Revenue target missing"
    assert df.columns.is_unique, "Duplicate column names"

    print("\nAssignment requirements: PASSED")

    # Step 5: Inspect dataset
    print("\nFirst 5 records:")
    print(df.head().to_string())

    # Step 6: Missing values
    print("\nMissing values:", df.isnull().sum().sum())

    # Step 7: Duplicate records
    print("Duplicate rows:", df.duplicated().sum())

    # Step 8: Target distribution
    print("\nRevenue distribution:")
    print(df["Revenue"].value_counts(dropna=False))

    print("\nRevenue percentage:")
    print(
        (df["Revenue"].value_counts(normalize=True) * 100)
        .round(2)
    )

    # Step 9: Data types
    print("\nData types:")
    print(df.dtypes)

    print("\nDataset verification completed successfully!")

if __name__ == "__main__":
    verify_dataset()
