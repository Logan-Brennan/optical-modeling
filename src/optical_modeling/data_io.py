from pathlib import Path
import pandas as pd


def load_optical_data(file_path):
    """
    Load optical-constant data from a CSV or Excel file.

    Parameters:
    file_path : str or path-like
        Path to the optical-constant data file.

    Returns:
    pandas.DataFrame
        Tabular optical-constant data loaded from the file.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        data = pd.read_csv(file_path)

    elif suffix in [".xlsx", ".xls"]:
        data = pd.read_excel(file_path)

    else:
        raise ValueError(
            "Unsupported file type. Use CSV or Excel files."
        )

    return data




