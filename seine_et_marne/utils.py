# =======================================================================================
# 2026 - Modélisation aléatoire, statistiques et processus
# Projet : modélisation de la Seine et de la Marne
# Utils : fonctions de chargement et de prétraitement des données
# =======================================================================================


def load_data(filename: str):
    """
    Load the data from the given filename and return a pandas DataFrame with the following columns:
    - datetime: the date and time of the measurement, as a datetime object
    - wse: the water surface elevation in meters, as a float
    - wse_u: the uncertainty of the water surface elevation, as a float
    - elevation: the elevation of the water surface compared to the minimum altitude, as a float
    """
    import pandas as pd

    df = pd.read_csv(f"./data/{filename}", sep=";")
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")
    df["wse"] = df["wse"].astype(float)
    df["wse_u"] = df["wse_u"].astype(float)
    # Add a column for the elevation compared to the minimum altitude
    df["elevation"] = df["wse"] - df["wse"].min()
    return df