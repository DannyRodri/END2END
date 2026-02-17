from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def crear_preprocesador(num_features, cat_features):

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_features),
        ("cat", categorical_pipeline, cat_features)
    ])

    return preprocessor

def validar_y_limpieza(df: pd.DataFrame, config_columnas: dict) -> pd.DataFrame:
    
    """
    Valida columnas, tipos y aplica limpieza básica.
    
    Parameters
    ----------
    df : pd.DataFrame
    config_columnas : dict
        Diccionario {columna: tipo_esperado}
    
    Returns
    -------
    pd.DataFrame limpio y validado
    """
    df = df.copy()

    columnas_requeridas = set(config_columnas.keys())
    columnas_df = set(df.columns)

    faltantes = columnas_requeridas - columnas_df
    extras = columnas_df - columnas_requeridas

    if faltantes:
        # logger.error("Faltan columnas requeridas: %s", faltantes)
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    if extras:
        # logger.warning("Columnas extra detectadas, serán eliminadas: %s", extras)
        df = df[list(columnas_requeridas)]

    for col, tipo in config_columnas.items():
        try:
            df[col] = df[col].astype(tipo)
        except Exception as e:
            # logger.error("Error convirtiendo columna %s a %s", col, tipo)
            raise TypeError(f"Error convirtiendo columna '{col}' a {tipo}: {e}")

    df = df.drop_duplicates()
    df = df.dropna(how="all")

    columnas_object = df.select_dtypes(include="object").columns
    for col in columnas_object:
        df[col] = df[col].str.strip()
        df[col] = df[col].str.lower()

    # columnas_numericas = df.select_dtypes(include=["int64", "float64"]).columns
    # for col in columnas_numericas:
    #     if df[col].isna().any():
    #         df[col] = df[col].fillna(df[col].median())

    return df
