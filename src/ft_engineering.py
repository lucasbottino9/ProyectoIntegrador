"""
===========================================================
Proyecto Integrador - MLOps
Henry

Archivo:
    ft_engineering.py

Descripción:
    Ingeniería de características y preprocesamiento del
    dataset para modelos de clasificación.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Ruta del dataset
BASE_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_PATH / "Base_de_datos.csv"
TARGET = "Pago_atiempo"

# CARGA DEL DATASE
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV.
    Parameters
    ----------
    path : Path
        Ruta del archivo CSV.
    Returns
    -------
    pd.DataFrame
        Dataset cargado.
    """
    df = pd.read_csv("C:/Users/Lucas/Proyectos/ProyectoIntegrador/data/raw/Base_de_datos.csv")
    print("=" * 60)
    print("DATASET CARGADO CORRECTAMENTE")
    print("=" * 60)
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")
    return df
    
# FUNCIONES AUXILIARES
def safe_division(numerator, denominator):
    """
    Evita divisiones por cero.
    Parameters
    ----------
    numerator
    denominator
    Returns
    -------
    Serie numérica
    """
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator
def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte las columnas de fecha al formato datetime.
    """
    if "fecha_prestamo" in df.columns:
        df["fecha_prestamo"] = pd.to_datetime(
            df["fecha_prestamo"],
            errors="coerce"
        )
        # Antigüedad del préstamo en días
        df["dias_desde_prestamo"] = (
            pd.Timestamp.today() - df["fecha_prestamo"]
        ).dt.days
    return df

# FEATURE ENGINEERING
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera nuevas variables para mejorar la capacidad
    predictiva de los modelos.
    """
    df = df.copy()

    # Conversión de fechas
    df = convert_dates(df)

    # Ratio cuota / salario
    df["ratio_cuota_salario"] = safe_division(
        df["cuota_pactada"],
        df["salario_cliente"]
    )
    
    # Ratio deuda / salario
    df["ratio_deuda_salario"] = safe_division(
        df["saldo_total"],
        df["salario_cliente"]
    )

    # Capital restante
    df["capital_restante"] = (
        df["capital_prestado"] -
        df["saldo_principal"]
    )
    
    # Mora total
    df["mora_total"] = (
        df["saldo_mora"] +
        df["saldo_mora_codeudor"]
    )

    # Total créditos
    df["total_creditos"] = (
        df["creditos_sectorFinanciero"] +
        df["creditos_sectorCooperativo"] +
        df["creditos_sectorReal"]
    )

    # Ratio salario vs Datacrédito
    df["ratio_salario_datacredito"] = safe_division(
        df["salario_cliente"],
        df["promedio_ingresos_datacredito"]
    )
    
    # Diferencia de puntajes
    df["diferencia_puntaje"] = (
        df["puntaje"] -
        df["puntaje_datacredito"]
    )

    # Promedio de mora
    df["mora_promedio"] = safe_division(
        df["mora_total"],
        df["total_creditos"] + 1
    )

    # Créditos vigentes sobre total créditos
    df["ratio_creditos_vigentes"] = safe_division(
        df["cant_creditosvigentes"],
        df["total_creditos"] + 1
    )

    # Porcentaje de capital adeudado
    df["porcentaje_capital_adeudado"] = safe_division(
        df["saldo_principal"],
        df["capital_prestado"]
    )

    # Relación saldo mora / saldo total
    df["ratio_mora_total"] = safe_division(
        df["saldo_mora"],
        df["saldo_total"]
    )

    # Reemplazar infinitos
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("=" * 60)
    print("FEATURE ENGINEERING COMPLETADO")
    print("=" * 60)
    print(f"Cantidad de variables: {df.shape[1]}")
    return df

# SEPARACIÓN FEATURES / TARGET
def split_features_target(
    df: pd.DataFrame
):
    """
    Separa las variables predictoras (X)
    y la variable objetivo (y).

    Returns
    -------
    X : pd.DataFrame
        Variables independientes.

    y : pd.Series
        Variable objetivo.
    """
    X = df.drop(
        TARGET,
        axis=1
    )
    y = df[TARGET]
    print("=" * 60)
    print("VARIABLES SEPARADAS")
    print("=" * 60)
    print(f"Variables predictoras: {X.shape[1]}")
    print(f"Variable objetivo: {TARGET}")
    return X, y
    
# DIVISIÓN TRAIN / TEST
def split_data(
    X,
    y
):
    """
    Divide los datos en entrenamiento y prueba.
    Mantiene la proporción de clases mediante stratify.
    """
    X_train, X_test, y_train, y_test = train_test_split
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print("=" * 60)
    print("TRAIN TEST SPLIT COMPLETADO")
    print("=" * 60)
    print(f"Train: {X_train.shape}")
    print(f"Test : {X_test.shape}"
    return (
        X_train,
        X_test,
        y_train,
        y_test
    )

# PIPELINE DE PREPROCESAMIENTO
def build_preprocessor(
    X: pd.DataFrame
):
    """
    Construye el pipeline de transformación.
    Numéricas:
        - Imputación por mediana
        - StandardScaler
    Categóricas:
        - Imputación por moda
        - OneHotEncoder
    """
    categorical_features = [
        "tipo_laboral",
        "tendencia_ingresos"
    ]
    numeric_features = [
        col for col in X.columns
        if col not in categorical_features
    ]

    # Pipeline variables numéricas
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Pipeline variables categóricas
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numericas",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categoricas",
                categorical_pipeline,
                categorical_features
            )
        ]
    )
    print("=" * 60)
    print("PREPROCESSOR CREADO")
    print("=" * 60)
    print(
        f"Variables numéricas: {len(numeric_features)}"
    )
    print(
        f"Variables categóricas: {len(categorical_features)}"
    )
    return preprocessor

# EJECUCIÓN DEL PIPELINE DE FEATURE ENGINEERING
def prepare_data():
    """
    Ejecuta todo el flujo de preparación:
    1. Carga dataset
    2. Feature engineering
    3. Separación X/y
    4. Train/Test Split
    5. Creación del preprocesador
    Returns
    -------
    Datos preparados para entrenamiento.
    """
    
    # Cargar datos
    df = load_data()

    # Crear variables nuevas
    df = feature_engineering(
        df
    )

    # Separar variables
    X, y = split_features_target(
        df
    )

    # División entrenamiento/prueba
    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )
    
    # Crear preprocessing
    preprocessor = build_preprocessor(
        X_train
    )
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )

# MAIN
if __name__ == "__main__":
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data()
    print("=" * 60)
    print("PIPELINE FT_ENGINEERING FINALIZADO")
    print("=" * 60)