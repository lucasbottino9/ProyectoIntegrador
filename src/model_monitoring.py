"""
===========================================================
Proyecto Integrador - MLOps
Henry

Archivo:
    model_monitoring.py

Descripción:
    Sistema de monitoreo del modelo.
    Detección de Data Drift mediante métricas estadísticas.

Incluye:
    - KS Test
    - PSI
    - Jensen-Shannon Divergence
    - Chi-Square Test
    - Visualización de drift
    - Alertas automáticas
Versión:
    1.1.0
===========================================================
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import (
    ks_2samp,
    chi2_contingency,
    entropy
)

# CONFIGURACIÓN
RANDOM_STATE = 42
DRIFT_THRESHOLD = {
    "bajo": 0.10,
    "medio": 0.25,
    "alto": 0.40
}
BASE_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_PATH / "Base_de_datos.csv"

# CARGA DE DATOS
def load_data(path=DATA_PATH):
    """
    Carga dataset para monitoreo.
    """
    df = pd.read_csv(path)
    print("="*60)
    print("DATOS CARGADOS PARA MONITOREO")
    print("="*60)
    print(df.shape)
    return df

# MUESTREO DE DATOS
def create_samples(
        df,
        sample_size=500,
        random_state=42
):
    """
    Genera muestras simulando
    población histórica y actual.
    En producción:
        histórico = datos de entrenamiento
        actual = datos nuevos recibidos
    """

    historical = df.sample(
        n=sample_size,
        random_state=random_stat
    )
    current = df.sample(
        n=sample_size,
        random_state=random_state+
    )
    return historical, current

# POPULATION STABILITY INDEX (PSI)
def calculate_psi(
        expected,
        actual,
        bins=10
):
    """
    Calcula Population Stability Index.
    Interpretación:
    <0.10  : sin drift
    0.10-0.25 : drift moderado
    >0.25 : drift significativo
    """
    breakpoints = np.linspace(
        0,
        100,
        bins + 1
    )
    expected_perc = np.histogram(
        expected,
        bins=np.percentile(
            expected,
            breakpoints
        )
    )[0] / len(expected)
    actual_perc = np.histogram(
        actual,
        bins=np.percentile(
            expected,
            breakpoints
        )
    )[0] / len(actual)
    expected_perc = np.where(
        expected_perc == 0,
        0.0001,
        expected_perc
    )
    actual_perc = np.where(
        actual_perc == 0,
        0.0001,
        actual_perc
    )
    psi = np.sum(
        (actual_perc - expected_perc)
        *
        np.log(
            actual_perc /
            expected_perc
        )
    )
    return psi
    
# JENSEN SHANNON DIVERGENCE
def calculate_js_divergence(
        expected,
        actual
):
    """
    Calcula divergencia Jensen-Shannon.
    """
    hist_range = (
        min(
            expected.min(),
            actual.min()
        ),
        max(
            expected.max(),
            actual.max()
        )
    )
    expected_hist, _ = np.histogram(
        expected,
        bins=20,
        range=hist_range,
        density=True
    )
    actual_hist, _ = np.histogram(
        actual,
        bins=20,
        range=hist_range,
        density=Tru
    )
    expected_hist += 1e-10
    actual_hist += 1e-10
    expected_hist /= expected_hist.sum()
    actual_hist /= actual_hist.sum()
    mean_distribution = (
        expected_hist +
        actual_hist
    ) / 2
    js = (
        entropy(
            expected_hist,
            mean_distribution
        )
        +
        entropy(
            actual_hist,
            mean_distribution
        )
    ) / 2
    return js

# KS TEST VARIABLES NUMÉRICAS
def calculate_ks_test(
        historical,
        current
):
    """
    Ejecuta Kolmogorov-Smirnov Test.
    """
    statistic, p_value = ks_2samp(
        historical,
        current
    )
    return statistic, p_value

# CHI CUADRADO VARIABLES CATEGÓRICAS
def calculate_chi_square(
        historical,
        current
):
    """
    Test Chi-cuadrado para variables categóricas.
    """
    dataframe = pd.DataFrame({
        "historical":
            historical.value_counts(),
        "current":
            current.value_counts()
    }).fillna(0)
    chi, p_value, _, _ = chi2_contingency(
        dataframe
    )
    return chi, p_value
    
# DETECCIÓN DE DRIFT
def detect_numeric_drift
        historical,
        current
):
    results = []
    numeric_columns = historical.select_dtypes(
        include=np.number
    ).columns
    for column in numeric_columns:
        ks, ks_p = calculate_ks_test(
            historical[column],
            current[column]
        )
        psi = calculate_psi(
            historical[column],
            current[column]
        )
        js = calculate_js_divergence(
            historical[column],
            current[column]
        )
        results.append({
            "variable": column,
            "KS": ks,
            "KS_p_value": ks_p,
            "PSI": psi,
            "JS": js
        })
    return pd.DataFrame(results)
    
# DRIFT CATEGÓRICO
def detect_categorical_drift(
        historical,
        current
):
    results = []
    categorical_columns = historical.select_dtypes(
        include="object"
    ).columns
    for column in categorical_columns:
        chi, p = calculate_chi_square(
            historical[column],
            current[column]
        )
        results.append({
            "variable": column,
            "CHI2": chi,
            "p_value": p
        })
    return pd.DataFrame(results)

# NIVEL DE ALERTA
def drift_alert(
        psi
):
    if psi < 0.10:
        return "BAJO"
    elif psi < 0.25
        return "MEDIO"
    else:
        return "ALTO"
# VISUALIZACIÓN
def plot_distribution(
        historical,
        current,
        variable
):
    plt.figure(figsize=(8,5))
    sns.histplot(
        historical[variable],
        kde=True,
        label="Histórico",
        stat="density"
    )
    sns.histplot(
        current[variable],
        kde=True,
        label="Actual",
        stat="density"
    )
    plt.title(
        f"Distribución Drift - {variable}"
    )
    plt.legend()
    plt.show()


# REPORTE COMPLETO
def generate_monitoring_report():

    """
    Ejecuta monitoreo completo.
    """
    df = load_data()
    historical, current = create_samples(df)
    numeric_report = detect_numeric_drift(
        historical,
        current
    )
    categorical_report = detect_categorical_drift(
        historical,
        current
    )
    numeric_report["alerta"] = numeric_report["PSI"].apply(
        drift_alert
    )
    print("="*60)
    print("REPORTE DATA DRIFT")
    print("="*60)
    print("\nVariables numéricas")
    print(numeric_report)
    print("\nVariables categóricas")
    print(categorical_report)
    return (
        numeric_report,
        categorical_report
    )
# MAIN
if __name__ == "__main__":
    numeric_report, categorical_report = (
        generate_monitoring_report()
    )
    print(
        "\nMONITOREO FINALIZADO"
    )