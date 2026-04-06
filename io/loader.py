# io/loader.py
import json
import csv
import pandas as pd
from pathlib import Path


def load_result(json_path: str) -> dict:
    """
    Carga un resultado completo desde un fichero JSON.

    Parameters
    ----------
    json_path : ruta al fichero JSON generado por save_result

    Returns
    -------
    dict con metadata y result completo (incluyendo fitness_history)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_results(results_dir: str = "results") -> list[dict]:
    """
    Carga todos los ficheros JSON de un directorio de resultados.

    Parameters
    ----------
    results_dir : carpeta donde están los ficheros JSON

    Returns
    -------
    Lista de dicts, uno por experimento
    """
    path = Path(results_dir)
    json_files = sorted(path.glob("*.json"))

    results = []
    for f in json_files:
        results.append(load_result(f))

    return results


def load_summary(results_dir: str = "results") -> pd.DataFrame:
    """
    Carga el CSV resumen como un DataFrame de pandas.

    Útil para comparar estrategias, hacer boxplots y tablas resumen.

    Parameters
    ----------
    results_dir : carpeta donde está el summary.csv

    Returns
    -------
    pd.DataFrame con una fila por experimento
    """
    csv_path = Path(results_dir) / "summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró {csv_path}. Ejecuta primero algún experimento.")
    return pd.read_csv(csv_path)