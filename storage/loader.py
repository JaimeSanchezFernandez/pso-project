# storage/loader.py
import json
import pandas as pd
from pathlib import Path


def cargar_resultado(ruta_json: str) -> dict:
    """
    Carga un resultado completo desde un fichero JSON.

    Parameters
    ----------
    ruta_json : ruta al fichero JSON generado por guardar_resultado

    Returns
    -------
    dict con metadata y resultado completo
    """
    with open(ruta_json, "r", encoding="utf-8") as f:
        return json.load(f)


def cargar_todos_resultados(carpeta_resultados: str = "results") -> list[dict]:
    """
    Carga todos los ficheros JSON de un directorio de resultados.

    Parameters
    ----------
    carpeta_resultados : carpeta donde están los ficheros JSON

    Returns
    -------
    Lista de dicts, uno por experimento
    """
    ruta = Path(carpeta_resultados)
    ficheros_json = sorted(ruta.glob("*.json"))

    resultados = []
    for f in ficheros_json:
        resultados.append(cargar_resultado(f))

    return resultados


def cargar_resumen(carpeta_resultados: str = "results") -> pd.DataFrame:
    """
    Carga el CSV resumen como un DataFrame de pandas.

    Parameters
    ----------
    carpeta_resultados : carpeta donde está el summary.csv

    Returns
    -------
    pd.DataFrame con una fila por experimento
    """
    ruta_csv = Path(carpeta_resultados) / "summary.csv"
    if not ruta_csv.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta_csv}. Ejecuta primero algún experimento."
        )
    return pd.read_csv(ruta_csv)