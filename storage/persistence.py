# storage/persistence.py
import json
import csv
import platform
import datetime
from pathlib import Path


def _info_sistema() -> dict:
    """Recoge información básica del sistema para reproducibilidad."""
    return {
        "version_python": platform.python_version(),
        "sistema":        platform.system(),
        "procesador":     platform.processor(),
        "maquina":        platform.machine(),
    }


def guardar_resultado(resultado: dict, carpeta_salida: str = "results") -> Path:
    """
    Guarda el resultado de un experimento en disco.

    Formato elegido: JSON para resultados completos (estructura anidada),
    CSV para métricas finales (compatible con pandas/Excel).

    Parameters
    ----------
    resultado      : dict devuelto por ejecutar_experimento
    carpeta_salida : carpeta donde guardar los resultados

    Returns
    -------
    Path del fichero JSON creado
    """
    ruta_salida = Path(carpeta_salida)
    ruta_salida.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    nombre_fn = resultado.get("funcion_objetivo", "unknown").replace("(", "_").replace(")", "")
    nombre_eval = resultado.get("evaluador", "unknown")
    nombre_fichero = f"{nombre_fn}_{nombre_eval}_{timestamp}"

    enriquecido = {
        "metadata": {
            "timestamp": timestamp,
            "sistema":   _info_sistema(),
        },
        "resultado": resultado,
    }

    ruta_json = ruta_salida / f"{nombre_fichero}.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(enriquecido, f, indent=2)

    ruta_csv = ruta_salida / "summary.csv"
    _añadir_a_csv(resultado, ruta_csv)

    return ruta_json


def _añadir_a_csv(resultado: dict, ruta_csv: Path) -> None:
    """Añade una fila de métricas finales al CSV resumen."""
    fila = {
        "funcion_objetivo":     resultado.get("funcion_objetivo"),
        "evaluador":            resultado.get("evaluador"),
        "num_particulas":       resultado.get("num_particulas"),
        "w":                    resultado.get("w"),
        "c1":                   resultado.get("c1"),
        "c2":                   resultado.get("c2"),
        "semilla":              resultado.get("semilla"),
        "fitness_global":       resultado.get("fitness_global"),
        "num_iteraciones":      resultado.get("num_iteraciones"),
        "tiempo_total":         resultado.get("tiempo_total"),
        "tiempo_evaluacion":    resultado.get("tiempo_evaluacion"),
        "tiempo_actualizacion": resultado.get("tiempo_actualizacion"),
        "overhead":             resultado.get("overhead"),
    }

    escribir_cabecera = not ruta_csv.exists()
    with open(ruta_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fila.keys())
        if escribir_cabecera:
            writer.writeheader()
        writer.writerow(fila)