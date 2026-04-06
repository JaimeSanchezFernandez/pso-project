# io/persistence.py
import json
import csv
import platform
import datetime
from pathlib import Path


def _get_hardware_info() -> dict:
    """Recoge información básica del sistema para reproducibilidad."""
    return {
        "python_version": platform.python_version(),
        "system": platform.system(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }


def save_result(result: dict, output_dir: str = "results") -> Path:
    """
    Guarda el resultado de un experimento en disco.

    Formato elegido: JSON
    Justificación: los resultados tienen estructura anidada
    (fitness_history es una lista, gbest_pos es un array).
    JSON maneja esto de forma natural sin aplanar la estructura,
    es legible por humanos y universalmente soportado.
    CSV se usa además para las métricas finales (sin historial)
    porque es más cómodo para análisis en pandas/Excel.

    Parameters
    ----------
    result     : dict devuelto por run_experiment
    output_dir : carpeta donde guardar los resultados

    Returns
    -------
    Path del fichero JSON creado
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Nombre único por timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    fn_name = result.get("objective_fn", "unknown").replace("(", "_").replace(")", "")
    evaluator_name = result.get("evaluator", "unknown")
    filename = f"{fn_name}_{evaluator_name}_{timestamp}"

    # Enriquecer con metadata del sistema
    enriched = {
        "metadata": {
            "timestamp": timestamp,
            "hardware": _get_hardware_info(),
        },
        "result": result,
    }

    # Guardar JSON completo (con historial)
    json_path = output_path / f"{filename}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    # Guardar CSV de métricas finales (sin historial, más ligero)
    csv_path = output_path / "summary.csv"
    _append_to_csv(result, csv_path)

    return json_path


def _append_to_csv(result: dict, csv_path: Path) -> None:
    """Añade una fila de métricas finales al CSV resumen."""
    row = {
        "objective_fn":     result.get("objective_fn"),
        "evaluator":        result.get("evaluator"),
        "n_particles":      result.get("n_particles"),
        "w":                result.get("w"),
        "c1":               result.get("c1"),
        "c2":               result.get("c2"),
        "seed":             result.get("seed"),
        "gbest_fit":        result.get("gbest_fit"),
        "n_iterations":     result.get("n_iterations"),
        "elapsed_seconds":  result.get("elapsed_seconds"),
    }

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)