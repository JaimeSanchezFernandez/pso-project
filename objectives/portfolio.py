# objectives/portfolio.py
"""
Optimización de cartera de inversión mediante PSO.

Caso de uso real: construir una cartera de 8 activos (acciones tecnológicas
y defensivas) que maximice el ratio de Sharpe, es decir, el retorno ajustado
por riesgo.

    Sharpe = (μ_p - r_f) / σ_p

donde μ_p es el retorno anualizado de la cartera, r_f es la tasa libre de
riesgo y σ_p es la volatilidad anualizada.

El PSO minimiza -Sharpe (porque minimiza por convención).

Restricción de suma unitaria:
    Los pesos de la cartera deben sumar 1 (100 % invertido).
    Se resuelve normalizando los pesos en cada evaluación, que equivale a
    proyectar sobre el simplex estándar. Esto es habitual en la literatura
    de PSO con restricciones de igualdad.

Datos:
    Al no tener acceso a APIs externas en este entorno, se generan series
    de retornos diarios sintéticos pero estadísticamente realistas, basadas
    en los parámetros históricos aproximados de cada activo (2015-2024).
    La misma semilla (SEED_DATA = 0) garantiza reproducibilidad.
    En producción se sustituiría _generar_retornos() por una llamada a
    yfinance u otro proveedor.
"""

import numpy as np
import pandas as pd

from objectives.functions import ObjectiveFunction

# ---------------------------------------------------------------------------
# Parámetros históricos aproximados de los activos (retorno y vol anuales)
# ---------------------------------------------------------------------------
ASSETS: dict[str, dict[str, float]] = {
    "AAPL":  {"mu": 0.28, "sigma": 0.28},
    "MSFT":  {"mu": 0.30, "sigma": 0.25},
    "GOOGL": {"mu": 0.22, "sigma": 0.26},
    "AMZN":  {"mu": 0.25, "sigma": 0.30},
    "JPM":   {"mu": 0.15, "sigma": 0.22},
    "JNJ":   {"mu": 0.08, "sigma": 0.14},
    "XOM":   {"mu": 0.10, "sigma": 0.24},
    "BRK-B": {"mu": 0.13, "sigma": 0.17},
}

# Correlaciones históricas aproximadas entre activos
_CORR = np.array([
    [1.00, 0.75, 0.70, 0.65, 0.45, 0.25, 0.20, 0.40],
    [0.75, 1.00, 0.72, 0.68, 0.48, 0.28, 0.22, 0.42],
    [0.70, 0.72, 1.00, 0.65, 0.42, 0.24, 0.20, 0.38],
    [0.65, 0.68, 0.65, 1.00, 0.40, 0.22, 0.18, 0.36],
    [0.45, 0.48, 0.42, 0.40, 1.00, 0.35, 0.30, 0.55],
    [0.25, 0.28, 0.24, 0.22, 0.35, 1.00, 0.28, 0.42],
    [0.20, 0.22, 0.20, 0.18, 0.30, 0.28, 1.00, 0.35],
    [0.40, 0.42, 0.38, 0.36, 0.55, 0.42, 0.35, 1.00],
])

SEED_DATA  = 0       # semilla para los datos sintéticos (fija, no del PSO)
N_DAYS     = 1260    # ~5 años de sesiones bursátiles
RISK_FREE  = 0.04    # tasa libre de riesgo anual (aprox. T-bill 2024)
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Generación de datos sintéticos
# ---------------------------------------------------------------------------

def _generar_retornos(seed: int = SEED_DATA) -> pd.DataFrame:
    """
    Simula N_DAYS días de retornos logarítmicos diarios para todos los activos
    usando un movimiento browniano geométrico multivariado con la estructura
    de correlación definida en _CORR.

    En producción esta función se sustituiría por:
        import yfinance as yf
        prices = yf.download(tickers, start=..., end=...)["Close"]
        return np.log(prices).diff().dropna()
    """
    rng    = np.random.default_rng(seed)
    names  = list(ASSETS.keys())
    sigmas = np.array([ASSETS[k]["sigma"] for k in names])
    mus    = np.array([ASSETS[k]["mu"]    for k in names])
    dt     = 1.0 / TRADING_DAYS

    cov_daily = np.outer(sigmas, sigmas) * _CORR * dt
    L         = np.linalg.cholesky(cov_daily)

    drift   = (mus - 0.5 * sigmas ** 2) * dt
    prices  = np.ones((N_DAYS + 1, len(names)))
    for t in range(1, N_DAYS + 1):
        z          = rng.standard_normal(len(names))
        prices[t]  = prices[t - 1] * np.exp(drift + L @ z)

    log_returns = pd.DataFrame(
        np.diff(np.log(prices), axis=0),
        columns=names,
    )
    return log_returns


# ---------------------------------------------------------------------------
# Función objetivo
# ---------------------------------------------------------------------------

class PortfolioSharpe(ObjectiveFunction):
    """
    Minimiza el ratio de Sharpe negativo de una cartera.

    Cada dimensión representa el peso de un activo. Los pesos se normalizan
    internamente para que sumen 1, por lo que el espacio de búsqueda efectivo
    es el simplex estándar. Los límites [0, 1] impiden posiciones cortas.

    Atributos públicos útiles para el análisis posterior:
        asset_names  : lista de tickers
        mu_anual     : vector de retornos anuales históricos (estimados)
        cov_anual    : matriz de covarianza anual (estimada)
        risk_free    : tasa libre de riesgo usada
    """

    def __init__(
        self,
        risk_free: float = RISK_FREE,
        data_seed: int   = SEED_DATA,
    ) -> None:
        n = len(ASSETS)
        # Límites: pesos entre 0 y 1 (sin posiciones cortas)
        super().__init__(dim=n, bounds=[(0.0, 1.0)] * n)

        self.asset_names = list(ASSETS.keys())
        self.risk_free   = risk_free

        # Estimar momentos a partir de los retornos sintéticos
        retornos         = _generar_retornos(seed=data_seed)
        self.mu_anual    = retornos.mean().values  * TRADING_DAYS
        self.cov_anual   = retornos.cov().values   * TRADING_DAYS
        self._retornos   = retornos.values          # (N_DAYS, n_assets)

    # ------------------------------------------------------------------
    # Métodos de utilidad
    # ------------------------------------------------------------------

    def normalizar_pesos(self, x: np.ndarray) -> np.ndarray:
        """
        Proyecta x sobre el simplex estándar (suma = 1, valores >= 0).
        Si todos los valores son <= 0, devuelve pesos iguales.
        """
        x = np.clip(x, 0.0, None)
        total = x.sum()
        if total <= 1e-12:
            return np.ones(self.dim) / self.dim
        return x / total

    def metricas(self, x: np.ndarray) -> dict:
        """
        Devuelve un desglose completo de la cartera para los pesos x.
        Útil para el análisis final del caso de uso.
        """
        w        = self.normalizar_pesos(x)
        mu_p     = float(w @ self.mu_anual)
        sigma_p  = float(np.sqrt(w @ self.cov_anual @ w))
        sharpe   = (mu_p - self.risk_free) / sigma_p if sigma_p > 1e-12 else 0.0
        return {
            "pesos":         dict(zip(self.asset_names, w.round(4))),
            "retorno_anual": round(mu_p,    4),
            "volatilidad":   round(sigma_p, 4),
            "sharpe":        round(sharpe,  4),
        }

    # ------------------------------------------------------------------
    # Interfaz ObjectiveFunction
    # ------------------------------------------------------------------

    def __call__(self, x: np.ndarray) -> float:
        """
        Evalúa -Sharpe para los pesos x.
        El PSO minimiza, así que minimizar -Sharpe = maximizar Sharpe.
        """
        w       = self.normalizar_pesos(x)
        mu_p    = float(w @ self.mu_anual)
        sigma_p = float(np.sqrt(w @ self.cov_anual @ w))
        if sigma_p < 1e-12:
            return 0.0
        sharpe  = (mu_p - self.risk_free) / sigma_p
        return -sharpe   # negativo porque PSO minimiza

    def __repr__(self) -> str:
        return f"PortfolioSharpe(n_assets={self.dim}, rf={self.risk_free})"