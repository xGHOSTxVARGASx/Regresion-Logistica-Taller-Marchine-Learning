"""
optimizacion.py
---------------
Wrapper alrededor de scipy.optimize.minimize para ajustar el modelo
logístico minimizando el riesgo empírico R_S(theta) definido en modelo.py.
"""

import numpy as np
from scipy.optimize import minimize

from .modelo import riesgo_logistico, gradiente_riesgo


def ajustar_logistica(x, y, theta_init=(0.0, 0.0), maxiter=None, metodo="BFGS"):
    """
    Ajusta theta minimizando R_S(theta; x, y) con scipy.optimize.minimize.

    Parámetros
    ----------
    x, y : array_like
        Datos de entrenamiento.
    theta_init : array_like, longitud 2
        Punto inicial (theta0, theta1). Por defecto (0, 0), como pide el taller.
    maxiter : int or None
        Número máximo de iteraciones del optimizador. Si es None, se usa el
        valor por defecto de scipy.
    metodo : str
        Método de optimización de scipy.optimize.minimize.

    Retorna
    -------
    res : OptimizeResult
        Objeto retornado por scipy.optimize.minimize. res.x es theta_hat y
        res.fun es R_S(theta_hat).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    options = {}
    if maxiter is not None:
        options["maxiter"] = maxiter

    res = minimize(
        riesgo_logistico,
        x0=np.array(theta_init, dtype=float),
        args=(x, y),
        jac=gradiente_riesgo,
        method=metodo,
        options=options,
    )
    return res
