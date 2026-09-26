"""
simulacion.py
-------------
Generación de datos sintéticos bajo el modelo generador logístico:

    X ~ U[-2, 2]
    Y | X = x ~ Bernoulli( sigma(theta0* + theta1* * x) )

Usado en el Ejercicio 3 (recuperación de parámetros) y en el Ejercicio 5
(verosimilitud y estabilidad numérica).
"""

import numpy as np

from .modelo import sigmoid


def generar_datos(n, theta_star, rng):
    """
    Genera n observaciones (x_i, y_i) bajo el modelo logístico verdadero.

    Parámetros
    ----------
    n : int
        Tamaño de la muestra.
    theta_star : array_like, longitud 2
        Parámetros verdaderos (theta0*, theta1*) del modelo generador.
    rng : numpy.random.Generator
        Generador de números aleatorios (usar numpy.random.default_rng(2026)
        para reproducibilidad, como pide el enunciado).

    Retorna
    -------
    x : ndarray de forma (n,)
    y : ndarray de forma (n,), valores en {0, 1}
    """
    theta0_star, theta1_star = theta_star

    x = rng.uniform(-2.0, 2.0, size=n)
    p = sigmoid(theta0_star + theta1_star * x)
    y = rng.binomial(1, p).astype(float)

    return x, y
