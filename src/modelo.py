"""
modelo.py
---------
Funciones básicas del modelo de regresión logística:
    p_theta(x) = sigma(theta0 + theta1 * x)

y del riesgo empírico logístico (log-loss / entropía cruzada):
    R_S(theta) = -(1/n) * sum_i [ y_i log p_theta(x_i) + (1-y_i) log(1 - p_theta(x_i)) ]

Todas las funciones están escritas de forma numéricamente estable, evitando
calcular exp(t) para valores grandes de t (lo que produciría overflow) y
evitando log(0).
"""

import numpy as np


def sigmoid(t):
    """
    Función sigmoide sigma(t) = 1 / (1 + exp(-t)), evaluada de forma estable.

    Para t >= 0 se usa 1 / (1 + exp(-t))  (exp(-t) no explota).
    Para t <  0 se usa exp(t) / (1 + exp(t)) (exp(t) no explota).
    """
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)

    pos = t >= 0
    neg = ~pos

    out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
    exp_t_neg = np.exp(t[neg])
    out[neg] = exp_t_neg / (1.0 + exp_t_neg)

    return out


def p_theta(x, theta):
    """
    Probabilidad p_theta(x) = sigma(theta0 + theta1 * x).

    Parámetros
    ----------
    x : array_like
    theta : array_like de longitud 2, (theta0, theta1)
    """
    x = np.asarray(x, dtype=float)
    theta0, theta1 = theta
    return sigmoid(theta0 + theta1 * x)


def log_sigmoid(t):
    """
    log(sigma(t)) calculado de forma estable como -log(1 + exp(-t)),
    usando np.logaddexp para evitar overflow/underflow.
    """
    t = np.asarray(t, dtype=float)
    return -np.logaddexp(0.0, -t)


def riesgo_logistico(theta, x, y):
    """
    Riesgo empírico logístico (log-loss promedio):

        R_S(theta) = -(1/n) sum_i [ y_i log p_i + (1 - y_i) log(1 - p_i) ]

    donde p_i = p_theta(x_i).

    Se calcula usando log_sigmoid tanto para log(p_i) como para
    log(1 - p_i) = log(sigma(-z_i)), lo que evita log(0) incluso cuando
    z_i = theta0 + theta1 * x_i toma valores grandes en magnitud.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    theta0, theta1 = theta
    z = theta0 + theta1 * x

    log_p = log_sigmoid(z)        # log p_theta(x)
    log_1mp = log_sigmoid(-z)     # log (1 - p_theta(x))

    return -np.mean(y * log_p + (1.0 - y) * log_1mp)


def gradiente_riesgo(theta, x, y):
    """
    Gradiente analítico de R_S respecto a theta = (theta0, theta1):

        dR/dtheta0 = (1/n) sum_i (p_i - y_i)
        dR/dtheta1 = (1/n) sum_i (p_i - y_i) * x_i

    Se pasa como argumento `jac` a scipy.optimize.minimize para acelerar
    y estabilizar la optimización.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = p_theta(x, theta)
    g0 = np.mean(p - y)
    g1 = np.mean((p - y) * x)
    return np.array([g0, g1])


def log_verosimilitud(theta, x, y):
    """
    Log-verosimilitud de una muestra Bernoulli con probabilidad p_theta(x_i):

        log L(theta) = sum_i [ y_i log p_i + (1 - y_i) log(1 - p_i) ]

    Nótese que log L(theta) = -n * R_S(theta).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    theta0, theta1 = theta
    z = theta0 + theta1 * x
    log_p = log_sigmoid(z)
    log_1mp = log_sigmoid(-z)
    return np.sum(y * log_p + (1.0 - y) * log_1mp)


def verosimilitud(theta, x, y):
    """
    Verosimilitud L(theta) = exp(log_verosimilitud(theta, x, y)).

    ADVERTENCIA: para n moderado o grande este producto de probabilidades
    puede subdesbordarse (underflow) a 0.0 en aritmética de punto flotante,
    aunque la log-verosimilitud siga siendo un número finito perfectamente
    calculable. Ver Ejercicio 5.
    """
    return np.exp(log_verosimilitud(theta, x, y))
