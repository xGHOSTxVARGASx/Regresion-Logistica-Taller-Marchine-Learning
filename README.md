# Exploración computacional de la regresión logística

**Curso:** Introducción al Machine Learning — Universidad El Bosque (2026-2)
**Integrantes:** Daniel Vargas

## Objetivo

Estudiar computacionalmente propiedades del modelo de regresión logística
(superficie del riesgo, comportamiento en datos separables, recuperación de
parámetros, interpretación de los coeficientes y estabilidad numérica de la
verosimilitud) usando únicamente `numpy`, `scipy` y `matplotlib`.

## Contenido del repositorio

- **`src/`**: código reutilizable, importado por el cuaderno.
  - `modelo.py`: sigmoide estable, `p_theta(x)`, riesgo empírico logístico
    `riesgo_logistico`, su gradiente, y funciones de (log-)verosimilitud.
  - `optimizacion.py`: `ajustar_logistica`, envoltorio de
    `scipy.optimize.minimize` para estimar `theta`.
  - `simulacion.py`: `generar_datos`, generación de muestras sintéticas bajo
    el modelo logístico verdadero.
- **`notebooks/experimentos.ipynb`**: desarrollo completo de los 5 ejercicios
  del taller. Solo importa las funciones de `src/`, corre los experimentos,
  produce las tablas/verificaciones pedidas y genera las figuras.
- **`resultados/`**: figuras `.png` generadas por el cuaderno (superficie y
  curvas de nivel del riesgo, ajustes en datos separables/no separables,
  efecto de `theta0` y `theta1`).

## Instalación

Con Python 3.14 instalado:

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

Para reproducir todos los resultados desde un kernel reiniciado:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/experimentos.ipynb
```

o, de forma interactiva:

```bash
jupyter notebook notebooks/experimentos.ipynb
```

y ejecutar "Kernel → Restart & Run All". Las figuras se guardan
automáticamente en `resultados/`.
