# Selección de características acústicas con algoritmos genéticos para la detección de la enfermedad de Parkinson a partir de la voz

Proyecto Final de Aprendizaje Automático (Grado en Ingeniería Matemática, UFV). Autor: Jorge Simón Gutiérrez.

## Datos
Dataset *Parkinson's Disease Classification* (UCI Machine Learning Repository).
Sakar, C., Serbes, G., Gunduz, A., Nizam, H., & Sakar, B. (2018). https://doi.org/10.24432/C5MS4X

## Estructura
- `data/`: el CSV original
- `src/utils.py`: carga de datos, reparto por paciente y métricas
- `src/01_eda.py`: análisis exploratorio (figuras en `figures/`, resumen en `results/`)
- `src/02_baseline.py`: línea base con regresión logística y las 752 características
- `src/03_ga.py`: primer algoritmo genético (DEAP) y comparación con información mutua y selección al azar

## Cómo ejecutarlo
```
pip install -r requirements.txt
python src/01_eda.py
python src/02_baseline.py
python src/03_ga.py
```
Ejecutar siempre desde la carpeta raíz del proyecto. La semilla aleatoria está fijada (`SEED = 42`).

## Decisiones importantes
- Los datos se dividen siempre **por paciente** (cada paciente tiene 3 grabaciones).
- El escalado de las características se hace dentro de cada validación para no filtrar información.
- Métricas: MCC (nota entre -1 y 1), exactitud equilibrada, F1 y AUC, por el desbalance entre enfermos y sanos.
