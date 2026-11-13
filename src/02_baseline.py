"""Base: regresión logística con las 752 características.
Se evalúa con validación cruzada por paciente dentro del entrenamiento y,
una sola vez, con los pacientes apartados para la prueba"""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from utils import (cargar_datos, reparto_entrenamiento_prueba, metricas, SEED)

df, X, y, grupos, sexo = cargar_datos()
X = X.to_numpy()
i_ent, i_test = reparto_entrenamiento_prueba(X, y, grupos)
print(f"Entrenamiento: {len(i_ent)} grabaciones, {len(set(grupos[i_ent]))} pacientes")
print(f"Prueba: {len(i_test)} grabaciones, {len(set(grupos[i_test]))} pacientes")
print(f"Con Parkinson en prueba: {y[i_test].mean()*100:.1f} %  en entrenamiento: {y[i_ent].mean()*100:.1f} %")

def modelo(C):
    # El escalado va dentro del modelo para aprender solo del entrenamiento
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))

def validacion_cruzada(C):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    res = []
    for a, b in cv.split(X[i_ent], y[i_ent], grupos[i_ent]):
        m = modelo(C).fit(X[i_ent][a], y[i_ent][a])
        pred = m.predict(X[i_ent][b])
        punt = m.decision_function(X[i_ent][b])
        res.append(metricas(y[i_ent][b], pred, punt))
    return pd.DataFrame(res).mean()

filas = []
for C in (0.0001, 0.001, 0.01, 0.1, 1):
    r = validacion_cruzada(C); r["C"] = C; filas.append(r)
tabla = pd.DataFrame(filas).set_index("C")
print("\nValidación cruzada por paciente (dentro del entrenamiento):")
print(tabla.round(3).to_string())

mejorC = tabla["MCC"].idxmax()
m = modelo(mejorC).fit(X[i_ent], y[i_ent])
prueba = metricas(y[i_test], m.predict(X[i_test]), m.decision_function(X[i_test]))
print(f"\nMejor C por validación cruzada: {mejorC}")
print("Resultado en la prueba final (752 características):")
print({k: round(v, 3) for k, v in prueba.items()})

tabla.round(3).to_csv("results/baseline_validacion_cruzada.csv")
pd.Series(prueba).round(3).to_csv("results/baseline_prueba.csv", header=["valor"])
