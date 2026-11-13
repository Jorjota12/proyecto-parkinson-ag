#Carga de datos, reparto por paciente y métricas
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (matthews_corrcoef, balanced_accuracy_score,
                             f1_score, roc_auc_score)

SEED = 42
RUTA_DATOS = "data/pd_speech_features.csv"


def cargar_datos(ruta=RUTA_DATOS):
    """Lee el CSV (la fila 2 tiene los nombres de las columnas) y separa
    las características X, el diagnóstico y, el paciente y el sexo."""
    df = pd.read_csv(ruta, header=1)
    X = df.drop(columns=["id", "gender", "class"])
    y = df["class"].to_numpy()
    grupos = df["id"].to_numpy()
    sexo = df["gender"].to_numpy()
    return df, X, y, grupos, sexo


def reparto_entrenamiento_prueba(X, y, grupos, semilla=SEED):
    """Aparta aproximadamente un 20 % de los PACIENTES para la prueba final.
    Las 3 grabaciones de cada paciente van siempre juntas y se mantiene
    más o menos el mismo reparto de enfermos y sanos."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=semilla)
    idx_entreno, idx_prueba = next(cv.split(X, y, grupos))
    return idx_entreno, idx_prueba


def metricas(y_real, y_pred, y_puntuacion):
    #Devuelve las métricas de rendimiento. MCC es la nota entre -1 y 1.
    return {
        "MCC": matthews_corrcoef(y_real, y_pred),
        "Exactitud equilibrada": balanced_accuracy_score(y_real, y_pred),
        "F1": f1_score(y_real, y_pred),
        "AUC": roc_auc_score(y_real, y_puntuacion),
    }
