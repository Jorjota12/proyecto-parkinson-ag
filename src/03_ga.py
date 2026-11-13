#Primer algoritmo genético para seleccionar características (se usará DEAP)
import random
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from deap import base, creator, tools
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import matthews_corrcoef
from utils import cargar_datos, reparto_entrenamiento_prueba, metricas, SEED

# Parámetros 
TAM_POBLACION = 40
GENERACIONES = 30
P_CRUCE = 0.7            # probabilidad de cruce
P_MUT_BIT = 0.003        # probabilidad de voltear cada bit (mas o menos 2 bits por individuo)
P_INICIAL_UNOS = 0.05    # al empezar, cada característica se elige con un 5 %
LAMBDA = 0.1             # penalización por usar muchas características
TAM_TORNEO = 2
N_ELITE = 1

random.seed(SEED); np.random.seed(SEED)
df, X, y, grupos, sexo = cargar_datos()
nombres = np.array(X.columns)
X = X.to_numpy()
P = X.shape[1]
i_ent, i_test = reparto_entrenamiento_prueba(X, y, grupos)
Xe, ye, ge = X[i_ent], y[i_ent], grupos[i_ent]

def modelo():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))

# Particiones de validación dentro del entrenamiento (fijas, por paciente)
particiones = list(StratifiedGroupKFold(n_splits=3, shuffle=True,
                                        random_state=SEED).split(Xe, ye, ge))
cache = {}

def aptitud(ind):
    #Aptitud = MCC medio en validación cruzada por paciente - penalización.
    clave = tuple(ind)
    if clave in cache:
        return cache[clave]
    cols = np.flatnonzero(ind)
    if len(cols) == 0:
        res = (-1.0,)
    else:
        notas = []
        for a, b in particiones:
            m = modelo().fit(Xe[a][:, cols], ye[a])
            notas.append(matthews_corrcoef(ye[b], m.predict(Xe[b][:, cols])))
        res = (float(np.mean(notas)) - LAMBDA * len(cols) / P,)
    cache[clave] = res
    return res

creator.create("Aptitud", base.Fitness, weights=(1.0,))
creator.create("Individuo", list, fitness=creator.Aptitud)
tb = base.Toolbox()
tb.register("bit", lambda: int(random.random() < P_INICIAL_UNOS))
tb.register("individuo", tools.initRepeat, creator.Individuo, tb.bit, n=P)
tb.register("poblacion", tools.initRepeat, list, tb.individuo)
tb.register("evaluate", aptitud)
tb.register("select", tools.selTournament, tournsize=TAM_TORNEO)
tb.register("mate", tools.cxOnePoint)
tb.register("mutate", tools.mutFlipBit, indpb=P_MUT_BIT)

pob = tb.poblacion(n=TAM_POBLACION)
for ind in pob:
    ind.fitness.values = tb.evaluate(ind)
historial = []
t0 = time.time()
for g in range(GENERACIONES):
    elite = [tb.clone(i) for i in tools.selBest(pob, N_ELITE)]
    hijos = [tb.clone(i) for i in tb.select(pob, len(pob) - N_ELITE)]
    for a, b in zip(hijos[::2], hijos[1::2]):
        if random.random() < P_CRUCE:
            tb.mate(a, b); del a.fitness.values, b.fitness.values
    for h in hijos:
        tb.mutate(h); del h.fitness.values
    for h in hijos:
        h.fitness.values = tb.evaluate(h)
    pob = elite + hijos
    mejor = tools.selBest(pob, 1)[0]
    historial.append((g + 1, mejor.fitness.values[0], sum(mejor),
                      np.mean([i.fitness.values[0] for i in pob])))
    print(f"Generación {g+1:2d}: mejor aptitud {mejor.fitness.values[0]:.3f} con {sum(mejor)} características")

mejor = tools.selBest(pob, 1)[0]
cols = np.flatnonzero(mejor)
print(f"\nTiempo: {time.time()-t0:.0f} s. Evaluaciones distintas: {len(cache)}")
print(f"Mejor solución: {len(cols)} de {P} características")

# Prueba final (una sola vez)
def evaluar_prueba(columnas):
    m = modelo().fit(Xe[:, columnas], ye)
    Xt = X[i_test][:, columnas]
    return metricas(y[i_test], m.predict(Xt), m.decision_function(Xt))

res_ga = evaluar_prueba(cols)
res_todas = evaluar_prueba(np.arange(P))
rng = np.random.default_rng(SEED)
res_azar = pd.DataFrame([evaluar_prueba(rng.choice(P, len(cols), replace=False))
                         for _ in range(30)]).mean().to_dict()
mi = mutual_info_classif(StandardScaler().fit_transform(Xe), ye, random_state=SEED)
res_mi = evaluar_prueba(np.argsort(mi)[::-1][:len(cols)])

tabla = pd.DataFrame({"Algoritmo genético": res_ga, "Todas (752)": res_todas,
                      "Información mutua": res_mi, "Selección al azar (media de 30)": res_azar}).T
tabla["Nº características"] = [len(cols), P, len(cols), len(cols)]
print("\nPrueba final con los pacientes apartados (mismo clasificador en todos):")
print(tabla.round(3).to_string())
tabla.round(3).to_csv("results/ga_prueba_comparacion.csv")
pd.Series(nombres[cols]).to_csv("results/ga_caracteristicas_elegidas.csv", index=False, header=["caracteristica"])
pd.DataFrame(historial, columns=["generacion", "mejor_aptitud", "n_caracteristicas", "aptitud_media"]).to_csv("results/ga_historial.csv", index=False)

h = pd.DataFrame(historial, columns=["g", "mejor", "n", "media"])
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].plot(h.g, h.mejor, color="#1f7a8c", label="mejor"); ax[0].plot(h.g, h.media, color="#c75d2c", label="media")
ax[0].set_xlabel("Generación"); ax[0].set_ylabel("Aptitud"); ax[0].legend()
ax[1].plot(h.g, h.n, color="#1f7a8c"); ax[1].set_xlabel("Generación"); ax[1].set_ylabel("Características del mejor")
plt.tight_layout(); plt.savefig("figures/fig4_evolucion_ga.png", dpi=200); plt.close()
