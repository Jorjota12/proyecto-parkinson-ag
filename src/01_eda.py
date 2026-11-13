#Análisis exploratorio de datos (EDA). Genera figuras y un resumen.
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import cargar_datos

df, X, y, grupos, sexo = cargar_datos()
salida = []
def p(texto):
    print(texto)
    salida.append(texto)

# Tamaño y estructura
p(f"Filas (grabaciones): {df.shape[0]}  Columnas: {df.shape[1]}")
p(f"Características de la voz: {X.shape[1]}")
p(f"Valores vacíos: {int(df.isna().sum().sum())}")
p(f"Pacientes: {df['id'].nunique()}  Grabaciones por paciente: "
  f"{df.groupby('id').size().unique().tolist()}")

# Reparto de clases (por paciente y por grabación)
por_paciente = df.groupby("id")["class"].first()
p(f"Pacientes con Parkinson: {(por_paciente==1).sum()}  sanos: {(por_paciente==0).sum()}")
p(f"Grabaciones con Parkinson: {(y==1).sum()}  sanas: {(y==0).sum()}")
p("Un modelo que dijera siempre 'Parkinson' acertaría "
  f"{(por_paciente==1).mean()*100:.1f} % de los pacientes")

# Sexo
tabla = pd.crosstab(df.groupby("id")["gender"].first(), por_paciente)
p("Pacientes por sexo (filas) y diagnóstico (columnas):\n" + tabla.to_string())

# Columnas constantes y duplicadas
constantes = X.columns[X.nunique() <= 1].tolist()
duplicadas = int(X.T.duplicated().sum())
p(f"Columnas constantes: {len(constantes)}  Columnas duplicadas: {duplicadas}")

# Escalas muy distintas
desv = X.std()
p(f"Desviación típica de las características: mínima {desv.min():.2e}, máxima {desv.max():.2e}")
p(f"Rango de valores: de {X.min().min():.2e} a {X.max().max():.2e}")

# Correlaciones entre características
corr = X.corr().abs().to_numpy().copy()
np.fill_diagonal(corr, 0)
max_corr = np.nanmax(corr, axis=0)
for umbral in (0.9, 0.95, 0.99):
    pares = int((np.triu(np.nan_to_num(corr)) > umbral).sum())
    p(f"Pares de características con |correlación| > {umbral}: {pares}")
p(f"Características con alguna otra casi idéntica (|r| > 0.95): {(max_corr>0.95).sum()} de {len(max_corr)}")

# Las 3 grabaciones de un paciente son muy parecidas
dentro = X.groupby(df["id"]).transform(lambda c: c - c.mean()).var()
total = X.var()
razon = (dentro / total).replace([np.inf, -np.inf], np.nan).dropna()
p(f"Variación dentro de un mismo paciente respecto a la total (mediana): {razon.median():.2f}")

# Familias de características (por el prefijo del nombre)
def familia(c):
    if c.startswith("tqwt"): return "TQWT"
    if c.startswith(("det_", "app_")): return "Wavelet"
    if c.startswith(("mean_", "std_")): return "MFCC"
    return "Otras (jitter, shimmer, HNR, formantes...)"
fam = X.columns.to_series().map(familia).value_counts()
p("Características por familia:\n" + fam.to_string())

# Figuras 
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].bar(["Sanos", "Parkinson"], [(por_paciente==0).sum(), (por_paciente==1).sum()], color=["#1f7a8c", "#c75d2c"])
ax[0].set_title("Pacientes por diagnóstico")
ax[1].bar(["Sanas", "Parkinson"], [(y==0).sum(), (y==1).sum()], color=["#1f7a8c", "#c75d2c"])
ax[1].set_title("Grabaciones por diagnóstico")
plt.tight_layout(); plt.savefig("figures/fig1_reparto_clases.png", dpi=200); plt.close()

nombres = ["PPE", "locPctJitter", "locShimmer", "meanNoiseToHarmHarmonicity"]
fig, ax = plt.subplots(1, 4, figsize=(11, 3.4))
for a, n in zip(ax, nombres):
    a.boxplot([X.loc[y==0, n], X.loc[y==1, n]], tick_labels=["Sano", "Parkinson"])
    a.set_title(n, fontsize=9)
plt.tight_layout(); plt.savefig("figures/fig2_rasgos_por_clase.png", dpi=200); plt.close()

plt.figure(figsize=(6, 3.6))
plt.hist(max_corr[~np.isnan(max_corr)], bins=30, color="#1f7a8c")
plt.axvline(0.95, color="#c75d2c", ls="--")
plt.xlabel("Mayor |correlación| con otra característica")
plt.ylabel("Nº de características")
plt.title("Redundancia entre características")
plt.tight_layout(); plt.savefig("figures/fig3_redundancia.png", dpi=200); plt.close()

open("results/eda_resumen.txt", "w", encoding="utf-8").write("\n".join(salida))
