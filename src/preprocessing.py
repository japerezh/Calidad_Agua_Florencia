# ============================================================
# preprocessing.py
# Proyecto: Calidad del Agua — Caquetá
# Contiene: Bloques 1, 2 y 3
# Ejecutar: python src/preprocessing.py
# ============================================================

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ============================================================
# BLOQUE 1 — CARGA Y VALIDACIÓN DE INTEGRIDAD
# ============================================================

RUTA_DATOS = 'data/raw/registro_calidad_agua_PROCESADO.xlsx'

df = pd.read_excel(RUTA_DATOS, sheet_name='Datos_Procesados')
df['Fecha_Hora'] = pd.to_datetime(df['Fecha_Hora'])
df = df.sort_values('Fecha_Hora').reset_index(drop=True)

COLUMNAS_ESPERADAS = [
    'Fecha_Hora', 'pH', 'Oxigeno_Disuelto_mgL', 'Turbidez_NTU',
    'Temperatura_C', 'Indice_Calidad_Agua', 'Clasificacion', 'Temporada'
]

assert list(df.columns) == COLUMNAS_ESPERADAS, \
    f"Columnas inesperadas: {df.columns.tolist()}"
assert df.isnull().sum().sum() == 0, "Valores nulos detectados"
assert df.duplicated().sum() == 0, "Duplicados detectados"

diffs = df['Fecha_Hora'].diff().dropna()
gaps  = diffs[diffs != pd.Timedelta('15min')]
assert len(gaps) == 0, f"Gaps temporales en {len(gaps)} posiciones"

rangos = {
    'pH':                   (6.5,  8.5),
    'Oxigeno_Disuelto_mgL': (7.0, 14.0),
    'Turbidez_NTU':         (0.0, 50.0),
    'Temperatura_C':        (20.0, 30.0),
    'Indice_Calidad_Agua':  (0.0, 100.0),
}
for col, (lo, hi) in rangos.items():
    fuera = df[(df[col] < lo) | (df[col] > hi)]
    assert len(fuera) == 0, \
        f"{col}: {len(fuera)} valores fuera de rango [{lo}, {hi}]"

def cls_desde_ica(ica):
    if ica >= 90: return 'EXCELENTE'
    if ica >= 80: return 'BUENA'
    return 'ACEPTABLE'

discrepancias = (
    df['Indice_Calidad_Agua'].apply(cls_desde_ica) != df['Clasificacion']
).sum()
assert discrepancias == 0, \
    f"{discrepancias} inconsistencias ICA-Clasificacion"

print("=" * 55)
print("  BLOQUE 1 — VALIDACIÓN DE INTEGRIDAD")
print("=" * 55)
print(f"  Registros        : {len(df):,}")
print(f"  Período          : {df['Fecha_Hora'].min().date()} → "
      f"{df['Fecha_Hora'].max().date()}")
print(f"  Nulos            : 0")
print(f"  Duplicados       : 0")
print(f"  Gaps temporales  : 0")
print(f"  ICA ↔ Clase      : 100% consistente")
print()
print("  DISTRIBUCIÓN DE CLASES:")
for cls, n in df['Clasificacion'].value_counts().items():
    print(f"    {cls:<12}: {n:>5}  ({n/len(df)*100:.1f}%)")
print()
print("  ESTADÍSTICAS SENSORIALES:")
cols_s = ['pH','Oxigeno_Disuelto_mgL','Turbidez_NTU','Temperatura_C']
print(df[cols_s].describe().round(3).to_string())
print("\n  ✓ Bloque 1 completado\n")

# ============================================================
# BLOQUE 2 — FEATURE ENGINEERING TEMPORAL
# ============================================================

df['hora_sin']   = np.sin(2 * np.pi * df['Fecha_Hora'].dt.hour / 24)
df['hora_cos']   = np.cos(2 * np.pi * df['Fecha_Hora'].dt.hour / 24)
df['dia_semana'] = df['Fecha_Hora'].dt.dayofweek
df['dia_mes']    = df['Fecha_Hora'].dt.day

LAGS = [1, 2, 4, 8, 16]
for lag in LAGS:
    df[f'pH_lag{lag}']   = df['pH'].shift(lag)
    df[f'turb_lag{lag}'] = df['Turbidez_NTU'].shift(lag)
    df[f'OD_lag{lag}']   = df['Oxigeno_Disuelto_mgL'].shift(lag)
    df[f'temp_lag{lag}'] = df['Temperatura_C'].shift(lag)

for d in [1, 4, 8]:
    df[f'turb_delta{d}'] = df['Turbidez_NTU'].diff(d)
df['pH_delta1']   = df['pH'].diff(1)
df['OD_delta1']   = df['Oxigeno_Disuelto_mgL'].diff(1)
df['temp_delta1'] = df['Temperatura_C'].diff(1)

for win in [4, 8, 16]:
    df[f'turb_mean_{win}'] = (
        df['Turbidez_NTU'].rolling(win, min_periods=win).mean())
    df[f'turb_std_{win}']  = (
        df['Turbidez_NTU'].rolling(win, min_periods=win).std())
    df[f'turb_max_{win}']  = (
        df['Turbidez_NTU'].rolling(win, min_periods=win).max())
    df[f'OD_mean_{win}']   = (
        df['Oxigeno_Disuelto_mgL'].rolling(win, min_periods=win).mean())

df['turb_log'] = np.log1p(df['Turbidez_NTU'])
df['anomalia'] = (df['Turbidez_NTU'] > 5.0).astype(int)

df_fe = df.dropna().reset_index(drop=True)

EXCLUIR = {
    'Fecha_Hora', 'Indice_Calidad_Agua', 'Clasificacion',
    'Temporada', 'anomalia', 'hora', 'dia_semana', 'dia_mes'
}
FEATURES = [c for c in df_fe.columns if c not in EXCLUIR]

# FEATURES_ANOM: excluye turbidez y sus derivadas para evitar fuga de etiqueta.
# La etiqueta anomalia = (Turbidez_NTU > 5.0) hace que cualquier feature
# derivada de turbidez sea una fuga directa hacia el target.
TURB_COLS     = {c for c in df_fe.columns if 'turb' in c.lower()}
EXCLUIR_ANOM  = EXCLUIR | TURB_COLS
FEATURES_ANOM = [c for c in df_fe.columns if c not in EXCLUIR_ANOM]

# Verificar ausencia de fuga en FEATURES (clf/reg)
for col in ['Indice_Calidad_Agua', 'Clasificacion']:
    assert col not in FEATURES, \
        f"DATA LEAKAGE DETECTADO: '{col}' en FEATURES"

# Verificar ausencia de fuga en FEATURES_ANOM
for col in TURB_COLS:
    assert col not in FEATURES_ANOM, \
        f"DATA LEAKAGE ANOMALIAS: '{col}' en FEATURES_ANOM"
assert 'anomalia' not in FEATURES_ANOM, \
    "DATA LEAKAGE: 'anomalia' en FEATURES_ANOM"

print("=" * 55)
print("  BLOQUE 2 — FEATURE ENGINEERING")
print("=" * 55)
print(f"  Registros tras FE  : {len(df_fe):,}")
print(f"  Features clf/reg   : {len(FEATURES)}")
print(f"  Features anomalías : {len(FEATURES_ANOM)}")
print(f"  Cols turbidez excl : {len(TURB_COLS)}")
print(f"  Anomalías          : {df_fe['anomalia'].sum()} "
      f"({df_fe['anomalia'].mean()*100:.1f}%)")
print(f"\n  Features clf/reg ({len(FEATURES)}):")
for f in FEATURES:
    print(f"    · {f}")
print(f"\n  Features anomalías ({len(FEATURES_ANOM)}) — sin turbidez:")
for f in FEATURES_ANOM:
    print(f"    · {f}")
print("\n  ✓ Bloque 2 completado\n")

# ============================================================
# BLOQUE 3 — SEPARACIÓN DE DATASETS Y NORMALIZACIÓN
# ============================================================

le = LabelEncoder()
X  = df_fe[FEATURES].values
y  = le.fit_transform(df_fe['Clasificacion'].values)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

split_idx   = int(len(df_fe) * 0.80)
X_train_reg = X[:split_idx];  X_test_reg = X[split_idx:]
y_train_reg = df_fe['Indice_Calidad_Agua'].values[:split_idx]
y_test_reg  = df_fe['Indice_Calidad_Agua'].values[split_idx:]

y_anom       = df_fe['anomalia'].values
X_anom       = df_fe[FEATURES_ANOM].values
X_train_anom = X_anom[:split_idx];  X_test_anom = X_anom[split_idx:]
y_train_anom = y_anom[:split_idx]
y_test_anom  = y_anom[split_idx:]

scaler_clf    = StandardScaler()
X_train_clf_s = scaler_clf.fit_transform(X_train_clf)
X_test_clf_s  = scaler_clf.transform(X_test_clf)

scaler_ts     = StandardScaler()
X_train_reg_s = scaler_ts.fit_transform(X_train_reg)
X_test_reg_s  = scaler_ts.transform(X_test_reg)

scaler_anom    = StandardScaler()
X_train_anom_s = scaler_anom.fit_transform(X_train_anom)
X_test_anom_s  = scaler_anom.transform(X_test_anom)

# Guardado del dataset procesado y scalers
df_fe.to_csv('data/processed/dataset_features.csv', index=False)
joblib.dump(scaler_clf,  'data/models/scaler_clf.pkl')
joblib.dump(scaler_ts,   'data/models/scaler_ts.pkl')
joblib.dump(scaler_anom, 'data/models/scaler_anom.pkl')
joblib.dump(le,          'data/models/label_encoder.pkl')

print("=" * 55)
print("  BLOQUE 3 — SEPARACIÓN Y NORMALIZACIÓN")
print("=" * 55)
print(f"  Clases: {dict(zip(le.classes_, le.transform(le.classes_)))}")
print()
print("  CONFIG A — CLASIFICACIÓN (split estratificado)")
print(f"    Train: {len(X_train_clf):>5}  |  Test: {len(X_test_clf):>4}")
for cls, enc in zip(le.classes_, le.transform(le.classes_)):
    print(f"    {cls:<12}: train={(y_train_clf==enc).sum()} "
          f" test={(y_test_clf==enc).sum()}")
print()
print("  CONFIG B — REGRESIÓN ICA (split temporal)")
print(f"    Train: {len(X_train_reg):>5}  |  Test: {len(X_test_reg):>4}")
print(f"    Test desde: {df_fe['Fecha_Hora'].iloc[split_idx].date()}")
print()
print("  CONFIG C — ANOMALÍAS (split temporal, sin turbidez)")
print(f"    Train: {len(X_train_anom):>5}  |  Test: {len(X_test_anom):>4}")
print(f"    Features: {len(FEATURES_ANOM)} (turbidez excluida)")
print(f"    Anomalías train: {y_train_anom.sum()} | "
      f"test: {y_test_anom.sum()}")
print()
print("  Archivos guardados:")
print("    data/processed/dataset_features.csv")
print("    data/models/scaler_clf.pkl")
print("    data/models/scaler_ts.pkl")
print("    data/models/scaler_anom.pkl")
print("    data/models/label_encoder.pkl")
print("\n  ✓ Bloque 3 completado")
print("\n  PREPROCESSING COMPLETO — listo para models.py")

# Exportar variables para uso en models.py si se importa
if __name__ == '__main__':
    print("\n  Ejecuta ahora: python src/models.py")