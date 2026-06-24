# Pipeline IA para Monitoreo de Calidad del Agua

**Tesis de Maestría — Universidad Autónoma de Occidente**  
Integración de Ciencia Ciudadana e Inteligencia Artificial para el Monitoreo en Tiempo Real de la Calidad del Agua en Caquetá

---

## Autor

| Campo | Detalle |
|---|---|
| **Nombre** | Jairo Andrés Pérez Hurtatis |
| **Programa** | Maestría en Inteligencia Artificial y Ciencia de Datos |
| **Universidad** | Universidad Autónoma de Occidente — Santiago de Cali |
| **Director** | Juan Manuel Núñez Velasco |
| **Correo** | jairo_and.perez@uao.edu.co |
| **Año** | 2026 |

---

## Descripción

Sistema de monitoreo hídrico inteligente que integra mediciones multiparamétricas de campo con modelos de Inteligencia Artificial para:

- Clasificar automáticamente el estado del agua (EXCELENTE / BUENA / ACEPTABLE)
- Predecir el Índice de Calidad del Agua (ICA)
- Detectar anomalías y eventos de contaminación
- Generar alertas con 3 niveles de riesgo (VERDE / AMARILLO / ROJO)

El sistema fue desarrollado y validado sobre datos reales del **Río Hacha, Florencia, Caquetá** — recolectados con el equipo multiparamétrico portátil **AquaSense Pro**, diseñado y construido por el autor.

---

## Resultados Principales

| Tarea | Mejor Modelo | Métrica |
|---|---|---|
| Clasificación estado agua | Gradient Boosting | F1-macro = 0.7751 |
| Predicción ICA | Random Forest | R² = 0.7896 / RMSE = 1.5129 |
| Detección anomalías | RF + SMOTE | F1 = 0.4706 / AP = 0.4626 |
| Recall sistema alertas (Capa 1 operativa) | Motor bicapa | 100% ROJO / 100% AMARILLO+ROJO |
| Distribución alertas (Capa 1 operativa) | Motor bicapa | VERDE 80.2% / AMARILLO 16.6% / ROJO 3.2% |
| Recall sistema alertas (Capa 2 ML) | Isolation Forest | 10.1% ROJO / 100% AMARILLO+ROJO |
| Distribución alertas (Capa 2 ML) | Isolation Forest | VERDE 37.9% / AMARILLO 50.0% / ROJO 12.1% |

> **Nota metodológica:** La detección de anomalías se evalúa con 26 features
> independientes de turbidez para evitar fuga de etiqueta
> (`anomalia = Turbidez_NTU > 5.0 NTU`). El recall del 100% en AMARILLO+ROJO
> corresponde a ambas capas del motor de alertas; la Capa 1 operativa es el
> componente principal con tasa de activación selectiva del 19.8%.

---

## Estructura del Proyecto

```
Calidad_Agua/
│
├── data/
│   ├── raw/
│   │   └── registro_calidad_agua_PROCESADO.xlsx
│   ├── processed/
│   │   └── dataset_features.csv
│   ├── models/
│   │   ├── rf_clasificador.pkl
│   │   ├── xgb_regresor.pkl
│   │   ├── isolation_forest.pkl
│   │   ├── rf_anomalias.pkl
│   │   ├── scaler_clf.pkl
│   │   ├── scaler_ts.pkl
│   │   ├── scaler_anom.pkl
│   │   └── label_encoder.pkl
│   └── logs/
│
├── src/
│   ├── preprocessing.py   ← Bloque 1-3: carga, FE, splits
│   ├── models.py          ← Bloque 5-7: clasificación, regresión, anomalías
│   ├── alerts.py          ← Motor de alertas bicapa (Capa 1 operativa + Capa 2 ML)
│   └── validation.py      ← Bloque 10: validación técnica OE4
│
├── reports/
│   ├── figures/           ← 12 figuras PNG
│   └── metrics/
│       ├── resultados_modelos.xlsx
│       ├── reporte_alertas.xlsx
│       └── validacion_tecnica.xlsx
│
├── docs/
│   └── Anteproyecto.pdf
│
├── main.py                ← Orquestador del pipeline completo
├── requirements.txt
└── README.md
```

---

## Instalación y Uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/japerezh/Calidad_Agua.git
cd Calidad_Agua
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Colocar el dataset

Copia el archivo en:
```
data/raw/registro_calidad_agua_PROCESADO.xlsx
```

### 4. Ejecutar el pipeline completo

```bash
python main.py
```

> `main.py` ejecuta los cuatro scripts en el orden correcto:
> `preprocessing.py` → `models.py` → `alerts.py` → `validation.py`

### Orden de ejecución de scripts individuales (obligatorio)

```bash
# Paso 1 — Preprocesamiento y feature engineering
python src/preprocessing.py

# Paso 2 — Entrenamiento y evaluación de modelos
python src/models.py

# Paso 3 — Motor de alertas bicapa
python src/alerts.py

# Paso 4 — Validación técnica completa (OE4)
python src/validation.py
```

> ⚠️ Los scripts deben ejecutarse en este orden. Cada uno depende
> de los archivos generados por el anterior. `preprocessing.py` genera
> los scalers y el dataset procesado; `models.py` genera los modelos .pkl;
> `alerts.py` y `validation.py` cargan esos modelos para evaluación.

---

## Dataset

| Campo | Valor |
|---|---|
| Registros totales | 2.500 |
| Registros efectivos (tras FE) | 2.484 |
| Período | 1 al 27 de abril de 2026 |
| Intervalo | 15 minutos |
| Ubicación | Río Hacha, Florencia, Caquetá, Colombia |
| Equipo | AquaSense Pro (diseño propio — ESP32 + MicroPython) |
| Anomalías detectadas | 79 registros (3.2%) |

### Variables del dataset

| Variable | Rango observado | Rango válido | Unidad |
|---|---|---|---|
| pH | 6.93 — 7.38 | 6.5 — 8.5 | Unidades pH |
| Oxígeno Disuelto | 9.07 — 12.09 | 7.0 — 14.0 | mg/L |
| Turbidez | 0.02 — 16.00 | 0.0 — 50.0 | NTU |
| Temperatura | 22.05 — 26.97 | 20.0 — 30.0 | °C |
| ICA | 73.6 — 99.3 | 0.0 — 100.0 | 0-100 |

### Feature Engineering

Se generaron **45 features predictivas** a partir de las 4 variables sensoriales:

- Lags temporales: 1, 2, 4, 8, 16 intervalos (15 min a 4 horas)
- Estadísticas de ventana deslizante: media, std, máximo (4, 8, 16 pasos)
- Gradientes instantáneos: delta entre registros consecutivos
- Codificación cíclica de la hora: `hora_sin`, `hora_cos`
- Logaritmo de turbidez: `turb_log`

**Para detección de anomalías se usan 26 features** (sin turbidez ni derivadas)
para evitar fuga de etiqueta.

---

## Modelos Implementados

### Clasificación del estado del agua (45 features)
- **Random Forest** (`n_estimators=200`, `class_weight='balanced'`, `min_samples_leaf=3`)
- **XGBoost** (`n_estimators=200`, `max_depth=5`, `learning_rate=0.05`)
- **Gradient Boosting** (`n_estimators=150`, `learning_rate=0.05`, `max_depth=4`)

### Regresión del ICA (45 features)
- **Random Forest Regressor** (`n_estimators=200`, `min_samples_leaf=3`)
- **XGBoost Regressor** (`n_estimators=200`, `learning_rate=0.05`, `max_depth=5`)
- **Gradient Boosting Regressor** (`n_estimators=150`, `learning_rate=0.05`, `max_depth=4`)

### Detección de anomalías (26 features — SIN turbidez)
- **Isolation Forest** (no supervisado, `contamination=0.032`)
- **Random Forest + SMOTE** (supervisado, balanceo de clases)
- **One-Class SVM** (semi-supervisado, `kernel='rbf'`, `nu=0.032`)

> **Nota metodológica crítica:** La etiqueta de anomalía se define como
> `Turbidez_NTU > 5.0 NTU`. Para evitar fuga de etiqueta, las 19 columnas
> derivadas de turbidez (`turb_log`, `turb_lag*`, `turb_delta*`,
> `turb_mean_*`, `turb_std_*`, `turb_max_*`) son excluidas del conjunto
> de features de detección de anomalías. El resultado F1 = 0.4706 es el
> valor metodológicamente válido y defendible.

---

## Sistema de Alertas — Arquitectura Bicapa

El motor de alertas implementa dos capas diferenciadas con propósitos distintos.

### Capa 1 — Regla operativa (turbidez medida directamente)

| Nivel | Condición | Registros | % | Acción |
|---|---|---|---|---|
| 🟢 VERDE | Turbidez ≤ 5.0 NTU y delta ≤ 1.5 NTU | 1.993 | 80.2% | Operación normal |
| 🟡 AMARILLO | Delta > 1.5 NTU o P(ACEPTABLE) > 0.20 | 412 | 16.6% | Monitoreo elevado |
| 🔴 ROJO | Turbidez > 5.0 NTU | 79 | 3.2% | Acción inmediata |

**Recall sobre 79 eventos reales:** 100% en ROJO / 100% en AMARILLO+ROJO combinados  
**Tasa de activación:** 19.8% (AMARILLO+ROJO) — componente operativo principal.

### Capa 2 — Experimento ML (Isolation Forest sin turbidez)

| Nivel | Condición | Registros | % | Acción |
|---|---|---|---|---|
| 🟢 VERDE | Score IA < 0.35 y P(ACEPTABLE) < 0.20 | 942 | 37.9% | Operación normal |
| 🟡 AMARILLO | Score 0.35–0.65 o delta > 1.5 NTU o P(ACEPTABLE) > 0.20 | 1.242 | 50.0% | Monitoreo elevado |
| 🔴 ROJO | Score ≥ 0.65 o (Score ≥ 0.45 y P(ACEPTABLE) > 0.30) | 300 | 12.1% | Acción inmediata |

**Recall sobre 79 eventos reales:** 10.1% en ROJO / 100% en AMARILLO+ROJO combinados  
**Tasa de activación:** 62.1% — resultado experimental con 26 features independientes de turbidez.

> El sistema detecta condiciones en el instante actual a intervalos de
> 15 minutos. La Capa 1 es el componente operativo principal; la Capa 2
> es un experimento ML complementario evaluado sin acceso a turbidez
> para garantizar validez metodológica. El recall del 100% en AMARILLO+ROJO
> de la Capa 2 se interpreta siempre junto a su tasa de activación del 62.1%.

---

## Validación Técnica (OE4)

| Esquema | Modelo | Resultado | Criterio | Estado |
|---|---|---|---|---|
| CV Estratificado 5-fold | XGBoost | F1 = 0.7019 ± 0.066 | F1 > 0.65 | ✅ |
| Consistencia temporal | Random Forest | F1 = 0.6772 ± 0.039 | Std < 0.10 | ✅ |
| Robustez σ=0.05 | Random Forest | F1 = 0.7497 (−0.5%) | Degr. < 15% | ✅ |
| Robustez σ=0.05 | Gradient Boosting | F1 = 0.7384 (−2.1%) | Degr. < 15% | ✅ |
| Regresión ICA | Random Forest | R² = 0.7896 | R² > 0.70 | ✅ |
| Regresión ruido σ=0.01 | Random Forest | R² = 0.790 | R² > 0.70 | ✅ |

---

## Outputs Generados

### Figuras (reports/figures/)

| Figura | Contenido |
|---|---|
| fig1_comparacion_clasificacion.png | Tabla comparativa clasificación |
| fig2_matrices_confusion.png | Matrices de confusión clasificación |
| fig3_importancia_features.png | Importancia de features (RF) |
| fig4_prediccion_vs_real.png | Predicción vs Real ICA |
| fig5_scatter_prediccion.png | Scatter predicción ICA |
| fig6_anomalias_tiempo.png | Anomalías en el tiempo |
| fig7_precision_recall.png | Curvas Precision-Recall |
| fig8_confusion_anomalias.png | Matrices confusión anomalías |
| fig9_analisis_deterioro.png | Análisis de deterioro |
| fig10_validacion_cruzada.png | Validación cruzada |
| fig11_robustez_ruido.png | Robustez frente a ruido |
| fig12_curvas_aprendizaje.png | Curvas de aprendizaje |

### Excel (reports/metrics/)

| Archivo | Hojas | Contenido |
|---|---|---|
| resultados_modelos.xlsx | 6 | Métricas por modelo y tarea |
| reporte_alertas.xlsx | 4 | Eventos y alertas detectadas (Capa 1 + Capa 2) |
| validacion_tecnica.xlsx | 6 | Validación OE4 completa |

---

## Dependencias

```bash
pip install -r requirements.txt
```

| Librería | Versión | Uso |
|---|---|---|
| pandas | 3.0.2 | Manipulación de datos |
| numpy | 2.4.4 | Operaciones numéricas |
| scikit-learn | 1.8.0 | Modelos ML y validación |
| xgboost | 3.2.0 | Modelos XGBoost |
| imbalanced-learn | 0.14.1 | SMOTE para desbalance |
| matplotlib | 3.10.9 | Generación de figuras |
| seaborn | 0.13.2 | Visualización estadística |
| openpyxl | 3.1.5 | Exportación Excel |
| joblib | 1.5.3 | Serialización de modelos |

---

## Semillas Aleatorias Fijadas

Todos los modelos usan `random_state=42` para garantizar reproducibilidad exacta:

```python
RandomForestClassifier(random_state=42)
XGBClassifier(random_state=42)
GradientBoostingClassifier(random_state=42)
IsolationForest(random_state=42)
SMOTE(random_state=42)
train_test_split(random_state=42)
StratifiedKFold(random_state=42)
```

---

## Objetivos Específicos Cubiertos

| OE | Descripción | Evidencia | Estado |
|---|---|---|---|
| OE1 | Diagnóstico brechas monitoreo hídrico | Encuesta 129 actores, análisis factorial | ✅ |
| OE2 | Modelos IA para análisis y detección anomalías | Pipeline 3 tareas, métricas validadas | ✅ |
| OE3 | Modelo conceptual híbrido IA–Ciencia Ciudadana | Diagrama flujo integrado | ✅ |
| OE4 | Validación técnica y aplicabilidad del prototipo | CV, TSS, robustez, curvas aprendizaje | ✅ |

---

## Referencias

- Brown, R.M. et al. (1972). A water quality index — do we dare? *Water Sewage Works*, 122(10), 339-343.
- IDEAM (2023). *Estudio Nacional del Agua 2022*. Instituto de Hidrología, Meteorología y Estudios Ambientales.
- CORPOAMAZONIA (2018). *Actualización del Plan de Ordenación y Manejo de la cuenca del río Hacha*.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
- Liu, F.T. et al. (2008). Isolation Forest. *ICDM 2008*.
- Chawla, N.V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357.
- Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation. *IJCAI*.
- Bergmeir, C. & Benítez, J.M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*.
- Ng, A. (2018). *Machine Learning Yearning*. deeplearning.ai.

---

## Licencia

Proyecto desarrollado como trabajo de grado para la Maestría en Inteligencia Artificial y Ciencia de Datos — Universidad Autónoma de Occidente. Uso académico.