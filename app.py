import time
import numpy as np
import pandas as pd
import streamlit as st

from supabase import create_client
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve,
    auc,
    roc_auc_score,
    average_precision_score,
)

import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# CONFIGURACION
# ============================================================

st.set_page_config(
    page_title="AI-FleetMonitor Pro",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ESTILO PROFESIONAL - SIN ICONOS
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background: #f4f6f9;
            color: #172033;
        }

        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.96);
            border-bottom: 1px solid #e2e8f0;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e2e8f0;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 1.7rem;
            padding-bottom: 3rem;
        }

        .app-header {
            background: linear-gradient(110deg, #0f172a 0%, #173a72 100%);
            border-radius: 16px;
            padding: 25px 30px;
            margin-bottom: 22px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.10);
        }

        .app-header-title {
            color: #ffffff;
            font-size: 30px;
            font-weight: 700;
            letter-spacing: -0.3px;
            margin: 0;
        }

        .app-header-subtitle {
            color: #cbd5e1;
            font-size: 14px;
            margin-top: 6px;
        }

        .section-title {
            color: #172033;
            font-size: 21px;
            font-weight: 700;
            margin-top: 10px;
            margin-bottom: 10px;
        }

        .section-note {
            color: #64748b;
            font-size: 13px;
            margin-bottom: 16px;
        }

        .panel {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.045);
        }

        .status-card {
            border-radius: 13px;
            padding: 17px 19px;
            border: 1px solid;
            margin-bottom: 15px;
        }

        .status-critical {
            background: #fff5f5;
            border-color: #fecaca;
            color: #991b1b;
        }

        .status-high {
            background: #fff8ed;
            border-color: #fed7aa;
            color: #9a3412;
        }

        .status-medium {
            background: #fffdf0;
            border-color: #fde68a;
            color: #854d0e;
        }

        .status-stable {
            background: #f3fbf6;
            border-color: #bbf7d0;
            color: #166534;
        }

        .status-main {
            font-size: 19px;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .status-sub {
            font-size: 13px;
            opacity: 0.9;
        }

        .mini-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 14px 15px;
            min-height: 92px;
        }

        .mini-label {
            color: #64748b;
            font-size: 12px;
            font-weight: 600;
        }

        .mini-value {
            color: #172033;
            font-size: 24px;
            font-weight: 750;
            margin-top: 5px;
        }

        .mini-help {
            color: #94a3b8;
            font-size: 11px;
            margin-top: 2px;
        }

        .evaluation-band {
            background: #ffffff;
            border: 1px solid #dbe4ef;
            border-left: 4px solid #2563eb;
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 18px;
        }

        .evaluation-band-title {
            font-weight: 700;
            color: #172033;
        }

        .evaluation-band-text {
            font-size: 13px;
            color: #64748b;
            margin-top: 4px;
        }

        .metric-box {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 13px;
            padding: 14px 16px;
            box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
        }

        .metric-label {
            color: #64748b;
            font-size: 12px;
            font-weight: 650;
        }

        .metric-value {
            color: #172033;
            font-size: 25px;
            font-weight: 750;
            margin-top: 4px;
        }

        .metric-desc {
            color: #94a3b8;
            font-size: 11px;
            margin-top: 3px;
        }

        .interpretation {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 14px;
            height: 100%;
        }

        .interpretation-title {
            color: #172033;
            font-weight: 700;
            font-size: 13px;
        }

        .interpretation-text {
            color: #64748b;
            font-size: 12px;
            line-height: 1.45;
            margin-top: 5px;
        }

        .stButton > button {
            border-radius: 9px;
            border: 1px solid #cbd5e1;
            font-weight: 600;
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 12px 14px;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            overflow: hidden;
        }

        .small-muted {
            color: #64748b;
            font-size: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CABECERA
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <div class="app-header-title">AI-FleetMonitor Pro</div>
        <div class="app-header-subtitle">
            Monitoreo de hardware, detección de anomalías, clasificación de riesgo
            y evaluación del modelo de inteligencia artificial.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SUPABASE
# ============================================================

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


@st.cache_data(ttl=60)
def cargar_datos(_cache_bust):
    resultado = supabase.table("equipos").select("*").execute()
    df = pd.DataFrame(resultado.data)

    if df.empty:
        return df

    df = df.rename(
        columns={
            "id_pc": "ID_PC",
            "fecha_hora": "Fecha_Hora",
            "uso_cpu_porcentaje": "Uso_CPU_Porcentaje",
            "uso_ram_porcentaje": "Uso_RAM_Porcentaje",
            "cpu_normalizado_porcentaje": "CPU_Normalizado_Porcentaje",
            "ticket_usuario": "Ticket_Usuario",
            "porcentaje_bateria": "Porcentaje_Bateria",
            "uso_disco_porcentaje": "Uso_Disco_Porcentaje",
            "usuario": "Usuario",
            "modelo": "Modelo",
            "serial": "Serial",
        }
    )

    columnas_numericas = [
        "Uso_CPU_Porcentaje",
        "Uso_RAM_Porcentaje",
        "CPU_Normalizado_Porcentaje",
        "Porcentaje_Bateria",
        "Uso_Disco_Porcentaje",
    ]

    for columna in columnas_numericas:
        if columna not in df.columns:
            df[columna] = np.nan
        df[columna] = pd.to_numeric(df[columna], errors="coerce")
        df[columna] = df[columna].clip(0, 100)

    if "Fecha_Hora" in df.columns:
        df["Fecha_Hora"] = pd.to_datetime(df["Fecha_Hora"], errors="coerce")

    if "Ticket_Usuario" not in df.columns:
        df["Ticket_Usuario"] = np.nan

    ticket_limpio = df["Ticket_Usuario"].fillna("").astype(str).str.strip()
    df["Tiene_Ticket"] = ticket_limpio.ne("")
    df["Clase_Real"] = np.where(df["Tiene_Ticket"], "CRÍTICO", "ESTABLE")

    return df


df = cargar_datos(int(time.time() // 60))

if st.button("Actualizar datos"):
    st.cache_data.clear()
    st.rerun()

if df.empty:
    st.warning(
        "Todavía no hay información disponible en Supabase. "
        "Ejecuta agente_monitor.py en al menos un equipo."
    )
    st.stop()


# ============================================================
# PREPROCESAMIENTO
# ============================================================

columnas_base = [
    "Uso_CPU_Porcentaje",
    "Uso_RAM_Porcentaje",
    "CPU_Normalizado_Porcentaje",
    "Porcentaje_Bateria",
    "Uso_Disco_Porcentaje",
]

for columna in columnas_base:
    mediana = df[columna].median()
    if pd.isna(mediana):
        mediana = 0
    df[columna] = df[columna].fillna(mediana).clip(0, 100)


# ============================================================
# VARIABLES DERIVADAS
# ============================================================

df["Presion_Recursos"] = df[
    ["Uso_CPU_Porcentaje", "Uso_RAM_Porcentaje", "Uso_Disco_Porcentaje"]
].max(axis=1)

df["Promedio_Recursos"] = df[
    ["Uso_CPU_Porcentaje", "Uso_RAM_Porcentaje", "Uso_Disco_Porcentaje"]
].mean(axis=1)

df["CPU_RAM_Conjunta"] = (
    df["Uso_CPU_Porcentaje"] * df["Uso_RAM_Porcentaje"] / 100
)

df["Diferencia_CPU"] = (
    df["CPU_Normalizado_Porcentaje"] - df["Uso_CPU_Porcentaje"]
).abs()

df["Componentes_Saturados"] = (
    (df["Uso_CPU_Porcentaje"] >= 85).astype(int)
    + (df["Uso_RAM_Porcentaje"] >= 85).astype(int)
    + (df["Uso_Disco_Porcentaje"] >= 90).astype(int)
)


# ============================================================
# GUIA TECNICA DE REFERENCIA
# ============================================================
# Esta capa NO reemplaza al modelo de ML. Sirve para que el sistema
# conozca explícitamente qué comportamiento es bueno, regular o malo.
#
# CPU / RAM:
#   0-59.99  = BUENO
#   60-84.99 = REGULAR
#   85-100   = MALO
#
# La clasificación final de IA sigue usando el modelo supervisado
# cuando hay suficientes tickets, pero estas variables se incorporan
# como evidencia técnica adicional.

def clasificar_recurso(valor):
    if valor < 60:
        return "BUENO"
    if valor < 85:
        return "REGULAR"
    return "MALO"


df["Estado_CPU_Tecnico"] = df["Uso_CPU_Porcentaje"].apply(clasificar_recurso)
df["Estado_RAM_Tecnico"] = df["Uso_RAM_Porcentaje"].apply(clasificar_recurso)

df["CPU_Mala"] = (df["Uso_CPU_Porcentaje"] >= 85).astype(int)
df["RAM_Mala"] = (df["Uso_RAM_Porcentaje"] >= 85).astype(int)
df["Disco_Malo"] = (df["Uso_Disco_Porcentaje"] >= 90).astype(int)

df["Recursos_Malos"] = (
    df["CPU_Mala"] +
    df["RAM_Mala"] +
    df["Disco_Malo"]
)

def clasificar_condicion_tecnica(row):
    malos = int(row["Recursos_Malos"])
    cpu = row["Uso_CPU_Porcentaje"]
    ram = row["Uso_RAM_Porcentaje"]

    if malos >= 2:
        return "MALO"
    if malos == 1:
        return "MALO"
    if cpu >= 60 or ram >= 60:
        return "REGULAR"
    return "BUENO"

df["Condicion_Tecnica"] = df.apply(
    clasificar_condicion_tecnica,
    axis=1
)


# ============================================================
# TENDENCIAS
# ============================================================

if "Fecha_Hora" in df.columns:
    df = df.sort_values(["ID_PC", "Fecha_Hora"]).copy()

    df["CPU_Anterior"] = df.groupby("ID_PC")["Uso_CPU_Porcentaje"].shift(1)
    df["RAM_Anterior"] = df.groupby("ID_PC")["Uso_RAM_Porcentaje"].shift(1)
    df["Disco_Anterior"] = df.groupby("ID_PC")["Uso_Disco_Porcentaje"].shift(1)

    df["Tendencia_CPU"] = (
        df["Uso_CPU_Porcentaje"] - df["CPU_Anterior"]
    ).fillna(0)
    df["Tendencia_RAM"] = (
        df["Uso_RAM_Porcentaje"] - df["RAM_Anterior"]
    ).fillna(0)
    df["Tendencia_Disco"] = (
        df["Uso_Disco_Porcentaje"] - df["Disco_Anterior"]
    ).fillna(0)
else:
    df["Tendencia_CPU"] = 0
    df["Tendencia_RAM"] = 0
    df["Tendencia_Disco"] = 0


# ============================================================
# VARIABLES DEL MODELO
# ============================================================

features = [
    "Uso_CPU_Porcentaje",
    "Uso_RAM_Porcentaje",
    "CPU_Normalizado_Porcentaje",
    "Porcentaje_Bateria",
    "Uso_Disco_Porcentaje",
    "Presion_Recursos",
    "Promedio_Recursos",
    "CPU_RAM_Conjunta",
    "Diferencia_CPU",
    "Componentes_Saturados",
    "Tendencia_CPU",
    "Tendencia_RAM",
    "Tendencia_Disco",
    "CPU_Mala",
    "RAM_Mala",
    "Disco_Malo",
    "Recursos_Malos",
]

X = df[features].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True)).fillna(0)

n_total = len(df)
cantidad_criticos = int((df["Clase_Real"] == "CRÍTICO").sum())
cantidad_estables = int((df["Clase_Real"] == "ESTABLE").sum())

# Con el volumen actual de la flota se permite entrenar desde 10 registros,
# siempre que existan al menos 2 ejemplos de cada clase.
# Esto permite evaluar con 14 equipos sin esperar a acumular 20.
puede_usar_supervisado = (
    n_total >= 10
    and cantidad_criticos >= 2
    and cantidad_estables >= 2
)


# ============================================================
# TRAIN / TEST
# ============================================================

if puede_usar_supervisado:
    # 30% de prueba. Con 14 registros suele producir 4-5 registros
    # de evaluación y mantiene representación de ambas clases cuando
    # cada clase tiene al menos 2 ejemplos.
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=0.30,
            random_state=42,
            stratify=df["Clase_Real"],
        )
    except ValueError:
        # Respaldo para conjuntos excepcionalmente pequeños.
        train_df, test_df = train_test_split(
            df,
            test_size=max(2 / max(n_total, 1), 0.25),
            random_state=42,
            stratify=df["Clase_Real"],
        )
else:
    train_df = df.copy()
    test_df = df.copy()

X_train = train_df[features].replace([np.inf, -np.inf], np.nan)
X_train = X_train.fillna(X.median()).fillna(0)

X_test = test_df[features].replace([np.inf, -np.inf], np.nan)
X_test = X_test.fillna(X.median()).fillna(0)

y_test_real = test_df["Clase_Real"].values


# ============================================================
# MODELO SUPERVISADO
# ============================================================

modelo_supervisado = None

if puede_usar_supervisado:
    modelo_supervisado = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    modelo_supervisado.fit(X_train, train_df["Clase_Real"])


# ============================================================
# ISOLATION FOREST
# ============================================================

proporcion_criticos = cantidad_criticos / n_total if n_total else 0.10
CONTAMINATION = float(np.clip(max(0.05, proporcion_criticos), 0.05, 0.25))

modelo_anomalia = IsolationForest(
    n_estimators=400,
    contamination=CONTAMINATION,
    max_samples="auto",
    random_state=42,
    n_jobs=-1,
)
modelo_anomalia.fit(X_train)

pred_anomalia = modelo_anomalia.predict(X)
raw_anomaly = -modelo_anomalia.decision_function(X)

p05 = np.percentile(raw_anomaly, 5)
p95 = np.percentile(raw_anomaly, 95)

if p95 > p05:
    anomaly_score = ((raw_anomaly - p05) / (p95 - p05)) * 100
else:
    anomaly_score = np.full(len(df), 50.0)

anomaly_score = np.clip(anomaly_score, 0, 100)
df["Score_Anomalia"] = anomaly_score


# ============================================================
# PROBABILIDAD SUPERVISADA
# ============================================================

if modelo_supervisado is not None:
    probabilidades = modelo_supervisado.predict_proba(X)
    clases_modelo = list(modelo_supervisado.classes_)

    if "CRÍTICO" in clases_modelo:
        indice_critico = clases_modelo.index("CRÍTICO")
        prob_critico = probabilidades[:, indice_critico]
    else:
        prob_critico = np.zeros(len(df))
else:
    prob_critico = np.zeros(len(df))


# ============================================================
# SCORE TECNICO
# ============================================================

def calcular_score_tecnico(row):
    cpu = row["Uso_CPU_Porcentaje"]
    ram = row["Uso_RAM_Porcentaje"]
    disco = row["Uso_Disco_Porcentaje"]
    cpu_norm = row["CPU_Normalizado_Porcentaje"]

    score = 0

    if cpu >= 95:
        score += 30
    elif cpu >= 85:
        score += 22
    elif cpu >= 75:
        score += 12
    elif cpu >= 65:
        score += 5

    if ram >= 95:
        score += 25
    elif ram >= 85:
        score += 18
    elif ram >= 75:
        score += 10
    elif ram >= 65:
        score += 5

    if disco >= 97:
        score += 25
    elif disco >= 90:
        score += 18
    elif disco >= 80:
        score += 10
    elif disco >= 70:
        score += 5

    if cpu_norm >= 95:
        score += 20
    elif cpu_norm >= 85:
        score += 12
    elif cpu_norm >= 75:
        score += 5

    return min(score, 100)


df["Score_Tecnico"] = df.apply(calcular_score_tecnico, axis=1)


# ============================================================
# RIESGO FINAL
# ============================================================

if modelo_supervisado is not None:
    df["Riesgo_IA"] = (
        prob_critico * 0.55
        + anomaly_score * 0.25
        + df["Score_Tecnico"] * 0.20
    )
else:
    df["Riesgo_IA"] = (
        anomaly_score * 0.55
        + df["Score_Tecnico"] * 0.45
    )

df["Riesgo_IA"] = np.clip(df["Riesgo_IA"], 0, 100)


def determinar_estado(riesgo):
    if riesgo >= 80:
        return "CRÍTICO"
    if riesgo >= 60:
        return "ALTO"
    if riesgo >= 35:
        return "MEDIO"
    return "ESTABLE"


def determinar_nivel(riesgo):
    if riesgo >= 80:
        return "Muy alto"
    if riesgo >= 60:
        return "Alto"
    if riesgo >= 35:
        return "Moderado"
    return "Bajo"


df["Estado"] = df["Riesgo_IA"].apply(determinar_estado)
df["Nivel_Riesgo"] = df["Riesgo_IA"].apply(determinar_nivel)


# ============================================================
# DIAGNOSTICO Y RECOMENDACIONES
# ============================================================

def generar_diagnostico(row):
    problemas = []

    cpu = row["Uso_CPU_Porcentaje"]
    ram = row["Uso_RAM_Porcentaje"]
    disco = row["Uso_Disco_Porcentaje"]
    cpu_norm = row["CPU_Normalizado_Porcentaje"]

    if cpu >= 95:
        problemas.append("La CPU presenta saturación crítica (>=95%).")
    elif cpu >= 85:
        problemas.append("La CPU presenta un nivel elevado (>=85%).")

    if ram >= 95:
        problemas.append(
            "La memoria RAM está en nivel crítico (>=95%), "
            "lo que indica alta presión de memoria."
        )
    elif ram >= 85:
        problemas.append("El consumo de RAM es elevado (>=85%).")

    if disco >= 97:
        problemas.append("La actividad del disco es extremadamente elevada (>=97%).")
    elif disco >= 90:
        problemas.append("La actividad del disco es elevada (>=90%).")

    if cpu_norm >= 90:
        problemas.append(
            "La CPU normalizada confirma un comportamiento de alta carga."
        )

    if row["Tendencia_CPU"] >= 15:
        problemas.append("La utilización de CPU está aumentando rápidamente.")

    if row["Tendencia_RAM"] >= 15:
        problemas.append("El consumo de RAM está aumentando respecto a la medición anterior.")

    if row["Tendencia_Disco"] >= 15:
        problemas.append("La actividad del almacenamiento presenta un incremento considerable.")

    if not problemas:
        return (
            "No se identifican señales técnicas críticas en las métricas actuales. "
            "El comportamiento observado es consistente con una operación estable."
        )

    return " ".join(problemas)


def generar_recomendaciones(row):
    recomendaciones = []
    cpu = row["Uso_CPU_Porcentaje"]
    ram = row["Uso_RAM_Porcentaje"]
    disco = row["Uso_Disco_Porcentaje"]
    bateria = row["Porcentaje_Bateria"]

    if cpu >= 85:
        recomendaciones.append(
            "Revisar procesos con mayor consumo de CPU y determinar si existe una "
            "aplicación o servicio generando carga sostenida."
        )

    if ram >= 85:
        recomendaciones.append(
            "Revisar aplicaciones con alto consumo de memoria, cerrar procesos "
            "innecesarios y verificar presión de memoria."
        )

    if disco >= 90:
        recomendaciones.append(
            "Revisar procesos con alta actividad de almacenamiento, espacio disponible "
            "y comportamiento de lectura/escritura del disco."
        )

    if bateria <= 15:
        recomendaciones.append(
            "Conectar el equipo a corriente y comprobar el estado de la batería."
        )

    if cpu >= 85 and ram >= 85:
        recomendaciones.append(
            "Investigar aplicaciones que estén generando simultáneamente alta carga "
            "de CPU y memoria."
        )

    if cpu >= 85 and ram >= 85 and disco >= 90:
        recomendaciones.append(
            "Realizar diagnóstico integral porque existe saturación simultánea "
            "de CPU, memoria y almacenamiento."
        )

    if not recomendaciones:
        recomendaciones.append(
            "No se requiere una intervención inmediata; mantener el monitoreo preventivo."
        )

    return recomendaciones


df["Diagnostico_IA"] = df.apply(generar_diagnostico, axis=1)
df["Recomendaciones_IA"] = df.apply(generar_recomendaciones, axis=1)


# ============================================================
# EVALUACION: MODELO SUPERVISADO Y SISTEMA DESPLEGADO
# ============================================================

if modelo_supervisado is not None:
    # Predicción pura del Random Forest sobre datos NO vistos.
    pred_test_supervisado = modelo_supervisado.predict(X_test)
    evaluacion_nombre = "Random Forest supervisado"
else:
    pred_test_if = modelo_anomalia.predict(X_test)
    pred_test_supervisado = np.where(pred_test_if == -1, "CRÍTICO", "ESTABLE")
    evaluacion_nombre = "Isolation Forest — evaluación preliminar"

# Alias utilizado por las secciones de evaluación existentes.
pred_test = pred_test_supervisado


def metricas_clasificacion(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=["CRÍTICO", "ESTABLE"],
    )
    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr_value = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, pos_label="CRÍTICO", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, pos_label="CRÍTICO", zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, pos_label="CRÍTICO", zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "specificity": specificity,
        "fpr": fpr_value,
        "mcc": matthews_corrcoef(
            (np.asarray(y_true) == "CRÍTICO").astype(int),
            (np.asarray(y_pred) == "CRÍTICO").astype(int),
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


metricas = metricas_clasificacion(
    y_test_real,
    pred_test_supervisado,
)


# ============================================================
# EVALUACION DEL SISTEMA DESPLEGADO
# ============================================================
# El estado general de la flota se calcula sobre TODOS los registros.
# La matriz de confusión se calcula SOLO sobre el conjunto de prueba.
# Por eso los conteos no tienen por qué coincidir.
#
# Para evitar confusión en la interfaz, mostramos explícitamente
# cuántos registros se usaron en cada vista.

if modelo_supervisado is not None:
    prob_test_full = modelo_supervisado.predict_proba(X_test)
    clases_test_full = list(modelo_supervisado.classes_)
    if "CRÍTICO" in clases_test_full:
        idx_critico_test = clases_test_full.index("CRÍTICO")
        prob_critico_test = prob_test_full[:, idx_critico_test]
    else:
        prob_critico_test = np.zeros(len(test_df))
else:
    prob_critico_test = np.zeros(len(test_df))


# ============================================================
# SIDEBAR

# ============================================================

st.sidebar.markdown("## Diagnóstico individual")

pc_seleccionado = st.sidebar.selectbox(
    "Equipo",
    sorted(df["ID_PC"].astype(str).unique()),
)

pc_data = df[df["ID_PC"].astype(str) == str(pc_seleccionado)]
equipo = pc_data.iloc[0]

st.sidebar.divider()
st.sidebar.markdown("## Registro de ticket")

with st.sidebar.form("form_ticket"):
    pc_ticket = st.selectbox(
        "Equipo",
        sorted(df["ID_PC"].astype(str).unique()),
        key="pc_ticket",
    )

    descripcion = st.text_area(
        "Descripción del problema",
        placeholder="Ejemplo: equipo lento, congelamiento, pantalla azul...",
    )

    enviado = st.form_submit_button(
        "Registrar ticket",
        use_container_width=True,
    )

    if enviado:
        if descripcion.strip():
            try:
                supabase.rpc(
                    "reportar_ticket",
                    {
                        "p_id_pc": pc_ticket,
                        "p_ticket": descripcion.strip(),
                    },
                ).execute()
                st.sidebar.success("Ticket registrado correctamente.")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"No fue posible registrar el ticket: {e}")
        else:
            st.sidebar.warning("Escribe una descripción antes de registrar el ticket.")


# ============================================================
# INDICADORES PRINCIPALES
# ============================================================

st.markdown('<div class="section-title">Estado general de la flota</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.metric("Total equipos", f"{len(df)}")
k2.metric(
    "En riesgo",
    f"{int((df['Riesgo_IA'] >= 35).sum())}",
    help="Incluye estados MEDIO, ALTO y CRÍTICO según el puntaje de riesgo."
)
k3.metric(
    "Críticos IA",
    f"{int((df['Estado'] == 'CRÍTICO').sum())}",
    help="Equipos que la lógica final de IA clasifica con riesgo >= 80/100."
)
k4.metric(
    "Con ticket",
    f"{cantidad_criticos}",
    help="Registros utilizados como referencia de la clase CRÍTICO."
)
k5.metric("CPU promedio", f"{df['Uso_CPU_Porcentaje'].mean():.1f}%")
k6.metric("RAM promedio", f"{df['Uso_RAM_Porcentaje'].mean():.1f}%")

st.caption(
    "El contador de la flota se calcula con todos los registros disponibles. "
    "La matriz de confusión utiliza únicamente los registros reservados para prueba; "
    "por diseño, sus totales no tienen que coincidir."
)

fleet_counts = (
    df["Estado"]
    .value_counts()
    .reindex(["ESTABLE", "MEDIO", "ALTO", "CRÍTICO"], fill_value=0)
    .reset_index()
)
fleet_counts.columns = ["Estado", "Cantidad"]

st.dataframe(
    fleet_counts,
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# DIAGNOSTICO INDIVIDUAL
# ============================================================

st.markdown('<div class="section-title">Diagnóstico individual</div>', unsafe_allow_html=True)

estado_pc = equipo["Estado"]
risk_class = {
    "CRÍTICO": "status-critical",
    "ALTO": "status-high",
    "MEDIO": "status-medium",
    "ESTABLE": "status-stable",
}.get(estado_pc, "status-stable")

st.markdown(
    f"""
    <div class="status-card {risk_class}">
        <div class="status-main">{estado_pc} — Riesgo {equipo['Riesgo_IA']:.1f}/100</div>
        <div class="status-sub">Nivel de riesgo: {equipo['Nivel_Riesgo']} | Equipo: {pc_seleccionado}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("CPU", f"{equipo['Uso_CPU_Porcentaje']:.1f}%")
r2.metric("RAM", f"{equipo['Uso_RAM_Porcentaje']:.1f}%")
r3.metric("CPU normalizada", f"{equipo['CPU_Normalizado_Porcentaje']:.1f}%")
r4.metric("Disco", f"{equipo['Uso_Disco_Porcentaje']:.1f}%")
r5.metric("Batería", f"{equipo['Porcentaje_Bateria']:.1f}%")

col_diag, col_rec = st.columns([1.15, 1])

with col_diag:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("**Análisis generado por la IA**")
    st.write(equipo["Diagnostico_IA"])
    st.markdown("**Componentes del riesgo**")
    st.write(
        f"Modelo supervisado: {prob_critico[df.index.get_loc(equipo.name)] * 100:.1f}% | "
        f"Anomalía: {equipo['Score_Anomalia']:.1f}/100 | "
        f"Reglas técnicas: {equipo['Score_Tecnico']:.1f}/100"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_rec:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("**Recomendaciones técnicas**")
    for rec in equipo["Recomendaciones_IA"]:
        st.write(f"• {rec}")
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# INTERPRETACION TECNICA DE CPU Y RAM
# ============================================================

st.markdown('<div class="section-title">Referencia técnica utilizada por la IA</div>', unsafe_allow_html=True)
guide = pd.DataFrame(
    {
        "Rango": [
            "0% - 59.99%",
            "60% - 84.99%",
            "85% - 100%",
        ],
        "CPU": [
            "BUENO",
            "REGULAR",
            "MALO",
        ],
        "RAM": [
            "BUENO",
            "REGULAR",
            "MALO",
        ],
        "Interpretación": [
            "Reposo o tareas ligeras.",
            "Carga aceptable; se debe observar si permanece.",
            "Carga alta; requiere revisión si es sostenida.",
        ],
    }
)
st.dataframe(guide, use_container_width=True, hide_index=True)

st.markdown(
    f'<div class="section-note">En la flota actual: CPU promedio {df["Uso_CPU_Porcentaje"].mean():.1f}% y RAM promedio {df["Uso_RAM_Porcentaje"].mean():.1f}%. '
    f'Equipos con CPU en estado MALO: {(df["Estado_CPU_Tecnico"] == "MALO").sum()} | '
    f'Equipos con RAM en estado MALO: {(df["Estado_RAM_Tecnico"] == "MALO").sum()}.</div>',
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

tab_dashboard, tab_evaluacion, tab_datos = st.tabs(
    ["Dashboard", "Evaluación y entrenamiento", "Inventario"]
)


# ============================================================
# DASHBOARD
# ============================================================

with tab_dashboard:
    st.markdown('<div class="section-title">Mapa de riesgo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">La posición representa la relación entre CPU y RAM. El tamaño representa el riesgo calculado.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 2])

    with left:
        detalle = pd.DataFrame(
            {
                "Indicador": [
                    "ID",
                    "Usuario",
                    "Modelo",
                    "Serial",
                    "Estado",
                    "Riesgo",
                    "Nivel",
                    "CPU",
                    "RAM",
                    "CPU normalizada",
                    "Disco",
                    "Batería",
                    "Ticket",
                ],
                "Valor": [
                    equipo["ID_PC"],
                    equipo.get("Usuario", "No disponible"),
                    equipo.get("Modelo", "No disponible"),
                    equipo.get("Serial", "No disponible"),
                    estado_pc,
                    f"{equipo['Riesgo_IA']:.1f}/100",
                    equipo["Nivel_Riesgo"],
                    f"{equipo['Uso_CPU_Porcentaje']:.1f}%",
                    f"{equipo['Uso_RAM_Porcentaje']:.1f}%",
                    f"{equipo['CPU_Normalizado_Porcentaje']:.1f}%",
                    f"{equipo['Uso_Disco_Porcentaje']:.1f}%",
                    f"{equipo['Porcentaje_Bateria']:.1f}%",
                    "Sí" if equipo["Tiene_Ticket"] else "No",
                ],
            }
        )
        st.dataframe(detalle, use_container_width=True, hide_index=True)

    with right:
        mapa = px.scatter(
            df,
            x="Uso_CPU_Porcentaje",
            y="Uso_RAM_Porcentaje",
            color="Estado",
            size="Riesgo_IA",
            hover_data=[
                "ID_PC",
                "Uso_Disco_Porcentaje",
                "Riesgo_IA",
                "Nivel_Riesgo",
                "Tiene_Ticket",
            ],
            color_discrete_map={
                "CRÍTICO": "#dc2626",
                "ALTO": "#ea580c",
                "MEDIO": "#ca8a04",
                "ESTABLE": "#2563eb",
            },
        )
        mapa.update_layout(
            height=500,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            xaxis_title="Uso CPU (%)",
            yaxis_title="Uso RAM (%)",
            legend_title="Estado",
            margin=dict(l=35, r=20, t=40, b=40),
        )
        st.plotly_chart(mapa, use_container_width=True)

    st.markdown('<div class="section-title">Priorización de atención</div>', unsafe_allow_html=True)

    priorizacion = df[
        [
            "ID_PC",
            "Estado",
            "Nivel_Riesgo",
            "Riesgo_IA",
            "Uso_CPU_Porcentaje",
            "Uso_RAM_Porcentaje",
            "Uso_Disco_Porcentaje",
            "Score_Tecnico",
            "Score_Anomalia",
            "Estado_CPU_Tecnico",
            "Estado_RAM_Tecnico",
            "Condicion_Tecnica",
            "Tiene_Ticket",
            "Diagnostico_IA",
        ]
    ].copy().sort_values("Riesgo_IA", ascending=False)

    st.dataframe(priorizacion, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Distribución del riesgo</div>', unsafe_allow_html=True)
    fig_riesgo = px.histogram(
        df,
        x="Riesgo_IA",
        color="Estado",
        nbins=20,
        color_discrete_map={
            "CRÍTICO": "#dc2626",
            "ALTO": "#ea580c",
            "MEDIO": "#ca8a04",
            "ESTABLE": "#2563eb",
        },
    )
    fig_riesgo.update_layout(
        height=390,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis_title="Riesgo IA (0 a 100)",
        yaxis_title="Cantidad de registros",
        margin=dict(l=35, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_riesgo, use_container_width=True)


# ============================================================
# EVALUACION Y ENTRENAMIENTO
# ============================================================

with tab_evaluacion:
    st.markdown('<div class="section-title">Evaluación y entrenamiento de la IA</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Esta sección concentra la evidencia de entrenamiento, validación, rendimiento y comportamiento del modelo. Con 14 equipos, la evaluación supervisada se ejecuta cuando existen al menos 2 ejemplos de cada clase.</div>',
        unsafe_allow_html=True,
    )

    estado_entrenamiento = "Modelo supervisado activo" if modelo_supervisado is not None else "Modo preliminar: detección de anomalías"
    texto_entrenamiento = (
        f"Registros totales: {n_total} | Críticos: {cantidad_criticos} | Estables: {cantidad_estables} | "
        f"Entrenamiento: {len(train_df)} | Prueba: {len(test_df)} | "
        f"Método: {evaluacion_nombre}"
    )

    st.markdown(
        f"""
        <div class="evaluation-band">
            <div class="evaluation-band-title">{estado_entrenamiento}</div>
            <div class="evaluation-band-text">{texto_entrenamiento}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if modelo_supervisado is None:
        faltan_registros = max(0, 10 - n_total)
        faltan_criticos = max(0, 2 - cantidad_criticos)
        faltan_estables = max(0, 2 - cantidad_estables)

        st.warning(
            f"El modelo supervisado todavía no puede entrenarse con el conjunto actual. "
            f"Requisito operativo: mínimo 10 registros, 2 críticos y 2 estables. "
            f"Actualmente hay {n_total} registros, {cantidad_criticos} críticos y "
            f"{cantidad_estables} estables. "
            f"Faltan aproximadamente {faltan_registros} registros, "
            f"{faltan_criticos} críticos y {faltan_estables} estables."
        )
    else:
        st.success(
            f"Modelo supervisado activo. La evaluación se realiza con "
            f"{len(train_df)} registros de entrenamiento y {len(test_df)} registros "
            f"reservados para prueba. La matriz de confusión y las curvas representan "
            f"exclusivamente el conjunto de prueba; el estado general de la flota usa "
            f"los {len(df)} registros disponibles."
        )

    # -----------------------
    # RESUMEN DEL EXPERIMENTO
    # -----------------------
    if modelo_supervisado is not None:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Registros para entrenar", len(train_df))
        s2.metric("Registros para probar", len(test_df))
        s3.metric("Críticos en prueba", int((y_test_real == "CRÍTICO").sum()))
        s4.metric("Estables en prueba", int((y_test_real == "ESTABLE").sum()))

        st.caption(
            "La matriz de confusión suma únicamente los registros de prueba. "
            "Por eso su total no tiene por qué ser igual al total de equipos mostrado en el dashboard."
        )

    # -----------------------
    # METRICAS
    # -----------------------
    st.markdown('<div class="section-title">Métricas principales de clasificación</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    metric_cards = [
        ("Accuracy", metricas["accuracy"], "Predicciones correctas sobre el total."),
        ("Precision", metricas["precision"], "De lo marcado como crítico, cuánto era realmente crítico."),
        ("Recall", metricas["recall"], "De los críticos reales, cuánto logró detectar la IA."),
        ("F1-Score", metricas["f1"], "Equilibrio entre Precision y Recall."),
    ]
    for col, (label, value, desc) in zip([m1, m2, m3, m4], metric_cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value:.2%}</div>
                    <div class="metric-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Métricas adicionales de evaluación</div>', unsafe_allow_html=True)

    a1, a2, a3, a4 = st.columns(4)
    extra_cards = [
        ("Balanced Accuracy", metricas["balanced_accuracy"], "Promedio equilibrado entre sensibilidad de ambas clases."),
        ("Specificity", metricas["specificity"], "Capacidad de reconocer correctamente los equipos estables."),
        ("MCC", metricas["mcc"], "Medida robusta de correlación entre predicción y realidad."),
        ("False Positive Rate", metricas["fpr"], "Proporción de equipos estables marcados incorrectamente como críticos."),
    ]
    for col, (label, value, desc) in zip([a1, a2, a3, a4], extra_cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value:.2%}</div>
                    <div class="metric-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Matriz de confusión</div>', unsafe_allow_html=True)

    cm = confusion_matrix(
        y_test_real,
        pred_test,
        labels=["CRÍTICO", "ESTABLE"],
    )

    cm_fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicción CRÍTICO", "Predicción ESTABLE"],
            y=["Real CRÍTICO", "Real ESTABLE"],
            colorscale="Blues",
            showscale=True,
            text=cm,
            texttemplate="%{text}",
            textfont=dict(size=18, color="#0f172a"),
            hovertemplate="Real: %{y}<br>Predicción: %{x}<br>Registros: %{z}<extra></extra>",
        )
    )
    cm_fig.update_layout(
        height=370,
        margin=dict(l=90, r=35, t=35, b=65),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
    )

    cm_left, cm_right = st.columns([1, 1.1])
    with cm_left:
        st.plotly_chart(cm_fig, use_container_width=True)
    with cm_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Lectura de la matriz**")
        st.write(f"Registros evaluados: {len(y_test_real)}")
        st.write(f"Verdaderos positivos: {metricas['tp']}")
        st.write(f"Verdaderos negativos: {metricas['tn']}")
        st.write(f"Falsos positivos: {metricas['fp']}")
        st.write(f"Falsos negativos: {metricas['fn']}")
        st.markdown(
            "La prioridad del sistema es reducir los falsos negativos, "
            "porque representan equipos con problema que la IA no detectó."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------
    # ROC Y PR
    # -----------------------
    st.markdown('<div class="section-title">Curvas de evaluación</div>', unsafe_allow_html=True)

    roc_col, pr_col = st.columns(2)

    if modelo_supervisado is not None:
        prob_test = modelo_supervisado.predict_proba(X_test)
        clases_test = list(modelo_supervisado.classes_)
        if "CRÍTICO" in clases_test:
            idx = clases_test.index("CRÍTICO")
            score_test = prob_test[:, idx]
            y_test_bin = (y_test_real == "CRÍTICO").astype(int)

            if len(np.unique(y_test_bin)) == 2:
                fpr, tpr, _ = roc_curve(y_test_bin, score_test)
                roc_auc = auc(fpr, tpr)

                with roc_col:
                    roc_fig = go.Figure()
                    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC {roc_auc:.3f}", line=dict(color="#2563eb", width=3)))
                    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line=dict(color="#94a3b8", dash="dash")))
                    roc_fig.update_layout(
                        title="ROC",
                        height=360,
                        margin=dict(l=45, r=20, t=50, b=45),
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                    )
                    st.plotly_chart(roc_fig, use_container_width=True)
                    st.caption(f"ROC-AUC: {roc_auc:.3f}")

                precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test_bin, score_test)
                pr_auc = auc(recall_curve, precision_curve)

                with pr_col:
                    pr_fig = go.Figure()
                    pr_fig.add_trace(go.Scatter(x=recall_curve, y=precision_curve, mode="lines", name=f"AUC {pr_auc:.3f}", line=dict(color="#0f766e", width=3)))
                    pr_fig.update_layout(
                        title="Precision-Recall",
                        height=360,
                        margin=dict(l=45, r=20, t=50, b=45),
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                        xaxis_title="Recall",
                        yaxis_title="Precision",
                    )
                    st.plotly_chart(pr_fig, use_container_width=True)
                    st.caption(f"PR-AUC: {pr_auc:.3f}")
            else:
                roc_col.info("No hay dos clases en el conjunto de prueba para calcular ROC/PR.")
                pr_col.info("No hay dos clases en el conjunto de prueba para calcular ROC/PR.")
    else:
        roc_col.info("ROC-AUC y PR-AUC supervisados se calculan sobre el conjunto de prueba cuando ambas clases están representadas.")
        pr_col.info("ROC-AUC y PR-AUC supervisados se calculan sobre el conjunto de prueba cuando ambas clases están representadas.")

    # -----------------------
    # VALIDACION CRUZADA
    # -----------------------
    st.markdown('<div class="section-title">Validación cruzada del entrenamiento</div>', unsafe_allow_html=True)

    if modelo_supervisado is not None:
        min_class = min(cantidad_criticos, cantidad_estables)
        # Con 14 equipos y 2 ejemplos de la clase minoritaria, se utilizan
        # 2 folds. Si hay más ejemplos, se aprovechan hasta 5 folds.
        folds = min(5, min_class)

        if folds >= 2:
            cv_model = RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )

            scoring = {
                "accuracy": "accuracy",
                "precision": make_scorer(
                    precision_score,
                    pos_label="CRÍTICO",
                    zero_division=0,
                ),
                "recall": make_scorer(
                    recall_score,
                    pos_label="CRÍTICO",
                    zero_division=0,
                ),
                "f1": make_scorer(
                    f1_score,
                    pos_label="CRÍTICO",
                    zero_division=0,
                ),
            }

            cv = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=42,
            )

            cv_result = cross_validate(
                cv_model,
                X,
                df["Clase_Real"],
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                error_score="raise",
            )

            cv_table = pd.DataFrame(
                {
                    "Métrica": ["Accuracy", "Precision", "Recall", "F1-Score"],
                    "Promedio": [
                        cv_result["test_accuracy"].mean(),
                        cv_result["test_precision"].mean(),
                        cv_result["test_recall"].mean(),
                        cv_result["test_f1"].mean(),
                    ],
                    "Desviación estándar": [
                        cv_result["test_accuracy"].std(),
                        cv_result["test_precision"].std(),
                        cv_result["test_recall"].std(),
                        cv_result["test_f1"].std(),
                    ],
                }
            )

            cv_display = cv_table.copy()
            cv_display["Promedio"] = cv_display["Promedio"].map(lambda x: f"{x:.2%}")
            cv_display["Desviación estándar"] = cv_display["Desviación estándar"].map(lambda x: f"{x:.2%}")
            cv_display["N folds"] = folds
            st.dataframe(cv_display, use_container_width=True, hide_index=True)
        else:
            st.info("La validación cruzada no puede ejecutarse todavía porque la clase minoritaria tiene menos de 2 registros.")
    else:
        st.info("La validación cruzada del modelo supervisado se activará cuando ambas clases estén presentes en la prueba y exista el modelo supervisado.")

    # -----------------------
    # IMPORTANCIA DE VARIABLES
    # -----------------------
    if modelo_supervisado is not None:
        st.markdown('<div class="section-title">Importancia de las variables</div>', unsafe_allow_html=True)

        importancia = pd.DataFrame(
            {
                "Variable": features,
                "Importancia": modelo_supervisado.feature_importances_,
            }
        ).sort_values("Importancia", ascending=False)

        imp_fig = px.bar(
            importancia.sort_values("Importancia", ascending=True),
            x="Importancia",
            y="Variable",
            orientation="h",
        )
        imp_fig.update_layout(
            height=480,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            margin=dict(l=10, r=30, t=20, b=45),
            xaxis_title="Importancia relativa",
            yaxis_title="",
        )
        st.plotly_chart(imp_fig, use_container_width=True)

        st.caption(
            "La importancia indica cuánto contribuye cada variable a las decisiones del Random Forest. "
            "No representa una causalidad médica ni una probabilidad individual de falla."
        )

    # -----------------------
    # ANALISIS DE UMBRAL
    # -----------------------
    if modelo_supervisado is not None and "CRÍTICO" in list(modelo_supervisado.classes_):
        st.markdown('<div class="section-title">Análisis del umbral de decisión</div>', unsafe_allow_html=True)
        st.write(
            "Se prueban distintos umbrales de probabilidad para determinar cuándo el modelo debe clasificar un registro como crítico. "
            "Esto permite seleccionar un punto que priorice la detección de problemas sin incrementar innecesariamente los falsos positivos."
        )

        classes = list(modelo_supervisado.classes_)
        critical_idx = classes.index("CRÍTICO")
        prob_crit_test = modelo_supervisado.predict_proba(X_test)[:, critical_idx]

        threshold_rows = []
        for threshold in np.arange(0.30, 0.76, 0.05):
            pred_threshold = np.where(prob_crit_test >= threshold, "CRÍTICO", "ESTABLE")
            met = metricas_clasificacion(y_test_real, pred_threshold)
            threshold_rows.append(
                {
                    "Umbral": round(float(threshold), 2),
                    "Accuracy": met["accuracy"],
                    "Precision": met["precision"],
                    "Recall": met["recall"],
                    "F1-Score": met["f1"],
                    "Falsos negativos": met["fn"],
                }
            )

        threshold_df = pd.DataFrame(threshold_rows)
        threshold_display = threshold_df.copy()
        for c in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            threshold_display[c] = threshold_display[c].map(lambda x: f"{x:.2%}")
        st.dataframe(threshold_display, use_container_width=True, hide_index=True)

    # -----------------------
    # CONTAMINATION
    # -----------------------
    st.markdown('<div class="section-title">Sensibilidad del parámetro contamination</div>', unsafe_allow_html=True)
    st.write(
        "Se compara cómo cambia la clasificación del Isolation Forest cuando se modifica la proporción esperada de observaciones anómalas. "
        "El valor utilizado actualmente se muestra en la tabla."
    )

    valores_prueba = sorted(set([0.05, 0.10, 0.15, 0.20, CONTAMINATION]))
    resultados_contamination = []

    for c in valores_prueba:
        if c <= 0 or c >= 0.50:
            continue

        modelo_temp = IsolationForest(
            n_estimators=250,
            contamination=c,
            random_state=42,
            n_jobs=-1,
        )
        modelo_temp.fit(X_train)
        pred_temp = modelo_temp.predict(X_test)
        pred_temp_label = np.where(pred_temp == -1, "CRÍTICO", "ESTABLE")
        met = metricas_clasificacion(y_test_real, pred_temp_label)

        resultados_contamination.append(
            {
                "Contamination": c,
                "Accuracy": met["accuracy"],
                "Precision": met["precision"],
                "Recall": met["recall"],
                "F1-Score": met["f1"],
                "Configuración actual": "ACTUAL" if abs(c - CONTAMINATION) < 0.0001 else "",
            }
        )

    contamination_df = pd.DataFrame(resultados_contamination)
    if not contamination_df.empty:
        contamination_display = contamination_df.copy()
        for c in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            contamination_display[c] = contamination_display[c].map(lambda x: f"{x:.2%}")
        st.dataframe(contamination_display, use_container_width=True, hide_index=True)


# ============================================================
# INVENTARIO
# ============================================================

with tab_datos:
    st.markdown('<div class="section-title">Inventario técnico completo</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Datos originales de Supabase junto con las variables calculadas por el sistema de IA.</div>',
        unsafe_allow_html=True,
    )

    inventario = df.copy().sort_values("Riesgo_IA", ascending=False)
    st.dataframe(inventario, use_container_width=True, hide_index=True)


# ============================================================
# PIE
# ============================================================

st.divider()
st.markdown(
    '<div class="small-muted">AI-FleetMonitor Pro — Monitoreo, evaluación y entrenamiento del modelo de clasificación de riesgo.</div>',
    unsafe_allow_html=True,
)
