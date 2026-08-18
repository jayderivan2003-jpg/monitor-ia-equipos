import time
import numpy as np
import pandas as pd
import streamlit as st

from supabase import create_client
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
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
    df = pd.DataFrame(columns=[
        "ID_PC", "Fecha_Hora", "Uso_CPU_Porcentaje", "Uso_RAM_Porcentaje",
        "CPU_Normalizado_Porcentaje", "Porcentaje_Bateria", "Uso_Disco_Porcentaje",
        "Ticket_Usuario", "Tiene_Ticket", "Clase_Real"
    ])


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
    if columna in df.columns:
        mediana = df[columna].median()
        if pd.isna(mediana):
            mediana = 0
        df[columna] = df[columna].fillna(mediana).clip(0, 100)
    else:
        df[columna] = 0.0


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
]

X = df[features].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True)).fillna(0)

n_total = len(df)
cantidad_criticos = int((df["Clase_Real"] == "CRÍTICO").sum()) if "Clase_Real" in df.columns else 0
cantidad_estables = int((df["Clase_Real"] == "ESTABLE").sum()) if "Clase_Real" in df.columns else 0

puede_usar_supervisado = (
    n_total >= 20 and cantidad_criticos >= 5 and cantidad_estables >= 5
)


# ============================================================
# TRAIN / TEST
# ============================================================

if puede_usar_supervisado:
    train_df, test_df = train_test_split(
        df,
        test_size=0.30,
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

y_test_real = test_df["Clase_Real"].values if "Clase_Real" in test_df.columns else np.array(["ESTABLE"] * len(test_df))


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

p05 = np.percentile(raw_anomaly, 5) if len(raw_anomaly) > 0 else 0
p95 = np.percentile(raw_anomaly, 95) if len(raw_anomaly) > 0 else 1

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
# EVALUACION: METRICAS BASE
# ============================================================

if modelo_supervisado is not None:
    pred_test = modelo_supervisado.predict(X_test)
    evaluacion_nombre = "Random Forest supervisado"
else:
    pred_test_if = modelo_anomalia.predict(X_test)
    pred_test = np.where(pred_test_if == -1, "CRÍTICO", "ESTABLE")
    evaluacion_nombre = "Isolation Forest — evaluación preliminar"


def metricas_clasificacion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=["CRÍTICO", "ESTABLE"])
    tn = int(cm[1, 1])
    fp = int(cm[1, 0])
    fn = int(cm[0, 1])
    tp = int(cm[0, 0])

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr_value = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label="CRÍTICO", zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label="CRÍTICO", zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label="CRÍTICO", zero_division=0),
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


metricas = metricas_clasificacion(y_test_real, pred_test)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## Diagnóstico individual")

pc_seleccionado = st.sidebar.selectbox(
    "Equipo",
    sorted(df["ID_PC"].astype(str).unique()) if not df.empty else [],
)

pc_data = df[df["ID_PC"].astype(str) == str(pc_seleccionado)] if not df.empty else pd.DataFrame()
equipo = pc_data.iloc[0] if not pc_data.empty else None

st.sidebar.divider()
st.sidebar.markdown("## Registro de ticket")

with st.sidebar.form("form_ticket"):
    pc_ticket = st.selectbox(
        "Equipo",
        sorted(df["ID_PC"].astype(str).unique()) if not df.empty else [],
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

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Total equipos", f"{len(df)}")
k2.metric("En riesgo", f"{int(df['Estado'].isin(['CRÍTICO', 'ALTO']).sum())}" if not df.empty else "0")
k3.metric("Críticos", f"{int((df['Estado'] == 'CRÍTICO').sum())}" if not df.empty else "0")
k4.metric("CPU promedio", f"{df['Uso_CPU_Porcentaje'].mean():.1f}%" if not df.empty else "0.0%")
k5.metric("RAM promedio", f"{df['Uso_RAM_Porcentaje'].mean():.1f}%" if not df.empty else "0.0%")


# ============================================================
# DIAGNOSTICO INDIVIDUAL
# ============================================================

st.markdown('<div class="section-title">Diagnóstico individual</div>', unsafe_allow_html=True)

if equipo is not None:
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
else:
    st.info("Selecciona un equipo válido para ver su diagnóstico.")


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
        if equipo is not None:
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
        if not df.empty:
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

    if not df.empty:
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
                "Tiene_Ticket",
                "Diagnostico_IA",
            ]
        ].copy().sort_values("Riesgo_IA", ascending=False)

        st.dataframe(priorizacion, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Distribución del riesgo</div>', unsafe_allow_html=True)
    if not df.empty:
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
        '<div class="section-note">Esta sección concentra la evidencia de entrenamiento, validación, rendimiento y comportamiento del modelo. Las métricas se calculan con los registros disponibles.</div>',
        unsafe_allow_html=True,
    )

    estado_entrenamiento = "Modelo supervisado activo" if modelo_supervisado is not None else "Modo preliminar: detección de anomalías"
    texto_entrenamiento = (
        f"Registros totales: {n_total} | Críticos: {cantidad_criticos} | Estables: {cantidad_estables} | "
        f"Entrenamiento: {len(train_df)} | Prueba: {len(test_df)}"
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
        st.info(
            "La clasificación supervisada se habilita cuando existen al menos 20 registros "
            "y suficiente representación de ambas clases. Hasta entonces, las métricas son una "
            "evaluación preliminar del Isolation Forest y no deben presentarse como validación final."
        )

    st.markdown('<div class="section-title">Métricas principales de clasificación</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    metric_cards = [
        ("Accuracy", f"{metricas['accuracy']*100:.1f}%", "Predicciones correctas sobre el total."),
        ("Precision", f"{metricas['precision']*100:.1f}%", "De lo marcado como crítico, cuánto era realmente crítico."),
        ("Recall", f"{metricas['recall']*100:.1f}%", "Capacidad de detectar los casos críticos reales."),
        ("F1-Score", f"{metricas['f1']*100:.1f}%", "Media armónica entre precisión y sensibilidad."),
    ]

    for col, (label, val, desc) in zip([m1, m2, m3, m4], metric_cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# INVENTARIO
# ============================================================

with tab_datos:
    st.markdown('<div class="section-title">Inventario y registros completos</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)
