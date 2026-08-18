import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from supabase import create_client
import time


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="AI-FleetMonitor Pro",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DISEÑO PROFESIONAL
# ============================================================

st.markdown("""
<style>

    /* --------------------------------------------------------
       FONDO GENERAL
       -------------------------------------------------------- */

    .stApp {
        background-color: #f5f7fb;
    }

    /* --------------------------------------------------------
       HEADER
       -------------------------------------------------------- */

    .main-header {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #1e3a8a 55%,
            #2563eb 100%
        );

        padding: 28px 32px;
        border-radius: 18px;
        margin-bottom: 24px;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.15);
    }

    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 32px;
        font-weight: 750;
    }

    .main-header p {
        color: #dbeafe;
        margin-top: 8px;
        margin-bottom: 0;
        font-size: 15px;
    }


    /* --------------------------------------------------------
       TARJETAS
       -------------------------------------------------------- */

    .info-card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
        margin-bottom: 15px;
    }

    .section-card {
        background: white;
        padding: 24px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
        margin-bottom: 20px;
    }


    /* --------------------------------------------------------
       RIESGO
       -------------------------------------------------------- */

    .risk-critical {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
        padding: 18px;
        border-radius: 14px;
        font-weight: 600;
    }

    .risk-high {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
        padding: 18px;
        border-radius: 14px;
        font-weight: 600;
    }

    .risk-medium {
        background: #fefce8;
        border: 1px solid #fde68a;
        color: #854d0e;
        padding: 18px;
        border-radius: 14px;
        font-weight: 600;
    }

    .risk-stable {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #166534;
        padding: 18px;
        border-radius: 14px;
        font-weight: 600;
    }


    /* --------------------------------------------------------
       MÉTRICAS
       -------------------------------------------------------- */

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        padding: 17px;
        border-radius: 14px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
    }

    [data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 750;
    }


    /* --------------------------------------------------------
       SIDEBAR
       -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }


    /* --------------------------------------------------------
       BOTONES
       -------------------------------------------------------- */

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #cbd5e1;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #2563eb;
        color: #2563eb;
    }


    /* --------------------------------------------------------
       TABLAS
       -------------------------------------------------------- */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }


    /* --------------------------------------------------------
       TÍTULOS
       -------------------------------------------------------- */

    h2 {
        color: #0f172a;
        font-weight: 750;
    }

    h3 {
        color: #1e293b;
        font-weight: 700;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>🖥️ AI-FleetMonitor Pro</h1>
    <p>
        Plataforma inteligente para monitoreo, detección de anomalías,
        clasificación de riesgo y diagnóstico técnico de equipos.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SUPABASE
# ============================================================

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_ANON_KEY
)


# ============================================================
# CARGA DE DATOS
# ============================================================

@st.cache_data(ttl=60)
def cargar_datos(_cache_bust):

    resultado = (
        supabase
        .table("equipos")
        .select("*")
        .execute()
    )

    df = pd.DataFrame(resultado.data)

    if df.empty:
        return df

    df = df.rename(columns={
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
    })

    columnas_numericas = [
        "Uso_CPU_Porcentaje",
        "Uso_RAM_Porcentaje",
        "CPU_Normalizado_Porcentaje",
        "Porcentaje_Bateria",
        "Uso_Disco_Porcentaje"
    ]

    for columna in columnas_numericas:

        if columna not in df.columns:
            df[columna] = np.nan

        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )

        df[columna] = df[columna].clip(
            0,
            100
        )

    if "Fecha_Hora" in df.columns:

        df["Fecha_Hora"] = pd.to_datetime(
            df["Fecha_Hora"],
            errors="coerce"
        )

    if "Ticket_Usuario" not in df.columns:
        df["Ticket_Usuario"] = np.nan

    tickets = (
        df["Ticket_Usuario"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["Tiene_Ticket"] = tickets.ne("")

    df["Clase_Real"] = np.where(
        df["Tiene_Ticket"],
        "CRÍTICO",
        "ESTABLE"
    )

    return df


# ============================================================
# CARGAR
# ============================================================

df = cargar_datos(
    int(time.time() // 60)
)


if st.button("🔄 Actualizar datos"):

    st.cache_data.clear()
    st.rerun()


if df.empty:

    st.warning(
        "Todavía no hay información disponible en Supabase. "
        "Ejecuta `agente_monitor.py` en al menos un equipo."
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
    "Uso_Disco_Porcentaje"
]


for columna in columnas_base:

    if columna not in df.columns:
        df[columna] = 0

    mediana = df[columna].median()

    if pd.isna(mediana):
        mediana = 0

    df[columna] = (
        df[columna]
        .fillna(mediana)
        .clip(0, 100)
    )


# ============================================================
# VARIABLES DERIVADAS
# ============================================================

df["Presion_Recursos"] = df[
    [
        "Uso_CPU_Porcentaje",
        "Uso_RAM_Porcentaje",
        "Uso_Disco_Porcentaje"
    ]
].max(axis=1)


df["Promedio_Recursos"] = df[
    [
        "Uso_CPU_Porcentaje",
        "Uso_RAM_Porcentaje",
        "Uso_Disco_Porcentaje"
    ]
].mean(axis=1)


df["CPU_RAM_Conjunta"] = (
    df["Uso_CPU_Porcentaje"]
    * df["Uso_RAM_Porcentaje"]
) / 100


df["Diferencia_CPU"] = (
    df["CPU_Normalizado_Porcentaje"]
    - df["Uso_CPU_Porcentaje"]
).abs()


df["Componentes_Saturados"] = (
    (df["Uso_CPU_Porcentaje"] >= 85).astype(int)
    +
    (df["Uso_RAM_Porcentaje"] >= 85).astype(int)
    +
    (df["Uso_Disco_Porcentaje"] >= 90).astype(int)
)


# ============================================================
# TENDENCIAS
# ============================================================

if "Fecha_Hora" in df.columns:

    df = df.sort_values(
        ["ID_PC", "Fecha_Hora"]
    ).copy()

    df["CPU_Anterior"] = (
        df.groupby("ID_PC")[
            "Uso_CPU_Porcentaje"
        ].shift(1)
    )

    df["RAM_Anterior"] = (
        df.groupby("ID_PC")[
            "Uso_RAM_Porcentaje"
        ].shift(1)
    )

    df["Disco_Anterior"] = (
        df.groupby("ID_PC")[
            "Uso_Disco_Porcentaje"
        ].shift(1)
    )

    df["Tendencia_CPU"] = (
        df["Uso_CPU_Porcentaje"]
        -
        df["CPU_Anterior"]
    ).fillna(0)

    df["Tendencia_RAM"] = (
        df["Uso_RAM_Porcentaje"]
        -
        df["RAM_Anterior"]
    ).fillna(0)

    df["Tendencia_Disco"] = (
        df["Uso_Disco_Porcentaje"]
        -
        df["Disco_Anterior"]
    ).fillna(0)

else:

    df["Tendencia_CPU"] = 0
    df["Tendencia_RAM"] = 0
    df["Tendencia_Disco"] = 0


# ============================================================
# FEATURES
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
    "Tendencia_Disco"
]


X = df[features].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median(numeric_only=True)
)

X = X.fillna(0)


# ============================================================
# DATOS PARA CLASIFICACIÓN
# ============================================================

n_total = len(df)

cantidad_criticos = int(
    (df["Clase_Real"] == "CRÍTICO").sum()
)

cantidad_estables = int(
    (df["Clase_Real"] == "ESTABLE").sum()
)


puede_usar_supervisado = (
    n_total >= 20
    and cantidad_criticos >= 5
    and cantidad_estables >= 5
)


# ============================================================
# TRAIN / TEST
# ============================================================

if puede_usar_supervisado:

    train_df, test_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["Clase_Real"]
    )

else:

    train_df = df.copy()
    test_df = df.copy()


X_train = train_df[features].copy()

X_test = test_df[features].copy()


X_train = (
    X_train
    .replace([np.inf, -np.inf], np.nan)
    .fillna(X.median())
    .fillna(0)
)


X_test = (
    X_test
    .replace([np.inf, -np.inf], np.nan)
    .fillna(X.median())
    .fillna(0)
)


# ============================================================
# VARIABLES DE EVALUACIÓN
# IMPORTANTE:
# SE DEFINEN SIEMPRE
# ============================================================

y_test_real = test_df[
    "Clase_Real"
].values


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
        n_jobs=-1
    )

    modelo_supervisado.fit(
        X_train,
        train_df["Clase_Real"]
    )


# ============================================================
# CONTAMINATION DINÁMICO
# ============================================================

proporcion_criticos = (
    cantidad_criticos / n_total
    if n_total > 0
    else 0.10
)


CONTAMINATION = float(
    np.clip(
        max(0.05, proporcion_criticos),
        0.05,
        0.25
    )
)


# ============================================================
# ISOLATION FOREST
# ============================================================

modelo_anomalia = IsolationForest(
    n_estimators=400,
    contamination=CONTAMINATION,
    max_samples="auto",
    random_state=42,
    n_jobs=-1
)

modelo_anomalia.fit(
    X_train
)


pred_anomalia = modelo_anomalia.predict(X)

raw_anomaly = (
    -modelo_anomalia.decision_function(X)
)


# ============================================================
# SCORE DE ANOMALÍA 0-100
# ============================================================

p05 = np.percentile(
    raw_anomaly,
    5
)

p95 = np.percentile(
    raw_anomaly,
    95
)


if p95 > p05:

    anomaly_score = (
        (raw_anomaly - p05)
        /
        (p95 - p05)
    ) * 100

else:

    anomaly_score = np.full(
        len(df),
        50.0
    )


anomaly_score = np.clip(
    anomaly_score,
    0,
    100
)


df["Score_Anomalia"] = anomaly_score


# ============================================================
# PREDICCIÓN SUPERVISADA
# ============================================================

if modelo_supervisado is not None:

    probabilidades = (
        modelo_supervisado
        .predict_proba(X)
    )

    clases_modelo = list(
        modelo_supervisado.classes_
    )

    if "CRÍTICO" in clases_modelo:

        indice_critico = clases_modelo.index(
            "CRÍTICO"
        )

        prob_critico = probabilidades[
            :,
            indice_critico
        ]

    else:

        prob_critico = np.zeros(
            len(df)
        )

else:

    prob_critico = np.zeros(
        len(df)
    )


# ============================================================
# SCORE TÉCNICO
# ============================================================

def calcular_score_tecnico(row):

    cpu = row["Uso_CPU_Porcentaje"]
    ram = row["Uso_RAM_Porcentaje"]
    disco = row["Uso_Disco_Porcentaje"]
    cpu_norm = row[
        "CPU_Normalizado_Porcentaje"
    ]

    score = 0

    # CPU
    if cpu >= 95:
        score += 30

    elif cpu >= 85:
        score += 22

    elif cpu >= 75:
        score += 12

    elif cpu >= 65:
        score += 5


    # RAM
    if ram >= 95:
        score += 25

    elif ram >= 85:
        score += 18

    elif ram >= 75:
        score += 10

    elif ram >= 65:
        score += 5


    # DISCO
    if disco >= 97:
        score += 25

    elif disco >= 90:
        score += 18

    elif disco >= 80:
        score += 10

    elif disco >= 70:
        score += 5


    # CPU NORMALIZADA
    if cpu_norm >= 95:
        score += 20

    elif cpu_norm >= 85:
        score += 12

    elif cpu_norm >= 75:
        score += 5


    return min(
        score,
        100
    )


df["Score_Tecnico"] = df.apply(
    calcular_score_tecnico,
    axis=1
)


# ============================================================
# SCORE FINAL
# ============================================================

if modelo_supervisado is not None:

    df["Riesgo_IA"] = (
        prob_critico * 0.55
        +
        anomaly_score * 0.25
        +
        df["Score_Tecnico"] * 0.20
    )

else:

    df["Riesgo_IA"] = (
        anomaly_score * 0.55
        +
        df["Score_Tecnico"] * 0.45
    )


df["Riesgo_IA"] = np.clip(
    df["Riesgo_IA"],
    0,
    100
)


# ============================================================
# CLASIFICACIÓN
# ============================================================

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


df["Estado"] = df[
    "Riesgo_IA"
].apply(determinar_estado)


df["Nivel_Riesgo"] = df[
    "Riesgo_IA"
].apply(determinar_nivel)


# ============================================================
# DIAGNÓSTICO
# ============================================================

def generar_diagnostico(row):

    problemas = []

    cpu = row["Uso_CPU_Porcentaje"]
    ram = row["Uso_RAM_Porcentaje"]
    disco = row["Uso_Disco_Porcentaje"]
    cpu_norm = row[
        "CPU_Normalizado_Porcentaje"
    ]

    tendencia_cpu = row[
        "Tendencia_CPU"
    ]

    tendencia_ram = row[
        "Tendencia_RAM"
    ]

    tendencia_disco = row[
        "Tendencia_Disco"
    ]


    if cpu >= 95:

        problemas.append(
            "La CPU presenta una saturación crítica "
            "(≥95%)."
        )

    elif cpu >= 85:

        problemas.append(
            "La CPU presenta un nivel elevado "
            "(≥85%)."
        )


    if ram >= 95:

        problemas.append(
            "La RAM está en nivel crítico "
            "(≥95%), indicando una alta presión de memoria."
        )

    elif ram >= 85:

        problemas.append(
            "El consumo de RAM es elevado "
            "(≥85%)."
        )


    if disco >= 97:

        problemas.append(
            "La actividad del disco es extremadamente elevada "
            "(≥97%)."
        )

    elif disco >= 90:

        problemas.append(
            "La actividad del disco es elevada "
            "(≥90%)."
        )


    if cpu_norm >= 90:

        problemas.append(
            "La CPU normalizada confirma un comportamiento "
            "de alta carga."
        )


    if tendencia_cpu >= 15:

        problemas.append(
            "La utilización de CPU está aumentando rápidamente."
        )


    if tendencia_ram >= 15:

        problemas.append(
            "El consumo de RAM está aumentando respecto "
            "a la medición anterior."
        )


    if tendencia_disco >= 15:

        problemas.append(
            "La actividad del almacenamiento presenta "
            "un incremento considerable."
        )


    if not problemas:

        return (
            "No se identifican señales técnicas críticas "
            "en las métricas actuales. El comportamiento "
            "del equipo es consistente con una operación estable."
        )


    return " ".join(problemas)


df["Diagnostico_IA"] = df.apply(
    generar_diagnostico,
    axis=1
)


# ============================================================
# RECOMENDACIONES
# ============================================================

def generar_recomendaciones(row):

    recomendaciones = []

    cpu = row[
        "Uso_CPU_Porcentaje"
    ]

    ram = row[
        "Uso_RAM_Porcentaje"
    ]

    disco = row[
        "Uso_Disco_Porcentaje"
    ]

    bateria = row[
        "Porcentaje_Bateria"
    ]


    if cpu >= 85:

        recomendaciones.append(
            "Revisar el Administrador de tareas y "
            "localizar los procesos con mayor consumo de CPU."
        )


    if ram >= 85:

        recomendaciones.append(
            "Revisar aplicaciones con alto consumo de memoria "
            "y cerrar procesos innecesarios."
        )


    if disco >= 90:

        recomendaciones.append(
            "Revisar aplicaciones con alta actividad de disco, "
            "estado del almacenamiento y espacio disponible."
        )


    if bateria <= 15:

        recomendaciones.append(
            "Conectar el equipo a corriente y verificar el "
            "estado de la batería."
        )


    if (
        cpu >= 85
        and ram >= 85
    ):

        recomendaciones.append(
            "Realizar una revisión de las aplicaciones que "
            "están generando simultáneamente alta carga de CPU y RAM."
        )


    if (
        cpu >= 85
        and ram >= 85
        and disco >= 90
    ):

        recomendaciones.append(
            "Realizar diagnóstico integral del equipo debido "
            "a saturación simultánea de CPU, memoria y almacenamiento."
        )


    if not recomendaciones:

        recomendaciones.append(
            "No se requiere una intervención inmediata. "
            "Mantener el monitoreo preventivo."
        )


    return recomendaciones


df["Recomendaciones_IA"] = df.apply(
    generar_recomendaciones,
    axis=1
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## 🔍 Diagnóstico"
)

pc_seleccionado = st.sidebar.selectbox(
    "Equipo:",
    sorted(
        df["ID_PC"]
        .astype(str)
        .unique()
    )
)


pc_data = df[
    df["ID_PC"].astype(str)
    ==
    str(pc_seleccionado)
]


st.sidebar.divider()

st.sidebar.markdown(
    "## 🎫 Reportar incidente"
)


with st.sidebar.form(
    "form_ticket"
):

    pc_ticket = st.selectbox(
        "Equipo:",
        sorted(
            df["ID_PC"]
            .astype(str)
            .unique()
        ),
        key="pc_ticket"
    )

    descripcion = st.text_area(
        "Descripción:",
        placeholder=(
            "Ejemplo: el equipo está lento, "
            "se congela, presenta pantalla azul..."
        )
    )

    enviado = st.form_submit_button(
        "🎫 Registrar ticket",
        use_container_width=True
    )


    if enviado:

        if descripcion.strip():

            try:

                supabase.rpc(
                    "reportar_ticket",
                    {
                        "p_id_pc": pc_ticket,
                        "p_ticket": descripcion.strip()
                    }
                ).execute()

                st.sidebar.success(
                    "Ticket registrado correctamente."
                )

                st.cache_data.clear()

                st.rerun()

            except Exception as e:

                st.sidebar.error(
                    f"Error al registrar: {e}"
                )

        else:

            st.sidebar.warning(
                "Debes escribir una descripción."
            )


# ============================================================
# EQUIPO SELECCIONADO
# ============================================================

equipo = pc_data.iloc[0]

indice_equipo = equipo.name

riesgo_pc = float(
    equipo["Riesgo_IA"]
)

estado_pc = equipo[
    "Estado"
]


# ============================================================
# KPI PRINCIPALES
# ============================================================

st.markdown(
    "### 📊 Estado general de la flota"
)


k1, k2, k3, k4, k5 = st.columns(5)


k1.metric(
    "🖥️ Equipos",
    len(df)
)

k2.metric(
    "🚨 En riesgo",
    int(
        df["Estado"]
        .isin(["CRÍTICO", "ALTO"])
        .sum()
    )
)

k3.metric(
    "🔴 Críticos",
    int(
        (
            df["Estado"] ==
            "CRÍTICO"
        ).sum()
    )
)

k4.metric(
    "⚙️ CPU promedio",
    f"{df['Uso_CPU_Porcentaje'].mean():.1f}%"
)

k5.metric(
    "🧠 RAM promedio",
    f"{df['Uso_RAM_Porcentaje'].mean():.1f}%"
)


# ============================================================
# DIAGNÓSTICO INDIVIDUAL
# ============================================================

st.markdown(
    "### 🧠 Diagnóstico inteligente"
)


estado_iconos = {
    "CRÍTICO": "🔴",
    "ALTO": "🟠",
    "MEDIO": "🟡",
    "ESTABLE": "🟢"
}


icono = estado_iconos.get(
    estado_pc,
    "⚪"
)


if estado_pc == "CRÍTICO":

    clase_css = "risk-critical"

elif estado_pc == "ALTO":

    clase_css = "risk-high"

elif estado_pc == "MEDIO":

    clase_css = "risk-medium"

else:

    clase_css = "risk-stable"


st.markdown(
    f"""
    <div class="{clase_css}">
        {icono} {estado_pc} —
        Riesgo IA: {riesgo_pc:.1f}/100 —
        Nivel: {equipo['Nivel_Riesgo']}
    </div>
    """,
    unsafe_allow_html=True
)


st.write("")


d1, d2, d3, d4 = st.columns(4)


d1.metric(
    "Riesgo IA",
    f"{riesgo_pc:.1f}/100"
)

d2.metric(
    "Anomalía",
    f"{equipo['Score_Anomalia']:.1f}/100"
)

d3.metric(
    "Riesgo técnico",
    f"{equipo['Score_Tecnico']:.1f}/100"
)

d4.metric(
    "Ticket",
    "Sí"
    if equipo["Tiene_Ticket"]
    else "No"
)


# ============================================================
# RECURSOS
# ============================================================

r1, r2, r3, r4, r5 = st.columns(5)


r1.metric(
    "CPU",
    f"{equipo['Uso_CPU_Porcentaje']:.1f}%"
)

r2.metric(
    "RAM",
    f"{equipo['Uso_RAM_Porcentaje']:.1f}%"
)

r3.metric(
    "CPU normalizada",
    f"{equipo['CPU_Normalizado_Porcentaje']:.1f}%"
)

r4.metric(
    "Disco",
    f"{equipo['Uso_Disco_Porcentaje']:.1f}%"
)

r5.metric(
    "Batería",
    f"{equipo['Porcentaje_Bateria']:.1f}%"
)


# ============================================================
# DIAGNÓSTICO
# ============================================================

with st.container():

    st.markdown(
        "#### 🔎 Análisis de la IA"
    )

    st.info(
        equipo["Diagnostico_IA"]
    )


    st.markdown(
        "#### 🛠️ Recomendaciones"
    )

    for recomendacion in equipo[
        "Recomendaciones_IA"
    ]:

        st.write(
            f"• {recomendacion}"
        )


# ============================================================
# TABS PRINCIPALES
# ============================================================

tab_dashboard, tab_evaluacion, tab_inventario = st.tabs(
    [
        "📊 Dashboard",
        "🤖 Evaluación de IA",
        "🖥️ Inventario"
    ]
)


# ============================================================
# TAB DASHBOARD
# ============================================================

with tab_dashboard:

    st.markdown(
        "### 🗺️ Mapa de riesgo"
    )

    c1, c2 = st.columns(
        [1, 2]
    )


    with c1:

        st.markdown(
            "#### Equipo seleccionado"
        )

        detalle = pd.DataFrame(
            {
                "Indicador": [
                    "ID",
                    "Usuario",
                    "Modelo",
                    "Serial",
                    "Estado IA",
                    "Nivel",
                    "Riesgo",
                    "CPU",
                    "RAM",
                    "CPU normalizada",
                    "Disco",
                    "Batería"
                ],

                "Valor": [
                    equipo["ID_PC"],
                    equipo.get(
                        "Usuario",
                        "No disponible"
                    ),
                    equipo.get(
                        "Modelo",
                        "No disponible"
                    ),
                    equipo.get(
                        "Serial",
                        "No disponible"
                    ),
                    estado_pc,
                    equipo["Nivel_Riesgo"],
                    f"{riesgo_pc:.1f}/100",
                    f"{equipo['Uso_CPU_Porcentaje']:.1f}%",
                    f"{equipo['Uso_RAM_Porcentaje']:.1f}%",
                    f"{equipo['CPU_Normalizado_Porcentaje']:.1f}%",
                    f"{equipo['Uso_Disco_Porcentaje']:.1f}%",
                    f"{equipo['Porcentaje_Bateria']:.1f}%"
                ]
            }
        )


        st.dataframe(
            detalle,
            use_container_width=True,
            hide_index=True
        )


    with c2:

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
                "Tiene_Ticket"
            ],
            color_discrete_map={
                "CRÍTICO": "#dc2626",
                "ALTO": "#f97316",
                "MEDIO": "#eab308",
                "ESTABLE": "#2563eb"
            }
        )


        mapa.update_layout(
            height=500,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis_title="Uso de CPU (%)",
            yaxis_title="Uso de RAM (%)",
            legend_title="Estado IA"
        )


        st.plotly_chart(
            mapa,
            use_container_width=True
        )


    # --------------------------------------------------------
    # PRIORIZACIÓN
    # --------------------------------------------------------

    st.markdown(
        "### 🚨 Priorización de atención"
    )


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
            "Tiene_Ticket",
            "Diagnostico_IA"
        ]
    ].copy()


    priorizacion = priorizacion.sort_values(
        "Riesgo_IA",
        ascending=False
    )


    st.dataframe(
        priorizacion,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # DISTRIBUCIÓN RIESGO
    # --------------------------------------------------------

    st.markdown(
        "### 📈 Distribución del riesgo"
    )


    fig_riesgo = px.histogram(
        df,
        x="Riesgo_IA",
        color="Estado",
        nbins=20,
        color_discrete_map={
            "CRÍTICO": "#dc2626",
            "ALTO": "#f97316",
            "MEDIO": "#eab308",
            "ESTABLE": "#2563eb"
        }
    )


    fig_riesgo.update_layout(
        height=400,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Riesgo IA (0 - 100)",
        yaxis_title="Cantidad de equipos"
    )


    st.plotly_chart(
        fig_riesgo,
        use_container_width=True
    )


# ============================================================
# TAB EVALUACIÓN
# ============================================================

with tab_evaluacion:

    st.markdown(
        "## 🤖 Evaluación y validación de la IA"
    )


    # --------------------------------------------------------
    # ESTADO DEL ENTRENAMIENTO
    # --------------------------------------------------------

    if modelo_supervisado is not None:

        st.success(
            f"""
            **Modelo supervisado activo**

            Registros totales: **{n_total}**

            Registros críticos: **{cantidad_criticos}**

            Registros estables: **{cantidad_estables}**

            Entrenamiento: **{len(train_df)}**

            Prueba: **{len(test_df)}**
            """
        )

    else:

        st.warning(
            f"""
            **Modelo supervisado todavía no disponible.**

            Registros actuales: **{n_total}**

            Críticos con ticket: **{cantidad_criticos}**

            Estables: **{cantidad_estables}**

            Se necesitan al menos 20 registros y suficientes
            ejemplos de ambas clases para entrenar una clasificación
            supervisada confiable.

            Mientras tanto, la plataforma utiliza Isolation Forest
            + reglas técnicas.
            """
        )


    # --------------------------------------------------------
    # EVALUACIÓN
    # --------------------------------------------------------

    st.markdown(
        "### 🎯 Calidad de clasificación"
    )


    if modelo_supervisado is not None:

        pred_test = (
            modelo_supervisado
            .predict(X_test)
        )

        y_pred_evaluacion = pred_test

        tipo_modelo = (
            "Random Forest supervisado"
        )

    else:

        # Evaluación preliminar mediante Isolation Forest
        # para conservar las métricas visibles incluso
        # cuando aún no existe suficiente historial.

        pred_test_anomalia = (
            modelo_anomalia
            .predict(X_test)
        )

        y_pred_evaluacion = np.where(
            pred_test_anomalia == -1,
            "CRÍTICO",
            "ESTABLE"
        )

        tipo_modelo = (
            "Isolation Forest — evaluación preliminar"
        )


    # --------------------------------------------------------
    # Las 4 métricas SIEMPRE EXISTEN
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test_real,
        y_pred_evaluacion
    )


    precision = precision_score(
        y_test_real,
        y_pred_evaluacion,
        pos_label="CRÍTICO",
        zero_division=0
    )


    recall = recall_score(
        y_test_real,
        y_pred_evaluacion,
        pos_label="CRÍTICO",
        zero_division=0
    )


    f1 = f1_score(
        y_test_real,
        y_pred_evaluacion,
        pos_label="CRÍTICO",
        zero_division=0
    )


    # --------------------------------------------------------
    # CUADRO DE 4 MÉTRICAS
    # --------------------------------------------------------

    metric1, metric2, metric3, metric4 = st.columns(4)


    metric1.metric(
        "🎯 Accuracy",
        f"{accuracy:.2%}"
    )


    metric2.metric(
        "🎯 Precision",
        f"{precision:.2%}"
    )


    metric3.metric(
        "🎯 Recall",
        f"{recall:.2%}"
    )


    metric4.metric(
        "🎯 F1-Score",
        f"{f1:.2%}"
    )


    st.caption(
        f"Método evaluado: **{tipo_modelo}**"
    )


    if modelo_supervisado is None:

        st.info(
            "Estas métricas son preliminares porque todavía no hay "
            "suficiente historial etiquetado para entrenar Random Forest. "
            "Cuando aumenten los tickets reales, la evaluación "
            "supervisada será más representativa."
        )


    # --------------------------------------------------------
    # INTERPRETACIÓN DE MÉTRICAS
    # --------------------------------------------------------

    st.markdown(
        "### 📖 Interpretación"
    )


    i1, i2, i3, i4 = st.columns(4)


    i1.markdown(
        """
        **Accuracy**

        Porcentaje total de predicciones correctas.
        """
    )


    i2.markdown(
        """
        **Precision**

        De los equipos que la IA marcó como críticos,
        cuántos realmente fueron críticos.
        """
    )


    i3.markdown(
        """
        **Recall**

        De los equipos realmente críticos,
        cuántos logró detectar la IA.
        """
    )


    i4.markdown(
        """
        **F1-Score**

        Equilibrio entre Precision y Recall.
        """
    )


    # --------------------------------------------------------
    # MATRIZ Y REPORTE
    # --------------------------------------------------------

    st.markdown(
        "### 🔢 Matriz de confusión y reporte"
    )


    col_cm, col_rep = st.columns(
        [1, 1]
    )


    labels = [
        "CRÍTICO",
        "ESTABLE"
    ]


    with col_cm:

        cm = confusion_matrix(
            y_test_real,
            y_pred_evaluacion,
            labels=labels
        )


        fig_cm, ax = plt.subplots(
            figsize=(6, 4.5)
        )


        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=labels,
            yticklabels=labels
        )


        ax.set_xlabel(
            "Predicción IA"
        )

        ax.set_ylabel(
            "Realidad"
        )

        ax.set_title(
            "Matriz de Confusión"
        )


        st.pyplot(
            fig_cm
        )


    with col_rep:

        reporte = classification_report(
            y_test_real,
            y_pred_evaluacion,
            labels=labels,
            output_dict=True,
            zero_division=0
        )


        reporte_df = (
            pd.DataFrame(reporte)
            .transpose()
            .round(3)
        )


        st.dataframe(
            reporte_df,
            use_container_width=True
        )


    # --------------------------------------------------------
    # ROC / AUC
    # --------------------------------------------------------

    st.markdown(
        "### 📈 Curva ROC y AUC"
    )


    if modelo_supervisado is not None:

        probabilidades_test = (
            modelo_supervisado
            .predict_proba(X_test)
        )

        clases_test = list(
            modelo_supervisado.classes_
        )


        if (
            "CRÍTICO" in clases_test
            and len(
                np.unique(y_test_real)
            ) == 2
        ):

            indice_critico_test = (
                clases_test.index(
                    "CRÍTICO"
                )
            )


            score_roc = (
                probabilidades_test[
                    :,
                    indice_critico_test
                ]
            )


            y_binario = (
                y_test_real ==
                "CRÍTICO"
            ).astype(int)


            fpr, tpr, _ = roc_curve(
                y_binario,
                score_roc
            )


            auc_score = auc(
                fpr,
                tpr
            )


            fig_roc = go.Figure()


            fig_roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=(
                        f"Random Forest "
                        f"(AUC = {auc_score:.3f})"
                    ),
                    line=dict(
                        color="#2563eb",
                        width=3
                    )
                )
            )


            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Azar (AUC = 0.50)",
                    line=dict(
                        color="#94a3b8",
                        dash="dash"
                    )
                )
            )


            fig_roc.update_layout(
                height=420,
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis_title="Tasa de falsos positivos",
                yaxis_title="Tasa de verdaderos positivos"
            )


            st.plotly_chart(
                fig_roc,
                use_container_width=True
            )


            st.metric(
                "AUC Score",
                f"{auc_score:.3f}"
            )


        else:

            st.info(
                "No hay suficientes clases representadas "
                "en el conjunto de prueba para calcular ROC/AUC."
            )


    else:

        st.info(
            "ROC/AUC supervisado estará disponible cuando "
            "existan suficientes tickets históricos de ambas clases."
        )


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    if modelo_supervisado is not None:

        st.markdown(
            "### 🧠 Importancia de las variables"
        )


        importancia = pd.DataFrame(
            {
                "Variable": features,
                "Importancia": (
                    modelo_supervisado
                    .feature_importances_
                )
            }
        ).sort_values(
            "Importancia",
            ascending=False
        )


        fig_importancia = px.bar(
            importancia,
            x="Importancia",
            y="Variable",
            orientation="h"
        )


        fig_importancia.update_layout(
            height=520,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )


        st.plotly_chart(
            fig_importancia,
            use_container_width=True
        )


    # --------------------------------------------------------
    # DISTRIBUCIÓN ANOMALÍA
    # --------------------------------------------------------

    st.markdown(
        "### 🔬 Distribución del puntaje de anomalía"
    )


    fig_hist = px.histogram(
        df,
        x="Score_Anomalia",
        color="Clase_Real",
        barmode="overlay",
        nbins=20,
        color_discrete_map={
            "CRÍTICO": "#dc2626",
            "ESTABLE": "#2563eb"
        }
    )


    fig_hist.update_layout(
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Puntaje de anomalía",
        yaxis_title="Cantidad"
    )


    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )


    # ========================================================
    # CONTAMINATION
    # ========================================================

    st.markdown(
        "### ⚙️ Sensibilidad del parámetro `contamination`"
    )


    st.write(
        "Esta prueba compara cómo cambia la detección de anomalías "
        "cuando modificamos la proporción esperada de equipos anómalos."
    )


    valores_prueba = sorted(
        set(
            [
                0.05,
                0.10,
                0.15,
                0.20,
                CONTAMINATION
            ]
        )
    )


    resultados_contamination = []


    # IMPORTANTE:
    # y_test_real SIEMPRE existe porque se creó antes.
    # Esto corrige exactamente el error que te apareció.

    for c in valores_prueba:

        if c <= 0 or c >= 0.50:
            continue


        modelo_temp = IsolationForest(
            n_estimators=250,
            contamination=c,
            random_state=42,
            n_jobs=-1
        )


        modelo_temp.fit(
            X_train
        )


        pred_temp = modelo_temp.predict(
            X_test
        )


        pred_temp_label = np.where(
            pred_temp == -1,
            "CRÍTICO",
            "ESTABLE"
        )


        accuracy_temp = accuracy_score(
            y_test_real,
            pred_temp_label
        )


        precision_temp = precision_score(
            y_test_real,
            pred_temp_label,
            pos_label="CRÍTICO",
            zero_division=0
        )


        recall_temp = recall_score(
            y_test_real,
            pred_temp_label,
            pos_label="CRÍTICO",
            zero_division=0
        )


        f1_temp = f1_score(
            y_test_real,
            pred_temp_label,
            pos_label="CRÍTICO",
            zero_division=0
        )


        resultados_contamination.append(
            {
                "Contamination": c,
                "Accuracy": accuracy_temp,
                "Precision": precision_temp,
                "Recall": recall_temp,
                "F1-Score": f1_temp,
                "Configuración actual": (
                    "✅ ACTUAL"
                    if abs(
                        c - CONTAMINATION
                    ) < 0.0001
                    else ""
                )
            }
        )


    resultados_df = pd.DataFrame(
        resultados_contamination
    )


    if not resultados_df.empty:

        resultados_mostrar = resultados_df.copy()


        for columna in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score"
        ]:

            resultados_mostrar[
                columna
            ] = (
                resultados_mostrar[
                    columna
                ] * 100
            ).round(2).astype(str) + "%"


        st.dataframe(
            resultados_mostrar,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# INVENTARIO
# ============================================================

with tab_inventario:

    st.markdown(
        "## 🖥️ Inventario técnico completo"
    )


    st.caption(
        "Información recibida directamente desde Supabase "
        "y enriquecida con los cálculos de la IA."
    )


    # Columnas más importantes primero
    columnas_prioritarias = [
        "ID_PC",
        "Fecha_Hora",
        "Usuario",
        "Modelo",
        "Serial",
        "Estado",
        "Nivel_Riesgo",
        "Riesgo_IA",
        "Score_Tecnico",
        "Score_Anomalia",
        "Uso_CPU_Porcentaje",
        "Uso_RAM_Porcentaje",
        "Uso_Disco_Porcentaje",
        "CPU_Normalizado_Porcentaje",
        "Porcentaje_Bateria",
        "Tiene_Ticket",
        "Diagnostico_IA"
    ]


    columnas_existentes = [
        c for c in columnas_prioritarias
        if c in df.columns
    ]


    inventario = df[
        columnas_existentes
    ].copy()


    inventario = inventario.sort_values(
        "Riesgo_IA",
        ascending=False
    )


    st.dataframe(
        inventario,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PIE
# ============================================================

st.divider()

st.caption(
    "AI-FleetMonitor Pro • Monitoreo inteligente • "
    "Detección de anomalías • Clasificación de riesgo • "
    "Diagnóstico técnico"
)
