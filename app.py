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
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    layout="wide",
    page_title="AI-FleetMonitor Pro - Diagnóstico de Hardware",
    page_icon="🖥️"
)

st.title("🖥️ AI-FleetMonitor Pro: Diagnóstico Inteligente de Hardware")
st.caption(
    "Sistema híbrido de detección de anomalías, clasificación de riesgo "
    "y diagnóstico técnico de equipos."
)


# ============================================================
# 1. CONEXIÓN A SUPABASE
# ============================================================

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_ANON_KEY
)


@st.cache_data(ttl=60)
def cargar_datos(_cache_bust):

    resultado = supabase.table("equipos").select("*").execute()

    df = pd.DataFrame(resultado.data)

    if df.empty:
        return df

    # --------------------------------------------------------
    # Renombrar columnas provenientes de Supabase
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Crear columnas que puedan faltar
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Limpiar valores fuera de rango
    # --------------------------------------------------------

    for columna in columnas_numericas:

        df[columna] = df[columna].clip(
            lower=0,
            upper=100
        )

    # --------------------------------------------------------
    # Fecha
    # --------------------------------------------------------

    if "Fecha_Hora" in df.columns:

        df["Fecha_Hora"] = pd.to_datetime(
            df["Fecha_Hora"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Ticket real
    #
    # Un ticket con texto = evidencia de problema reportado.
    # --------------------------------------------------------

    if "Ticket_Usuario" not in df.columns:
        df["Ticket_Usuario"] = np.nan

    ticket_limpio = (
        df["Ticket_Usuario"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["Tiene_Ticket"] = ticket_limpio.ne("")

    df["Clase_Real"] = np.where(
        df["Tiene_Ticket"],
        "CRÍTICO",
        "ESTABLE"
    )

    return df


# ============================================================
# CARGA
# ============================================================

df = cargar_datos(
    int(time.time() // 60)
)


if st.button("🔄 Actualizar ahora"):

    st.cache_data.clear()
    st.rerun()


if df.empty:

    st.warning(
        "Todavía no hay datos en Supabase. "
        "Ejecuta `agente_monitor.py` en al menos una PC "
        "para empezar a recibir información."
    )

    st.stop()


# ============================================================
# 2. PREPARACIÓN DE DATOS PARA IA
# ============================================================

# ------------------------------------------------------------
# Rellenar faltantes
# ------------------------------------------------------------

columnas_base = [
    "Uso_CPU_Porcentaje",
    "Uso_RAM_Porcentaje",
    "CPU_Normalizado_Porcentaje",
    "Porcentaje_Bateria",
    "Uso_Disco_Porcentaje"
]

for columna in columnas_base:

    if columna not in df.columns:
        df[columna] = np.nan


for columna in columnas_base:

    mediana = df[columna].median()

    if pd.isna(mediana):
        mediana = 0

    df[columna] = df[columna].fillna(mediana)


# ============================================================
# 3. VARIABLES INTELIGENTES DERIVADAS
# ============================================================

# ------------------------------------------------------------
# Presión general de recursos
# ------------------------------------------------------------

df["Presion_Recursos"] = df[
    [
        "Uso_CPU_Porcentaje",
        "Uso_RAM_Porcentaje",
        "Uso_Disco_Porcentaje"
    ]
].max(axis=1)


# ------------------------------------------------------------
# Promedio general de utilización
# ------------------------------------------------------------

df["Promedio_Recursos"] = df[
    [
        "Uso_CPU_Porcentaje",
        "Uso_RAM_Porcentaje",
        "Uso_Disco_Porcentaje"
    ]
].mean(axis=1)


# ------------------------------------------------------------
# Combinación CPU + RAM
# ------------------------------------------------------------

df["CPU_RAM_Conjunta"] = (
    df["Uso_CPU_Porcentaje"] *
    df["Uso_RAM_Porcentaje"]
) / 100


# ------------------------------------------------------------
# Diferencia CPU normalizada / CPU actual
# ------------------------------------------------------------

df["Diferencia_CPU"] = (
    df["CPU_Normalizado_Porcentaje"] -
    df["Uso_CPU_Porcentaje"]
).abs()


# ------------------------------------------------------------
# Nivel de saturación
# ------------------------------------------------------------

df["Componentes_Saturados"] = (
    (df["Uso_CPU_Porcentaje"] >= 85).astype(int) +
    (df["Uso_RAM_Porcentaje"] >= 85).astype(int) +
    (df["Uso_Disco_Porcentaje"] >= 90).astype(int)
)


# ============================================================
# 4. TENDENCIA POR EQUIPO
# ============================================================

# Si tenemos fecha, podemos calcular comportamiento reciente.

if "Fecha_Hora" in df.columns:

    df = df.sort_values(
        ["ID_PC", "Fecha_Hora"]
    ).copy()

    df["CPU_Anterior"] = (
        df.groupby("ID_PC")["Uso_CPU_Porcentaje"]
        .shift(1)
    )

    df["RAM_Anterior"] = (
        df.groupby("ID_PC")["Uso_RAM_Porcentaje"]
        .shift(1)
    )

    df["Disco_Anterior"] = (
        df.groupby("ID_PC")["Uso_Disco_Porcentaje"]
        .shift(1)
    )

    df["Tendencia_CPU"] = (
        df["Uso_CPU_Porcentaje"] -
        df["CPU_Anterior"]
    ).fillna(0)

    df["Tendencia_RAM"] = (
        df["Uso_RAM_Porcentaje"] -
        df["RAM_Anterior"]
    ).fillna(0)

    df["Tendencia_Disco"] = (
        df["Uso_Disco_Porcentaje"] -
        df["Disco_Anterior"]
    ).fillna(0)

else:

    df["Tendencia_CPU"] = 0
    df["Tendencia_RAM"] = 0
    df["Tendencia_Disco"] = 0


# ============================================================
# 5. FEATURES DE IA
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
# 6. CONFIGURACIÓN DEL MODELO
# ============================================================

n_total = len(df)

conteo_clases = df["Clase_Real"].value_counts()

cantidad_criticos = conteo_clases.get(
    "CRÍTICO",
    0
)

cantidad_estables = conteo_clases.get(
    "ESTABLE",
    0
)


# Se necesita suficiente información de ambas clases
puede_usar_supervisado = (
    n_total >= 20
    and cantidad_criticos >= 5
    and cantidad_estables >= 5
)


# ============================================================
# 7. TRAIN / TEST
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
# 8. MODELO SUPERVISADO
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
# 9. ISOLATION FOREST
# ============================================================

# En vez de obligar a exactamente 10% de anomalías,
# calculamos una contaminación razonable según la cantidad
# de equipos con ticket, limitándola a un rango sano.

proporcion_criticos = (
    cantidad_criticos / n_total
    if n_total > 0
    else 0.1
)

CONTAMINATION = float(
    np.clip(
        max(0.05, proporcion_criticos),
        0.05,
        0.25
    )
)


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


# ============================================================
# 10. PREDICCIÓN SUPERVISADA
# ============================================================

if modelo_supervisado is not None:

    probabilidades = modelo_supervisado.predict_proba(
        X
    )

    clases_modelo = list(
        modelo_supervisado.classes_
    )

    if "CRÍTICO" in clases_modelo:

        indice_critico = clases_modelo.index("CRÍTICO")

        prob_critico = probabilidades[:, indice_critico]

    else:

        prob_critico = np.zeros(len(df))

else:

    # Sin suficiente historial supervisado,
    # empezamos desde el componente de anomalía.
    prob_critico = np.zeros(len(df))


# ============================================================
# 11. DETECCIÓN DE ANOMALÍAS
# ============================================================

pred_anomalia = modelo_anomalia.predict(X)

raw_anomaly = -modelo_anomalia.decision_function(X)


# ------------------------------------------------------------
# Convertir anomaly score a una escala 0-100
# ------------------------------------------------------------

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
        (raw_anomaly - p05) /
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


# ============================================================
# 12. SCORE TÉCNICO POR REGLAS
# ============================================================

def calcular_score_tecnico(row):

    cpu = row["Uso_CPU_Porcentaje"]
    ram = row["Uso_RAM_Porcentaje"]
    disco = row["Uso_Disco_Porcentaje"]
    cpu_norm = row["CPU_Normalizado_Porcentaje"]

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

    # Disco
    if disco >= 97:
        score += 25
    elif disco >= 90:
        score += 18
    elif disco >= 80:
        score += 10
    elif disco >= 70:
        score += 5

    # CPU normalizada
    if cpu_norm >= 95:
        score += 20
    elif cpu_norm >= 85:
        score += 12
    elif cpu_norm >= 75:
        score += 5

    return min(score, 100)


df["Score_Tecnico"] = df.apply(
    calcular_score_tecnico,
    axis=1
)


# ============================================================
# 13. SCORE FINAL DE RIESGO
# ============================================================

if modelo_supervisado is not None:

    # El aprendizaje histórico tiene mayor peso,
    # pero nunca se ignoran anomalías ni síntomas técnicos.

    df["Riesgo_IA"] = (
        prob_critico * 0.55
        + anomaly_score * 0.25
        + df["Score_Tecnico"] * 0.20
    )

else:

    # Cuando aún no hay suficientes tickets:
    # el sistema depende principalmente del comportamiento
    # estadístico y del diagnóstico técnico.

    df["Riesgo_IA"] = (
        anomaly_score * 0.55
        + df["Score_Tecnico"] * 0.45
    )


df["Riesgo_IA"] = np.clip(
    df["Riesgo_IA"],
    0,
    100
)


# ============================================================
# 14. ESTADO DEL EQUIPO
# ============================================================

def determinar_estado(riesgo):

    if riesgo >= 80:
        return "CRÍTICO"

    elif riesgo >= 60:
        return "ALTO"

    elif riesgo >= 35:
        return "MEDIO"

    else:
        return "ESTABLE"


df["Estado"] = df["Riesgo_IA"].apply(
    determinar_estado
)


# ============================================================
# 15. NIVEL DE RIESGO
# ============================================================

def nivel_riesgo(riesgo):

    if riesgo >= 80:
        return "Muy alto"

    elif riesgo >= 60:
        return "Alto"

    elif riesgo >= 35:
        return "Moderado"

    else:
        return "Bajo"


df["Nivel_Riesgo"] = df["Riesgo_IA"].apply(
    nivel_riesgo
)


# ============================================================
# 16. DIAGNÓSTICO AUTOMÁTICO
# ============================================================

def generar_diagnostico(row):

    problemas = []

    cpu = row["Uso_CPU_Porcentaje"]
    ram = row["Uso_RAM_Porcentaje"]
    disco = row["Uso_Disco_Porcentaje"]
    cpu_norm = row["CPU_Normalizado_Porcentaje"]

    tendencia_cpu = row["Tendencia_CPU"]
    tendencia_ram = row["Tendencia_RAM"]
    tendencia_disco = row["Tendencia_Disco"]

    # CPU
    if cpu >= 95:

        problemas.append(
            "CPU en nivel crítico (≥95%). "
            "Existe una saturación importante del procesador."
        )

    elif cpu >= 85:

        problemas.append(
            "CPU elevada (≥85%). "
            "Puede existir carga excesiva de procesos o aplicaciones."
        )

    # RAM
    if ram >= 95:

        problemas.append(
            "Memoria RAM en nivel crítico (≥95%). "
            "Existe alta posibilidad de presión de memoria."
        )

    elif ram >= 85:

        problemas.append(
            "Memoria RAM elevada (≥85%). "
            "Puede existir consumo excesivo de aplicaciones."
        )

    # Disco
    if disco >= 97:

        problemas.append(
            "Actividad de disco extremadamente alta (≥97%). "
            "Puede existir saturación de E/S o procesos intensivos."
        )

    elif disco >= 90:

        problemas.append(
            "Actividad de disco elevada (≥90%). "
            "Conviene revisar procesos que realizan operaciones de lectura/escritura."
        )

    # CPU normalizada
    if cpu_norm >= 90:

        problemas.append(
            "El comportamiento de CPU normalizado también es elevado, "
            "lo que refuerza la señal de sobrecarga."
        )

    # Tendencias
    if tendencia_cpu >= 15:

        problemas.append(
            "La CPU está aumentando rápidamente respecto a la medición anterior."
        )

    if tendencia_ram >= 15:

        problemas.append(
            "El consumo de RAM está aumentando respecto a la medición anterior."
        )

    if tendencia_disco >= 15:

        problemas.append(
            "La actividad de disco está aumentando rápidamente."
        )

    # Sin síntomas claros
    if not problemas:

        return (
            "No se observan niveles críticos de utilización en los "
            "recursos monitoreados. El comportamiento actual es consistente "
            "con una operación estable."
        )

    return " ".join(problemas)


df["Diagnostico_IA"] = df.apply(
    generar_diagnostico,
    axis=1
)


# ============================================================
# 17. RECOMENDACIONES AUTOMÁTICAS
# ============================================================

def generar_recomendaciones(row):

    recomendaciones = []

    cpu = row["Uso_CPU_Porcentaje"]
    ram = row["Uso_RAM_Porcentaje"]
    disco = row["Uso_Disco_Porcentaje"]

    # CPU
    if cpu >= 85:

        recomendaciones.append(
            "Revisar en el Administrador de tareas los procesos "
            "que presentan mayor consumo de CPU."
        )

    # RAM
    if ram >= 85:

        recomendaciones.append(
            "Cerrar aplicaciones innecesarias y revisar procesos "
            "que estén consumiendo una cantidad elevada de memoria."
        )

    # Disco
    if disco >= 90:

        recomendaciones.append(
            "Revisar actividad del almacenamiento, espacio disponible "
            "y aplicaciones que estén generando alta E/S de disco."
        )

    # Batería
    bateria = row["Porcentaje_Bateria"]

    if bateria <= 15:

        recomendaciones.append(
            "Conectar el equipo a corriente y verificar el estado de la batería."
        )

    # CPU + RAM
    if cpu >= 85 and ram >= 85:

        recomendaciones.append(
            "Investigar aplicaciones que simultáneamente generan alta "
            "carga de CPU y memoria."
        )

    # Los tres
    if cpu >= 85 and ram >= 85 and disco >= 90:

        recomendaciones.append(
            "Realizar diagnóstico integral porque existe saturación "
            "simultánea de CPU, RAM y almacenamiento."
        )

    # Ninguna alerta
    if not recomendaciones:

        recomendaciones.append(
            "Continuar con el monitoreo. No se requiere una intervención "
            "inmediata con las métricas actuales."
        )

    return recomendaciones


df["Recomendaciones_IA"] = df.apply(
    generar_recomendaciones,
    axis=1
)


# ============================================================
# 18. SIDEBAR
# ============================================================

st.sidebar.header("🔍 Diagnóstico Individual")

pc_seleccionado = st.sidebar.selectbox(
    "Filtrar Equipo ID:",
    sorted(df["ID_PC"].astype(str).unique())
)

pc_data = df[
    df["ID_PC"].astype(str) ==
    str(pc_seleccionado)
]


# ============================================================
# 19. REPORTAR TICKET
# ============================================================

st.sidebar.divider()

st.sidebar.header("🎫 Reportar un Ticket")

with st.sidebar.form("form_ticket"):

    pc_ticket = st.selectbox(
        "Equipo con problema:",
        sorted(df["ID_PC"].astype(str).unique()),
        key="pc_ticket"
    )

    descripcion = st.text_area(
        "Describe el problema:",
        placeholder=(
            "Ej: El equipo está lento, "
            "se congela o presenta pantalla azul..."
        )
    )

    enviado = st.form_submit_button(
        "🎫 Reportar ticket"
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
                    f"Ticket registrado correctamente para {pc_ticket}"
                )

                st.cache_data.clear()
                st.rerun()

            except Exception as e:

                st.sidebar.error(
                    f"No se pudo registrar el ticket: {e}"
                )

        else:

            st.sidebar.warning(
                "Escribe una descripción antes de enviar."
            )


# ============================================================
# 20. OBTENER EQUIPO SELECCIONADO
# ============================================================

equipo = pc_data.iloc[0]


riesgo_pc = float(
    equipo["Riesgo_IA"]
)


estado_pc = equipo["Estado"]


# ============================================================
# 21. KPIs
# ============================================================

col_a, col_b, col_c, col_d, col_e = st.columns(5)

col_a.metric(
    "Total Equipos",
    len(df)
)

col_b.metric(
    "Equipos en Riesgo",
    int(
        (df["Estado"].isin(["CRÍTICO", "ALTO"])).sum()
    )
)

col_c.metric(
    "Riesgo Crítico",
    int(
        (df["Estado"] == "CRÍTICO").sum()
    )
)

col_d.metric(
    "Promedio CPU",
    f"{df['Uso_CPU_Porcentaje'].mean():.1f}%"
)

col_e.metric(
    "Promedio RAM",
    f"{df['Uso_RAM_Porcentaje'].mean():.1f}%"
)


# ============================================================
# 22. DIAGNÓSTICO INDIVIDUAL
# ============================================================

st.divider()

st.subheader(
    f"🧠 Diagnóstico Inteligente — {pc_seleccionado}"
)


# ------------------------------------------------------------
# Estado
# ------------------------------------------------------------

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


d1, d2, d3, d4 = st.columns(4)

d1.metric(
    "Estado IA",
    f"{icono} {estado_pc}"
)

d2.metric(
    "Riesgo",
    f"{riesgo_pc:.1f}/100"
)

d3.metric(
    "Nivel",
    equipo["Nivel_Riesgo"]
)

d4.metric(
    "Anomalía",
    f"{anomaly_score[df.index.get_loc(equipo.name)]:.1f}/100"
)


# ============================================================
# 23. DETALLE DE RECURSOS
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
    "CPU Normalizada",
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
# 24. EXPLICACIÓN DE LA IA
# ============================================================

st.markdown("### 🔎 ¿Por qué la IA dio este resultado?")

st.info(
    equipo["Diagnostico_IA"]
)


# ============================================================
# 25. RECOMENDACIONES
# ============================================================

st.markdown("### 🛠️ Recomendaciones de solución")

for recomendacion in equipo["Recomendaciones_IA"]:

    st.write(
        f"• {recomendacion}"
    )


# ============================================================
# 26. PUNTAJES INTERNOS
# ============================================================

st.markdown("### 📊 Componentes del riesgo")

score1, score2, score3 = st.columns(3)

score1.metric(
    "Modelo supervisado",
    f"{float(prob_critico[df.index.get_loc(equipo.name)]):.1f}%"
)

score2.metric(
    "Anomalía estadística",
    f"{anomaly_score[df.index.get_loc(equipo.name)]:.1f}/100"
)

score3.metric(
    "Riesgo técnico",
    f"{equipo['Score_Tecnico']:.1f}/100"
)


# ============================================================
# 27. DASHBOARD PRINCIPAL
# ============================================================

c1, c2 = st.columns([1, 2])


with c1:

    st.subheader(
        f"Detalles técnicos: {pc_seleccionado}"
    )

    detalle = pd.DataFrame({
        "Indicador": [
            "Equipo",
            "Usuario",
            "Modelo",
            "Serial",
            "Estado IA",
            "Nivel de riesgo",
            "Riesgo IA",
            "Uso CPU",
            "Uso RAM",
            "CPU normalizada",
            "Uso disco",
            "Batería",
            "Componentes saturados",
            "Tiene ticket"
        ],

        "Valor": [
            equipo["ID_PC"],
            equipo.get("Usuario", "No disponible"),
            equipo.get("Modelo", "No disponible"),
            equipo.get("Serial", "No disponible"),
            estado_pc,
            equipo["Nivel_Riesgo"],
            f"{riesgo_pc:.1f}/100",
            f"{equipo['Uso_CPU_Porcentaje']:.1f}%",
            f"{equipo['Uso_RAM_Porcentaje']:.1f}%",
            f"{equipo['CPU_Normalizado_Porcentaje']:.1f}%",
            f"{equipo['Uso_Disco_Porcentaje']:.1f}%",
            f"{equipo['Porcentaje_Bateria']:.1f}%",
            int(equipo["Componentes_Saturados"]),
            "Sí" if equipo["Tiene_Ticket"] else "No"
        ]
    })

    st.dataframe(
        detalle,
        use_container_width=True,
        hide_index=True
    )


with c2:

    st.subheader("🗺️ Mapa de Riesgo IA")

    fig = px.scatter(
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
            "CRÍTICO": "#FF4B4B",
            "ALTO": "#FFA726",
            "MEDIO": "#FFD54F",
            "ESTABLE": "#0068C9"
        }
    )

    fig.update_layout(
        xaxis_title="Uso CPU (%)",
        yaxis_title="Uso RAM (%)",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# 28. TABLA DE EQUIPOS PRIORIZADOS
# ============================================================

st.markdown("### 🚨 Priorización de equipos")

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


# ============================================================
# 29. EVALUACIÓN DEL MODELO
# ============================================================

st.divider()

st.subheader(
    "📈 Evaluación y Validación del Modelo de IA"
)


if modelo_supervisado is None:

    st.warning(
        f"""
El sistema todavía no tiene suficiente historial etiquetado para entrenar
un modelo supervisado confiable.

Registros disponibles: {n_total}
Críticos con ticket: {cantidad_criticos}
Estables: {cantidad_estables}

Mientras se recopilan más tickets, el sistema utiliza un modelo híbrido
basado en detección de anomalías + reglas técnicas.
"""
    )

else:

    st.success(
        f"""
El modelo supervisado está activo.

Registros totales: {n_total}
Críticos: {cantidad_criticos}
Estables: {cantidad_estables}

Entrenamiento: {len(train_df)} registros
Prueba: {len(test_df)} registros
"""
    )


# ============================================================
# 30. EVALUACIÓN REAL SOBRE TEST
# ============================================================

if modelo_supervisado is not None:

    pred_test = modelo_supervisado.predict(
        X_test
    )

    y_test_real = test_df["Clase_Real"].values

    y_test_pred = pred_test

    acc = accuracy_score(
        y_test_real,
        y_test_pred
    )

    prec = precision_score(
        y_test_real,
        y_test_pred,
        pos_label="CRÍTICO",
        zero_division=0
    )

    rec = recall_score(
        y_test_real,
        y_test_pred,
        pos_label="CRÍTICO",
        zero_division=0
    )

    f1 = f1_score(
        y_test_real,
        y_test_pred,
        pos_label="CRÍTICO",
        zero_division=0
    )


    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Accuracy",
        f"{acc:.2%}"
    )

    m2.metric(
        "Precision CRÍTICO",
        f"{prec:.2%}"
    )

    m3.metric(
        "Recall CRÍTICO",
        f"{rec:.2%}"
    )

    m4.metric(
        "F1-Score CRÍTICO",
        f"{f1:.2%}"
    )


    # --------------------------------------------------------
    # Matriz + reporte
    # --------------------------------------------------------

    col_cm, col_rep = st.columns([1, 1])

    labels = [
        "CRÍTICO",
        "ESTABLE"
    ]


    with col_cm:

        st.markdown(
            "**Matriz de Confusión**"
        )

        cm = confusion_matrix(
            y_test_real,
            y_test_pred,
            labels=labels
        )

        fig_cm, ax = plt.subplots(
            figsize=(5, 4)
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
            "Realidad (Tickets)"
        )

        st.pyplot(
            fig_cm
        )


    with col_rep:

        st.markdown(
            "**Reporte de Clasificación**"
        )

        reporte = classification_report(
            y_test_real,
            y_test_pred,
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


    # ========================================================
    # 31. ROC / AUC
    # ========================================================

    st.markdown(
        "### 📊 Curva ROC + AUC"
    )

    prob_test = modelo_supervisado.predict_proba(
        X_test
    )

    clases_test = list(
        modelo_supervisado.classes_
    )

    if "CRÍTICO" in clases_test:

        idx_critico = clases_test.index(
            "CRÍTICO"
        )

        score_test = prob_test[
            :, idx_critico
        ]

        y_test_bin = (
            y_test_real == "CRÍTICO"
        ).astype(int)

        if len(np.unique(y_test_bin)) == 2:

            fpr, tpr, _ = roc_curve(
                y_test_bin,
                score_test
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
                    name=f"Modelo (AUC = {auc_score:.3f})",
                    line=dict(
                        color="#0068C9",
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
                        color="gray",
                        dash="dash"
                    )
                )
            )

            fig_roc.update_layout(
                xaxis_title="Tasa de Falsos Positivos",
                yaxis_title="Tasa de Verdaderos Positivos",
                height=400
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
                "El conjunto de prueba no contiene las dos clases necesarias para calcular ROC/AUC."
            )


# ============================================================
# 32. DISTRIBUCIÓN DE RIESGO
# ============================================================

st.markdown(
    "### 📊 Distribución del Puntaje de Riesgo"
)

fig_hist = px.histogram(
    df,
    x="Riesgo_IA",
    color="Estado",
    nbins=20,
    color_discrete_map={
        "CRÍTICO": "#FF4B4B",
        "ALTO": "#FFA726",
        "MEDIO": "#FFD54F",
        "ESTABLE": "#0068C9"
    }
)

fig_hist.update_layout(
    xaxis_title="Riesgo IA (0-100)",
    yaxis_title="Cantidad de equipos"
)

st.plotly_chart(
    fig_hist,
    use_container_width=True
)


# ============================================================
# 33. SCORE DE ANOMALÍA
# ============================================================

st.markdown(
    "### 🔬 Distribución del Puntaje de Anomalía"
)

fig_anomalia = px.histogram(
    df,
    x="Score_Anomalia" if "Score_Anomalia" in df.columns else anomaly_score,
    color="Clase_Real",
    barmode="overlay",
    nbins=20,
    color_discrete_map={
        "CRÍTICO": "#FF4B4B",
        "ESTABLE": "#0068C9"
    }
)

st.plotly_chart(
    fig_anomalia,
    use_container_width=True
)


# ============================================================
# 34. GUARDAR SCORE DE ANOMALÍA EN DF
# ============================================================

# Se hace después porque no afecta al modelo.

df["Score_Anomalia"] = anomaly_score


# ============================================================
# 35. SENSIBILIDAD DE CONTAMINATION
# ============================================================

st.markdown(
    "### ⚙️ Sensibilidad del parámetro `contamination`"
)

st.write(
    "Esta prueba permite observar cómo cambia la detección de anomalías "
    "cuando se modifica la proporción esperada de valores anómalos."
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


for c in valores_prueba:

    if c <= 0 or c >= 0.5:
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

    # Solo se compara contra tickets cuando existen
    # ambas clases y test es válido.

    if len(np.unique(y_test_real)) >= 2:

        pred_temp_label = np.where(
            pred_temp == -1,
            "CRÍTICO",
            "ESTABLE"
        )

        acc_temp = accuracy_score(
            y_test_real,
            pred_temp_label
        )

    else:

        acc_temp = np.nan


    resultados_contamination.append(
        {
            "contamination": c,
            "accuracy": acc_temp,
            "usado_actualmente": (
                "Sí"
                if abs(c - CONTAMINATION) < 0.0001
                else ""
            )
        }
    )


st.dataframe(
    pd.DataFrame(
        resultados_contamination
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 36. EXPLICACIÓN GENERAL DEL SISTEMA
# ============================================================

st.divider()

st.subheader(
    "🤖 Cómo toma decisiones la IA"
)

st.write(
    """
El sistema utiliza tres fuentes de información para determinar el riesgo:

1. **Modelo supervisado:** aprende de los equipos que históricamente
   fueron reportados mediante tickets.

2. **Detección de anomalías:** identifica equipos cuyo comportamiento
   se aleja significativamente del comportamiento normal de la flota.

3. **Reglas técnicas:** analiza niveles críticos de CPU, RAM, disco,
   CPU normalizada y tendencias recientes.

El resultado de estos componentes se combina en un puntaje de
riesgo de 0 a 100. Después se clasifica el equipo como ESTABLE,
MEDIO, ALTO o CRÍTICO.

Esto permite que la decisión no dependa únicamente de una variable
individual.
"""
)


# ============================================================
# 37. INVENTARIO COMPLETO
# ============================================================

st.divider()

st.subheader(
    "🖥️ Inventario Técnico Completo"
)

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)
