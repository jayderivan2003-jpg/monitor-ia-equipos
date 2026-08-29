import time
import numpy as np
import pandas as pd
import streamlit as st

from supabase import create_client
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None
from sklearn.neighbors import LocalOutlierFactor  # NUEVO: para detección por densidad
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
    make_scorer,
)

import plotly.express as px
import plotly.graph_objects as go

# Intentar importar SMOTE para balanceo de clases (más entrenamiento)
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    # Si no está instalado, se muestra un aviso pero el código sigue funcionando


# ============================================================
# CONFIGURACION
# ============================================================

st.set_page_config(
    page_title="AI-FleetMonitor Pro - Detección Avanzada",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ESTILO (mejorado con más elementos visuales)
# ============================================================

st.markdown(
    """
    <style>
        .stApp { background: #f4f6f9; color: #172033; }
        [data-testid="stHeader"] { background: #ffffff; border-bottom: 1px solid #e2e8f0; }
        section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
        .block-container { max-width: 1500px; padding-top: 1.5rem; padding-bottom: 3rem; }

        .app-header {
            background: linear-gradient(110deg, #0f172a 0%, #173a72 100%);
            border-radius: 16px; padding: 25px 30px; margin-bottom: 22px;
            box-shadow: 0 8px 24px rgba(15,23,42,.10);
        }
        .app-header-title { color: #fff; font-size: 30px; font-weight: 700; margin: 0; }
        .app-header-subtitle { color: #dbe4f0; font-size: 14px; margin-top: 7px; }

        .section-title { color: #172033; font-size: 21px; font-weight: 700; margin: 16px 0 10px; }
        .section-note { color: #64748b; font-size: 13px; margin-bottom: 14px; }
        .panel { background: #fff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px; box-shadow: 0 4px 14px rgba(15,23,42,.045); }
        .status-card { border-radius: 13px; padding: 16px 18px; border: 1px solid; margin-bottom: 15px; }
        .status-critical { background:#fff1f2; border-color:#fecdd3; color:#9f1239; }
        .status-high { background:#fff7ed; border-color:#fed7aa; color:#9a3412; }
        .status-medium { background:#fefce8; border-color:#fde68a; color:#854d0e; }
        .status-stable { background:#f0fdf4; border-color:#bbf7d0; color:#166534; }
        .small-muted { color:#64748b; font-size:12px; }

        [data-testid="stMetric"] { background:#fff; border:1px solid #dbe3ee; border-radius:12px; padding:14px; box-shadow:0 3px 10px rgba(15,23,42,.04); }
        [data-testid="stMetricLabel"] { color:#475569 !important; }
        [data-testid="stMetricValue"] { color:#172033 !important; }
        [data-testid="stMetricDelta"] { color:#475569 !important; }

        .stButton > button { background:#ffffff !important; color:#172033 !important; border:1px solid #cbd5e1 !important; border-radius:9px !important; font-weight:600 !important; }
        .stButton > button:hover { border-color:#2563eb !important; color:#2563eb !important; }
        button[kind="primary"] { background:#1e3a8a !important; color:#ffffff !important; border-color:#1e3a8a !important; }

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] span { color:#172033 !important; }
        section[data-testid="stSidebar"] textarea,
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] [role="combobox"] { color:#172033 !important; }

        [data-testid="stTabs"] button { color:#334155 !important; font-weight:600 !important; }
        [data-testid="stTabs"] button[aria-selected="true"] { color:#1e3a8a !important; }
        .stCaption, .stMarkdown, p { color:#334155; }

        [data-testid="stDataFrame"] { border:1px solid #e2e8f0; border-radius:10px; overflow:hidden; }

        /* NUEVO: estilos para alertas */
        .alert-box {
            padding: 10px 15px;
            border-radius: 8px;
            margin: 5px 0;
            border-left: 4px solid;
        }
        .alert-critical { background: #fef2f2; border-color: #dc2626; }
        .alert-warning { background: #fffbeb; border-color: #f59e0b; }
        .alert-info { background: #eff6ff; border-color: #3b82f6; }

        .anomaly-highlight {
            background: #fef2f2;
            border: 2px solid #dc2626;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <div class="app-header-title">🔍 AI-FleetMonitor Pro</div>
        <div class="app-header-subtitle">
            Monitoreo de hardware • Detección avanzada de anomalías • Clasificación de riesgo
            • Entrenamiento y evaluación con IA de máxima sensibilidad
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SUPABASE
# ============================================================

try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
except Exception:
    st.error("No se encontraron SUPABASE_URL y SUPABASE_ANON_KEY en st.secrets.")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


# ============================================================
# CARGA DE DATOS
# ============================================================

@st.cache_data(ttl=60)
def cargar_historial(_cache_bust):
    """Carga todas las mediciones históricas. Es la fuente para entrenar la IA."""
    resultado = (
        supabase.table("mediciones_equipos")
        .select("*")
        .order("fecha_hora", desc=False)
        .execute()
    )
    df = pd.DataFrame(resultado.data)
    return df


@st.cache_data(ttl=60)
def cargar_equipos(_cache_bust):
    """Carga el inventario actual. Sirve para mantener los datos de equipos/tickets."""
    resultado = supabase.table("equipos").select("*").execute()
    df = pd.DataFrame(resultado.data)
    return df


try:
    historial = cargar_historial(int(time.time() // 60))
except Exception as exc:
    st.error(f"No fue posible leer mediciones_equipos: {exc}")
    st.stop()

try:
    equipos_bd = cargar_equipos(int(time.time() // 60))
except Exception:
    equipos_bd = pd.DataFrame()

if st.button("🔄 Actualizar datos"):
    st.cache_data.clear()
    st.rerun()

if historial.empty:
    st.warning("La tabla mediciones_equipos esta vacia. Ejecuta generar_20_equipos.py antes de entrenar la IA.")
    st.stop()


# ============================================================
# NORMALIZACION DE NOMBRES
# ============================================================

historial = historial.rename(
    columns={
        "id": "ID_Medicion",
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
        "temperatura_cpu": "Temperatura_CPU",
        "procesos_activos": "Procesos_Activos",
        "estado_tecnico": "Estado_Tecnico",
        "severidad": "Severidad",
    }
)

if "ID_PC" not in historial.columns:
    historial["ID_PC"] = "SIN_ID"

historial["ID_PC"] = historial["ID_PC"].astype(str)

if "Fecha_Hora" in historial.columns:
    historial["Fecha_Hora"] = pd.to_datetime(historial["Fecha_Hora"], errors="coerce", utc=True)

numeric_cols = [
    "Uso_CPU_Porcentaje",
    "Uso_RAM_Porcentaje",
    "CPU_Normalizado_Porcentaje",
    "Porcentaje_Bateria",
    "Uso_Disco_Porcentaje",
    "Temperatura_CPU",
    "Procesos_Activos",
]

for col in numeric_cols:
    if col not in historial.columns:
        historial[col] = np.nan
    historial[col] = pd.to_numeric(historial[col], errors="coerce")

for col in [
    "Uso_CPU_Porcentaje",
    "Uso_RAM_Porcentaje",
    "CPU_Normalizado_Porcentaje",
    "Porcentaje_Bateria",
    "Uso_Disco_Porcentaje",
]:
    historial[col] = historial[col].clip(0, 100)

if "Ticket_Usuario" not in historial.columns:
    historial["Ticket_Usuario"] = None

if "Estado_Tecnico" not in historial.columns:
    historial["Estado_Tecnico"] = None

if "Severidad" not in historial.columns:
    historial["Severidad"] = None

historial["Tiene_Ticket"] = (
    historial["Ticket_Usuario"].fillna("").astype(str).str.strip().ne("")
)


# ============================================================
# ETIQUETAS: EL MODELO APRENDE DEL ESTADO TECNICO (MEJORADO)
# ============================================================

def estado_tecnico_reglas(row):
    cpu = float(row["Uso_CPU_Porcentaje"])
    ram = float(row["Uso_RAM_Porcentaje"])
    disco = float(row["Uso_Disco_Porcentaje"])
    temp = float(row.get("Temperatura_CPU", 0))  # Ahora incluye temperatura

    # Umbrales más sensibles
    if cpu >= 90 or ram >= 90 or disco >= 95 or temp >= 85:
        return "CRITICO"
    if cpu >= 85 or ram >= 85 or disco >= 90 or temp >= 75:
        return "ALERTA"
    if cpu >= 60 or ram >= 60 or disco >= 60 or temp >= 65:
        return "REGULAR"
    return "ESTABLE"


historial["Estado_Tecnico"] = historial.apply(
    lambda row: row["Estado_Tecnico"]
    if isinstance(row["Estado_Tecnico"], str) and row["Estado_Tecnico"].strip()
    else estado_tecnico_reglas(row),
    axis=1,
)

historial["Estado_Tecnico"] = (
    historial["Estado_Tecnico"]
    .astype(str)
    .str.upper()
    .replace({"CRÍTICO": "CRITICO"})
)

# Objetivo binario
historial["Clase_Real"] = np.where(
    historial["Estado_Tecnico"].eq("CRITICO"),
    "CRÍTICO",
    "ESTABLE",
)


# ============================================================
# VARIABLES DERIVADAS E HISTORICAS (MEJORADAS - MÁS SENSIBLES)
# ============================================================

historial = historial.sort_values(["ID_PC", "Fecha_Hora"], na_position="first").copy()

for col in numeric_cols:
    mediana = historial[col].median()
    if pd.isna(mediana):
        mediana = 0
    historial[col] = historial[col].fillna(mediana)

historial["Presion_Recursos"] = historial[
    ["Uso_CPU_Porcentaje", "Uso_RAM_Porcentaje", "Uso_Disco_Porcentaje"]
].max(axis=1)

historial["Promedio_Recursos"] = historial[
    ["Uso_CPU_Porcentaje", "Uso_RAM_Porcentaje", "Uso_Disco_Porcentaje"]
].mean(axis=1)

historial["CPU_RAM_Conjunta"] = (
    historial["Uso_CPU_Porcentaje"] * historial["Uso_RAM_Porcentaje"] / 100
)

historial["Diferencia_CPU"] = (
    historial["CPU_Normalizado_Porcentaje"] - historial["Uso_CPU_Porcentaje"]
).abs()

# === UMBRALES MÁS SENSIBLES (80% en lugar de 85%) ===
historial["CPU_Mala"] = (historial["Uso_CPU_Porcentaje"] >= 80).astype(int)
historial["RAM_Mala"] = (historial["Uso_RAM_Porcentaje"] >= 80).astype(int)
historial["Disco_Malo"] = (historial["Uso_Disco_Porcentaje"] >= 85).astype(int)
historial["Recursos_Malos"] = historial[["CPU_Mala", "RAM_Mala", "Disco_Malo"]].sum(axis=1)

historial["Estado_CPU_Tecnico"] = pd.cut(
    historial["Uso_CPU_Porcentaje"],
    bins=[-np.inf, 50, 70, 80, np.inf],
    labels=["EXCELENTE", "BUENO", "REGULAR", "MALO"],
    right=False,
)

historial["Estado_RAM_Tecnico"] = pd.cut(
    historial["Uso_RAM_Porcentaje"],
    bins=[-np.inf, 50, 70, 80, np.inf],
    labels=["EXCELENTE", "BUENO", "REGULAR", "MALO"],
    right=False,
)

# Persistencia y tendencia
historial["CPU_Anterior"] = historial.groupby("ID_PC")["Uso_CPU_Porcentaje"].shift(1)
historial["RAM_Anterior"] = historial.groupby("ID_PC")["Uso_RAM_Porcentaje"].shift(1)
historial["Disco_Anterior"] = historial.groupby("ID_PC")["Uso_Disco_Porcentaje"].shift(1)

historial["Tendencia_CPU"] = (historial["Uso_CPU_Porcentaje"] - historial["CPU_Anterior"]).fillna(0)
historial["Tendencia_RAM"] = (historial["Uso_RAM_Porcentaje"] - historial["RAM_Anterior"]).fillna(0)
historial["Tendencia_Disco"] = (historial["Uso_Disco_Porcentaje"] - historial["Disco_Anterior"]).fillna(0)

# Ventana movil de 5 mediciones
grouped = historial.groupby("ID_PC", group_keys=False)
historial["CPU_Media_5"] = grouped["Uso_CPU_Porcentaje"].transform(lambda s: s.rolling(5, min_periods=1).mean())
historial["RAM_Media_5"] = grouped["Uso_RAM_Porcentaje"].transform(lambda s: s.rolling(5, min_periods=1).mean())
historial["Disco_Media_5"] = grouped["Uso_Disco_Porcentaje"].transform(lambda s: s.rolling(5, min_periods=1).mean())

historial["CPU_Alta_5"] = grouped["Uso_CPU_Porcentaje"].transform(lambda s: s.rolling(5, min_periods=1).apply(lambda x: np.mean(x >= 80)))
historial["RAM_Alta_5"] = grouped["Uso_RAM_Porcentaje"].transform(lambda s: s.rolling(5, min_periods=1).apply(lambda x: np.mean(x >= 80)))
historial["Disco_Alto_5"] = grouped["Uso_Disco_Porcentaje"].transform(lambda s: s.rolling(5, min_periods=1).apply(lambda x: np.mean(x >= 85)))

historial["Componentes_Saturados"] = (
    (historial["Uso_CPU_Porcentaje"] >= 80).astype(int)
    + (historial["Uso_RAM_Porcentaje"] >= 80).astype(int)
    + (historial["Uso_Disco_Porcentaje"] >= 85).astype(int)
)


# ============================================================
# FEATURES (se mantienen igual)
# ============================================================

features = [
    "Uso_CPU_Porcentaje",
    "Uso_RAM_Porcentaje",
    "CPU_Normalizado_Porcentaje",
    "Porcentaje_Bateria",
    "Uso_Disco_Porcentaje",
    "Temperatura_CPU",
    "Procesos_Activos",
    "Presion_Recursos",
    "Promedio_Recursos",
    "CPU_RAM_Conjunta",
    "Diferencia_CPU",
    "Componentes_Saturados",
    "CPU_Mala",
    "RAM_Mala",
    "Disco_Malo",
    "Recursos_Malos",
    "Tendencia_CPU",
    "Tendencia_RAM",
    "Tendencia_Disco",
    "CPU_Media_5",
    "RAM_Media_5",
    "Disco_Media_5",
    "CPU_Alta_5",
    "RAM_Alta_5",
    "Disco_Alto_5",
]

X_all = historial[features].replace([np.inf, -np.inf], np.nan)
X_all = X_all.fillna(X_all.median(numeric_only=True)).fillna(0)

n_registros = len(historial)
n_equipos = historial["ID_PC"].nunique()
cantidad_criticos = int(historial["Clase_Real"].eq("CRÍTICO").sum())
cantidad_estables = int(historial["Clase_Real"].eq("ESTABLE").sum())

puede_usar_supervisado = (
    n_registros >= 10
    and cantidad_criticos >= 2
    and cantidad_estables >= 2
    and n_equipos >= 4
)


# ============================================================
# TRAIN/TEST POR EQUIPO PARA EVITAR DATA LEAKAGE
# ============================================================

train_df = historial.copy()
test_df = historial.copy()

if puede_usar_supervisado:
    equipos_labels = (
        historial.groupby("ID_PC")["Clase_Real"]
        .agg(lambda s: "CRÍTICO" if np.mean(s.eq("CRÍTICO")) >= 0.5 else "ESTABLE")
        .reset_index()
    )

    try:
        train_pcs, test_pcs = train_test_split(
            equipos_labels["ID_PC"],
            test_size=0.30,
            random_state=42,
            stratify=equipos_labels["Clase_Real"],
        )
    except ValueError:
        train_pcs, test_pcs = train_test_split(
            equipos_labels["ID_PC"],
            test_size=0.30,
            random_state=42,
        )

    train_pcs = set(train_pcs)
    test_pcs = set(test_pcs)

    train_df = historial[historial["ID_PC"].isin(train_pcs)].copy()
    test_df = historial[historial["ID_PC"].isin(test_pcs)].copy()

    # Seguridad: si por el tamaño de la muestra la prueba pierde una clase
    if test_df["Clase_Real"].nunique() < 2 or train_df["Clase_Real"].nunique() < 2:
        for seed in range(1, 101):
            try:
                tr_pcs, te_pcs = train_test_split(
                    equipos_labels["ID_PC"],
                    test_size=0.30,
                    random_state=seed,
                    stratify=equipos_labels["Clase_Real"],
                )
                tr = historial[historial["ID_PC"].isin(set(tr_pcs))]
                te = historial[historial["ID_PC"].isin(set(te_pcs))]
                if tr["Clase_Real"].nunique() == 2 and te["Clase_Real"].nunique() == 2:
                    train_df, test_df = tr.copy(), te.copy()
                    break
            except Exception:
                continue

X_train = train_df[features].replace([np.inf, -np.inf], np.nan)
X_train = X_train.fillna(X_all.median(numeric_only=True)).fillna(0)
X_test = test_df[features].replace([np.inf, -np.inf], np.nan)
X_test = X_test.fillna(X_all.median(numeric_only=True)).fillna(0)
y_train = train_df["Clase_Real"].values
y_test_real = test_df["Clase_Real"].values


# ============================================================
# ENTRENAMIENTO CON MÁXIMA SENSIBILIDAD (MEJORADO)
# ============================================================

modelo_supervisado = None
tiempo_entrenamiento_supervisado = 0.0
N_ARBOLES_RF = 1500  # Aumentado de 1000 a 1500 para más entrenamiento


# ============================================================
# 1. ISOLATION FOREST CON ALTA SENSIBILIDAD
# ============================================================

proporcion_criticos = cantidad_criticos / max(n_registros, 1)
# Aumentar contamination para detectar más anomalías (máximo 35%)
CONTAMINATION = float(np.clip(max(0.10, proporcion_criticos * 1.5), 0.05, 0.35))

modelo_anomalia_if = IsolationForest(
    n_estimators=1500,  # Aumentado
    contamination=CONTAMINATION,
    max_samples="auto",
    bootstrap=True,  # Mejora la robustez
    random_state=42,
    n_jobs=-1,
)
modelo_anomalia_if.fit(X_train)


# ============================================================
# 2. LOCAL OUTLIER FACTOR (NUEVO - detección por densidad)
# ============================================================

n_neighbors = min(20, max(5, len(X_train) // 10))
modelo_lof = LocalOutlierFactor(
    n_neighbors=n_neighbors,
    contamination=CONTAMINATION,
    novelty=True
)
modelo_lof.fit(X_train)


# ============================================================
# 3. RANDOM FOREST CON PESOS EXTREMOS Y SMOTE (MEJORADO)
# ============================================================

if puede_usar_supervisado and train_df["Clase_Real"].nunique() == 2:
    X_train_rf = X_train.copy()
    y_train_rf = y_train.copy()

    # Aplicar SMOTE si hay pocos críticos (balanceo de clases)
    if cantidad_criticos < 15 and SMOTE_AVAILABLE:
        try:
            k_neighbors = min(3, cantidad_criticos - 1)
            if k_neighbors >= 1:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_train_rf, y_train_rf = smote.fit_resample(X_train, y_train)
                st.info(f"✅ SMOTE aplicado: {len(X_train_rf)} muestras balanceadas")
        except Exception as e:
            st.warning(f"⚠️ No se pudo aplicar SMOTE: {e}")

    modelo_supervisado = RandomForestClassifier(
        n_estimators=N_ARBOLES_RF,
        max_depth=15,  # Aumentado de 12 a 15
        min_samples_split=2,  # Reducido de 4 a 2 (más sensible)
        min_samples_leaf=1,   # Reducido de 2 a 1 (más sensible)
        class_weight={'CRÍTICO': 5, 'ESTABLE': 1},  # Peso extremo a críticos
        random_state=42,
        n_jobs=-1,
        min_impurity_decrease=0.0001,  # Acepta divisiones más pequeñas
    )

    inicio_entrenamiento = time.perf_counter()
    modelo_supervisado.fit(X_train_rf, y_train_rf)
    tiempo_entrenamiento_supervisado = time.perf_counter() - inicio_entrenamiento


# ============================================================
# CALCULAR SCORES COMBINADOS (ENSAMBLE)
# ============================================================

# Isolation Forest
raw_anomaly_if = -modelo_anomalia_if.decision_function(X_all)
p05_if = np.percentile(raw_anomaly_if, 5)
p95_if = np.percentile(raw_anomaly_if, 95)
if p95_if > p05_if:
    anomaly_score_if = ((raw_anomaly_if - p05_if) / (p95_if - p05_if)) * 100
else:
    anomaly_score_if = np.full(n_registros, 50.0)
anomaly_score_if = np.clip(anomaly_score_if, 0, 100)

# Local Outlier Factor
raw_anomaly_lof = -modelo_lof.decision_function(X_all)
p05_lof = np.percentile(raw_anomaly_lof, 5)
p95_lof = np.percentile(raw_anomaly_lof, 95)
if p95_lof > p05_lof:
    anomaly_score_lof = ((raw_anomaly_lof - p05_lof) / (p95_lof - p05_lof)) * 100
else:
    anomaly_score_lof = np.full(n_registros, 50.0)
anomaly_score_lof = np.clip(anomaly_score_lof, 0, 100)

# Combinar scores (ensamble: 60% IF + 40% LOF)
historial["Score_Anomalia"] = (0.6 * anomaly_score_if + 0.4 * anomaly_score_lof)


# ============================================================
# PROBABILIDAD SUPERVISADA
# ============================================================

if modelo_supervisado is not None and "CRÍTICO" in list(modelo_supervisado.classes_):
    probs = modelo_supervisado.predict_proba(X_all)
    idx_critical = list(modelo_supervisado.classes_).index("CRÍTICO")
    prob_critico = probs[:, idx_critical]
else:
    prob_critico = np.zeros(n_registros)


# ============================================================
# SCORE TECNICO MEJORADO (MÁS SENSIBLE)
# ============================================================

def calcular_score_tecnico_avanzado(row):
    cpu = row["Uso_CPU_Porcentaje"]
    ram = row["Uso_RAM_Porcentaje"]
    disco = row["Uso_Disco_Porcentaje"]
    cpu_norm = row["CPU_Normalizado_Porcentaje"]
    persistencia = max(row["CPU_Alta_5"], row["RAM_Alta_5"], row["Disco_Alto_5"])
    temp = row.get("Temperatura_CPU", 0)

    score = 0

    # CPU (más sensible)
    if cpu >= 80:
        score += 30
    elif cpu >= 70:
        score += 25
    elif cpu >= 60:
        score += 18
    elif cpu >= 50:
        score += 10
    elif cpu >= 40:
        score += 5

    # RAM (más sensible)
    if ram >= 80:
        score += 30
    elif ram >= 70:
        score += 22
    elif ram >= 60:
        score += 15
    elif ram >= 50:
        score += 8

    # Disco (más sensible)
    if disco >= 85:
        score += 25
    elif disco >= 75:
        score += 18
    elif disco >= 65:
        score += 12
    elif disco >= 55:
        score += 6

    # Temperatura (NUEVO)
    if temp >= 80:
        score += 15
    elif temp >= 70:
        score += 10
    elif temp >= 60:
        score += 5

    # CPU Normalizado
    if cpu_norm >= 90:
        score += 15
    elif cpu_norm >= 80:
        score += 10
    elif cpu_norm >= 70:
        score += 5

    # Persistencia
    score += min(10, int(round(persistencia * 10)))

    # BONIFICACIÓN POR MÚLTIPLES PROBLEMAS (sinergia)
    problemas = 0
    if cpu >= 70: problemas += 1
    if ram >= 70: problemas += 1
    if disco >= 75: problemas += 1
    if temp >= 70: problemas += 1

    if problemas >= 3:
        score += 15
    elif problemas >= 2:
        score += 8

    return min(score, 100)


historial["Score_Tecnico"] = historial.apply(calcular_score_tecnico_avanzado, axis=1)


# ============================================================
# RIESGO IA FINAL CON PESOS AJUSTADOS
# ============================================================

# Factor de confianza: si hay pocos datos, menos peso a supervisado
confianza_supervisado = min(1.0, cantidad_criticos / 30)

if modelo_supervisado is not None:
    historial["Riesgo_IA"] = (
        prob_critico * (0.35 * confianza_supervisado) +
        historial["Score_Anomalia"] * (0.35 + 0.15 * (1 - confianza_supervisado)) +
        historial["Score_Tecnico"] * (0.30 + 0.10 * (1 - confianza_supervisado))
    )
else:
    historial["Riesgo_IA"] = (
        historial["Score_Anomalia"] * 0.50 +
        historial["Score_Tecnico"] * 0.50
    )

historial["Riesgo_IA"] = np.clip(historial["Riesgo_IA"], 0, 100)


# ============================================================
# UMBRALES MÁS SENSIBLES
# ============================================================

def determinar_estado_sensible(riesgo):
    if riesgo >= 70:  # Antes era 80
        return "CRÍTICO"
    if riesgo >= 50:  # Antes era 60
        return "ALTO"
    if riesgo >= 30:  # Antes era 35
        return "MEDIO"
    return "ESTABLE"


def determinar_nivel_sensible(riesgo):
    if riesgo >= 70:
        return "Muy alto"
    if riesgo >= 50:
        return "Alto"
    if riesgo >= 30:
        return "Moderado"
    return "Bajo"


historial["Estado"] = historial["Riesgo_IA"].apply(determinar_estado_sensible)
historial["Nivel_Riesgo"] = historial["Riesgo_IA"].apply(determinar_nivel_sensible)


# ============================================================
# ALERTAS TEMPRANAS (NUEVO - DETECCIÓN PREVENTIVA)
# ============================================================

def generar_alertas_tempranas(row):
    """Detecta problemas ANTES de que sean críticos"""
    alertas = []

    cpu = row["Uso_CPU_Porcentaje"]
    ram = row["Uso_RAM_Porcentaje"]
    disco = row["Uso_Disco_Porcentaje"]
    temp = row.get("Temperatura_CPU", 0)

    # Tendencias al alza (problemas futuros)
    if row["Tendencia_CPU"] > 10:
        alertas.append(f"⚠️ CPU subiendo rápidamente (+{row['Tendencia_CPU']:.1f}%)")

    if row["Tendencia_RAM"] > 10:
        alertas.append(f"⚠️ RAM subiendo rápidamente (+{row['Tendencia_RAM']:.1f}%)")

    if row["Tendencia_Disco"] > 10:
        alertas.append(f"⚠️ Disco subiendo rápidamente (+{row['Tendencia_Disco']:.1f}%)")

    # Niveles elevados PERSISTENTES (problema crónico)
    if 70 <= cpu < 85 and row["CPU_Alta_5"] > 0.5:
        alertas.append(f"📊 CPU elevada y persistente ({cpu:.1f}%)")

    if 70 <= ram < 85 and row["RAM_Alta_5"] > 0.5:
        alertas.append(f"📊 RAM elevada y persistente ({ram:.1f}%)")

    if 75 <= disco < 90 and row["Disco_Alto_5"] > 0.5:
        alertas.append(f"📊 Disco elevado y persistente ({disco:.1f}%)")

    # Combinaciones peligrosas
    if cpu >= 65 and ram >= 65:
        alertas.append("🔴 CPU + RAM elevadas simultáneamente")

    if cpu >= 65 and temp >= 65:
        alertas.append("🔴 CPU + Temperatura elevadas simultáneamente")

    # Temperatura
    if temp > 70:
        alertas.append(f"🌡️ Temperatura alta ({temp:.1f}°C)")

    # Score de anomalía alto
    if row["Score_Anomalia"] > 65:
        alertas.append(f"🔍 Comportamiento anómalo detectado (score: {row['Score_Anomalia']:.1f})")

    return alertas if alertas else ["✅ Sin alertas detectadas"]


historial["Alertas_Tempranas"] = historial.apply(generar_alertas_tempranas, axis=1)


# ============================================================
# DIAGNOSTICO / RECOMENDACIONES (MEJORADOS)
# ============================================================

def generar_diagnostico_avanzado(row):
    problemas = []
    cpu, ram, disco = row["Uso_CPU_Porcentaje"], row["Uso_RAM_Porcentaje"], row["Uso_Disco_Porcentaje"]
    temp = row.get("Temperatura_CPU", 0)

    if cpu < 50:
        problemas.append("CPU en rango excelente (<50%).")
    elif cpu < 70:
        problemas.append("CPU en rango bueno (50%-69.99%).")
    elif cpu < 80:
        problemas.append("CPU en rango regular (70%-79.99%), monitorear evolución.")
    else:
        problemas.append("CPU en rango malo (>=80%); requiere atención.")

    if ram < 50:
        problemas.append("RAM en rango excelente (<50%).")
    elif ram < 70:
        problemas.append("RAM en rango bueno (50%-69.99%).")
    elif ram < 80:
        problemas.append("RAM en rango regular (70%-79.99%), monitorear evolución.")
    else:
        problemas.append("RAM en rango malo (>=80%); requiere atención.")

    if disco < 55:
        problemas.append("Disco en rango excelente (<55%).")
    elif disco < 70:
        problemas.append("Disco en rango bueno (55%-69.99%).")
    elif disco < 85:
        problemas.append("Disco en rango regular (70%-84.99%), monitorear evolución.")
    else:
        problemas.append("Disco en rango malo (>=85%); requiere atención.")

    if temp > 0:
        if temp < 50:
            problemas.append("Temperatura en rango excelente (<50°C).")
        elif temp < 65:
            problemas.append("Temperatura en rango bueno (50°C-64.99°C).")
        elif temp < 75:
            problemas.append("Temperatura en rango regular (65°C-74.99°C), monitorear.")
        else:
            problemas.append("Temperatura en rango malo (>=75°C); requiere atención.")

    if row["CPU_Alta_5"] >= 0.6:
        problemas.append("La CPU ha permanecido elevada en la mayoría de las últimas 5 mediciones.")
    if row["RAM_Alta_5"] >= 0.6:
        problemas.append("La RAM ha permanecido elevada en la mayoría de las últimas 5 mediciones.")
    if row["Disco_Alto_5"] >= 0.6:
        problemas.append("La actividad de disco ha permanecido elevada en la mayoría de las últimas 5 mediciones.")

    return " ".join(problemas) if problemas else "No se detectan problemas significativos."


def generar_recomendaciones_avanzado(row):
    recomendaciones = []
    cpu, ram, disco = row["Uso_CPU_Porcentaje"], row["Uso_RAM_Porcentaje"], row["Uso_Disco_Porcentaje"]
    temp = row.get("Temperatura_CPU", 0)

    if cpu >= 80:
        recomendaciones.append("Revisar los procesos con mayor consumo de CPU y comprobar si la carga permanece sostenida.")
    if ram >= 80:
        recomendaciones.append("Revisar aplicaciones con alto consumo de memoria y verificar presión de memoria.")
    if disco >= 85:
        recomendaciones.append("Revisar procesos con alta actividad de almacenamiento y espacio disponible.")
    if temp >= 75:
        recomendaciones.append("Verificar el sistema de refrigeración y limpieza de ventiladores.")
    if row["CPU_Alta_5"] >= 0.6 or row["RAM_Alta_5"] >= 0.6:
        recomendaciones.append("Priorizar una revisión técnica porque la carga elevada es persistente.")
    if cpu >= 80 and ram >= 80 and disco >= 85:
        recomendaciones.append("Realizar diagnóstico integral del equipo debido a saturación simultánea.")
    if not recomendaciones:
        recomendaciones.append("No se requiere una intervención inmediata; mantener el monitoreo preventivo.")
    return recomendaciones


historial["Diagnostico_IA"] = historial.apply(generar_diagnostico_avanzado, axis=1)
historial["Recomendaciones_IA"] = historial.apply(generar_recomendaciones_avanzado, axis=1)


# ============================================================
# VISTA ACTUAL DE CADA EQUIPO
# ============================================================

current_df = (
    historial.sort_values(["ID_PC", "Fecha_Hora"], na_position="first")
    .groupby("ID_PC", as_index=False)
    .tail(1)
    .copy()
)

if not equipos_bd.empty:
    eq = equipos_bd.rename(columns={"id_pc": "ID_PC", "ticket_usuario": "Ticket_Usuario_Equipo"})
    if "ID_PC" in eq.columns:
        eq["ID_PC"] = eq["ID_PC"].astype(str)
        merge_cols = [c for c in ["ID_PC", "Ticket_Usuario_Equipo"] if c in eq.columns]
        current_df = current_df.merge(eq[merge_cols], on="ID_PC", how="left")
        current_df["Tiene_Ticket_Equipo"] = current_df.get("Ticket_Usuario_Equipo", "").fillna("").astype(str).str.strip().ne("")
    else:
        current_df["Tiene_Ticket_Equipo"] = False
else:
    current_df["Tiene_Ticket_Equipo"] = False


# ============================================================
# EVALUACION DEL MODELO EN TEST
# ============================================================

def metricas_clasificacion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=["CRÍTICO", "ESTABLE"])
    tp, fn = int(cm[0, 0]), int(cm[0, 1])
    fp, tn = int(cm[1, 0]), int(cm[1, 1])
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


if modelo_supervisado is not None:
    pred_test = modelo_supervisado.predict(X_test)
    evaluacion_nombre = "Random Forest supervisado (mejorado)"
else:
    pred_test = np.where(modelo_anomalia_if.predict(X_test) == -1, "CRÍTICO", "ESTABLE")
    evaluacion_nombre = "Isolation Forest como evaluación preliminar"

metricas = metricas_clasificacion(y_test_real, pred_test)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## Diagnóstico individual")

pc_options = sorted(current_df["ID_PC"].astype(str).unique())
pc_seleccionado = st.sidebar.selectbox("Equipo", pc_options)
pc_data = current_df[current_df["ID_PC"].astype(str) == str(pc_seleccionado)]
equipo = pc_data.iloc[0]

st.sidebar.divider()
st.sidebar.markdown("## Registro de ticket")

with st.sidebar.form("form_ticket"):
    pc_ticket = st.selectbox("Equipo", pc_options, key="pc_ticket")
    descripcion = st.text_area("Descripción del problema", placeholder="Ejemplo: equipo lento, congelamiento, pantalla azul...")
    enviado = st.form_submit_button("Registrar ticket", use_container_width=True)

    if enviado:
        if descripcion.strip():
            try:
                supabase.rpc("reportar_ticket", {"p_id_pc": pc_ticket, "p_ticket": descripcion.strip()}).execute()
                st.sidebar.success("Ticket registrado correctamente.")
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.sidebar.error(f"No fue posible registrar el ticket: {exc}")
        else:
            st.sidebar.warning("Escribe una descripción antes de registrar el ticket.")


# ============================================================
# DASHBOARD
# ============================================================

st.markdown('<div class="section-title">Estado general de la flota</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total equipos", f"{len(current_df)}")
k2.metric("En riesgo", f"{int((current_df['Riesgo_IA'] >= 30).sum())}")
k3.metric("Críticos IA", f"{int((current_df['Estado'] == 'CRÍTICO').sum())}")
k4.metric("Con ticket", f"{int(current_df['Tiene_Ticket_Equipo'].sum())}")
k5.metric("CPU promedio", f"{current_df['Uso_CPU_Porcentaje'].mean():.1f}%")
k6.metric("RAM promedio", f"{current_df['Uso_RAM_Porcentaje'].mean():.1f}%")

st.caption(
    f"La IA se entrena con {n_registros:,} mediciones históricas de {n_equipos} equipos. "
    "El dashboard muestra únicamente la última medición de cada equipo."
)

fleet_counts = (
    current_df["Estado"].value_counts()
    .reindex(["ESTABLE", "MEDIO", "ALTO", "CRÍTICO"], fill_value=0)
    .reset_index()
)
fleet_counts.columns = ["Estado IA", "Cantidad"]

st.dataframe(fleet_counts, use_container_width=True, hide_index=True)


# ============================================================
# DIAGNOSTICO INDIVIDUAL
# ============================================================

st.markdown('<div class="section-title">Diagnóstico individual</div>', unsafe_allow_html=True)

estado_css = {
    "CRÍTICO": "status-critical",
    "ALTO": "status-high",
    "MEDIO": "status-medium",
    "ESTABLE": "status-stable",
}

st.markdown(
    f'<div class="status-card {estado_css[equipo["Estado"]]}">'
    f'<strong>{equipo["Estado"]}</strong> — Riesgo IA {float(equipo["Riesgo_IA"]):.1f}/100 — '
    f'Nivel {equipo["Nivel_Riesgo"]} — Equipo {equipo["ID_PC"]}</div>',
    unsafe_allow_html=True,
)

r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("CPU", f"{equipo['Uso_CPU_Porcentaje']:.1f}%")
r2.metric("RAM", f"{equipo['Uso_RAM_Porcentaje']:.1f}%")
r3.metric("Disco", f"{equipo['Uso_Disco_Porcentaje']:.1f}%")
r4.metric("Riesgo técnico", f"{equipo['Score_Tecnico']:.1f}/100")
r5.metric("Anomalía", f"{equipo['Score_Anomalia']:.1f}/100")

c1, c2 = st.columns([1, 1])
with c1:
    st.markdown("#### Diagnóstico")
    st.info(equipo["Diagnostico_IA"])
with c2:
    st.markdown("#### Recomendaciones")
    for rec in equipo["Recomendaciones_IA"]:
        st.write(f"- {rec}")

st.markdown("#### Alertas tempranas")
for alerta in equipo["Alertas_Tempranas"]:
    if "Sin alertas" in alerta:
        st.success(f"✅ {alerta}")
    else:
        st.warning(f"⚠️ {alerta}")


# ============================================================
# TABS
# ============================================================

tab_dashboard, tab_evaluacion, tab_datos = st.tabs(["Dashboard", "Evaluación y entrenamiento", "Inventario"])

with tab_dashboard:
    left, right = st.columns([1, 2])

    with left:
        st.markdown('<div class="section-title">Detalle del equipo</div>', unsafe_allow_html=True)
        detalle = pd.DataFrame({
            "Indicador": [
                "ID", "Usuario", "Modelo", "Serial", "Fecha de medición", "Estado IA",
                "Condición técnica CPU", "Condición técnica RAM", "Riesgo IA", "Ticket"
            ],
            "Valor": [
                equipo["ID_PC"],
                equipo.get("Usuario", "No disponible"),
                equipo.get("Modelo", "No disponible"),
                equipo.get("Serial", "No disponible"),
                str(equipo.get("Fecha_Hora", "No disponible")),
                equipo["Estado"],
                equipo["Estado_CPU_Tecnico"],
                equipo["Estado_RAM_Tecnico"],
                f"{equipo['Riesgo_IA']:.1f}/100",
                "SI" if equipo["Tiene_Ticket_Equipo"] else "NO",
            ],
        })
        st.dataframe(detalle, use_container_width=True, hide_index=True)

    with right:
        st.markdown('<div class="section-title">Mapa de riesgo de la flota</div>', unsafe_allow_html=True)
        mapa = px.scatter(
            current_df,
            x="Uso_CPU_Porcentaje",
            y="Uso_RAM_Porcentaje",
            color="Estado",
            size="Riesgo_IA",
            hover_data=["ID_PC", "Uso_Disco_Porcentaje", "Riesgo_IA", "Nivel_Riesgo", "Estado_Tecnico"],
            color_discrete_map={
                "CRÍTICO": "#dc2626",
                "ALTO": "#f97316",
                "MEDIO": "#eab308",
                "ESTABLE": "#2563eb",
            },
        )
        mapa.update_layout(height=470, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
        st.plotly_chart(mapa, use_container_width=True)

    st.markdown('<div class="section-title">Priorización de atención</div>', unsafe_allow_html=True)
    prioridad = current_df[[
        "ID_PC", "Estado", "Nivel_Riesgo", "Riesgo_IA", "Score_Tecnico", "Score_Anomalia",
        "Uso_CPU_Porcentaje", "Uso_RAM_Porcentaje", "Uso_Disco_Porcentaje", "Tiene_Ticket_Equipo",
        "Diagnostico_IA"
    ]].copy().sort_values("Riesgo_IA", ascending=False)
    st.dataframe(prioridad, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Distribución del riesgo actual</div>', unsafe_allow_html=True)
    risk_fig = px.histogram(
        current_df,
        x="Riesgo_IA",
        color="Estado",
        nbins=12,
        color_discrete_map={
            "CRÍTICO": "#dc2626",
            "ALTO": "#f97316",
            "MEDIO": "#eab308",
            "ESTABLE": "#2563eb",
        },
    )
    risk_fig.update_layout(height=380, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
    st.plotly_chart(risk_fig, use_container_width=True)


# ============================================================
# EVALUACION (MEJORADA - MÁS INFORMACIÓN)
# ============================================================

with tab_evaluacion:
    st.markdown('<div class="section-title">Evaluación y entrenamiento de la IA</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-note">Fuente de entrenamiento: mediciones_equipos. '
        f'{n_registros:,} mediciones de {n_equipos} equipos. El objetivo es detectar CRÍTICO frente a NO CRÍTICO.</div>',
        unsafe_allow_html=True,
    )

    if modelo_supervisado is not None:
        st.success(
            f"✅ Modelo supervisado activo. Entrenamiento: {len(train_df):,} mediciones de "
            f"{train_df['ID_PC'].nunique()} equipos. Prueba: {len(test_df):,} mediciones de "
            f"{test_df['ID_PC'].nunique()} equipos. No se mezclaron equipos entre entrenamiento y prueba."
        )

        st.markdown('<div class="section-title">Proceso de entrenamiento</div>', unsafe_allow_html=True)
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Algoritmo", "Random Forest")
        e2.metric("Árboles entrenados", f"{N_ARBOLES_RF:,}")
        e3.metric("Profundidad máxima", "15")
        e4.metric("Tiempo de entrenamiento", f"{tiempo_entrenamiento_supervisado:.3f} s")

        st.info(
            f"✅ El modelo entrenó {N_ARBOLES_RF:,} árboles usando {len(features)} variables "
            f"predictoras, con class_weight extremo (5:1 para críticos), semilla 42 y procesamiento paralelo. "
            f"Se utilizó SMOTE para balancear clases {'(activado)' if SMOTE_AVAILABLE and cantidad_criticos < 15 else '(no aplicado)'}."
        )
    else:
        st.warning(
            f"⚠️ El modelo supervisado no está activo. Registros: {n_registros}; equipos: {n_equipos}; "
            f"críticos: {cantidad_criticos}; no críticos: {cantidad_estables}."
        )

    st.markdown('<div class="section-title">Calidad de clasificación</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metricas['accuracy']:.2%}")
    m2.metric("Precision", f"{metricas['precision']:.2%}")
    m3.metric("Recall", f"{metricas['recall']:.2%}")
    m4.metric("F1-Score", f"{metricas['f1']:.2%}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Balanced Accuracy", f"{metricas['balanced_accuracy']:.2%}")
    m6.metric("Specificity", f"{metricas['specificity']:.2%}")
    m7.metric("MCC", f"{metricas['mcc']:.3f}")
    m8.metric("False Positive Rate", f"{metricas['fpr']:.2%}")

    st.caption(
        f"Modelo evaluado: {evaluacion_nombre}. La clase positiva es CRÍTICO. "
        f"La prueba contiene {len(test_df):,} mediciones y {test_df['ID_PC'].nunique()} equipos."
    )

    # Matriz de confusion
    st.markdown('<div class="section-title">Matriz de confusión</div>', unsafe_allow_html=True)
    cm = confusion_matrix(y_test_real, pred_test, labels=["CRÍTICO", "ESTABLE"])
    cm_df = pd.DataFrame(cm, index=["Real CRÍTICO", "Real NO CRÍTICO"], columns=["Predicción CRÍTICO", "Predicción NO CRÍTICO"])
    st.dataframe(cm_df, use_container_width=False, width=620, hide_index=False)
    st.caption(
        f"TP={metricas['tp']} | FN={metricas['fn']} | FP={metricas['fp']} | TN={metricas['tn']}"
    )

    # Curvas ROC / PR
    st.markdown('<div class="section-title">Curvas de evaluación</div>', unsafe_allow_html=True)
    roc_col, pr_col = st.columns(2)

    if modelo_supervisado is not None and len(np.unique(y_test_real)) == 2 and "CRÍTICO" in list(modelo_supervisado.classes_):
        classes = list(modelo_supervisado.classes_)
        idx_critical = classes.index("CRÍTICO")
        prob_test = modelo_supervisado.predict_proba(X_test)[:, idx_critical]
        y_bin = (np.asarray(y_test_real) == "CRÍTICO").astype(int)

        fpr, tpr, _ = roc_curve(y_bin, prob_test)
        roc_auc = auc(fpr, tpr)
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC {roc_auc:.3f}", line=dict(color="#1e3a8a", width=3)))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line=dict(color="#94a3b8", dash="dash")))
        roc_fig.update_layout(height=350, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", title="ROC", xaxis_title="Falsos positivos", yaxis_title="Verdaderos positivos")
        roc_col.plotly_chart(roc_fig, use_container_width=True)
        roc_col.metric("ROC-AUC", f"{roc_auc:.3f}")

        precision_curve, recall_curve, _ = precision_recall_curve(y_bin, prob_test)
        pr_auc = average_precision_score(y_bin, prob_test)
        pr_fig = go.Figure()
        pr_fig.add_trace(go.Scatter(x=recall_curve, y=precision_curve, mode="lines", name=f"AUC {pr_auc:.3f}", line=dict(color="#0f766e", width=3)))
        pr_fig.update_layout(height=350, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", title="Precision-Recall", xaxis_title="Recall", yaxis_title="Precision")
        pr_col.plotly_chart(pr_fig, use_container_width=True)
        pr_col.metric("PR-AUC", f"{pr_auc:.3f}")
    else:
        roc_col.warning("ROC-AUC requiere modelo supervisado activo y ambas clases presentes en el conjunto de prueba.")
        pr_col.warning("PR-AUC requiere modelo supervisado activo y ambas clases presentes en el conjunto de prueba.")

    # Validacion cruzada por equipos (mejorada)
    st.markdown('<div class="section-title">Validación cruzada del entrenamiento</div>', unsafe_allow_html=True)

    if modelo_supervisado is not None:
        group_labels = (
            historial.groupby("ID_PC")["Clase_Real"]
            .agg(lambda s: "CRÍTICO" if np.mean(s.eq("CRÍTICO")) >= 0.5 else "ESTABLE")
        )
        min_group_class = int(group_labels.value_counts().min()) if not group_labels.empty else 0
        folds = min(5, min_group_class)

        if folds >= 2:
            cv_model = RandomForestClassifier(
                n_estimators=1500,
                max_depth=15,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight={'CRÍTICO': 5, 'ESTABLE': 1},
                random_state=42,
                n_jobs=-1,
            )

            try:
                if StratifiedGroupKFold is not None:
                    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
                    cv_iter = cv.split(X_all, historial["Clase_Real"], groups=historial["ID_PC"])
                else:
                    cv = GroupKFold(n_splits=folds)
                    cv_iter = cv.split(X_all, historial["Clase_Real"], groups=historial["ID_PC"])

                cv_results = []
                for train_idx, val_idx in cv_iter:
                    cv_model.fit(X_all.iloc[train_idx], historial["Clase_Real"].iloc[train_idx])
                    val_pred = cv_model.predict(X_all.iloc[val_idx])
                    cv_results.append(metricas_clasificacion(historial["Clase_Real"].iloc[val_idx], val_pred))

                cv_table = pd.DataFrame([
                    {
                        "Metrica": "Accuracy",
                        "Promedio": np.mean([r["accuracy"] for r in cv_results]),
                        "Desv. estándar": np.std([r["accuracy"] for r in cv_results]),
                    },
                    {
                        "Metrica": "Precision",
                        "Promedio": np.mean([r["precision"] for r in cv_results]),
                        "Desv. estándar": np.std([r["precision"] for r in cv_results]),
                    },
                    {
                        "Metrica": "Recall",
                        "Promedio": np.mean([r["recall"] for r in cv_results]),
                        "Desv. estándar": np.std([r["recall"] for r in cv_results]),
                    },
                    {
                        "Metrica": "F1-Score",
                        "Promedio": np.mean([r["f1"] for r in cv_results]),
                        "Desv. estándar": np.std([r["f1"] for r in cv_results]),
                    },
                ])
                cv_display = cv_table.copy()
                cv_display["Promedio"] = cv_display["Promedio"].map(lambda x: f"{x:.2%}")
                cv_display["Desv. estándar"] = cv_display["Desv. estándar"].map(lambda x: f"{x:.2%}")
                cv_display["Folds"] = folds
                st.dataframe(cv_display, use_container_width=True, hide_index=True)
                st.caption(
                    f"Validación cruzada con {folds} folds y {N_ARBOLES_RF:,} árboles por fold. "
                    "La separación se realiza por equipos completos para reducir fuga de información."
                )
            except Exception as exc:
                st.warning(f"No fue posible completar la validación cruzada: {exc}")
        else:
            st.warning("Se necesita al menos 2 equipos representativos de cada clase para validación cruzada.")
    else:
        st.warning("El modelo supervisado aún no está activo.")

    # Evolución del entrenamiento por número de árboles
    if modelo_supervisado is not None:
        st.markdown('<div class="section-title">Evolución del entrenamiento por número de árboles</div>', unsafe_allow_html=True)

        arboles_prueba = [100, 250, 500, 750, 1000, 1500]
        evolucion = []

        for n_trees in arboles_prueba:
            rf_tmp = RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=15,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight={'CRÍTICO': 5, 'ESTABLE': 1},
                random_state=42,
                n_jobs=-1,
            )
            rf_tmp.fit(X_train, y_train)
            pred_tmp = rf_tmp.predict(X_test)
            met_tmp = metricas_clasificacion(y_test_real, pred_tmp)
            evolucion.append({
                "Árboles": n_trees,
                "Accuracy": met_tmp["accuracy"],
                "Precision": met_tmp["precision"],
                "Recall": met_tmp["recall"],
                "F1-Score": met_tmp["f1"],
            })

        evolucion_df = pd.DataFrame(evolucion)
        evolucion_view = evolucion_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            evolucion_view[col] = evolucion_view[col].map(lambda x: f"{x:.2%}")

        st.dataframe(evolucion_view, use_container_width=True, hide_index=True)

        evol_fig = go.Figure()
        for metric_name in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            evol_fig.add_trace(
                go.Scatter(
                    x=evolucion_df["Árboles"],
                    y=evolucion_df[metric_name],
                    mode="lines+markers",
                    name=metric_name,
                )
            )
        evol_fig.update_layout(
            height=380,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            title="Desempeño según cantidad de árboles",
            xaxis_title="Número de árboles",
            yaxis_title="Métrica",
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(evol_fig, use_container_width=True)

        st.caption(
            "Esta comparación permite evidenciar que el modelo fue evaluado con diferentes "
            "niveles de entrenamiento. En Random Forest se habla de árboles/estimadores."
        )

    # Importancia de variables
    if modelo_supervisado is not None:
        st.markdown('<div class="section-title">Importancia de las variables</div>', unsafe_allow_html=True)
        importancia = pd.DataFrame({"Variable": features, "Importancia": modelo_supervisado.feature_importances_}).sort_values("Importancia", ascending=True)
        fig_imp = px.bar(importancia, x="Importancia", y="Variable", orientation="h")
        fig_imp.update_layout(height=560, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", xaxis_title="Importancia relativa", yaxis_title="")
        st.plotly_chart(fig_imp, use_container_width=True)

    # Umbral
    if modelo_supervisado is not None and "CRÍTICO" in list(modelo_supervisado.classes_):
        st.markdown('<div class="section-title">Análisis del umbral de decisión</div>', unsafe_allow_html=True)
        classes = list(modelo_supervisado.classes_)
        idx_critical = classes.index("CRÍTICO")
        prob_test = modelo_supervisado.predict_proba(X_test)[:, idx_critical]
        threshold_rows = []
        for threshold in np.arange(0.30, 0.81, 0.05):
            pred_thr = np.where(prob_test >= threshold, "CRÍTICO", "ESTABLE")
            met = metricas_clasificacion(y_test_real, pred_thr)
            threshold_rows.append({
                "Umbral": round(float(threshold), 2),
                "Accuracy": met["accuracy"],
                "Precision": met["precision"],
                "Recall": met["recall"],
                "F1-Score": met["f1"],
                "Falsos negativos": met["fn"],
            })
        thr_df = pd.DataFrame(threshold_rows)
        thr_view = thr_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            thr_view[col] = thr_view[col].map(lambda x: f"{x:.2%}")
        st.dataframe(thr_view, use_container_width=True, hide_index=True)

    # Contamination
    st.markdown('<div class="section-title">Sensibilidad del parámetro contamination</div>', unsafe_allow_html=True)
    contamination_rows = []
    for c in sorted(set([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, CONTAMINATION])):
        model_temp = IsolationForest(n_estimators=250, contamination=c, random_state=42, n_jobs=-1)
        model_temp.fit(X_train)
        pred_temp = np.where(model_temp.predict(X_test) == -1, "CRÍTICO", "ESTABLE")
        met = metricas_clasificacion(y_test_real, pred_temp)
        contamination_rows.append({
            "Contamination": c,
            "Accuracy": met["accuracy"],
            "Precision": met["precision"],
            "Recall": met["recall"],
            "F1-Score": met["f1"],
            "Configuración actual": "ACTUAL" if abs(c - CONTAMINATION) < 0.0001 else "",
        })
    cont_df = pd.DataFrame(contamination_rows)
    cont_view = cont_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        cont_view[col] = cont_view[col].map(lambda x: f"{x:.2%}")
    st.dataframe(cont_view, use_container_width=True, hide_index=True)

    # Calidad de datos
    st.markdown('<div class="section-title">Calidad y composición del conjunto de datos</div>', unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Mediciones", f"{n_registros:,}")
    q2.metric("Equipos", f"{n_equipos}")
    q3.metric("Críticas", f"{cantidad_criticos:,}")
    q4.metric("No críticas", f"{cantidad_estables:,}")

    estado_dist = historial["Estado_Tecnico"].value_counts().reindex(["ESTABLE", "REGULAR", "ALERTA", "CRITICO"], fill_value=0).reset_index()
    estado_dist.columns = ["Estado técnico", "Cantidad"]
    st.dataframe(estado_dist, use_container_width=True, hide_index=True)


# ============================================================
# INVENTARIO
# ============================================================

with tab_datos:
    st.markdown('<div class="section-title">Inventario técnico actual</div>', unsafe_allow_html=True)
    st.caption("La IA utiliza el historial completo de mediciones; esta sección muestra la última medición disponible por equipo.")

    inv_cols = [
        "ID_PC", "Fecha_Hora", "Usuario", "Modelo", "Serial", "Estado", "Nivel_Riesgo",
        "Riesgo_IA", "Score_Tecnico", "Score_Anomalia", "Estado_Tecnico", "Uso_CPU_Porcentaje",
        "Uso_RAM_Porcentaje", "Uso_Disco_Porcentaje", "Temperatura_CPU", "Procesos_Activos",
        "Estado_CPU_Tecnico", "Estado_RAM_Tecnico", "Tiene_Ticket_Equipo", "Diagnostico_IA"
    ]
    inv_cols = [c for c in inv_cols if c in current_df.columns]
    inventario = current_df[inv_cols].sort_values("Riesgo_IA", ascending=False)
    st.dataframe(inventario, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Historial de mediciones</div>', unsafe_allow_html=True)
    history_preview_cols = [
        "ID_PC", "Fecha_Hora", "Uso_CPU_Porcentaje", "Uso_RAM_Porcentaje", "Uso_Disco_Porcentaje",
        "CPU_Normalizado_Porcentaje", "Estado_Tecnico", "Severidad", "Tiene_Ticket", "Riesgo_IA"
    ]
    history_preview_cols = [c for c in history_preview_cols if c in historial.columns]
    st.dataframe(
        historial[history_preview_cols].sort_values("Fecha_Hora", ascending=False).head(500),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PIE
# ============================================================

st.divider()
st.markdown(
    '<div class="small-muted">AI-FleetMonitor Pro — monitoreo, entrenamiento, validación y diagnóstico técnico con detección avanzada de anomalías.</div>',
    unsafe_allow_html=True,
)
