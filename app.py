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
from sklearn.neighbors import LocalOutlierFactor
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

# Intentar importar SMOTE para balanceo de clases
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

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
# ESTILO
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
            Detección avanzada de anomalías • Monitoreo de hardware • Clasificación de riesgo 
            • Entrenamiento y evaluación de IA con máxima sensibilidad
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
# ETIQUETAS: EL MODELO APRENDE DEL ESTADO TECNICO
# ============================================================

def estado_tecnico_reglas(row):
    cpu = float(row["Uso_CPU_Porcentaje"])
    ram = float(row["Uso_RAM_Porcentaje"])
    disco = float(row["Uso_Disco_Porcentaje"])
    temp = float(row.get("Temperatura_CPU", 0))

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
# VARIABLES DERIVADAS E HISTORICAS (MEJORADAS)
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

# Umbrales MÁS SENSIBLES (80% en lugar de 85%)
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
# FEATURES
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
# TRAIN/TEST POR EQUIPO
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
# ENTRENAMIENTO CON MÁXIMA SENSIBILIDAD
# ============================================================

modelo_supervisado = None
tiempo_entrenamiento_supervisado = 0.0
N_ARBOLES_RF = 1000

# ============================================================
# 1. ISOLATION FOREST CON ALTA SENSIBILIDAD
# ============================================================

proporcion_criticos = cantidad_criticos / max(n_registros, 1)
CONTAMINATION = float(np.clip(max(0.10, proporcion_criticos * 1.5), 0.05, 0.35))

modelo_anomalia_if = IsolationForest(
    n_estimators=1000,
    contamination=CONTAMINATION,
    max_samples="auto",
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)
modelo_anomalia_if.fit(X_train)

# ============================================================
# 2. LOCAL OUTLIER FACTOR (detección por densidad)
# ============================================================

n_neighbors = min(20, max(5, len(X_train) // 10))
modelo_lof = LocalOutlierFactor(
    n_neighbors=n_neighbors,
    contamination=CONTAMINATION,
    novelty=True
)
modelo_lof.fit(X_train)

# ============================================================
# 3. RANDOM FOREST CON PESOS EXTREMOS Y SMOTE
# ============================================================

if puede_usar_supervisado and train_df["Clase_Real"].nunique() == 2:
    X_train_rf = X_train.copy()
    y_train_rf = y_train.copy()
    
    if cantidad_criticos < 15 and SMOTE_AVAILABLE:
        try:
            k_neighbors = min(3, cantidad_criticos - 1)
            if k_neighbors >= 1:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_train_rf, y_train_rf = smote.fit_resample(X_train, y_train)
        except Exception:
            pass
    
    modelo_supervisado = RandomForestClassifier(
        n_estimators=N_ARBOLES_RF,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight={'CRÍTICO': 5, 'ESTABLE': 1},
        random_state=42,
        n_jobs=-1,
        min_impurity_decrease=0.0001,
    )

    inicio_entrenamiento = time.perf_counter()
    modelo_supervisado.fit(X_train_rf, y_train_rf)
    tiempo_entrenamiento_supervisado = time.perf_counter() - inicio_entrenamiento

# ============================================================
# CALCULAR SCORES COMBINADOS
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

# Combinar scores (ensamble)
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
# SCORE TECNICO MEJORADO (más sensible)
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
    
    # Temperatura
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
    
    # BONIFICACIÓN POR MÚLTIPLES PROBLEMAS
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
    if riesgo >= 70:
        return "CRÍTICO"
    if riesgo >= 50:
        return "ALTO"
    if riesgo >= 30:
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
# ALERTAS TEMPRANAS
# ============================================================

def generar_alertas_tempranas(row):
    """Detecta problemas ANTES de que sean críticos"""
    alertas = []
    
    cpu = row["Uso_CPU_Porcentaje"]
    ram = row["Uso_RAM_Porcentaje"]
    disco = row["Uso_Disco_Porcentaje"]
    temp = row.get("Temperatura_CPU", 0)
    
    # Tendencias al alza
    if row["Tendencia_CPU"] > 10:
        alertas.append(f"⚠️ CPU subiendo rápidamente (+{row['Tendencia_CPU']:.1f}%)")
    
    if row["Tendencia_RAM"] > 10:
        alertas.append(f"⚠️ RAM subiendo rápidamente (+{row['Tendencia_RAM']:.1f}%)")
    
    # Niveles elevados PERSISTENTES
    if 70 <= cpu < 85 and row["CPU_Alta_5"] > 0.5:
        alertas.append(f"📊 CPU elevada y persistente ({cpu:.1f}%)")
    
    if 70 <= ram < 85 and row["RAM_Alta_5"] > 0.5:
        alertas.append(f"📊 RAM elevada y persistente ({ram:.1f}%)")
    
    # Combinaciones peligrosas
    if cpu >= 65 and ram >= 65:
        alertas.append("🔴 CPU + RAM elevadas simultáneamente")
    
    # Temperatura
    if temp > 70:
        alertas.append(f"🌡️ Temperatura alta ({temp:.1f}°C)")
    
    # Score de anomalía alto
    if row["Score_Anomalia"] > 65:
        alertas.append(f"🔍 Comportamiento anómalo detectado (score: {row['Score_Anomalia']:.1f})")
    
    return alertas if alertas else ["✅ Sin alertas detectadas"]

historial["Alertas_Tempranas"] = historial.apply(generar_alertas_tempranas, axis=1)

# ============================================================
# DIAGNOSTICO / RECOMENDACIONES
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

    if disco >= 95:
        problemas.append("Actividad de disco muy alta (>=95
