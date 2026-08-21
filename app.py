0 10px; }
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <div class="app-header-title">AI-FleetMonitor Pro</div>
        <div class="app-header-subtitle">
            Monitoreo de hardware, deteccion de anomalias, clasificacion de riesgo,
            entrenamiento y evaluacion del modelo de inteligencia artificial.
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

if st.button("Actualizar datos"):
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
# ETIQUETAS: EL MODELO APRENDE DEL ESTADO TECNICO, NO DEL TICKET
# ============================================================

def estado_tecnico_reglas(row):
    cpu = float(row["Uso_CPU_Porcentaje"])
    ram = float(row["Uso_RAM_Porcentaje"])
    disco = float(row["Uso_Disco_Porcentaje"])

    if cpu >= 90 or ram >= 90 or disco >= 95:
        return "CRITICO"
    if cpu >= 85 or ram >= 85 or disco >= 90:
        return "ALERTA"
    if cpu >= 60 or ram >= 60 or disco >= 60:
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

# Objetivo binario para las metricas clasicas del semillero:
# CRITICO vs NO CRITICO.
historial["Clase_Real"] = np.where(
    historial["Estado_Tecnico"].eq("CRITICO"),
    "CRÍTICO",
    "ESTABLE",
)


# ============================================================
# VARIABLES DERIVADAS E HISTORICAS
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

historial["CPU_Mala"] = (historial["Uso_CPU_Porcentaje"] >= 85).astype(int)
historial["RAM_Mala"] = (historial["Uso_RAM_Porcentaje"] >= 85).astype(int)
historial["Disco_Malo"] = (historial["Uso_Disco_Porcentaje"] >= 90).astype(int)
historial["Recursos_Malos"] = historial[["CPU_Mala", "RAM_Mala", "Disco_Malo"]].sum(axis=1)

historial["Estado_CPU_Tecnico"] = pd.cut(
    historial["Uso_CPU_Porcentaje"],
    bins=[-np.inf, 60, 85, np.inf],
    labels=["BUENO", "REGULAR", "MALO"],
    right=False,
)

historial["Estado_RAM_Tecnico"] = pd.cut(
    historial["Uso_RAM_Porcentaje"],
    bins=[-np.inf, 60, 85, np.inf],
    labels=["BUENO", "REGULAR", "MALO"],
    right=False,
)

# Persistencia y tendencia: diferencia frente a mediciones anteriores.
historial["CPU_Anterior"] = historial.groupby("ID_PC")["Uso_CPU_Porcentaje"].shift(1)
historial["RAM_Anterior"] = historial.groupby("ID_PC")["Uso_RAM_Porcentaje"].shift(1)
historial["Disco_Anterior"] = historial.groupby("ID_PC")["Uso_Disco_Porcentaje"].shift(1)

historial["Tendencia_CPU"] = (historial["Uso_CPU_Porcentaje"] - historial["CPU_Anterior"]).fillna(0)
historial["Tendencia_RAM"] = (historial["Uso_RAM_Porcentaje"] - historial["RAM_Anterior"]).fillna(0)
historial["Tendencia_Disco"] = (historial["Uso_Disco_Porcentaje"] - historial["Disco_Anterior"]).fillna(0)

# Ventana movil de 5 mediciones.
grouped = historial.groupby("ID_PC", group_keys=False)
historial["CPU_Media_5"] = groupby_rolling = grouped["Uso_CPU_Porcentaje"].transform(lambda s: s.rolling(5, min_periods=1).mean())
historial["RAM_Media_5"] = grouped["Uso_RAM_Porcentaje"].transform(lambda s: s.rolling(5, min_periods=1).mean())
historial["Disco_Media_5"] = grouped["Uso_Disco_Porcentaje"].transform(lambda s: s.rolling(5, min_periods=1).mean())

historial["CPU_Alta_5"] = grouped["Uso_CPU_Porcentaje"].transform(lambda s: s.rolling(5, min_periods=1).apply(lambda x: np.mean(x >= 85)))
historial["RAM_Alta_5"] = grouped["Uso_RAM_Porcentaje"].transform(lambda s: s.rolling(5, min_periods=1).apply(lambda x: np.mean(x >= 85)))
historial["Disco_Alto_5"] = grouped["Uso_Disco_Porcentaje"].transform(lambda s: s.rolling(5, min_periods=1).apply(lambda x: np.mean(x >= 90)))

historial["Componentes_Saturados"] = (
    (historial["Uso_CPU_Porcentaje"] >= 85).astype(int)
    + (historial["Uso_RAM_Porcentaje"] >= 85).astype(int)
    + (historial["Uso_Disco_Porcentaje"] >= 90).astype(int)
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

    # Seguridad: si por el tamaño de la muestra la prueba pierde una clase,
    # buscamos otra semilla que mantenga ambas clases.
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
# RANDOM FOREST
# ============================================================

modelo_supervisado = None
tiempo_entrenamiento_supervisado = 0.0
N_ARBOLES_RF = 1000

if puede_usar_supervisado and train_df["Clase_Real"].nunique() == 2:
    modelo_supervisado = RandomForestClassifier(
        n_estimators=N_ARBOLES_RF,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    inicio_entrenamiento = time.perf_counter()
    modelo_supervisado.fit(X_train, y_train)
    tiempo_entrenamiento_supervisado = time.perf_counter() - inicio_entrenamiento


# ============================================================
# ISOLATION FOREST
# ============================================================

proporcion_criticos = cantidad_criticos / max(n_registros, 1)
CONTAMINATION = float(np.clip(max(0.05, proporcion_criticos), 0.05, 0.25))

modelo_anomalia = IsolationForest(
    n_estimators=1000,
    contamination=CONTAMINATION,
    max_samples="auto",
    random_state=42,
    n_jobs=-1,
)
modelo_anomalia.fit(X_train)

raw_anomaly = -modelo_anomalia.decision_function(X_all)
p05 = np.percentile(raw_anomaly, 5)
p95 = np.percentile(raw_anomaly, 95)
if p95 > p05:
    anomaly_score = ((raw_anomaly - p05) / (p95 - p05)) * 100
else:
    anomaly_score = np.full(n_registros, 50.0)
anomaly_score = np.clip(anomaly_score, 0, 100)
historial["Score_Anomalia"] = anomaly_score


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
# SCORE TECNICO
# ============================================================

def calcular_score_tecnico(row):
    cpu = row["Uso_CPU_Porcentaje"]
    ram = row["Uso_RAM_Porcentaje"]
    disco = row["Uso_Disco_Porcentaje"]
    cpu_norm = row["CPU_Normalizado_Porcentaje"]
    persistencia = max(row["CPU_Alta_5"], row["RAM_Alta_5"], row["Disco_Alto_5"])

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

    # Persistencia: aumenta el riesgo solo cuando el nivel alto se mantiene.
    score += min(10, int(round(persistencia * 10)))

    return min(score, 100)


historial["Score_Tecnico"] = historial.apply(calcular_score_tecnico, axis=1)


# ============================================================
# RIESGO IA FINAL
# ============================================================

if modelo_supervisado is not None:
    historial["Riesgo_IA"] = (
        prob_critico * 0.55
        + historial["Score_Anomalia"] * 0.20
        + historial["Score_Tecnico"] * 0.25
    )
else:
    historial["Riesgo_IA"] = (
        historial["Score_Anomalia"] * 0.50
        + historial["Score_Tecnico"] * 0.50
    )

historial["Riesgo_IA"] = np.clip(historial["Riesgo_IA"], 0, 100)


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


historial["Estado"] = historial["Riesgo_IA"].apply(determinar_estado)
historial["Nivel_Riesgo"] = historial["Riesgo_IA"].apply(determinar_nivel)


# ============================================================
# DIAGNOSTICO / RECOMENDACIONES
# ============================================================

def generar_diagnostico(row):
    problemas = []
    cpu, ram, disco = row["Uso_CPU_Porcentaje"], row["Uso_RAM_Porcentaje"], row["Uso_Disco_Porcentaje"]

    if cpu < 60:
        problemas.append("CPU en rango bueno (<60%).")
    elif cpu < 85:
        problemas.append("CPU en rango regular (60%-84.99%), compatible con carga moderada.")
    else:
        problemas.append("CPU en rango malo (>=85%); si la condición es persistente puede provocar lentitud.")

    if ram < 60:
        problemas.append("RAM en rango bueno (<60%).")
    elif ram < 85:
        problemas.append("RAM en rango regular (60%-84.99%), aceptable bajo carga.")
    else:
        problemas.append("RAM en rango malo (>=85%); puede aumentar la paginación y reducir el rendimiento.")

    if disco >= 95:
        problemas.append("Actividad de disco muy alta (>=95%).")
    elif disco >= 90:
        problemas.append("Actividad de disco alta (90%-94.99%).")

    if row["CPU_Alta_5"] >= 0.6:
        problemas.append("La CPU ha permanecido elevada en la mayoría de las últimas 5 mediciones.")
    if row["RAM_Alta_5"] >= 0.6:
        problemas.append("La RAM ha permanecido elevada en la mayoría de las últimas 5 mediciones.")
    if row["Disco_Alto_5"] >= 0.6:
        problemas.append("La actividad de disco ha permanecido elevada en la mayoría de las últimas 5 mediciones.")

    return " ".join(problemas)


def generar_recomendaciones(row):
    recomendaciones = []
    cpu, ram, disco = row["Uso_CPU_Porcentaje"], row["Uso_RAM_Porcentaje"], row["Uso_Disco_Porcentaje"]

    if cpu >= 85:
        recomendaciones.append("Revisar los procesos con mayor consumo de CPU y comprobar si la carga permanece sostenida.")
    if ram >= 85:
        recomendaciones.append("Revisar aplicaciones con alto consumo de memoria y verificar presión de memoria.")
    if disco >= 90:
        recomendaciones.append("Revisar procesos con alta actividad de almacenamiento y espacio disponible.")
    if row["CPU_Alta_5"] >= 0.6 or row["RAM_Alta_5"] >= 0.6:
        recomendaciones.append("Priorizar una revisión técnica porque la carga elevada es persistente y no corresponde solo a un pico aislado.")
    if cpu >= 85 and ram >= 85 and disco >= 90:
        recomendaciones.append("Realizar diagnóstico integral del equipo debido a saturación simultánea de CPU, RAM y disco.")
    if not recomendaciones:
        recomendaciones.append("No se requiere una intervención inmediata; mantener el monitoreo preventivo.")
    return recomendaciones


historial["Diagnostico_IA"] = historial.apply(generar_diagnostico, axis=1)
historial["Recomendaciones_IA"] = historial.apply(generar_recomendaciones, axis=1)


# ============================================================
# VISTA ACTUAL DE CADA EQUIPO
# ============================================================

# La IA se entrena con TODAS las mediciones; el dashboard muestra solo
# la última medición de cada PC para que "Total equipos" sea realmente el
# número de PCs y no el número de filas históricas.
current_df = (
    historial.sort_values(["ID_PC", "Fecha_Hora"], na_position="first")
    .groupby("ID_PC", as_index=False)
    .tail(1)
    .copy()
)

# Tickets de la tabla equipos, si existe, se usan para complementar la vista actual.
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
    evaluacion_nombre = "Random Forest supervisado"
else:
    pred_test = np.where(modelo_anomalia.predict(X_test) == -1, "CRÍTICO", "ESTABLE")
    evaluacion_nombre = "Isolation Forest como evaluacion preliminar"

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
k2.metric("En riesgo", f"{int((current_df['Riesgo_IA'] >= 35).sum())}")
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
# EVALUACION
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
            f"Modelo supervisado activo. Entrenamiento: {len(train_df):,} mediciones de "
            f"{train_df['ID_PC'].nunique()} equipos. Prueba: {len(test_df):,} mediciones de "
            f"{test_df['ID_PC'].nunique()} equipos. No se mezclaron equipos entre entrenamiento y prueba."
        )

        st.markdown('<div class="section-title">Proceso de entrenamiento</div>', unsafe_allow_html=True)
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Algoritmo", "Random Forest")
        e2.metric("Árboles entrenados", f"{N_ARBOLES_RF:,}")
        e3.metric("Profundidad máxima", "12")
        e4.metric("Tiempo de entrenamiento", f"{tiempo_entrenamiento_supervisado:.3f} s")

        st.info(
            f"El modelo entrenó {N_ARBOLES_RF:,} árboles usando {len(features)} variables "
            f"predictoras, con class_weight='balanced', semilla 42 y procesamiento paralelo "
            f"({-1} = todos los núcleos disponibles)."
        )
    else:
        st.warning(
            f"El modelo supervisado no está activo. Registros: {n_registros}; equipos: {n_equipos}; "
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

    # Matriz de confusion compacta y consistente con las metricas.
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

    # Validacion cruzada por equipos
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
                n_estimators=1000,
                max_depth=12,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )

            scoring = {
                "accuracy": make_scorer(accuracy_score),
                "precision": make_scorer(precision_score, pos_label="CRÍTICO", zero_division=0),
                "recall": make_scorer(recall_score, pos_label="CRÍTICO", zero_division=0),
                "f1": make_scorer(f1_score, pos_label="CRÍTICO", zero_division=0),
            }

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
                    "La separación se realiza por equipos completos para reducir fuga de información "
                    "entre mediciones del mismo equipo."
                )
            except Exception as exc:
                st.warning(f"No fue posible completar la validación cruzada: {exc}")
        else:
            st.warning("Se necesita al menos 2 equipos representativos de cada clase para validación cruzada.")
    else:
        st.warning("El modelo supervisado aún no está activo.")

    # Evidencia adicional del entrenamiento: evolución del desempeño
    # al aumentar la cantidad de árboles. No se inventan épocas:
    # Random Forest se entrena mediante árboles, no mediante épocas.
    if modelo_supervisado is not None:
        st.markdown('<div class="section-title">Evolución del entrenamiento por número de árboles</div>', unsafe_allow_html=True)

        arboles_prueba = [100, 250, 500, 750, 1000]
        evolucion = []

        for n_trees in arboles_prueba:
            rf_tmp = RandomForestClassifier(
                n_estimators=n_trees,
                max_depth=12,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight="balanced",
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
            "niveles de entrenamiento. En Random Forest se habla de árboles/estimadores, "
            "no de épocas como en redes neuronales."
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
    for c in sorted(set([0.05, 0.10, 0.15, 0.20, CONTAMINATION])):
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
    '<div class="small-muted">AI-FleetMonitor Pro — monitoreo, entrenamiento, validación y diagnóstico técnico.</div>',
    unsafe_allow_html=True,
)
