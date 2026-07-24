import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, roc_curve, auc)
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from supabase import create_client
import time

st.set_page_config(layout="wide", page_title="Monitor IA - Gestión Avanzada")
st.title("🖥️ AI-FleetMonitor Pro: Diagnóstico de Hardware")

# 1. Conexión a Supabase y carga de datos
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


@st.cache_data(ttl=60)  # refresca solo desde Supabase cada 60 segundos
def cargar_datos(_cache_bust):
    resultado = supabase.table("equipos").select("*").execute()
    df = pd.DataFrame(resultado.data)

    # Renombrar de vuelta a los nombres usados en el resto de la app
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

    # Si hay un Ticket_Usuario, lo marcamos como REALMENTE CRÍTICO
    # Si no tiene ticket, el equipo está ESTABLE en la realidad
    df['Clase_Real'] = df['Ticket_Usuario'].apply(lambda x: 'CRÍTICO' if pd.notnull(x) else 'ESTABLE')
    return df


# _cache_bust cambia cada 60s (por el ttl) forzando una consulta fresca a Supabase
df = cargar_datos(int(time.time() // 60))

if st.button("🔄 Actualizar ahora"):
    st.cache_data.clear()
    st.rerun()

if df.empty:
    st.warning(
        "⚠️ Todavía no hay datos en Supabase. "
        "Corre `agente_monitor.py` en al menos una PC para que empiece a subir información."
    )
    st.stop()

# 2. IA: Detección de Anomalías (Isolation Forest) con validación train/test
features = ['Uso_CPU_Porcentaje', 'Uso_RAM_Porcentaje', 'CPU_Normalizado_Porcentaje']
CONTAMINATION = 0.1

n_total = len(df)
puede_dividir = n_total >= 10 and df['Clase_Real'].nunique() == 2

if puede_dividir:
    try:
        train_df, test_df = train_test_split(
            df, test_size=0.3, random_state=42, stratify=df['Clase_Real']
        )
    except ValueError:
        # Si alguna clase tiene muy pocos registros para estratificar, se divide sin estratificar
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
else:
    train_df, test_df = df, df  # dataset muy pequeño todavía: se entrena y evalúa sobre todo (con advertencia visible)

modelo_final = IsolationForest(contamination=CONTAMINATION, random_state=42)
modelo_final.fit(train_df[features])

# Aplica el modelo entrenado a TODA la flota (esto es lo que opera el negocio)
df['Riesgo_IA'] = modelo_final.predict(df[features])
df['Estado'] = df['Riesgo_IA'].apply(lambda x: 'CRÍTICO' if x == -1 else 'ESTABLE')
df['Score_Anomalia'] = -modelo_final.decision_function(df[features])  # mayor score = mas riesgoso

# 3. Sidebar
st.sidebar.header("🔍 Diagnóstico Individual")
pc_seleccionado = st.sidebar.selectbox("Filtrar Equipo ID:", df['ID_PC'].unique())
pc_data = df[df['ID_PC'] == pc_seleccionado]

# 3.1 Formulario para reportar tickets (escribe en Supabase via función segura)
st.sidebar.divider()
st.sidebar.header("🎫 Reportar un Ticket")
with st.sidebar.form("form_ticket"):
    pc_ticket = st.selectbox("Equipo con problema:", df['ID_PC'].unique(), key="pc_ticket")
    descripcion = st.text_area("Describe el problema:", placeholder="Ej: No enciende, pantalla azul...")
    enviado = st.form_submit_button("📩 Reportar ticket")

    if enviado:
        if descripcion.strip():
            supabase.rpc("reportar_ticket", {"p_id_pc": pc_ticket, "p_ticket": descripcion.strip()}).execute()
            st.sidebar.success(f"✅ Ticket registrado para {pc_ticket}")
            st.cache_data.clear()
            st.rerun()
        else:
            st.sidebar.warning("Escribe una descripción antes de enviar.")

# 4. KPIs
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Total Equipos", len(df))
col_b.metric("Equipos en Riesgo", df[df['Estado'] == 'CRÍTICO'].shape[0])
col_c.metric("Promedio CPU", f"{df['Uso_CPU_Porcentaje'].mean():.1f}%")
col_d.metric("Promedio RAM", f"{df['Uso_RAM_Porcentaje'].mean():.1f}%")

# 5. DASHBOARD PRINCIPAL
c1, c2 = st.columns([1, 2])
with c1:
    st.subheader(f"Detalles: {pc_seleccionado}")
    st.write(pc_data.T)

with c2:
    st.subheader("Mapa de Riesgo (IA)")
    fig = px.scatter(df, x="Uso_CPU_Porcentaje", y="Uso_RAM_Porcentaje", color="Estado",
                      size='Uso_RAM_Porcentaje', color_discrete_map={'CRÍTICO': '#FF4B4B', 'ESTABLE': '#0068C9'})
    st.plotly_chart(fig, use_container_width=True)

# 6. EVALUACIÓN Y VALIDACIÓN DEL MODELO DE IA
st.divider()
st.subheader("📊 Evaluación y Validación del Modelo (Isolation Forest)")

if not puede_dividir:
    st.warning(
        f"⚠️ Solo hay {n_total} equipo(s) con datos todavía, o falta variedad de clases (CRÍTICO/ESTABLE). "
        "Para una validación rigurosa (train/test split) se necesitan al menos 10 registros con ambas clases "
        "representadas. Mientras tanto, el modelo se entrena y evalúa sobre el mismo conjunto completo — "
        "esto es solo una vista preliminar, no una validación real."
    )
else:
    st.caption(
        f"✅ Validado con división train/test: {len(train_df)} equipos para entrenar, "
        f"{len(test_df)} equipos separados para evaluar (nunca vistos durante el entrenamiento)."
    )

labels = ['CRÍTICO', 'ESTABLE']
y_test_real = test_df['Clase_Real'].values
pred_test = modelo_final.predict(test_df[features])
y_test_pred = np.where(pred_test == -1, 'CRÍTICO', 'ESTABLE')
score_test = -modelo_final.decision_function(test_df[features])

# 6.1 Métricas clave (calculadas SOLO sobre el set de prueba)
acc = accuracy_score(y_test_real, y_test_pred)
prec = precision_score(y_test_real, y_test_pred, pos_label='CRÍTICO', zero_division=0)
rec = recall_score(y_test_real, y_test_pred, pos_label='CRÍTICO', zero_division=0)
f1 = f1_score(y_test_real, y_test_pred, pos_label='CRÍTICO', zero_division=0)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{acc:.2%}")
m2.metric("Precision (CRÍTICO)", f"{prec:.2%}")
m3.metric("Recall (CRÍTICO)", f"{rec:.2%}")
m4.metric("F1-Score (CRÍTICO)", f"{f1:.2%}")

col_cm, col_rep = st.columns([1, 1])

with col_cm:
    st.markdown("**Matriz de Confusión** (sobre el set de prueba)")
    cm = confusion_matrix(y_test_real, y_test_pred, labels=labels)
    fig_cm, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicción IA')
    plt.ylabel('Realidad (Tickets)')
    st.pyplot(fig_cm)

with col_rep:
    st.markdown("**Reporte de Clasificación** (sobre el set de prueba)")
    reporte = classification_report(y_test_real, y_test_pred, labels=labels, output_dict=True, zero_division=0)
    reporte_df = pd.DataFrame(reporte).transpose().round(2)
    st.dataframe(reporte_df, use_container_width=True)

# 6.2 Curva ROC + AUC ("Score IA")
st.markdown("### 🎯 Score IA: Curva ROC y AUC")
st.write(
    "El modelo no solo predice CRÍTICO/ESTABLE, también calcula un puntaje continuo de qué tan anómalo "
    "es cada equipo. La curva ROC muestra qué tan bien ese puntaje separa los equipos críticos de los "
    "estables en distintos umbrales de decisión. El AUC resume esa capacidad en un solo número: "
    "1.0 = separación perfecta, 0.5 = equivalente a adivinar al azar."
)

y_test_bin = (y_test_real == 'CRÍTICO').astype(int)
if len(np.unique(y_test_bin)) == 2:
    fpr, tpr, _ = roc_curve(y_test_bin, score_test)
    auc_score = auc(fpr, tpr)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Modelo (AUC = {auc_score:.2f})',
                                  line=dict(color='#0068C9', width=3)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Azar (AUC = 0.50)',
                                  line=dict(color='gray', dash='dash')))
    fig_roc.update_layout(xaxis_title='Tasa de Falsos Positivos', yaxis_title='Tasa de Verdaderos Positivos',
                           height=400)
    st.plotly_chart(fig_roc, use_container_width=True)
    st.metric("AUC Score", f"{auc_score:.3f}")
else:
    st.info("El set de prueba solo tiene una clase representada, no se puede calcular la curva ROC todavía.")

# 6.3 Distribución de los puntajes de anomalía
st.markdown("### 📈 Distribución del Puntaje de Anomalía")
st.write("Qué tan separados están los puntajes de riesgo entre equipos CRÍTICOS y ESTABLES (sobre toda la flota).")
fig_hist = px.histogram(df, x="Score_Anomalia", color="Clase_Real", barmode="overlay", nbins=20,
                         color_discrete_map={'CRÍTICO': '#FF4B4B', 'ESTABLE': '#0068C9'})
st.plotly_chart(fig_hist, use_container_width=True)

# 6.4 Sensibilidad del parámetro contamination
st.markdown("### 🔧 Justificación del parámetro `contamination`")
st.write("Comparación de accuracy en el set de prueba usando distintos valores de `contamination`, "
         "para justificar por qué se eligió el valor actual en vez de uno arbitrario.")

valores_prueba = sorted(set([0.05, 0.1, 0.2, 0.3, CONTAMINATION,
                              round(min(df['Clase_Real'].value_counts(normalize=True).get('CRÍTICO', 0.1), 0.5), 2)]))
resultados_contamination = []
for c in valores_prueba:
    if c <= 0 or c >= 0.5:
        continue
    m = IsolationForest(contamination=c, random_state=42)
    m.fit(train_df[features])
    pred_c = m.predict(test_df[features])
    pred_c_label = np.where(pred_c == -1, 'CRÍTICO', 'ESTABLE')
    acc_c = accuracy_score(y_test_real, pred_c_label)
    resultados_contamination.append({"contamination": c, "accuracy": round(acc_c, 3),
                                      "usado_actualmente": "✅" if c == CONTAMINATION else ""})

st.dataframe(pd.DataFrame(resultados_contamination), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Inventario Técnico Completo")
st.dataframe(df, use_container_width=True)
