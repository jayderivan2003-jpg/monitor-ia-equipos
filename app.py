import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
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

# 2. IA: Detección de Anomalías (Isolation Forest)
features = ['Uso_CPU_Porcentaje', 'Uso_RAM_Porcentaje', 'CPU_Normalizado_Porcentaje']
model = IsolationForest(contamination=0.1, random_state=42)
df['Riesgo_IA'] = model.fit_predict(df[features])
df['Estado'] = df['Riesgo_IA'].apply(lambda x: 'CRÍTICO' if x == -1 else 'ESTABLE')

# 3. Sidebar
st.sidebar.header("🔍 Diagnóstico Individual")
pc_seleccionado = st.sidebar.selectbox("Filtrar Equipo ID:", df['ID_PC'].unique())
pc_data = df[df['ID_PC'] == pc_seleccionado]

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

# 6. EVALUACIÓN DEL MODELO DE IA
st.divider()
st.subheader("📊 Evaluación del Modelo (Isolation Forest)")
st.write("Comparación entre la predicción de la IA y la realidad (equipos con/sin ticket de soporte).")

y_real = df['Clase_Real']
y_pred = df['Estado']
labels = ['CRÍTICO', 'ESTABLE']

# 6.1 Métricas clave
acc = accuracy_score(y_real, y_pred)
prec = precision_score(y_real, y_pred, pos_label='CRÍTICO', zero_division=0)
rec = recall_score(y_real, y_pred, pos_label='CRÍTICO', zero_division=0)
f1 = f1_score(y_real, y_pred, pos_label='CRÍTICO', zero_division=0)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{acc:.2%}")
m2.metric("Precision (CRÍTICO)", f"{prec:.2%}")
m3.metric("Recall (CRÍTICO)", f"{rec:.2%}")
m4.metric("F1-Score (CRÍTICO)", f"{f1:.2%}")

col_cm, col_rep = st.columns([1, 1])

with col_cm:
    st.markdown("**Matriz de Confusión**")
    cm = confusion_matrix(y_real, y_pred, labels=labels)
    fig_cm, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicción IA')
    plt.ylabel('Realidad (Tickets)')
    st.pyplot(fig_cm)

with col_rep:
    st.markdown("**Reporte de Clasificación**")
    reporte = classification_report(y_real, y_pred, labels=labels, output_dict=True, zero_division=0)
    reporte_df = pd.DataFrame(reporte).transpose().round(2)
    st.dataframe(reporte_df, use_container_width=True)

st.divider()
st.subheader("Inventario Técnico Completo")
st.dataframe(df, use_container_width=True)
