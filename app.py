import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

st.set_page_config(layout="wide", page_title="Monitor IA - Gestión Avanzada")

st.title("🖥️ AI-FleetMonitor Pro: Diagnóstico de Hardware")

# 1. Cargar datos
@st.cache_data
def cargar_datos():
    return pd.read_csv('BASEDEDATOSPROYECTO.csv')

df = cargar_datos()

# 2. IA: Detección de Anomalías (Isolation Forest)
features = ['Uso_CPU_Porcentaje', 'Uso_RAM_Porcentaje', 'CPU_Normalizadontajercentaje']
model = IsolationForest(contamination=0.1, random_state=42)
df['Riesgo_IA'] = model.fit_predict(df[features])
df['Estado'] = df['Riesgo_IA'].apply(lambda x: 'CRÍTICO' if x == -1 else 'ESTABLE')

# 3. Sidebar: Análisis 1x1
st.sidebar.header("🔍 Diagnóstico Individual")
pc_seleccionado = st.sidebar.selectbox("Filtrar Equipo ID:", df['ID_PC'].unique())
pc_data = df[df['ID_PC'] == pc_seleccionado]

# 4. DASHBOARD SUPERIOR (KPIs)
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
    st.subheader("⚠️ Diagnóstico de Picos")
    cpu_val = pc_data['Uso_CPU_Porcentaje'].values[0]
    if cpu_val > 80: st.error(f"¡ALERTA: Pico de CPU detectado! ({cpu_val}%)")
    elif cpu_val > 60: st.warning(f"Atención: CPU elevado ({cpu_val}%)")
    else: st.success(f"CPU Estable ({cpu_val}%)")

with c2:
    st.subheader("Mapa de Riesgo (IA)")
    fig = px.scatter(df, x="Uso_CPU_Porcentaje", y="Uso_RAM_Porcentaje", color="Estado",
                     size='Uso_RAM_Porcentaje', hover_data=['ID_PC', 'Ticket_Usuario'],
                     color_discrete_map={'CRÍTICO': '#FF4B4B', 'ESTABLE': '#0068C9'})
    fig.add_trace(go.Scatter(x=pc_data['Uso_CPU_Porcentaje'], y=pc_data['Uso_RAM_Porcentaje'],
                             mode='markers', marker=dict(size=18, color='yellow', line=dict(width=2, color='black')),
                             name="Seleccionado"))
    st.plotly_chart(fig, use_container_width=True)

# 6. EVALUACIÓN DE ASERTIVIDAD (NUEVO)
st.divider()
st.subheader("📊 Reporte de Asertividad del Modelo (IA)")
st.write("Visualización de la Matriz de Confusión para validar el rendimiento predictivo:")

# Asumiendo que en tu CSV tienes una columna 'Clase_Real' (ej: 'ESTABLE', 'CRÍTICO')
if 'Clase_Real' in df.columns:
    cm = confusion_matrix(df['Clase_Real'], df['Estado'])
    fig_cm, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['CRÍTICO', 'ESTABLE'], yticklabels=['CRÍTICO', 'ESTABLE'])
    plt.xlabel('Predicción IA')
    plt.ylabel('Realidad')
    st.pyplot(fig_cm)
else:
    st.warning("Para ver la matriz de asertividad, agrega la columna 'Clase_Real' a tu CSV.")

st.subheader("Inventario Técnico Completo")
st.dataframe(df, use_container_width=True)
