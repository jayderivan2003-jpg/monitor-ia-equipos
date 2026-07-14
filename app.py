import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Monitor IA - Gestión Avanzada")

st.title("🖥️ AI-FleetMonitor Pro: Diagnóstico de Hardware")

# 1. Cargar datos
@st.cache_data
def cargar_datos():
    return pd.read_csv('BASEDEDATOSPROYECTO.csv')

df = cargar_datos()

# 2. IA: Detección de Anomalías
features = ['Uso_CPU_Porcentaje', 'Uso_RAM_Porcentaje', 'CPU_Normalizadontajercentaje']
model = IsolationForest(contamination=0.1, random_state=42)
df['Riesgo_IA'] = model.fit_predict(df[features])
df['Estado'] = df['Riesgo_IA'].apply(lambda x: 'CRÍTICO' if x == -1 else 'ESTABLE')

# 3. Sidebar: Análisis 1x1 Dinámico
st.sidebar.header("🔍 Diagnóstico Individual")
pc_seleccionado = st.sidebar.selectbox("Filtrar Equipo ID:", df['ID_PC'].unique())
pc_data = df[df['ID_PC'] == pc_seleccionado]

# 4. Dashboard Superior: Indicadores (KPIs)
col1, col2, col3 = st.columns(3)
col1.metric("Equipos Monitoreados", len(df))
col2.metric("Incidentes Detectados", df[df['Estado'] == 'CRÍTICO'].shape[0], delta_color="inverse")
col3.metric("Promedio CPU Flota", f"{df['Uso_CPU_Porcentaje'].mean():.1f}%")

# 5. Dashboard Principal
c1, c2 = st.columns([1, 2])

with c1:
    st.subheader(f"Info: {pc_seleccionado}")
    st.write(pc_data[['Modelo', 'Ticket_Usuario', 'Estado']])
    
    st.subheader("⚠️ Alerta de Picos")
    if pc_data['Uso_CPU_Porcentaje'].values[0] > 80:
        st.error("¡ALERTA: Pico de CPU detectado!")
    else:
        st.success("Rendimiento dentro de parámetros.")

with c2:
    st.subheader("Análisis de Riesgo Dinámico")
    # Gráfico que resalta el equipo seleccionado
    fig = px.scatter(df, x="Uso_CPU_Porcentaje", y="Uso_RAM_Porcentaje", color="Estado",
                     size='Uso_RAM_Porcentaje', hover_data=['ID_PC', 'Ticket_Usuario'],
                     color_discrete_map={'CRÍTICO': '#FF4B4B', 'ESTABLE': '#0068C9'})
    
    # Resaltar el seleccionado
    fig.add_trace(go.Scatter(x=pc_data['Uso_CPU_Porcentaje'], y=pc_data['Uso_RAM_Porcentaje'],
                             mode='markers', marker=dict(size=15, color='yellow', line=dict(width=2, color='black')),
                             name="Equipo Seleccionado"))
    
    st.plotly_chart(fig, use_container_width=True)

# 6. Inventario Completo
st.subheader("Histórico de Eventos")
st.dataframe(df, use_container_width=True)
