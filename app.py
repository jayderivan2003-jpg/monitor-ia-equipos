import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(layout="wide", page_title="Monitor IA Equipos")

st.title("🖥️ AI-FleetMonitor Pro: Diagnóstico Técnico")

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

# 3. Sidebar: Análisis 1x1
st.sidebar.header("🔍 Análisis Detallado")
pc_seleccionado = st.sidebar.selectbox("Seleccionar ID de PC:", df['ID_PC'].unique())
pc_data = df[df['ID_PC'] == pc_seleccionado]

st.sidebar.subheader(f"Datos de {pc_seleccionado}")
st.sidebar.write(pc_data.T) # Transpuesta para ver mejor los datos

# 4. Dashboard Principal
col1, col2 = st.columns(2)

with col1:
    st.subheader("Estadísticas de la Flota")
    total_criticos = df[df['Estado'] == 'CRÍTICO'].shape[0]
    st.metric("Equipos en Riesgo (Críticos)", total_criticos, delta=f"-{total_criticos}", delta_color="inverse")
    st.dataframe(df[['ID_PC', 'Modelo', 'Ticket_Usuario', 'Estado']], use_container_width=True)

with col2:
    st.subheader("Análisis de Riesgo (IA)")
    fig = px.scatter(df, x="Uso_CPU_Porcentaje", y="Uso_RAM_Porcentaje", color="Estado",
                     hover_data=['ID_PC', 'Modelo'],
                     color_discrete_map={'CRÍTICO': 'red', 'ESTABLE': 'blue'})
    st.plotly_chart(fig, use_container_width=True)

# 5. Inventario Completo
st.subheader("Inventario Técnico Completo")
st.dataframe(df, use_container_width=True)
