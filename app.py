import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🖥️ Sistema Integral de Monitoreo IA")

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

# 3. Interfaz
st.subheader("Análisis de Rendimiento")
col1, col2 = st.columns([1, 2])
with col1:
    tipo = st.selectbox("Filtrar por incidencia:", df['Ticket_Usuario'].unique())
    st.write(df[df['Ticket_Usuario'] == tipo])
with col2:
    fig = px.scatter(df, x="Uso_CPU_Porcentaje", y="Uso_RAM_Porcentaje", color="Estado",
                     color_discrete_map={'CRÍTICO': 'red', 'ESTABLE': 'blue'})
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Inventario Completo")
st.dataframe(df, use_container_width=True)
