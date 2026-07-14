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

# 5. DASHBOARD PRINCIPAL (Gráficos y Detalles)
c1, c2 = st.columns([1, 2])

with c1:
    st.subheader(f"Detalles: {pc_seleccionado}")
    st.write(pc_data.T)
    
    # Lógica de Diagnóstico de Picos (AÑADIDO)
    st.subheader("⚠️ Diagnóstico de Picos")
    cpu_val = pc_data['Uso_CPU_Porcentaje'].values[0]
    if cpu_val > 80:
        st.error(f"¡ALERTA: Pico de CPU detectado! ({cpu_val}%)")
    elif cpu_val > 60:
        st.warning(f"Atención: CPU elevado ({cpu_val}%)")
    else:
        st.success(f"CPU Estable ({cpu_val}%)")

with c2:
    st.subheader("Mapa de Riesgo (IA)")
    fig = px.scatter(df, x="Uso_CPU_Porcentaje", y="Uso_RAM_Porcentaje", color="Estado",
                     size='Uso_RAM_Porcentaje', hover_data=['ID_PC', 'Ticket_Usuario'],
                     color_discrete_map={'CRÍTICO': '#FF4B4B', 'ESTABLE': '#0068C9'})
    
    # Resaltar equipo seleccionado
    fig.add_trace(go.Scatter(x=pc_data['Uso_CPU_Porcentaje'], y=pc_data['Uso_RAM_Porcentaje'],
                             mode='markers', marker=dict(size=18, color='yellow', line=dict(width=2, color='black')),
                             name="Seleccionado"))
    st.plotly_chart(fig, use_container_width=True)

# 6. INVENTARIO COMPLETO (Se mantiene igual)
st.subheader("Inventario Técnico Completo")
st.dataframe(df, use_container_width=True)
