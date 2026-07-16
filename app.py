Python
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
    df = pd.read_csv('BASEDEDATOSPROYECTO.csv')
    # PASO 2: Generar la base de verdad automáticamente para la Matriz
    # Si hay un Ticket_Usuario, lo marcamos como REALMENTE CRÍTICO
    df['Clase_Real'] = df['Ticket_Usuario'].apply(lambda x: 'CRÍTICO' if pd.notnull(x) else 'ESTABLE')
    return df

df = cargar_datos()

# 2. IA: Detección de Anomalías (Isolation Forest)
features = ['Uso_CPU_Porcentaje', 'Uso_RAM_Porcentaje', 'CPU_Normalizadontajercentaje']
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

# 6. EVALUACIÓN DE ASERTIVIDAD (Ahora funcionará porque creamos Clase_Real en el Paso 1)
st.divider()
st.subheader("📊 Reporte de Asertividad del Modelo (Matriz de Confusión)")
st.write("Comparación entre la predicción de la IA y los reportes reales de tickets.")

cm = confusion_matrix(df['Clase_Real'], df['Estado'], labels=['CRÍTICO', 'ESTABLE'])
fig_cm, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
            xticklabels=['CRÍTICO', 'ESTABLE'], yticklabels=['CRÍTICO', 'ESTABLE'])
plt.xlabel('Predicción IA')
plt.ylabel('Realidad (Tickets)')
st.pyplot(fig_cm)

st.subheader("Inventario Técnico Completo")
st.dataframe(df, use_container_width=True)
