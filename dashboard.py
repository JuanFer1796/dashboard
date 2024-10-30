import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="Dashboard de Control", layout="wide", initial_sidebar_state="expanded")

# Funciones de utilidad
def calcular_tendencia(serie):
    """Calcula la tendencia lineal de una serie temporal."""
    x = np.arange(len(serie))
    z = np.polyfit(x, serie, 1)
    return z[0]  # Retorna la pendiente

def calcular_estacionalidad(df, columna):
    """Calcula la estacionalidad mensual."""
    return df.groupby(df['Fecha'].dt.month)[columna].mean()

def generar_prediccion_prophet(df, periodo_futuro=30):
    """Genera predicciones usando Prophet."""
    df_prophet = df[['Fecha', 'Cantidad']].rename(columns={'Fecha': 'ds', 'Cantidad': 'y'})
    modelo = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    modelo.fit(df_prophet)
    futuro = modelo.make_future_dataframe(periods=periodo_futuro)
    forecast = modelo.predict(futuro)
    return forecast

# Estilo y configuración
st.title("🎯 Dashboard de Control de KPIs")
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración")

# Subir archivo CSV
uploaded_file = st.sidebar.file_uploader("📂 Cargar archivo CSV", type="csv")

if uploaded_file:
    # Cargar y preparar datos
    try:
        df = pd.read_csv(uploaded_file)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df['Año-Mes'] = df['Fecha'].dt.to_period('M')
        df['Año'] = df['Fecha'].dt.year
        df['Mes'] = df['Fecha'].dt.month
        
        # Validación de columnas requeridas
        required_columns = ['Fecha', 'Cantidad', 'Objetivo', 'Categoría']
        if not all(col in df.columns for col in required_columns):
            st.error("El archivo CSV debe contener las columnas: Fecha, Cantidad, Objetivo, Categoría")
            st.stop()

        # Pestañas principales
        tab1, tab2, tab3, tab4 = st.tabs(["📊 KPIs Principales", "📈 Análisis Temporal", "🎯 Objetivos", "🔮 Predicciones"])

        with tab1:
            # KPIs principales en tres columnas
            col1, col2, col3 = st.columns(3)
            
            # Métricas principales
            total_cantidad = df['Cantidad'].sum()
            promedio_mensual = df.groupby('Año-Mes')['Cantidad'].mean().mean()
            progreso_objetivo = (df['Cantidad'].sum() / df['Objetivo'].sum()) * 100
            
            # Cálculo de tendencias
            tendencia = calcular_tendencia(df['Cantidad'])
            delta_color = "normal" if tendencia > 0 else "inverse"
            
          with col1:
            st.metric(
                "💰 Total Acumulado",
                f"{total_cantidad:,.2f}",
                f"Tendencia: {tendencia:+.2%}",
                delta_color="normal" if tendencia >= 0 else "inverse"
            )
        
        with col2:
            st.metric(
                "📊 Promedio Mensual",
                f"{promedio_mensual:,.2f}",
                f"vs Objetivo: {(promedio_mensual / df['Objetivo'].mean() - 1):+.2%}",
                delta_color="normal" if promedio_mensual >= df['Objetivo'].mean() else "inverse"
            )
        
        with col3:
            st.metric(
                "🎯 Progreso vs Objetivo",
                f"{progreso_objetivo:.1f}%",
                f"{progreso_objetivo - 100:+.1f}% vs 100%",
                delta_color="normal" if progreso_objetivo >= 100 else "inverse"
            )
            # Gráfico de desempeño por categoría
            st.subheader("📊 Desempeño por Categoría")
            fig_cat = px.bar(
                df.groupby('Categoría').agg({
                    'Cantidad': 'sum',
                    'Objetivo': 'sum'
                }).reset_index(),
                x='Categoría',
                y=['Cantidad', 'Objetivo'],
                barmode='group',
                title="Cantidad vs Objetivo por Categoría"
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        with tab2:
            st.subheader("📈 Análisis Temporal")
            
            # Serie temporal con área sombreada
            fig_tiempo = go.Figure()
            fig_tiempo.add_trace(go.Scatter(
                x=df['Fecha'],
                y=df['Cantidad'],
                fill='tonexty',
                name='Cantidad',
                line=dict(color='rgb(26, 118, 255)')
            ))
            fig_tiempo.add_trace(go.Scatter(
                x=df['Fecha'],
                y=df['Objetivo'],
                name='Objetivo',
                line=dict(color='red', dash='dash')
            ))
            fig_tiempo.update_layout(
                title='Evolución Temporal de Cantidad vs Objetivo',
                xaxis_title='Fecha',
                yaxis_title='Valor'
            )
            st.plotly_chart(fig_tiempo, use_container_width=True)

            # Análisis de estacionalidad
            st.subheader("📅 Patrón Estacional")
            estacionalidad = calcular_estacionalidad(df, 'Cantidad')
            fig_estacional = px.line(
                x=estacionalidad.index,
                y=estacionalidad.values,
                labels={'x': 'Mes', 'y': 'Promedio'},
                title="Patrón Estacional Mensual"
            )
            st.plotly_chart(fig_estacional, use_container_width=True)

        with tab3:
            st.subheader("🎯 Seguimiento de Objetivos")
            
            # Heatmap de cumplimiento mensual
            df_mensual = df.groupby(['Año', 'Mes']).agg({
                'Cantidad': 'sum',
                'Objetivo': 'sum'
            }).reset_index()
            
            df_mensual['Cumplimiento'] = (df_mensual['Cantidad'] / df_mensual['Objetivo']) * 100
            
            # Crear matriz para el heatmap
            años = df_mensual['Año'].unique()
            meses = range(1, 13)
            matriz_cumplimiento = np.zeros((len(años), 12))
            
            for idx, año in enumerate(años):
                datos_año = df_mensual[df_mensual['Año'] == año]
                for mes in meses:
                    dato_mes = datos_año[datos_año['Mes'] == mes]
                    if not dato_mes.empty:
                        matriz_cumplimiento[idx, mes-1] = dato_mes['Cumplimiento'].iloc[0]
                    else:
                        matriz_cumplimiento[idx, mes-1] = np.nan
            
            fig_heat = px.imshow(
                matriz_cumplimiento,
                labels=dict(x="Mes", y="Año", color="% Cumplimiento"),
                x=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'],
                y=años,
                aspect="auto",
                color_continuous_scale="RdYlBu_r"
            )
            fig_heat.update_layout(
                title="Heatmap de Cumplimiento Mensual (%)",
                xaxis_title="Mes",
                yaxis_title="Año"
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        with tab4:
            st.subheader("🔮 Predicciones")
            
            # Configuración de predicción
            periodo_prediccion = st.slider(
                "Seleccione período de predicción (días)",
                min_value=7,
                max_value=90,
                value=30
            )
            
            # Generar predicciones
            try:
                forecast = generar_prediccion_prophet(df, periodo_prediccion)
                
                fig_pred = go.Figure()
                # Datos históricos
                fig_pred.add_trace(go.Scatter(
                    x=df['Fecha'],
                    y=df['Cantidad'],
                    name='Histórico',
                    line=dict(color='blue')
                ))
                # Predicción
                fig_pred.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    name='Predicción',
                    line=dict(color='red', dash='dash')
                ))
                # Intervalo de confianza
                fig_pred.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_upper'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(255,0,0,0)',
                    showlegend=False
                ))
                fig_pred.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_lower'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(255,0,0,0)',
                    name='Intervalo de Confianza'
                ))
                
                fig_pred.update_layout(
                    title="Predicción de Valores Futuros",
                    xaxis_title="Fecha",
                    yaxis_title="Valor"
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Métricas de precisión
                mae = mean_absolute_error(df['Cantidad'], forecast['yhat'][:len(df)])
                rmse = np.sqrt(mean_squared_error(df['Cantidad'], forecast['yhat'][:len(df)]))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("MAE (Error Absoluto Medio)", f"{mae:.2f}")
                with col2:
                    st.metric("RMSE (Error Cuadrático Medio)", f"{rmse:.2f}")
                
            except Exception as e:
                st.error(f"Error en la generación de predicciones: {str(e)}")

    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")
        
else:
    # Mensaje informativo cuando no hay archivo
    st.info("""
        👆 Por favor, sube un archivo CSV con las siguientes columnas:
        - Fecha: Fecha de la medición (YYYY-MM-DD)
        - Cantidad: Valor numérico de la métrica
        - Objetivo: Valor objetivo
        - Categoría: Categoría o segmento
    """)
    
    # Ejemplo de formato
    st.markdown("""
    ### Ejemplo de formato CSV:
    ```
    Fecha,Cantidad,Objetivo,Categoría
    2024-01-01,100,120,Producto A
    2024-01-02,95,120,Producto A
    2024-01-01,80,90,Producto B
    ```
    """)
