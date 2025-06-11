"""
Dashboard interactivo para análisis del índice VIX.
Ejecutar con: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import pickle

# Configuración de la página
st.set_page_config(
    page_title="VIX Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class VIXDashboard:
    """
    Clase principal del dashboard del VIX
    """
    
    def __init__(self):
        self.df_raw = None
        self.df_enriched = None
        self.model_info = None
        self.predictions = None
    
    def load_data(self):
        """Carga los datos del VIX"""
        try:
            # Cargar datos crudos
            if os.path.exists("src/piv_2025_1_2/static/data/vix_data.csv"):
                self.df_raw = pd.read_csv("src/piv_2025_1_2/static/data/vix_data.csv", decimal=',')
                
                # Conversión de fecha más flexible
                if 'Date' in self.df_raw.columns:
                    try:
                        # Intentar conversión automática primero
                        self.df_raw['Date'] = pd.to_datetime(self.df_raw['Date'])
                        st.success("✅ Conversión de fecha automática exitosa")
                    except Exception as date_error:
                        st.warning(f"⚠️ Error en conversión automática: {date_error}")
                        try:
                            # Intentar con formato ISO8601
                            self.df_raw['Date'] = pd.to_datetime(self.df_raw['Date'], format='ISO8601')
                            st.success("✅ Conversión de fecha ISO8601 exitosa")
                        except Exception as iso_error:
                            st.warning(f"⚠️ Error ISO8601: {iso_error}")
                            try:
                                # Intentar con formato mixto
                                self.df_raw['Date'] = pd.to_datetime(self.df_raw['Date'], format='mixed')
                                st.success("✅ Conversión de fecha mixta exitosa")
                            except Exception as mixed_error:
                                st.error(f"❌ Todos los formatos de fecha fallaron: {mixed_error}")
                                # Mostrar muestra de datos para debugging
                                st.write("🔍 Muestra de datos de fecha:")
                                st.write(self.df_raw['Date'].head())
                                return False
            else:
                st.error("❌ No se encontró el archivo vix_data.csv")
                return False
            
            # Cargar datos enriquecidos
            if os.path.exists("src/piv_2025_1_2/static/data/vix_data_enricher.csv"):
                self.df_enriched = pd.read_csv("src/piv_2025_1_2/static/data/vix_data_enricher.csv", decimal=',')
                
                # Conversión de fecha para datos enriquecidos
                if 'Date' in self.df_enriched.columns:
                    try:
                        self.df_enriched['Date'] = pd.to_datetime(self.df_enriched['Date'])
                        st.success("✅ Conversión de fecha en datos enriquecidos exitosa")
                    except:
                        try:
                            self.df_enriched['Date'] = pd.to_datetime(self.df_enriched['Date'], format='ISO8601')
                            st.success("✅ Conversión ISO8601 en datos enriquecidos exitosa")
                        except:
                            try:
                                self.df_enriched['Date'] = pd.to_datetime(self.df_enriched['Date'], format='mixed')
                                st.success("✅ Conversión mixta en datos enriquecidos exitosa")
                            except:
                                st.warning("⚠️ Error en conversión de fecha de datos enriquecidos")
            else:
                # Si no hay datos enriquecidos, usar los datos crudos
                if self.df_raw is not None:
                    st.info("ℹ️ Creando datos enriquecidos desde datos crudos...")
                    self.df_enriched = self.df_raw.copy()
                    
                    # Agregar indicadores básicos
                    if 'Close' in self.df_enriched.columns:
                        self.df_enriched['Daily_Change'] = self.df_enriched['Close'].pct_change() * 100
                        self.df_enriched['MA20'] = self.df_enriched['Close'].rolling(window=20).mean()
                        self.df_enriched['MA50'] = self.df_enriched['Close'].rolling(window=50).mean()
                        
                        # RSI
                        delta = self.df_enriched['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        self.df_enriched['RSI'] = 100 - (100 / (1 + rs))
                        
                        # VIX Level
                        self.df_enriched['VIX_Level'] = pd.cut(
                            self.df_enriched['Close'], 
                            bins=[0, 20, 30, 100], 
                            labels=['Low', 'Normal', 'High']
                        )
                        
                        # Volatilidad
                        self.df_enriched['Volatility_20d'] = self.df_enriched['Close'].rolling(window=20).std()
                        
                        # Columnas adicionales para gráficos
                        self.df_enriched['MACD'] = 0  # Placeholder
                        self.df_enriched['ATR'] = self.df_enriched['Close'].rolling(window=14).std()
                        self.df_enriched['BB_Position'] = 0.5  # Placeholder
                        
                        # Día de la semana
                        if 'Date' in self.df_enriched.columns:
                            self.df_enriched['WeekdayName'] = self.df_enriched['Date'].dt.day_name()
                        
                        st.success("✅ Indicadores técnicos agregados")
            
            # Cargar información del modelo
            if os.path.exists("src/piv_2025_1_2/static/models/model_metadata.json"):
                import json
                with open("src/piv_2025_1_2/static/models/model_metadata.json", 'r') as f:
                    self.model_info = json.load(f)
                st.success("✅ Metadata del modelo cargada")
            
            # Verificar que tenemos datos
            if self.df_enriched is not None and len(self.df_enriched) > 0:
                st.success(f"🎉 Datos cargados exitosamente: {len(self.df_enriched)} registros")
                st.write(f"📅 Periodo: {self.df_enriched['Date'].min()} - {self.df_enriched['Date'].max()}")
                st.write(f"🔤 Columnas: {list(self.df_enriched.columns)}")
                return True
            else:
                st.error("❌ No se pudieron cargar datos válidos")
                return False
            
        except Exception as e:
            st.error(f"Error al cargar datos: {e}")
            import traceback
            st.error(f"Traceback completo: {traceback.format_exc()}")
            return False
    
    def calculate_kpis(self):
        """Calcula los KPIs principales"""
        if self.df_enriched is None:
            return None
        
        df = self.df_enriched.copy()
        
        # KPI 1: Valor actual del VIX
        current_vix = df['Close'].iloc[-1]
        
        # KPI 2: Cambio diario
        daily_change = df['Daily_Change'].iloc[-1]
        
        # KPI 3: Volatilidad promedio (20 días)
        avg_volatility = df['Volatility_20d'].iloc[-1]
        
        # KPI 4: Media móvil 20 días
        ma20 = df['MA20'].iloc[-1]
        
        # KPI 5: RSI actual
        current_rsi = df['RSI'].iloc[-1]
        
        # KPI 6: Nivel del VIX
        vix_level = df['VIX_Level'].iloc[-1]
        
        # KPI 7: Retorno acumulado (último mes)
        monthly_return = (df['Close'].iloc[-1] / df['Close'].iloc[-22] - 1) * 100 if len(df) >= 22 else 0
        
        # KPI 8: Desviación estándar (último mes)
        monthly_std = df['Close'].tail(22).std()
        
        return {
            'current_vix': current_vix,
            'daily_change': daily_change,
            'avg_volatility': avg_volatility,
            'ma20': ma20,
            'current_rsi': current_rsi,
            'vix_level': vix_level,
            'monthly_return': monthly_return,
            'monthly_std': monthly_std
        }
    
    def create_main_chart(self):
        """Crea el gráfico principal del VIX"""
        if self.df_enriched is None:
            return None
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('VIX y Medias Móviles', 'RSI', 'Volatilidad'),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        df = self.df_enriched.tail(252)  # Último año
        
        # Gráfico 1: VIX y medias móviles
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['Close'], name='VIX', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['MA20'], name='MA20', line=dict(color='red', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['MA50'], name='MA50', line=dict(color='green', width=1)),
            row=1, col=1
        )
        
        # Gráfico 2: RSI
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        # Líneas de sobrecompra y sobreventa
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # Gráfico 3: Volatilidad
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['Volatility_20d'], name='Volatilidad 20d', 
                      line=dict(color='orange')),
            row=3, col=1
        )
        
        fig.update_layout(height=800, showlegend=True, title_text="Análisis Técnico del VIX")
        fig.update_xaxes(title_text="Fecha", row=3, col=1)
        fig.update_yaxes(title_text="VIX", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="Volatilidad", row=3, col=1)
        
        return fig
    
    def create_distribution_chart(self):
        """Crea gráfico de distribución del VIX"""
        if self.df_enriched is None:
            return None
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Distribución del VIX', 'VIX por Día de la Semana'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        df = self.df_enriched.copy()
        
        # Histograma del VIX
        fig.add_trace(
            go.Histogram(x=df['Close'], nbinsx=30, name='Distribución VIX'),
            row=1, col=1
        )
        
        # Box plot por día de la semana
        fig.add_trace(
            go.Box(x=df['WeekdayName'], y=df['Close'], name='VIX por Día'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        return fig
    
    def create_correlation_heatmap(self):
        """Crea mapa de calor de correlaciones"""
        if self.df_enriched is None:
            return None
        
        # Seleccionar variables numéricas relevantes
        numeric_cols = ['Close', 'Daily_Change', 'MA20', 'MA50', 'Volatility_20d', 
                       'RSI', 'MACD', 'ATR', 'BB_Position']
        
        # Filtrar columnas que existen
        available_cols = [col for col in numeric_cols if col in self.df_enriched.columns]
        
        if len(available_cols) < 2:
            return None
        
        corr_matrix = self.df_enriched[available_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title="Matriz de Correlación - Indicadores del VIX",
            height=500
        )
        
        return fig
    
    def create_prediction_chart(self):
        """Crea gráfico con predicciones del modelo"""
        if self.df_enriched is None or not os.path.exists("src/piv_2025_1_2/static/models/model.pkl"):
            return None
        
        try:
            # Cargar modelo y hacer predicciones
            from modeller import Modeller
            modeller = Modeller()
            
            # Obtener los últimos 30 días para mostrar contexto
            recent_data = self.df_enriched.tail(30).copy()
            predictions = modeller.predecir(recent_data)
            
            # Crear fechas futuras para las predicciones
            last_date = recent_data['Date'].iloc[-1]
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=len(predictions),
                freq='D'
            )
            
            fig = go.Figure()
            
            # Datos históricos
            fig.add_trace(
                go.Scatter(
                    x=recent_data['Date'], 
                    y=recent_data['Close'],
                    name='VIX Histórico',
                    line=dict(color='blue')
                )
            )
            
            # Predicciones
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=predictions,
                    name='Predicciones',
                    line=dict(color='red', dash='dash')
                )
            )
            
            fig.update_layout(
                title="VIX: Histórico vs Predicciones",
                xaxis_title="Fecha",
                yaxis_title="VIX",
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.warning(f"No se pudieron generar predicciones: {e}")
            return None
    
    def render_sidebar(self):
        """Renderiza la barra lateral"""
        st.sidebar.title("📊 VIX Analytics")
        st.sidebar.markdown("---")
        
        # Información del dataset
        if self.df_enriched is not None:
            st.sidebar.subheader("📈 Información del Dataset")
            st.sidebar.write(f"**Registros:** {len(self.df_enriched):,}")
            st.sidebar.write(f"**Periodo:** {self.df_enriched['Date'].min().date()} - {self.df_enriched['Date'].max().date()}")
            st.sidebar.write(f"**Features:** {len(self.df_enriched.columns)}")
        
        st.sidebar.markdown("---")
        
        # Información del modelo
        if self.model_info:
            st.sidebar.subheader("🤖 Modelo Predictivo")
            st.sidebar.write(f"**Tipo:** {self.model_info.get('model_type', 'N/A')}")
            st.sidebar.write(f"**RMSE:** {self.model_info.get('metrics', {}).get('test_rmse', 0):.4f}")
            st.sidebar.write(f"**R²:** {self.model_info.get('metrics', {}).get('test_r2', 0):.4f}")
            st.sidebar.write(f"**Entrenado:** {self.model_info.get('training_date', 'N/A')[:10]}")
        
        st.sidebar.markdown("---")
        
        # Controles
        st.sidebar.subheader("⚙️ Controles")
        
        if st.sidebar.button("🔄 Recargar Datos"):
            st.experimental_rerun()
        
        # Filtros de fecha
        if self.df_enriched is not None:
            date_range = st.sidebar.slider(
                "Seleccionar periodo (días)",
                min_value=30,
                max_value=len(self.df_enriched),
                value=252,
                step=30
            )
            return date_range
        
        return 252
    
    def render_kpi_cards(self, kpis):
        """Renderiza las tarjetas de KPIs"""
        if kpis is None:
            return
        
        st.subheader("📊 Indicadores Clave de Rendimiento (KPIs)")
        
        # Primera fila de KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta_color = "normal" if kpis['daily_change'] == 0 else ("inverse" if kpis['daily_change'] > 0 else "normal")
            st.metric(
                label="VIX Actual",
                value=f"{kpis['current_vix']:.2f}",
                delta=f"{kpis['daily_change']:.2f}%",
                delta_color=delta_color
            )
        
        with col2:
            st.metric(
                label="Media Móvil 20d",
                value=f"{kpis['ma20']:.2f}",
                delta=f"{kpis['current_vix'] - kpis['ma20']:.2f}"
            )
        
        with col3:
            rsi_color = "normal"
            if kpis['current_rsi'] > 70:
                rsi_color = "inverse"
            elif kpis['current_rsi'] < 30:
                rsi_color = "normal"
            
            st.metric(
                label="RSI",
                value=f"{kpis['current_rsi']:.1f}",
                delta="Sobrecompra" if kpis['current_rsi'] > 70 else ("Sobreventa" if kpis['current_rsi'] < 30 else "Normal")
            )
        
        with col4:
            level_emoji = "🟢" if kpis['vix_level'] == "Low" else ("🟡" if kpis['vix_level'] == "Normal" else "🔴")
            st.metric(
                label="Nivel VIX",
                value=f"{level_emoji} {kpis['vix_level']}",
                delta=f"Vol: {kpis['avg_volatility']:.2f}"
            )
        
        # Segunda fila de KPIs
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric(
                label="Retorno Mensual",
                value=f"{kpis['monthly_return']:.2f}%",
                delta_color="inverse" if kpis['monthly_return'] > 0 else "normal"
            )
        
        with col6:
            st.metric(
                label="Volatilidad 20d",
                value=f"{kpis['avg_volatility']:.2f}",
                delta_color="inverse"
            )
        
        with col7:
            st.metric(
                label="Desv. Estándar Mensual",
                value=f"{kpis['monthly_std']:.2f}",
                delta_color="inverse"
            )
        
        with col8:
            # Calcular número de días de datos disponibles
            days_available = len(self.df_enriched) if self.df_enriched is not None else 0
            st.metric(
                label="Datos Disponibles",
                value=f"{days_available} días",
                delta=f"Última actualización: {datetime.now().strftime('%H:%M')}"
            )
    
    def run(self):
        """Ejecuta el dashboard principal"""
        st.title("📈 VIX Analytics Dashboard")
        st.markdown("### Análisis Interactivo del Índice de Volatilidad CBOE")
        
        # Cargar datos
        if not self.load_data():
            st.error("⚠️ No se pudieron cargar los datos. Ejecute primero el pipeline ETL.")
            st.stop()
        
        # Renderizar sidebar
        date_range = self.render_sidebar()
        
        # Filtrar datos según el rango seleccionado
        if self.df_enriched is not None:
            self.df_enriched = self.df_enriched.tail(date_range)
        
        # Calcular KPIs
        kpis = self.calculate_kpis()
        
        # Renderizar KPIs
        self.render_kpi_cards(kpis)
        
        st.markdown("---")
        
        # Gráficos principales
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Análisis Técnico")
            main_chart = self.create_main_chart()
            if main_chart:
                st.plotly_chart(main_chart, use_container_width=True)
        
        with col2:
            st.subheader("📈 Distribución y Patrones")
            dist_chart = self.create_distribution_chart()
            if dist_chart:
                st.plotly_chart(dist_chart, use_container_width=True)
        
        # Segunda fila de gráficos
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("🔗 Correlaciones")
            corr_chart = self.create_correlation_heatmap()
            if corr_chart:
                st.plotly_chart(corr_chart, use_container_width=True)
        
        with col4:
            st.subheader("🔮 Predicciones")
            pred_chart = self.create_prediction_chart()
            if pred_chart:
                st.plotly_chart(pred_chart, use_container_width=True)
            else:
                st.info("💡 Entrene un modelo primero para ver predicciones")
        
        # Tabla de datos recientes
        st.markdown("---")
        st.subheader("📋 Datos Recientes")
        
        if self.df_enriched is not None:
            # Mostrar las últimas 10 filas con columnas seleccionadas
            recent_data = self.df_enriched[['Date', 'Close', 'Daily_Change', 'MA20', 'RSI', 'VIX_Level']].tail(10)
            st.dataframe(recent_data, use_container_width=True)
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: gray; font-size: 0.8em;'>
            📊 VIX Analytics Dashboard | Proyecto Integrador V | 
            Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            """,
            unsafe_allow_html=True
        )

# Función principal para ejecutar el dashboard
def main():
    """Función principal del dashboard"""
    dashboard = VIXDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()