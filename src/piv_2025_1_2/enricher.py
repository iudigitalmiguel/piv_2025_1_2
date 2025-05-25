"""
Módulo para enriquecer y calcular KPIs sobre datos del índice VIX.
Versión mejorada para Actividad 2 con fuentes adicionales y más transformaciones.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class Enricher:
    """
    Clase para calcular indicadores clave de rendimiento (KPIs) sobre datos del VIX.
    Versión mejorada con variables temporales e indicadores adicionales.
    """
    
    def __init__(self, logger=None):
        """
        Inicializa el procesador de enriquecimiento de datos.
        
        Args:
            logger (Logger, optional): Instancia de Logger para registrar eventos
        """
        self.logger = logger
        
        if logger:
            logger.info('Enricher', '__init__', 'Instancia de Enricher inicializada')
    
    def calcular_kpi(self, df):
        """
        Calcula KPIs avanzados para los datos del VIX incluyendo variables temporales
        e indicadores adicionales.
        
        Args:
            df (pandas.DataFrame): DataFrame con datos históricos del VIX
            
        Returns:
            pandas.DataFrame: DataFrame enriquecido con KPIs calculados
        """
        if df.empty:
            if self.logger:
                self.logger.warning('Enricher', 'calcular_kpi', 'DataFrame vacío, no se pueden calcular KPIs')
            return df
        
        if self.logger:
            self.logger.info('Enricher', 'calcular_kpi', 'Iniciando cálculo de KPIs avanzados')
        
        # Crear copia para no modificar el original
        enriched_df = df.copy()
        
        try:
            # Asegurar que Date es datetime
            enriched_df['Date'] = pd.to_datetime(enriched_df['Date'])
            
            # === VARIABLES TEMPORALES ===
            enriched_df = self._add_temporal_features(enriched_df)
            
            # === INDICADORES TÉCNICOS BÁSICOS ===
            enriched_df = self._add_basic_indicators(enriched_df)
            
            # === INDICADORES TÉCNICOS AVANZADOS ===
            enriched_df = self._add_advanced_indicators(enriched_df)
            
            # === VARIABLES DE CONTEXTO MACRO ===
            enriched_df = self._add_macro_features(enriched_df)
            
            # === FEATURES PARA MACHINE LEARNING ===
            enriched_df = self._add_ml_features(enriched_df)
            
            if self.logger:
                self.logger.info('Enricher', 'calcular_kpi', f'KPIs avanzados calculados: {len(enriched_df.columns)} columnas totales')
            
            return enriched_df
            
        except Exception as e:
            if self.logger:
                self.logger.error('Enricher', 'calcular_kpi', f'Error al calcular KPIs: {e}')
            return df
    
    def _add_temporal_features(self, df):
        """Agregar variables temporales"""
        if self.logger:
            self.logger.info('Enricher', '_add_temporal_features', 'Agregando variables temporales')
        
        # Variables temporales básicas
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Lunes, 6=Domingo
        df['DayOfMonth'] = df['Date'].dt.day
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        
        # Variables temporales categóricas
        df['WeekdayName'] = df['Date'].dt.day_name()
        df['MonthName'] = df['Date'].dt.month_name()
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Variables cíclicas (para capturar patrones estacionales)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        
        # Días especiales
        df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
        df['IsQuarterEnd'] = df['Date'].dt.is_quarter_end.astype(int)
        df['IsYearEnd'] = df['Date'].dt.is_year_end.astype(int)
        
        return df
    
    def _add_basic_indicators(self, df):
        """Agregar indicadores técnicos básicos"""
        if self.logger:
            self.logger.info('Enricher', '_add_basic_indicators', 'Agregando indicadores técnicos básicos')
        
        # Variación diaria
        df['Daily_Change'] = df['Close'].pct_change() * 100
        df['Daily_Change_Abs'] = df['Daily_Change'].abs()
        
        # Medias móviles
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # Señales de cruces de medias móviles
        df['MA5_above_MA20'] = (df['MA5'] > df['MA20']).astype(int)
        df['MA20_above_MA50'] = (df['MA20'] > df['MA50']).astype(int)
        
        # Volatilidad
        df['Volatility_5d'] = df['Close'].rolling(window=5).std()
        df['Volatility_10d'] = df['Close'].rolling(window=10).std()
        df['Volatility_20d'] = df['Close'].rolling(window=20).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _add_advanced_indicators(self, df):
        """Agregar indicadores técnicos avanzados"""
        if self.logger:
            self.logger.info('Enricher', '_add_advanced_indicators', 'Agregando indicadores técnicos avanzados')
        
        # Bandas de Bollinger
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Average True Range (ATR)
        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = np.abs(df['High'] - df['Close'].shift())
        df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift())
        df['True_Range'] = np.maximum(df['High_Low'], 
                                    np.maximum(df['High_Close'], df['Low_Close']))
        df['ATR'] = df['True_Range'].rolling(window=14).mean()
        
        # Momentum indicators
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['Rate_of_Change'] = df['Close'].pct_change(periods=10) * 100
        
        # Support and Resistance levels
        df['Resistance_20'] = df['High'].rolling(window=20).max()
        df['Support_20'] = df['Low'].rolling(window=20).min()
        df['Support_Resistance_Ratio'] = (df['Close'] - df['Support_20']) / (df['Resistance_20'] - df['Support_20'])
        
        return df
    
    def _add_macro_features(self, df):
        """Agregar variables de contexto macroeconómico"""
        if self.logger:
            self.logger.info('Enricher', '_add_macro_features', 'Agregando variables macro (S&P 500)')
        
        try:
            # Obtener datos del S&P 500 para contexto
            start_date = df['Date'].min() - timedelta(days=30)
            end_date = df['Date'].max() + timedelta(days=1)
            
            sp500 = yf.Ticker("^GSPC")
            sp500_data = sp500.history(start=start_date, end=end_date)
            sp500_data = sp500_data.reset_index()
            sp500_data['Date'] = pd.to_datetime(sp500_data['Date']).dt.date
            
            # Preparar datos del VIX para merge
            df['Date_merge'] = pd.to_datetime(df['Date']).dt.date
            
            # Merge con datos del S&P 500
            sp500_relevant = sp500_data[['Date', 'Close']].rename(columns={'Close': 'SP500_Close'})
            sp500_relevant['Date'] = pd.to_datetime(sp500_relevant['Date']).dt.date
            
            df = df.merge(sp500_relevant, left_on='Date_merge', right_on='Date', how='left', suffixes=('', '_SP500'))
            
            # Calcular correlación VIX-S&P500
            df['SP500_Daily_Change'] = df['SP500_Close'].pct_change() * 100
            df['VIX_SP500_Correlation'] = df['Daily_Change'].rolling(window=20).corr(df['SP500_Daily_Change'])
            
            # Ratio VIX/S&P500 volatilidad
            df['SP500_Volatility'] = df['SP500_Close'].rolling(window=20).std()
            df['VIX_SP500_Vol_Ratio'] = df['Volatility_20d'] / df['SP500_Volatility']
            
            # Eliminar columnas auxiliares
            df = df.drop(['Date_merge', 'Date_SP500'], axis=1)
            
            if self.logger:
                self.logger.info('Enricher', '_add_macro_features', 'Variables macro agregadas exitosamente')
                
        except Exception as e:
            if self.logger:
                self.logger.warning('Enricher', '_add_macro_features', f'No se pudieron agregar variables macro: {e}')
            # Crear columnas vacías para mantener consistencia
            df['SP500_Close'] = np.nan
            df['SP500_Daily_Change'] = np.nan
            df['VIX_SP500_Correlation'] = np.nan
            df['SP500_Volatility'] = np.nan
            df['VIX_SP500_Vol_Ratio'] = np.nan
        
        return df
    
    def _add_ml_features(self, df):
        """Agregar features específicas para machine learning"""
        if self.logger:
            self.logger.info('Enricher', '_add_ml_features', 'Agregando features para ML')
        
        # Lags del VIX (valores pasados)
        for lag in [1, 2, 3, 5, 10]:
            df[f'VIX_Lag_{lag}'] = df['Close'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'VIX_Mean_{window}d'] = df['Close'].rolling(window).mean()
            df[f'VIX_Std_{window}d'] = df['Close'].rolling(window).std()
            df[f'VIX_Min_{window}d'] = df['Close'].rolling(window).min()
            df[f'VIX_Max_{window}d'] = df['Close'].rolling(window).max()
            df[f'VIX_Skew_{window}d'] = df['Close'].rolling(window).skew()
        
        # Variables de distancia desde máximos/mínimos
        df['Days_Since_High'] = df.groupby(df['Close'].expanding().max().values, group_keys=False).cumcount()
        df['Days_Since_Low'] = df.groupby(df['Close'].expanding().min().values, group_keys=False).cumcount()
        
        # Percentiles móviles
        df['VIX_Percentile_20d'] = df['Close'].rolling(window=20).rank(pct=True)
        df['VIX_Percentile_50d'] = df['Close'].rolling(window=50).rank(pct=True)
        
        # Clasificación por niveles (mejorada)
        low_percentile = np.percentile(df['Close'].dropna(), 20)
        high_percentile = np.percentile(df['Close'].dropna(), 80)
        
        df['VIX_Level'] = 'Normal'
        df.loc[df['Close'] <= low_percentile, 'VIX_Level'] = 'Low'
        df.loc[df['Close'] >= high_percentile, 'VIX_Level'] = 'High'
        
        # Variables dummy para VIX_Level
        df['VIX_Level_Low'] = (df['VIX_Level'] == 'Low').astype(int)
        df['VIX_Level_High'] = (df['VIX_Level'] == 'High').astype(int)
        
        return df