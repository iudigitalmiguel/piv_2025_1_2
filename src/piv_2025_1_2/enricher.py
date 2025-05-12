"""
Módulo para enriquecer y calcular KPIs sobre datos del índice VIX.
"""
import pandas as pd
import numpy as np

class Enricher:
    """
    Clase para calcular indicadores clave de rendimiento (KPIs) sobre datos del VIX.
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
        Calcula varios KPIs para los datos del VIX.
        
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
            self.logger.info('Enricher', 'calcular_kpi', 'Iniciando cálculo de KPIs')
        
        # Crear copia para no modificar el original
        enriched_df = df.copy()
        
        try:
            # Calcular variación diaria
            enriched_df['Daily_Change'] = enriched_df['Close'].pct_change() * 100
            
            # Calcular media móvil de 5 y 20 días
            enriched_df['MA5'] = enriched_df['Close'].rolling(window=5).mean()
            enriched_df['MA20'] = enriched_df['Close'].rolling(window=20).mean()
            
            # Calcular volatilidad (desviación estándar) de 20 días
            enriched_df['Volatility_20d'] = enriched_df['Close'].rolling(window=20).std()
            
            # Calcular RSI (Relative Strength Index) de 14 días
            delta = enriched_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            enriched_df['RSI'] = 100 - (100 / (1 + rs))
            
            # Marcar valores extremos (percentiles 20 y 80)
            low_percentile = np.percentile(enriched_df['Close'].dropna(), 20)
            high_percentile = np.percentile(enriched_df['Close'].dropna(), 80)
            
            enriched_df['VIX_Level'] = 'Normal'
            enriched_df.loc[enriched_df['Close'] <= low_percentile, 'VIX_Level'] = 'Low'
            enriched_df.loc[enriched_df['Close'] >= high_percentile, 'VIX_Level'] = 'High'
            
            if self.logger:
                self.logger.info('Enricher', 'calcular_kpi', 'KPIs calculados exitosamente')
            
            return enriched_df
            
        except Exception as e:
            if self.logger:
                self.logger.error('Enricher', 'calcular_kpi', f'Error al calcular KPIs: {e}')
            return df