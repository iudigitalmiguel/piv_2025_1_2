"""
Módulo para la recolección y persistencia de datos del índice VIX de Yahoo Finance.
"""
import pandas as pd
import requests
from datetime import datetime, timedelta
import time

class Collector:
    """
    Clase para recolectar datos históricos del índice VIX de Yahoo Finance.
    """
    
    def __init__(self, logger=None):
        """
        Inicializa el colector de datos VIX.
        
        Args:
            logger (Logger, optional): Instancia de Logger para registrar eventos
        """
        self.logger = logger
        
        # URL del índice VIX en Yahoo Finance
        self.url = "https://finance.yahoo.com/quote/%5EVIX/history/"
        
        if logger:
            logger.info('Collector', '__init__', 'Instancia de Collector inicializada')
    
    def collertor_data(self):
        """
        Descarga y retorna los datos históricos del índice VIX.
        
        Returns:
            pandas.DataFrame: DataFrame con los datos descargados
        """
        if self.logger:
            self.logger.info('Collector', 'collertor_data', 'Iniciando descarga de datos del VIX')
        
        # Configurar encabezados para simular un navegador
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            # Para descarga directa de datos históricos usando la API de Yahoo Finance
            # Definir el rango de fechas (1 año hacia atrás)
            end_date = int(datetime.now().timestamp())
            start_date = int((datetime.now() - timedelta(days=365)).timestamp())
            
            # Construir URL para la API
            ticker = "%5EVIX"  # ^VIX codificado para URL
            query_url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_date}&period2={end_date}&interval=1d&events=history"
            
            if self.logger:
                self.logger.info('Collector', 'collertor_data', f'Consultando URL: {query_url}')
            
            # Realizar solicitud
            response = requests.get(query_url, headers=headers)
            
            if response.status_code == 200:
                # Crear DataFrame a partir de los datos descargados
                df = pd.read_csv(pd.io.common.StringIO(response.text))
                df['Date'] = pd.to_datetime(df['Date'])
                
                if self.logger:
                    self.logger.info('Collector', 'collertor_data', f'Datos descargados exitosamente: {len(df)} registros')
                
                return df
            else:
                if self.logger:
                    self.logger.error('Collector', 'collertor_data', f'Error al descargar datos: Código {response.status_code}')
                return pd.DataFrame()
                
        except Exception as e:
            if self.logger:
                self.logger.error('Collector', 'collertor_data', f'Error durante la descarga de datos: {e}')
            return pd.DataFrame()