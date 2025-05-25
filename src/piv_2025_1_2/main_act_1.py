import os
from logger import Logger
from collector import Collector
from enricher import Enricher
import pandas as pd

def main():
    # Asegúrate de que el directorio para guardar los datos exista
    os.makedirs("src/piv_2025_1_2/static/data", exist_ok=True)
    
    # Inicializar logger
    logger = Logger()
    logger.info('Main', 'main', 'Inicializar clase Logger')
    
    # Inicializar colector y enricher
    collector = Collector(logger=logger)
    enricher = Enricher(logger=logger)
    
    # Obtener datos del VIX
    df = collector.collertor_data()
    
    if not df.empty:
        # Guardar datos crudos del VIX
        df.to_csv("src/piv_2025_1_2/static/data/vix_data.csv", index=False, float_format='%.2f', decimal=',')
        logger.info('Main', 'main', 'Datos crudos guardados exitosamente')
        
        # Enriquecer datos del VIX con KPIs adicionales
        df_2 = enricher.calcular_kpi(df)
        
        # Guardar datos enriquecidos del VIX
        df_2.to_csv("src/piv_2025_1_2/static/data/vix_data_enricher.csv", index=False, float_format='%.2f', decimal=',')
        logger.info('Main', 'main', 'Datos enriquecidos guardados exitosamente')
    else:
        logger.error('Main', 'main', 'No se pudieron obtener datos. No se generarán archivos CSV.')

if __name__ == "__main__":
    main()