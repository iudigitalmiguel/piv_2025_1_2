"""
Script principal que orquesta el proceso ETL completo para el índice VIX.
Versión actualizada para Actividad 2: incluye modelo predictivo.
"""
import os
import sys
from logger import Logger
from collector import Collector
from enricher import Enricher
from modeller import Modeller
import pandas as pd
from datetime import datetime
import argparse

def setup_directories():
    """Crea los directorios necesarios para el proyecto"""
    directories = [
        "src/piv_2025_1_2/static/data",
        "src/piv_2025_1_2/static/models",
        "src/piv_2025_1_2/static/reports",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_etl_pipeline(logger, train_model=True, model_type='random_forest'):
    """
    Ejecuta el pipeline ETL completo
    
    Args:
        logger: Instancia del logger
        train_model: Si debe entrenar el modelo predictivo
        model_type: Tipo de modelo a entrenar
    """
    logger.info('Main', 'run_etl_pipeline', 'Iniciando pipeline ETL completo')
    
    try:
        # 1. EXTRACT - Recolectar datos
        logger.info('Main', 'run_etl_pipeline', '=== FASE 1: EXTRACCIÓN DE DATOS ===')
        collector = Collector(logger=logger)
        df_raw = collector.collertor_data()
        
        if df_raw.empty:
            logger.error('Main', 'run_etl_pipeline', 'No se pudieron obtener datos. Pipeline abortado.')
            return False
        
        logger.info('Main', 'run_etl_pipeline', f'Datos extraídos: {len(df_raw)} registros')
        
        # Guardar datos crudos
        raw_data_path = "src/piv_2025_1_2/static/data/vix_data.csv"
        df_raw.to_csv(raw_data_path, index=False, float_format='%.2f', decimal=',')
        logger.info('Main', 'run_etl_pipeline', f'Datos crudos guardados: {raw_data_path}')
        
        # 2. TRANSFORM - Enriquecer datos
        logger.info('Main', 'run_etl_pipeline', '=== FASE 2: TRANSFORMACIÓN Y ENRIQUECIMIENTO ===')
        enricher = Enricher(logger=logger)
        df_enriched = enricher.calcular_kpi(df_raw)
        
        if df_enriched.empty:
            logger.error('Main', 'run_etl_pipeline', 'Error en enriquecimiento de datos')
            return False
        
        logger.info('Main', 'run_etl_pipeline', 
                   f'Datos enriquecidos: {len(df_enriched)} registros, {len(df_enriched.columns)} columnas')
        
        # Guardar datos enriquecidos
        enriched_data_path = "src/piv_2025_1_2/static/data/vix_data_enricher.csv"
        df_enriched.to_csv(enriched_data_path, index=False, float_format='%.2f', decimal=',')
        logger.info('Main', 'run_etl_pipeline', f'Datos enriquecidos guardados: {enriched_data_path}')
        
        # 3. MODEL - Entrenar modelo predictivo (opcional)
        if train_model:
            logger.info('Main', 'run_etl_pipeline', '=== FASE 3: ENTRENAMIENTO DE MODELO ===')
            modeller = Modeller(logger=logger)
            
            try:
                # Entrenar modelo
                metrics = modeller.entrenar(df_enriched, target_days=1, model_type=model_type)
                
                logger.info('Main', 'run_etl_pipeline', 
                           f'Modelo entrenado exitosamente. RMSE: {metrics["test_rmse"]:.4f}, R²: {metrics["test_r2"]:.4f}')
                
                # Generar algunas predicciones de ejemplo
                recent_data = df_enriched.tail(10)
                predictions = modeller.predecir(recent_data, load_model=False)
                
                logger.info('Main', 'run_etl_pipeline', 
                           f'Predicciones generadas: {len(predictions)} valores')
                
                # Guardar reporte de modelo
                generate_model_report(metrics, model_type, logger)
                
            except Exception as e:
                logger.error('Main', 'run_etl_pipeline', f'Error en entrenamiento de modelo: {e}')
                return False
        
        # 4. LOAD - Generar reportes finales
        logger.info('Main', 'run_etl_pipeline', '=== FASE 4: GENERACIÓN DE REPORTES ===')
        generate_summary_report(df_raw, df_enriched, logger)
        
        logger.info('Main', 'run_etl_pipeline', '✅ Pipeline ETL completado exitosamente')
        return True
        
    except Exception as e:
        logger.error('Main', 'run_etl_pipeline', f'Error crítico en pipeline: {e}')
        return False

def generate_model_report(metrics, model_type, logger):
    """Genera reporte del modelo entrenado"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"src/piv_2025_1_2/static/reports/model_report_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE MODELO PREDICTIVO - ÍNDICE VIX\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Fecha de entrenamiento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tipo de modelo: {model_type}\n\n")
            
            f.write("MÉTRICAS DE EVALUACIÓN:\n")
            f.write("-" * 25 + "\n")
            f.write(f"RMSE (Test): {metrics['test_rmse']:.4f}\n")
            f.write(f"MAE (Test): {metrics['test_mae']:.4f}\n")
            f.write(f"R² (Test): {metrics['test_r2']:.4f}\n")
            f.write(f"MAPE (Test): {metrics['test_mape']:.2f}%\n")
            f.write(f"Precisión Direccional: {metrics['directional_accuracy']:.2f}%\n\n")
            
            f.write("INTERPRETACIÓN DE MÉTRICAS:\n")
            f.write("-" * 30 + "\n")
            f.write("• RMSE: Error cuadrático medio (menor es mejor)\n")
            f.write("• MAE: Error absoluto medio (menor es mejor)\n")
            f.write("• R²: Coeficiente de determinación (1.0 es perfecto)\n")
            f.write("• MAPE: Error porcentual absoluto medio\n")
            f.write("• Precisión Direccional: % de predicciones correctas de dirección\n\n")
            
            # Evaluación de calidad
            if metrics['test_r2'] > 0.7:
                quality = "Excelente"
            elif metrics['test_r2'] > 0.5:
                quality = "Buena"
            elif metrics['test_r2'] > 0.3:
                quality = "Aceptable"
            else:
                quality = "Requiere mejora"
            
            f.write(f"CALIDAD DEL MODELO: {quality}\n\n")
            
            f.write("JUSTIFICACIÓN DE MÉTRICAS:\n")
            f.write("-" * 30 + "\n")
            f.write("Se seleccionó RMSE como métrica principal porque:\n")
            f.write("1. Penaliza más los errores grandes (importante para VIX)\n")
            f.write("2. Está en las mismas unidades que el VIX\n")
            f.write("3. Es estándar para problemas de regresión\n\n")
            f.write("MAE complementa RMSE al ser menos sensible a outliers.\n")
            f.write("R² indica qué porcentaje de varianza explica el modelo.\n")
        
        logger.info('Main', 'generate_model_report', f'Reporte de modelo guardado: {report_path}')
        
    except Exception as e:
        logger.error('Main', 'generate_model_report', f'Error generando reporte: {e}')

def generate_summary_report(df_raw, df_enriched, logger):
    """Genera reporte resumen del pipeline"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"src/piv_2025_1_2/static/reports/pipeline_summary_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE RESUMEN - PIPELINE ETL VIX\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATOS PROCESADOS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Registros extraídos: {len(df_raw):,}\n")
            f.write(f"Período: {df_raw['Date'].min()} a {df_raw['Date'].max()}\n")
            f.write(f"Columnas originales: {len(df_raw.columns)}\n")
            f.write(f"Columnas enriquecidas: {len(df_enriched.columns)}\n")
            f.write(f"Features añadidas: {len(df_enriched.columns) - len(df_raw.columns)}\n\n")
            
            f.write("ESTADÍSTICAS DEL VIX:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Valor mínimo: {df_raw['Close'].min():.2f}\n")
            f.write(f"Valor máximo: {df_raw['Close'].max():.2f}\n")
            f.write(f"Valor promedio: {df_raw['Close'].mean():.2f}\n")
            f.write(f"Desviación estándar: {df_raw['Close'].std():.2f}\n")
            f.write(f"Último valor: {df_raw['Close'].iloc[-1]:.2f}\n\n")
            
            # Distribución por niveles
            if 'VIX_Level' in df_enriched.columns:
                level_counts = df_enriched['VIX_Level'].value_counts()
                f.write("DISTRIBUCIÓN POR NIVELES:\n")
                f.write("-" * 30 + "\n")
                for level, count in level_counts.items():
                    pct = (count / len(df_enriched)) * 100
                    f.write(f"{level}: {count:,} días ({pct:.1f}%)\n")
                f.write("\n")
            
            f.write("ARCHIVOS GENERADOS:\n")
            f.write("-" * 20 + "\n")
            f.write("• src/piv_2025_1_2/static/data/vix_data.csv - Datos crudos\n")
            f.write("• src/piv_2025_1_2/static/data/vix_data_enricher.csv - Datos enriquecidos\n")
            f.write("• src/piv_2025_1_2/static/models/model.pkl - Modelo entrenado\n")
            f.write("• src/piv_2025_1_2/static/models/model_metadata.json - Metadata del modelo\n")
            f.write("• src/piv_2025_1_2/static/reports/ - Reportes generados\n\n")
            
            f.write("SIGUIENTE PASO:\n")
            f.write("-" * 15 + "\n")
            f.write("Ejecutar dashboard interactivo:\n")
            f.write("streamlit run models/dashboard.py\n")
        
        logger.info('Main', 'generate_summary_report', f'Reporte resumen guardado: {report_path}')
        
    except Exception as e:
        logger.error('Main', 'generate_summary_report', f'Error generando reporte: {e}')

def run_predictions_only(logger):
    """Ejecuta solo predicciones con modelo existente"""
    logger.info('Main', 'run_predictions_only', 'Ejecutando predicciones con modelo existente')
    
    try:
        # Verificar que existan los datos y modelo
        if not os.path.exists("src/piv_2025_1_2/static/data/vix_data_enricher.csv"):
            logger.error('Main', 'run_predictions_only', 'No se encontraron datos enriquecidos')
            return False
        
        if not os.path.exists("src/piv_2025_1_2/static/models/model.pkl"):
            logger.error('Main', 'run_predictions_only', 'No se encontró modelo entrenado')
            return False
        
        # Cargar datos
        df_enriched = pd.read_csv("src/piv_2025_1_2/static/data/vix_data_enricher.csv", decimal=',')
        df_enriched['Date'] = pd.to_datetime(df_enriched['Date'])
        
        # Inicializar modelo
        modeller = Modeller(logger=logger)
        
        # Hacer predicciones para los últimos 30 días
        recent_data = df_enriched.tail(30)
        predictions = modeller.predecir(recent_data)
        
        # Guardar predicciones
        predictions_df = pd.DataFrame({
            'Prediccion': predictions,
            'Fecha_Prediccion': pd.date_range(
                start=recent_data['Date'].iloc[-1] + pd.Timedelta(days=1),
                periods=len(predictions)
            )
        })
        
        predictions_path = "src/piv_2025_1_2/static/data/predictions.csv"
        predictions_df.to_csv(predictions_path, index=False, float_format='%.2f', decimal=',')
        
        logger.info('Main', 'run_predictions_only', 
                   f'Predicciones generadas y guardadas: {predictions_path}')
        
        # Mostrar información
        print(f"\n📊 PREDICCIONES GENERADAS:")
        print(f"Número de predicciones: {len(predictions)}")
        print(f"Rango de predicciones: {predictions.min():.2f} - {predictions.max():.2f}")
        print(f"Predicción promedio: {predictions.mean():.2f}")
        print(f"Archivo guardado: {predictions_path}")
        
        return True
        
    except Exception as e:
        logger.error('Main', 'run_predictions_only', f'Error en predicciones: {e}')
        return False

def main():
    """Función principal"""
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Pipeline ETL para análisis del VIX')
    parser.add_argument('--mode', choices=['full', 'etl', 'model', 'predict', 'dashboard'], 
                       default='full', help='Modo de ejecución')
    parser.add_argument('--model-type', choices=['random_forest', 'gradient_boosting', 'linear', 'ridge'],
                       default='random_forest', help='Tipo de modelo a entrenar')
    parser.add_argument('--no-model', action='store_true', help='No entrenar modelo')
    
    args = parser.parse_args()
    
    # Configurar directorios
    setup_directories()
    
    # Inicializar logger
    logger = Logger()
    logger.info('Main', 'main', f'=== INICIANDO PIPELINE VIX - MODO: {args.mode.upper()} ===')
    
    success = False
    
    try:
        if args.mode == 'full':
            # Pipeline completo: ETL + Modelo
            success = run_etl_pipeline(logger, train_model=not args.no_model, model_type=args.model_type)
            
        elif args.mode == 'etl':
            # Solo ETL (sin modelo)
            success = run_etl_pipeline(logger, train_model=False)
            
        elif args.mode == 'model':
            # Solo entrenar modelo (requiere datos existentes)
            if not os.path.exists("src/piv_2025_1_2/static/data/vix_data_enricher.csv"):
                logger.error('Main', 'main', 'No se encontraron datos enriquecidos. Ejecute ETL primero.')
                return
            
            df_enriched = pd.read_csv("src/piv_2025_1_2/static/data/vix_data_enricher.csv", decimal=',')
            modeller = Modeller(logger=logger)
            metrics = modeller.entrenar(df_enriched, model_type=args.model_type)
            generate_model_report(metrics, args.model_type, logger)
            success = True
            
        elif args.mode == 'predict':
            # Solo predicciones
            success = run_predictions_only(logger)
            
        elif args.mode == 'dashboard':
            # Lanzar dashboard
            print("\n🚀 Iniciando dashboard...")
            print("📊 Abra su navegador en: http://localhost:8501")
            print("⚠️  Presione Ctrl+C para detener el dashboard")
            os.system("streamlit run models/dashboard.py")
            success = True
        
        if success:
            print("\n" + "="*60)
            print("✅ PROCESO COMPLETADO EXITOSAMENTE")
            print("="*60)
            
            if args.mode in ['full', 'etl', 'model']:
                print("\n📁 ARCHIVOS GENERADOS:")
                print("   • src/piv_2025_1_2/static/data/vix_data.csv")
                print("   • src/piv_2025_1_2/static/data/vix_data_enricher.csv")
                if not args.no_model and args.mode in ['full', 'model']:
                    print("   • src/piv_2025_1_2/static/models/model.pkl")
                    print("   • src/piv_2025_1_2/static/models/model_metadata.json")
                print("   • src/piv_2025_1_2/static/reports/ (reportes)")
                
                print("\n🚀 SIGUIENTE PASO:")
                print("   python main.py --mode dashboard")
                print("   O ejecute: streamlit run models/dashboard.py")
                
        else:
            print("\n❌ PROCESO FALLÓ - Revise los logs para más detalles")
            
    except KeyboardInterrupt:
        logger.info('Main', 'main', 'Proceso interrumpido por el usuario')
        print("\n⚠️ Proceso interrumpido por el usuario")
        
    except Exception as e:
        logger.error('Main', 'main', f'Error crítico: {e}')
        print(f"\n❌ Error crítico: {e}")
        
    finally:
        logger.info('Main', 'main', '=== FINALIZANDO PIPELINE VIX ===')

if __name__ == "__main__":
    main()