"""
Módulo para entrenar y usar modelos predictivos del índice VIX.
"""
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class Modeller:
    """
    Clase para entrenar y usar modelos predictivos del índice VIX.
    """
    
    def __init__(self, logger=None):
        """
        Inicializa el modelador.
        
        Args:
            logger (Logger, optional): Instancia de Logger para registrar eventos
        """
        self.logger = logger
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.model_metadata = {}
        
        if logger:
            logger.info('Modeller', '__init__', 'Instancia de Modeller inicializada')
    
    def entrenar(self, df_enriched, target_days=1, model_type='random_forest'):
        """
        Entrena un modelo para predecir el VIX y guarda el artefacto.
        
        Args:
            df_enriched (pandas.DataFrame): DataFrame con datos enriquecidos
            target_days (int): Días hacia adelante a predecir (default: 1)
            model_type (str): Tipo de modelo ('random_forest', 'gradient_boosting', 'linear', 'ridge')
            
        Returns:
            dict: Métricas del modelo entrenado
        """
        if self.logger:
            self.logger.info('Modeller', 'entrenar', f'Iniciando entrenamiento con modelo {model_type}')
        
        try:
            # Crear directorio para modelos si no existe
            os.makedirs("src/piv_2025_1_2/static/models", exist_ok=True)
            
            # Preparar datos para entrenamiento
            X, y = self._preparar_datos_entrenamiento(df_enriched, target_days)
            
            if X.empty or len(y) == 0:
                raise ValueError("No hay suficientes datos para entrenar el modelo")
            
            # Dividir datos en entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False  # No shuffle para datos temporales
            )
            
            # Escalar características
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Entrenar modelo
            self.model = self._crear_modelo(model_type)
            self.model.fit(X_train_scaled, y_train)
            
            # Hacer predicciones
            y_pred_train = self.model.predict(X_train_scaled)
            y_pred_test = self.model.predict(X_test_scaled)
            
            # Calcular métricas
            metricas = self._calcular_metricas(y_train, y_pred_train, y_test, y_pred_test)
            
            # Guardar metadata del modelo
            self.model_metadata = {
                'model_type': model_type,
                'target_days': target_days,
                'feature_columns': list(X.columns),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'training_date': datetime.now().isoformat(),
                'metrics': metricas
            }
            
            # Guardar modelo y metadata
            self._guardar_modelo()
            
            if self.logger:
                self.logger.info('Modeller', 'entrenar', 
                               f'Modelo entrenado exitosamente. RMSE: {metricas["test_rmse"]:.4f}')
            
            return metricas
            
        except Exception as e:
            if self.logger:
                self.logger.error('Modeller', 'entrenar', f'Error durante entrenamiento: {e}')
            raise e
    
    def predecir(self, df_new, load_model=True):
        """
        Carga el modelo guardado y realiza predicciones.
        
        Args:
            df_new (pandas.DataFrame): DataFrame con datos nuevos
            load_model (bool): Si debe cargar el modelo desde archivo
            
        Returns:
            numpy.array: Predicciones del modelo
        """
        if self.logger:
            self.logger.info('Modeller', 'predecir', 'Iniciando predicciones')
        
        try:
            # Cargar modelo si es necesario
            if load_model or self.model is None:
                self._cargar_modelo()
            
            # Preparar datos para predicción
            X_pred = self._preparar_datos_prediccion(df_new)
            
            # Escalar datos
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # Realizar predicciones
            predicciones = self.model.predict(X_pred_scaled)
            
            if self.logger:
                self.logger.info('Modeller', 'predecir', 
                               f'Predicciones realizadas: {len(predicciones)} valores')
            
            return predicciones
            
        except Exception as e:
            if self.logger:
                self.logger.error('Modeller', 'predecir', f'Error durante predicción: {e}')
            raise e
    
    def _preparar_datos_entrenamiento(self, df, target_days):
        """Prepara los datos para entrenamiento del modelo"""
        if self.logger:
            self.logger.info('Modeller', '_preparar_datos_entrenamiento', 
                           f'Preparando datos para predecir {target_days} días adelante')
        
        # Crear variable objetivo (VIX de N días en el futuro)
        df = df.copy()
        df['target'] = df['Close'].shift(-target_days)
        
        # Eliminar filas sin target
        df = df.dropna(subset=['target'])
        
        # Seleccionar features para el modelo
        feature_columns = self._seleccionar_features(df)
        self.feature_columns = feature_columns
        
        # Preparar X e y
        X = df[feature_columns].copy()
        y = df['target'].copy()
        
        # Eliminar filas con NaN
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if self.logger:
            self.logger.info('Modeller', '_preparar_datos_entrenamiento', 
                           f'Datos preparados: {len(X)} muestras, {len(feature_columns)} features')
        
        return X, y
    
    def _preparar_datos_prediccion(self, df):
        """Prepara datos para realizar predicciones"""
        if not self.feature_columns:
            raise ValueError("Modelo no entrenado o feature_columns no disponibles")
        
        # Verificar que todas las features están disponibles
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Features faltantes en datos: {missing_features}")
        
        # Seleccionar solo las features del modelo
        X = df[self.feature_columns].copy()
        
        # Manejar valores faltantes (usar forward fill y luego backward fill)
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        return X
    
    def _seleccionar_features(self, df):
        """Selecciona las features más relevantes para el modelo"""
        # Features básicas del VIX
        basic_features = ['Open', 'High', 'Low', 'Volume']
        
        # Features temporales
        temporal_features = ['Month', 'Quarter', 'DayOfWeek', 'IsWeekend',
                           'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos']
        
        # Features técnicas
        technical_features = ['Daily_Change', 'MA5', 'MA10', 'MA20', 'MA50',
                            'Volatility_5d', 'Volatility_10d', 'Volatility_20d',
                            'RSI', 'MACD', 'MACD_Signal', 'ATR',
                            'BB_Position', 'BB_Width']
        
        # Features de lag
        lag_features = [col for col in df.columns if 'VIX_Lag_' in col]
        
        # Features estadísticas
        stat_features = [col for col in df.columns if any(x in col for x in ['_Mean_', '_Std_', '_Min_', '_Max_'])]
        
        # Features macro (si están disponibles)
        macro_features = ['SP500_Daily_Change', 'VIX_SP500_Correlation']
        
        # Combinar todas las features
        all_features = (basic_features + temporal_features + technical_features + 
                       lag_features + stat_features + macro_features)
        
        # Filtrar solo las que existen en el DataFrame
        available_features = [col for col in all_features if col in df.columns]
        
        if self.logger:
            self.logger.info('Modeller', '_seleccionar_features', 
                           f'Features seleccionadas: {len(available_features)}')
        
        return available_features
    
    def _crear_modelo(self, model_type):
        """Crea el modelo según el tipo especificado"""
        if model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=5,
                random_state=42
            )
        elif model_type == 'linear':
            return LinearRegression()
        elif model_type == 'ridge':
            return Ridge(alpha=1.0, random_state=42)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
    
    def _calcular_metricas(self, y_train, y_pred_train, y_test, y_pred_test):
        """Calcula métricas de evaluación del modelo"""
        metricas = {
            # Métricas de entrenamiento
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_r2': r2_score(y_train, y_pred_train),
            
            # Métricas de prueba
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_r2': r2_score(y_test, y_pred_test),
            
            # Métricas adicionales
            'test_mape': np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100,
            'directional_accuracy': np.mean(np.sign(np.diff(y_test)) == np.sign(np.diff(y_pred_test))) * 100
        }
        
        return metricas
    
    def _guardar_modelo(self):
        """Guarda el modelo entrenado y metadata"""
        modelo_completo = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'metadata': self.model_metadata
        }
        
        with open('src/piv_2025_1_2/static/models/model.pkl', 'wb') as f:
            pickle.dump(modelo_completo, f)
        
        # Guardar también metadata en JSON para fácil lectura
        import json
        with open('src/piv_2025_1_2/static/models/model_metadata.json', 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        
        if self.logger:
            self.logger.info('Modeller', '_guardar_modelo', 'Modelo guardado en src/piv_2025_1_2/static/models/model.pkl')
    
    def _cargar_modelo(self):
        """Carga el modelo guardado"""
        if not os.path.exists('src/piv_2025_1_2/static/models/model.pkl'):
            raise FileNotFoundError("No se encontró modelo guardado en static/models/model.pkl")
        
        with open('static/models/model.pkl', 'rb') as f:
            modelo_completo = pickle.load(f)
        
        self.model = modelo_completo['model']
        self.scaler = modelo_completo['scaler']
        self.feature_columns = modelo_completo['feature_columns']
        self.model_metadata = modelo_completo['metadata']
        
        if self.logger:
            self.logger.info('Modeller', '_cargar_modelo', 'Modelo cargado exitosamente')
    
    def get_model_info(self):
        """Retorna información del modelo"""
        if not self.model_metadata:
            return "No hay modelo entrenado"
        
        return self.model_metadata
    
    def get_feature_importance(self):
        """Retorna la importancia de las features (si el modelo lo soporta)"""
        if self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        else:
            return None