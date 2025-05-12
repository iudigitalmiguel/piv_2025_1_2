"""
Módulo de configuración de logging para el proyecto.
"""
import logging
from datetime import datetime
import os

class Logger:
    """
    Clase para configurar y gestionar logs del sistema.
    """
    
    def __init__(self, log_dir='logs'):
        """
        Inicializa el sistema de logging.
        
        Args:
            log_dir (str): Directorio donde se guardarán los logs
        """
        # Crear directorio de logs si no existe
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Configurar nombre del archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"vix_data_collector_{timestamp}.log")
        
        # Configurar el logger
        self.logger = logging.getLogger('vix_collector')
        self.logger.setLevel(logging.INFO)
        
        # Handler para archivo
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formato para los logs
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        
        # Agregar handlers al logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, class_name, method_name, message):
        """
        Registra un mensaje de nivel INFO.
        
        Args:
            class_name (str): Nombre de la clase
            method_name (str): Nombre del método
            message (str): Mensaje a registrar
        """
        self.logger.info(f"{class_name} - {method_name} - {message}")
    
    def error(self, class_name, method_name, message):
        """
        Registra un mensaje de nivel ERROR.
        
        Args:
            class_name (str): Nombre de la clase
            method_name (str): Nombre del método
            message (str): Mensaje a registrar
        """
        self.logger.error(f"{class_name} - {method_name} - {message}")
    
    def warning(self, class_name, method_name, message):
        """
        Registra un mensaje de nivel WARNING.
        
        Args:
            class_name (str): Nombre de la clase
            method_name (str): Nombre del método
            message (str): Mensaje a registrar
        """
        self.logger.warning(f"{class_name} - {method_name} - {message}")