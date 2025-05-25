from setuptools import setup, find_packages

setup(
    name="piv_2025",
    version="0.2.0",  # Actualizado para Actividad 2
    author="Andres Callejas",
    author_email="",
    description="Proyecto Integrador V - Análisis predictivo del índice VIX con dashboard interactivo",
    long_description="Sistema ETL para análisis del índice VIX con modelo predictivo y dashboard interactivo usando Streamlit",
    py_modules=["actividad_1", "actividad_2"],
    packages=find_packages(),  # Agregado para encontrar paquetes automáticamente
    install_requires=[
        # === DEPENDENCIAS BÁSICAS ===
        "pandas==2.2.3",
        "numpy>=1.24.0",  # Agregado - esencial para cálculos numéricos
        "openpyxl",
        "requests==2.32.3",
        "beautifulsoup4==4.13.3",
        
        # === DATOS FINANCIEROS ===
        "pandas_datareader>=0.10.0",  # Para obtener datos financieros
        "yfinance>=0.2.30",           # Alternativa para Yahoo Finance
        
        # === MACHINE LEARNING ===
        "scikit-learn>=1.3.0",       # Actualizado - modelos predictivos
        
        # === DASHBOARD INTERACTIVO ===
        "streamlit>=1.28.0",         # Nuevo - dashboard web interactivo
        "plotly>=5.17.0",            # Nuevo - gráficos interactivos
        
        # === VISUALIZACIÓN ===
        "matplotlib>=3.7.0",         # Nuevo - gráficos estáticos
        "seaborn>=0.12.0",           # Nuevo - visualización estadística
        
        # === UTILIDADES ADICIONALES ===
        "python-dateutil>=2.8.0",   # Nuevo - manejo de fechas
        "pytz>=2023.3",              # Nuevo - zonas horarias
    ],
    extras_require={
        # Dependencias opcionales para desarrollo
        "dev": [
            "jupyter>=1.0.0",        # Para notebooks de desarrollo
            "ipython>=8.0.0",        # IPython mejorado
            "pytest>=7.0.0",         # Testing
        ],
        # Dependencias adicionales para análisis avanzado
        "advanced": [
            "tensorflow>=2.13.0",    # Para modelos de deep learning
            "xgboost>=1.7.0",        # Gradient boosting avanzado
            "lightgbm>=4.0.0",       # Otro algoritmo de boosting
        ]
    },
    python_requires=">=3.8",        # Versión mínima de Python
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="vix, finance, etl, machine-learning, dashboard, streamlit, trading",
    project_urls={
        "Documentation": "https://github.com/tu-usuario/piv_2025",
        "Source": "https://github.com/tu-usuario/piv_2025",
        "Tracker": "https://github.com/tu-usuario/piv_2025/issues",
    },
)