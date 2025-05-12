# piv_2025_1_2
Proyecto integrador V finanzas-tomar un indicador de yahooy analizar
# Recolector y Analizador de Datos del Índice VIX

Este proyecto automatiza la recolección de datos históricos del Índice de Volatilidad VIX (CBOE Volatility Index) desde Yahoo Finance. El VIX es un índice en tiempo real que representa la expectativa del mercado sobre la volatilidad a 30 días, y es ampliamente conocido como el "índice del miedo" de Wall Street.

## Objetivos del Proyecto

- Automatizar la recolección diaria de datos históricos del VIX
- Aplicar técnicas de enriquecimiento para calcular KPIs relevantes
- Mantener un histórico actualizado en formatos accesibles (CSV)
- Implementar un sistema de logs para auditoría del proceso

## Estructura del Proyecto
proyecto/
├── .github/
│   └── workflows/
│       └── update_data.yml     # Automatización de la recolección diaria
├── src/
│   └── edu_piv/
│       ├── static/
│       │   └── data/
│       │       ├── vix_data.csv           # Datos crudos del VIX
│       │       └── vix_data_enricher.csv  # Datos enriquecidos con KPIs
│       ├── collector.py        # Recolector de datos del VIX
│       ├── enricher.py         # Calculador de KPIs
│       ├── logger.py           # Sistema de logging
│       └── main.py             # Punto de entrada principal
├── logs/                     # Directorio de logs generados
├── docs/
│   └── report_entrega1.pdf   # Informe en formato APA
└── README.md

## Tecnologías Utilizadas

- Python 3.9
- Pandas y NumPy para manipulación y análisis de datos
- Requests para comunicación con la API de Yahoo Finance
- GitHub Actions para automatización de la recolección
- SQLite/CSV para almacenamiento de datos

## Ejecutar el Proyecto

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar recolección manualmente
python src/edu_piv/main.py
