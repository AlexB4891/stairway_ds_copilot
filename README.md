# Stairway to Data Science 

## Escalón 3: Usando Github Copilot desde Visual Studio Code

## Descripción General
Este script de Python se utiliza para realizar análisis avanzado de series de tiempo en diferentes conjuntos de datos económicos y empresariales. Utiliza bibliotecas populares como pandas, numpy, matplotlib, seaborn, statsmodels, Prophet, pyreadstat, janitor y pmdarima.

## Características
- **Limpieza de Datos:** Emplea técnicas de limpieza y transformación de datos.
- **Análisis de Series de Tiempo:** Incluye descomposición de series de tiempo, detección de valores atípicos y pruebas de estacionariedad.
- **Modelado y Predicción:** Utiliza modelos ARIMA y Prophet para pronósticos de series de tiempo.
- **Visualización de Datos:** Genera gráficos para la interpretación y análisis de las series de tiempo.

## Datasets Incluidos
- `ts_series_pichincha.csv`: Empresas activas en Pichincha.
- `ts_series_iva.csv`: Datos y series del IVA.
- `ts_series_actividad_C1050.csv`: Industria de lácteos.
- `ts_series_actividad_C1071.csv`: Industria de panadería.
- `ts_series_actividad_C1103.csv`: Industria de bebidas malteadas y de malta.

## Requisitos
- Python 3.x
- Bibliotecas: pandas, numpy, matplotlib, seaborn, statsmodels, Prophet, pyreadstat, janitor, pmdarima.

## Instalación de Dependencias
```bash
pip install pandas numpy matplotlib seaborn statsmodels prophet pyreadstat janitor pmdarima
```

## Uso del Script
1. **Limpieza de Datos:** El script inicia con la limpieza y preparación de los datos.
2. **Análisis de Series de Tiempo:** Se realizan análisis detallados sobre las series temporales, incluyendo descomposición y pruebas de estacionariedad.
3. **Predicción de Series de Tiempo:** Se utilizan modelos de predicción como ARIMA y Prophet para generar pronósticos futuros.

## Funciones Clave
- `transform_fecha(df)`: Transforma y establece las fechas como índices en un DataFrame.
- `graficar_serie_tiempo(df, columna, titulo, etiqueta_x, etiqueta_y)`: Función para graficar series de tiempo.
- `imputar_valores_faltantes(df)`: Imputa valores faltantes en series de tiempo.
- `crear_variable_lag(df, columna, lag)`: Crea una variable lag para análisis de series temporales.
- `generate_forecast_df(model, steps)`: Genera un DataFrame de pronóstico para un número especificado de pasos en el futuro.

## Ejemplo de Uso
```python
# Cargar un dataset
ts_series_pichincha = pd.read_csv("data/ts_series_pichincha.csv")

# Limpieza y transformación de datos
ts_series_pichincha = transform_fecha(ts_series_pichincha)

# Análisis de series de tiempo
graficar_serie_tiempo(ts_series_pichincha, 'saldo', 'Empresas activas en Pichincha', 'Fecha', 'Número de Empresas')

# Generar predicciones
model = auto_arima(ts_series_pichincha['saldo'])
forecast = generate_forecast_df(model, 12)
```

## Nota Importante
Este script es una herramienta de análisis de datos y no debe ser utilizado como única fuente para tomar decisiones financieras o empresariales. Se recomienda siempre realizar un análisis exhaustivo y consultar con un experto en la materia.

## Contribuciones
Las contribuciones para mejorar el script son bienvenidas. Por favor, sientase libre de hacer fork del repositorio, realizar cambios y solicitar un pull request para la revisión.

## Licencia
Este script se distribuye bajo la licencia MIT.
