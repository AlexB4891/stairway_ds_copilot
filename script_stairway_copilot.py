# ------------------------------------------------------------------------------------- #
#                   Stairway to data science: Tercer escalón, el cielo                  #
# ------------------------------------------------------------------------------------- #

# Modulos que vamos a usar: 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from prophet import Prophet
import pyreadstat
from janitor import clean_names
from pmdarima import auto_arima
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import statsmodels.api as sm
from pandas.plotting import lag_plot
from scipy import stats
import pmdarima as pm
from pmdarima import auto_arima
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller


# Empresas activas en Pichincha:
ts_series_pichincha = pd.read_csv("data/ts_series_pichincha.csv")
# Datos y series del IVA:
ts_series_iva = pd.read_csv("data/ts_series_iva.csv")
# Lacteos:
ts_series_actividad_C1050 = pd.read_csv("data/ts_series_actividad_C1050.csv")
# Panadería:
ts_series_actividad_C1071 = pd.read_csv("data/ts_series_actividad_C1071.csv")
# Bebidas malteadas y de malta:
ts_series_actividad_C1103 = pd.read_csv("data/ts_series_actividad_C1103.csv")

#############################################################
# Limpieza de datos:
#############################################################
# Las fechas como son:

ts_series_iva["fecha"] = ts_series_iva["anio_fiscal"].astype(str) + "-" + ts_series_iva["mes_fiscal"].astype(str)
ts_series_iva["fecha"] = pd.to_datetime(ts_series_iva["fecha"])
ts_series_iva = ts_series_iva.set_index('fecha')
# Escribimos la función:
def transform_fecha(df):
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.set_index('fecha')
    return df

# Este es el estado inicial:

# Aplicacamos la función
ts_series_pichincha = transform_fecha(ts_series_pichincha)
# Deberemos usar a todas las series de tiempo:
ts_series_actividad_C1050 = transform_fecha(ts_series_actividad_C1050)
ts_series_actividad_C1071 = transform_fecha(ts_series_actividad_C1071)
ts_series_actividad_C1103 = transform_fecha(ts_series_actividad_C1103)

# Ejercicio 1. Podemos mejorar este codigo haciendo un loop

# Ahora vamos a graficar las series de tiempo:

def graficar_serie_tiempo(df, columna, titulo, etiqueta_x, etiqueta_y):
   
    # Crea el gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(df[columna])
    
    # Añade títulos y etiquetas
    plt.title(titulo)
    plt.xlabel(etiqueta_x)
    plt.ylabel(etiqueta_y)
    
    # Muestra el gráfico
    plt.show()


# Ejercicio 2. Podemos mejorar este gráfico con la etiqueta de dolar
    
graficar_serie_tiempo(ts_series_pichincha, 'saldo', 'Empresas activas', 'Fecha', 'Empresas')
                      
ts_series_iva.actividad_economica.unique()

ts_series_iva['actividad_economica'] = ts_series_iva['actividad_economica'].replace({
    'C1050': 'ELABORACIÓN DE PRODUCTOS LÁCTEOS',
    'C1071': 'ELABORACIÓN DE PRODUCTOS DE PANADERÍA',
    'C1103': 'ELABORACIÓN DE BEBIDAS MALTEADAS Y DE MALTA'
})

# No se necesita fecha por que es el index
reducido = ts_series_iva[['actividad_economica', 'imp_a_pagar_por_percepcion_699']]

# Varias series al mismo tiempo:

reducido = reducido.pivot(columns='actividad_economica', values='imp_a_pagar_por_percepcion_699')

plt.clf()

plt.style.use('default') 

plt.figure(figsize=(16, 8), dpi=150) 
  
# using .plot method to plot stock prices. 
# we have passed colors as a list 
reducido .plot(label='aapl', color=['orange', 'green','red']) 
  
# adding title 
plt.title('Impuesto a pagar por percepción 699') 
  
# adding label to x-axis 
plt.xlabel('Fecha') 
  
# adding legend. 
plt.legend() 

plt.show()

# Ejercicio 3: Construir una función para graficar varias series de tiempo


# Revisión de los gaps:


fechas_completas = pd.date_range(start = ts_series_iva.index.min(), end = ts_series_iva.index.max(),   freq = 'MS')

gaps = fechas_completas.difference(ts_series_iva.index) # OK, todo en orden por obvias razones

gaps_empresas = fechas_completas.difference(ts_series_pichincha.index) # Tenemos un problema, hay datos incompletos


# Ejercicio 4: Construir una función para imputar los gaps

def imputar_valores_faltantes(df):
    df = df.reindex(fechas_completas)
    df = df.interpolate(method='linear', limit_direction='both')
    return df

# Hay NaNs que puedo hacer?

ts_series_pichincha = ts_series_pichincha.dropna()

complete_pichincha = imputar_valores_faltantes(ts_series_pichincha)

fechas_completas.difference(complete_pichincha.index) 

complete_pichincha.loc[gaps_empresas]

# Grafique la serie de tiempo de empresas activas en Pichincha con los valores imputados
graficar_serie_tiempo(complete_pichincha, 'saldo', 'Empresas activas', 'Fecha', 'Empresas')

# Ejercicio 5: Buscando la anhelada estacionariedad
# Usamos el nombre de la función para decirle a copilot que hacer
# def crear_variable_lag(df, columna, lag)

crear_variable_lag(complete_pichincha, 'saldo', 1)

graficar_serie_tiempo(complete_pichincha, 'saldo_menos_1', 'Cambio en el núemro de empresas', 'Fecha', 'Empresas')

# Filtramos para el periodo 2012, y septiembre 2023 por la estrcutura del IVA
complete_pichincha = complete_pichincha.loc['2012-01-01':'2023-09-01']

graficar_serie_tiempo(complete_pichincha, 'saldo_menos_1', 'Cambio en el núemro de empresas', 'Fecha', 'Empresas')

# Vemos el efecto pandemia en 2020
# Vamos a ver el valor atipico

# Asegúrate de que tu serie de tiempo esté ordenada por fecha
df = complete_pichincha.sort_index()

# Descompone la serie de tiempo
res = sm.tsa.seasonal_decompose(df['saldo_menos_1'])

# Ejercicio 6: Grafica la serie de tiempo de la tendencia y sus componentes aditivos

residuos = res.resid
season = res.seasonal



# Los valores atípicos son los puntos donde los residuos son grandes
valores_atipicos = residuos[np.abs(residuos) > np.std(residuos)*3]

fig = res.plot()

fig.show()

plt.clf()

df_out = df.loc[valores_atipicos.index]

def plot_outliers(outliers, data, columna, method='KNN'):
    ax = data[columna].plot(alpha=0.6)
    
    data.loc[outliers.index, columna].plot(ax=ax, style='rx')


    plt.title(f'Atípicos en {columna} - {method}')
    plt.xlabel('Fecha')
    plt.ylabel(columna)
    plt.legend(['Serie', 'Atípico'])
    plt.show()

plot_outliers(df_out, df, 'saldo_menos_1')

# Algunos graficos bonitos con seaborn:


# Histograma
sns.histplot(df['saldo_menos_1'], kde=True)

plt.show()

plt.clf()

# Pero queremos un boxenplot:
# Ejercicio 7: Grafica el boxenplot de la serie de tiempo y un lagplot



lag_plot(df['saldo_menos_1'])

plt.show()

plt.clf()


model = auto_arima(df['saldo_menos_1'],
                   seasonal=True, m=12)


# Pronósticos de ingresos para 1, 2 y 5 años
forecast_3m = model.predict(steps=3)  # 3 meses
forecast_6m = model.predict(steps=6)  # 6 meses
forecast_1yr = model.predict(steps=12)  # 1 año

forecast_3m_df = pd.DataFrame(forecast_3m, columns=['Prediction'])
forecast_6m_df = pd.DataFrame(forecast_6m, columns=['Prediction'])
forecast_1yr_df = pd.DataFrame(forecast_1yr, columns=['Prediction'])

forecast_index_3m = pd.date_range(start=df.index[-1], periods=3+1, freq='MS')[1:]
forecast_index_6m = pd.date_range(start=df.index[-1], periods=6+1, freq='MS')[1:]
forecast_index_1yr = pd.date_range(start=df.index[-1], periods=12+1, freq='MS')[1:]

forecast_3m_df = pd.DataFrame(forecast_3m, index=forecast_index_3m, columns=['Prediction'])
forecast_6m_df = pd.DataFrame(forecast_6m, index=forecast_index_6m, columns=['Prediction'])
forecast_1yr_df = pd.DataFrame(forecast_1yr, index=forecast_index_1yr, columns=['Prediction'])

## !Ufff cuanto codigo!

# Objetivo: crear una función que genere un dataframe con los pronósticos de un modelo
generate_forecast_df(model,12)


# Ejercicio 8: Explain el codigein
def check_stationarity(df):
    kps = kpss(df)
    adf = adfuller(df)
    
    kpss_pv, adf_pv = kps[1], adf[1]
    kpssh, adfh = 'Stationary', 'Non-stationary'
    
    if adf_pv < 0.05:
        # Reject ADF Null Hypothesis
        adfh = 'Stationary'
    if kpss_pv < 0.05:
        # Reject KPSS Null Hypothesis
        kpssh = 'Non-stationary'
    return (kpssh, adfh)  


def plot_comparison(methods, plot_type='line'):
    n = len(methods) // 2
    fig, ax = plt.subplots(n,2, sharex=True, figsize=(10,5))
    for i, method in enumerate(methods):
        method.dropna(inplace=True)
        name = [n for n in globals() if globals()[n] is method]
        v, r = i // 2, i % 2
        kpss_s, adf_s = check_stationarity(method)
        method.plot(kind=plot_type,
                    ax=ax[v,r],
                    legend=False,
                    title=f'{name[0]} --> KPSS: {kpss_s}, ADF{adf_s}')
        ax[v,r].title.set_size(20)
        method.rolling(52).mean().plot(ax=ax[v,r],legend=False)
    plt.show() 


df =  df['saldo']
first_order_diff = df.diff().dropna()

differencing_twice = df.diff(52).diff().dropna()

rolling = df.rolling(window=12).mean()

subtract_rolling_mean = df - rolling

log_transform = np.log(df)

decomp = sm.tsa.seasonal_decompose(df)

sd_detrend = decomp.observed - decomp.trend


cyclic, trend = hpfilter(df)

methods = [first_order_diff, differencing_twice,
           subtract_rolling_mean, log_transform,
           sd_detrend, cyclic]

plot_comparison(methods)

# Ejercicio 9: ¿Como puedo ver todos los metodos de un objeto como modelo que resulta de autoarima?
#  ¿como ver los argumentos?

# Función final de pronósticos:

# Nuestro objetivo sera configurar el modelo de prediccion de auto arima
def generate_forecast_df(model, steps):
    forecast, conf_int = model.predict(n_periods=steps+6, alpha=0.05, return_conf_int=True)
    
    forecast_index = pd.date_range(start=df.index[-1], periods=steps+7, freq='MS')[1:]
    forecast_df = pd.DataFrame({'Prediction': forecast, 'Lower Bound': conf_int[:, 0], 'Upper Bound': conf_int[:, 1]}, index=forecast_index)
    
    return forecast_df.dropna()

predicciones = generate_forecast_df(model,12)

sd_detrend.plot(style='--', label='Original')

predicciones['Prediction'].plot(label='Predicción')

plt.legend()

plt.show()

