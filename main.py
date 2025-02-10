import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gp
import plotly.express as px
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

# Función para leer el dataset
def leer_dataset(url):
    datos = pd.read_csv(url)
    gdf = gp.GeoDataFrame(datos, geometry=gp.points_from_xy(datos.Longitud, datos.Latitud))
    gdf.set_crs(epsg=4326, inplace=True)
    return gdf

# Función para interpolar datos faltantes
def interpolar_datos(datos):
    datos_interpolados = datos.select_dtypes(include=[np.number])
    datos_interpolados = datos_interpolados.interpolate(method='linear', limit_direction='both', axis=0)
    datos_interpolados = datos_interpolados.interpolate(method='polynomial', order=3, limit_direction='both', axis=0)
    return datos.fillna(datos_interpolados)

# Cargar datos
URL = 'https://raw.githubusercontent.com/gabrielawad/programacion-para-ingenieria/refs/heads/main/archivos-datos/aplicaciones/analisis_clientes.csv'
df = leer_dataset(URL)
df = interpolar_datos(df)

# Título de la aplicación
st.title("Análisis de Clientes")

# Correlación entre Edad e Ingreso
st.subheader("Correlación Edad vs Ingreso Anual")
correlacion = df[['Edad', 'Ingreso_Anual_USD']].corr().iloc[0,1]
st.write(f"Coeficiente de correlación: {correlacion:.2f}")
fig = px.scatter(df, x='Edad', y='Ingreso_Anual_USD', color='Género')
st.plotly_chart(fig)

# Mapa global de clientes
st.subheader("Mapa de Ubicación de Clientes")
fig = px.scatter_mapbox(df, lat=df.geometry.y, lon=df.geometry.x, color='Frecuencia_Compra', 
                        mapbox_style="open-street-map", zoom=3)
st.plotly_chart(fig)

# Análisis de Clúster según frecuencia de compra
st.subheader("Análisis de Clúster")
n_clusters = st.slider("Selecciona el número de clústers", 2, 10, 3)
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(df[['Ingreso_Anual_USD', 'Frecuencia_Compra']])
df['Cluster'] = kmeans.labels_
fig = px.scatter(df, x='Ingreso_Anual_USD', y='Frecuencia_Compra', color=df['Cluster'].astype(str))
st.plotly_chart(fig)

# Gráfico de barras por género y frecuencia de compra
st.subheader("Distribución de Frecuencia de Compra por Género")
fig = px.bar(df, x='Género', y='Frecuencia_Compra', color='Género', barmode='group')
st.plotly_chart(fig)

# Mapa de calor según ingresos
st.subheader("Mapa de Calor de Ingresos")
fig = px.density_mapbox(df, lat=df.geometry.y, lon=df.geometry.x, z='Ingreso_Anual_USD', 
                        radius=10, mapbox_style="open-street-map")
st.plotly_chart(fig)

# Cálculo de distancias entre compradores de mayores ingresos
st.subheader("Distancia entre Compradores de Altos Ingresos")
top_ingresos = df.nlargest(10, 'Ingreso_Anual_USD')[['geometry']]
dist_matrix = squareform(pdist(top_ingresos.geometry.apply(lambda p: [p.x, p.y])))
st.write(pd.DataFrame(dist_matrix, columns=top_ingresos.index, index=top_ingresos.index))
