import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Configuración de la interfaz
st.title("👨‍🔬 Laboratorio de K-Means (No Supervisado)")
st.write("Mueve la barra lateral para ver cómo el algoritmo agrupa los datos por cercanía.")

# Barra lateral de controles
k_value = st.sidebar.slider("Número de Clusters (K):", 2, 10, 4)
puntos = st.sidebar.number_input("Cantidad de datos:", 100, 500, 300)

# Generación y entrenamiento del modelo
X, _ = make_blobs(n_samples=puntos, centers=4, cluster_std=0.60, random_state=42)
kmeans = KMeans(n_clusters=k_value, n_init=10)
y_pred = kmeans.fit_predict(X)

# Renderizado de la gráfica
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           s=250, c='red', marker='X', label='Centroides')
ax.legend()
st.pyplot(fig)
