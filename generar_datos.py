import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configuración de los datos
num_meses = 12  # Datos para 12 meses
categorias = ["Producto A", "Producto B", "Producto C"]
fecha_inicio = datetime(2023, 1, 1)

# Generar listas de datos
fechas = [fecha_inicio + timedelta(days=30 * i) for i in range(num_meses)]
cantidades = np.random.randint(50, 150, size=num_meses)  # Cantidad aleatoria entre 50 y 150
objetivos = np.random.randint(100, 200, size=num_meses)  # Objetivo aleatorio entre 100 y 200
categorias_data = [random.choice(categorias) for _ in range(num_meses)]

# Crear DataFrame
data = {
    "Fecha": fechas,
    "Cantidad": cantidades,
    "Objetivo": objetivos,
    "Categoría": categorias_data,
}

df = pd.DataFrame(data)

# Guardar como CSV
df.to_csv("datos_dummy.csv", index=False)

print("Archivo de datos dummy generado: datos_dummy.csv")
