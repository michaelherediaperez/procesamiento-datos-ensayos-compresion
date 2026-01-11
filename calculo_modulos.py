"""
Este código calcula los móduos de elasticidad y de corte de los especímenes de 
maderas ya guaduas ensayada en laboratorio, sacando gráficos y obteniendo 
promedios para diseño.

Por: Michael Heredia Pérez
email: michael.hper@gmail.com  
fecha: noviembre 2025
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression

# Crear carpetas para guardar gráficos
os.makedirs("graficos_ensayos_compresion", exist_ok=True)
os.makedirs("graficos_modulos_elasticidad", exist_ok=True)

import matplotlib as mpl
# Configure matplotlib for STIX font - comprehensive setup
mpl.rcParams.update({
    # Primary font configuration
    "font.family": "serif",              # Use serif family
    "font.serif": ["STIX", "STIXGeneral", "STIX Two Text"], # STIX font priority
    "mathtext.fontset": "stix",          # Math expressions in STIX
    
    # Explicit font specification for all text elements
    "axes.labelsize": 18,
    "axes.titlesize": 18, 
    "legend.fontsize": 16,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "font.size": 16,
    
    # Line properties
    "lines.linewidth": 1.5
})

# -----
# Funciones para leer y procesar los datos de los ensayos de compresión.


def leer_ensayo_compresion(ruta_archivo):
    """
    Lee un archivo de ensayo de compresión con el formato dado:
    - 4 líneas de información
    - Datos a partir de línea 9
    - Columnas: carga [kN], actuador [mm], tiempo [s]
    Retorna arrays de carga, desplazamiento y tiempo.
    """

    with open(ruta_archivo, "r") as f:
        lineas = f.readlines()

    # Saltar encabezado (primeras 8 líneas)
    datos = [linea.strip().split() for linea in lineas[8:]]

    datos = np.array(datos, dtype=float)

    carga_kN = datos[:, 0]
    actuador_mm = datos[:, 1]
    tiempo_s = datos[:, 2]

    return carga_kN, actuador_mm, tiempo_s


def graficar_material(material, lista_archivos, carpeta_base):
    """
    Grafica todos los ensayos correspondientes a un material
    y guarda la figura en JPG y PDF.
    """

    plt.figure(figsize=(8, 6))

    for archivo in lista_archivos:
        ruta = os.path.join(carpeta_base, archivo)

        carga_kN, actuador_mm, tiempo_s = leer_ensayo_compresion(ruta)

        plt.plot(actuador_mm, carga_kN, label=archivo.replace(".txt", ""), linewidth=2)

    plt.title(f"Ensayos de compresión: {material}", fontsize=14)
    plt.xlabel("Deformación del actuador [mm]", fontsize=12)
    plt.ylabel("Carga [kN]", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Guardar en las dos extensiones
    ruta_jpg = f"graficos_ensayos_compresion/{material}.jpg"
    ruta_pdf = f"graficos_ensayos_compresion/{material}.pdf"

    plt.savefig(ruta_jpg, dpi=300)
    plt.savefig(ruta_pdf)

    plt.close()


ensayos_compresion = {
    "Cedro": ["cedro_1.txt", "cedro_2.txt", "cedro_3.txt"],
    "Laurel": ["laurel_1.txt", "laurel_2.txt", "laurel_3.txt"],
    "Nogal": ["nogal_1.txt", "nogal_2.txt", "nogal_3.txt"],
    "Guadua 1": ["G11.txt", "G12.txt"],
    "Guadua 2": ["G21.txt", "G22.txt"],
    "Guadua 3": ["G31.txt", "G32.txt"],
}


def graficar_todos(carpeta_base="ensayos-compresion-organizados"):
    for material, archivos in ensayos_compresion.items():
        graficar_material(material, archivos, carpeta_base)


# Llamada principal.
graficar_todos()


# -----
# Cálculo del módulo de elasticidad en cada ensayo y módulo promedio.


AREA = (0.05 * 0.05)      # [m²]  sección 5×5 cm
L0 = 0.15                 # [m]   longitud 15 cm


def obtener_esfuerzo_deformacion(carga_kN, actuador_mm):
    """
    Convierte carga y desplazamiento en esfuerzo \sigma [Pa] y deformación 
    \varepsilon.
    """
    carga_N = carga_kN * 1000.0
    delta_m = actuador_mm / 1000.0
    deform = delta_m / L0
    sigma = carga_N / AREA
    return sigma, deform


def encontrar_rango_lineal(sigma, deform, inicio_frac=0.05, ventana=90):
    """
    Encuentra automáticamente el tramo lineal (rango elástico) mediante:
    - Descartar el primer 30% de datos
    - Probar ventanas deslizantes
    - Elegir la que maximiza R²
    
    Retorna:
        sigma_lin, deform_lin, modelo (sklearn)
    """

    inicio = int(len(sigma) * inicio_frac)

    sigma = sigma[inicio:]
    deform = deform[inicio:]

    mejor_R2 = -1
    mejor_modelo = None
    mejor_sigma = None
    mejor_deform = None

    for i in range(len(sigma) - ventana):
        sig_w = sigma[i:i+ventana]
        def_w = deform[i:i+ventana]

        X = def_w.reshape(-1,1)
        y = sig_w

        modelo = LinearRegression()
        modelo.fit(X, y)
        R2 = modelo.score(X, y)

        # Guardar si es el mejor R² (región más lineal)
        if R2 > mejor_R2:
            mejor_R2 = R2
            mejor_sigma = sig_w
            mejor_deform = def_w
            mejor_modelo = modelo

    return mejor_sigma, mejor_deform, mejor_modelo


def graficar_ajuste_esfuerzo_deformacion(sigma, deform, sigma_lin, deform_lin, modelo, nombre):
    """
    Grafica σ–ε y la recta ajustada del rango elástico.
    Guarda la figura como JPG y PDF.
    """

    plt.figure(figsize=(8,6))

    plt.plot(deform, sigma, label="Curva completa", color="gray", alpha=0.6)
    plt.scatter(deform_lin, sigma_lin, s=10, color="blue", label="Datos lineales usados")

    deform_pred = np.linspace(deform_lin.min(), deform_lin.max(), 100).reshape(-1,1)
    sigma_pred = modelo.predict(deform_pred)

    plt.plot(deform_pred, sigma_pred, color="red", linewidth=2, label="Ajuste lineal")

    plt.title(f"Linealización del ensayo: {nombre}")
    plt.xlabel("Deformación unitaria ε [-]")
    plt.ylabel("Esfuerzo σ [Pa]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Nombre sin extensión
    base = nombre.replace(".txt", "")

    # Guardar
    ruta_jpg = f"graficos_modulos_elasticidad/{base}.jpg"
    ruta_pdf = f"graficos_modulos_elasticidad/{base}.pdf"

    plt.savefig(ruta_jpg, dpi=300)
    plt.savefig(ruta_pdf)

    plt.close()



# -------------------------------------------------------------------
# Subrutina: parámetros específicos para ciertos ensayos problemáticos
# -------------------------------------------------------------------

def obtener_parametros_especificos(nombre_archivo):
    """
    Devuelve los parámetros inicio_frac y ventana.
    Si un archivo necesita valores particulares, se configuran aquí.
    """

    # Parámetros por defecto (los que ya ajustaste)
    params_default = {
        "inicio_frac": 0.05,
        "ventana": 90
    }

    # Diccionario de casos especiales
    casos_especiales = {
        "laurel_1.txt": {"inicio_frac": 0.15, "ventana": 50}
    }

    # Retornar parámetros especiales si existen
    if nombre_archivo in casos_especiales:
        return casos_especiales[nombre_archivo]

    # Si no está en la lista, usar los parámetros globales
    return params_default


def calcular_E_ensayo(ruta_archivo, nombre_archivo):
    """
    - Lee los datos
    - Convierte a \sigma-\verepsilon
    - Encuentra rango lineal real
    - Calcula E con sklearn
    - Grafica \sigma-\verepsilon + recta ajustada
    """

    carga_kN, actuador_mm, tiempo_s = leer_ensayo_compresion(ruta_archivo)
    sigma, deform = obtener_esfuerzo_deformacion(carga_kN, actuador_mm)

    # Encontrar rango lineal
    #sigma_lin, deform_lin, modelo = encontrar_rango_lineal(sigma, deform)

    # Obtener parámetros correctos según el archivo
    params = obtener_parametros_especificos(nombre_archivo)

    sigma_lin, deform_lin, modelo = encontrar_rango_lineal(
        sigma, deform,
        inicio_frac=params["inicio_frac"],
        ventana=params["ventana"]
    )

    # Módulo de elasticidad
    E = modelo.coef_[0]   # Pa

    # Gráfica
    graficar_ajuste_esfuerzo_deformacion(sigma, deform, sigma_lin, deform_lin, modelo, nombre_archivo)

    return E


def obtener_modulos_por_material(material, lista_archivos, carpeta_base="ensayos-compresion-organizados"):
    """
    Retorna lista de módulos E de cada archivo y el promedio.
    """
    modulos = []

    for archivo in lista_archivos:
        ruta = os.path.join(carpeta_base, archivo)
        E = calcular_E_ensayo(ruta, archivo)
        modulos.append(E)

    return modulos, np.mean(modulos)


print("\n--- Módulos de elasticidad por material ---\n")

# for material, archivos in ensayos_compresion.items():
#     modulos, promedio = obtener_modulos_por_material(material, archivos)
#     print(f"{material}:")
#     for a, E in zip(archivos, modulos):
#         print(f"  {a}: {E/1e9:.2f} GPa")
#     print(f"Promedio: {promedio/1e9:.2f} GPa\n")

# Guardar resultados en archivo .txt
with open("resultados_modulos_elasticidad.txt", "w") as f:

    f.write("--- Módulos de elasticidad por material ---\n\n")

    for material, archivos in ensayos_compresion.items():

        modulos, promedio = obtener_modulos_por_material(material, archivos)

        f.write(f"{material}:\n")
        print(f"{material}:")

        for archivo, E in zip(archivos, modulos):
            linea = f"  {archivo}: {E/1e9:.3f} GPa\n"
            f.write(linea)
            print(linea, end="")

        f.write(f"  → Promedio: {promedio/1e9:.3f} GPa\n\n")
        print(f"  → Promedio: {promedio/1e9:.3f} GPa\n")
        
# --------------------------------------------------------
# Cálculo del módulo de corte a partir del módulo promedio E
# --------------------------------------------------------

def calcular_rango_modulo_corte(E_promedio):
    """
    Dado E_promedio [Pa], devuelve (G_min, G_max) en Pa
    según el criterio: G entre 1/25 * E y 1/16 * E.
    """
    G_min = E_promedio / 25.0
    G_max = E_promedio / 16.0
    return G_min, G_max


# Calcular y guardar resultados en archivo .txt
with open("resultados_modulo_corte.txt", "w") as f:
    f.write("--- Resultados: Módulo de corte (estimado) ---\n\n")

    for material, archivos in ensayos_compresion.items():
        # obtener E promedio (ya en Pa) usando la función existente
        modulos, promedio_E = obtener_modulos_por_material(material, archivos)

        G_min, G_max = calcular_rango_modulo_corte(promedio_E)

        # Escribir en archivo (convertir a GPa para una lectura cómoda)
        f.write(f"{material}:\n")
        f.write(f"  E_promedio: {promedio_E/1e9:.3f} GPa\n")
        f.write(f"  G_min (1/25 * E): {G_min/1e9:.3f} GPa\n")
        f.write(f"  G_max (1/16 * E): {G_max/1e9:.3f} GPa\n\n")

        # Imprimir en pantalla también
        print(f"{material}:")
        print(f"  E_promedio: {promedio_E/1e9:.3f} GPa")
        print(f"  G_min (1/25 * E): {G_min/1e9:.3f} GPa")
        print(f"  G_max (1/16 * E): {G_max/1e9:.3f} GPa\n")

print("Archivo 'resultados_modulo_corte.txt' guardado.")
