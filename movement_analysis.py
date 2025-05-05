import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean

def fix_density_plot(visualizations_folder):
    """
    Corrección específica para el error de la barra de color en el mapa de densidad
    
    Args:
        visualizations_folder: Carpeta con las visualizaciones
    """
    print("Corrigiendo mapa de densidad de posiciones...")
    
    # Crear figura
    plt.figure(figsize=(14, 10))
    
    # Cargar estadísticas para obtener datos de posiciones de patos
    stats_file = os.path.join(visualizations_folder, 'estadisticas_patos.csv')
    
    if not os.path.exists(stats_file):
        print(f"Error: No se encontró el archivo de estadísticas en {stats_file}")
        return False
    
    # Cargar datos de trayectorias desde el archivo JSON original
    data_file = "/home/alfonso/Duck-Tracker/batch_output/merged_results/merged_tracking_data.json"
    
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error cargando archivo de datos: {str(e)}")
        return False
    
    # Extraer datos de frames
    frames_data = data.get('frames', {})
    
    # Extraer todas las posiciones x, y
    all_x = []
    all_y = []
    
    for frame_idx, frame_data in frames_data.items():
        positions = frame_data.get('positions', {})
        for duck_id, duck_data in positions.items():
            position = duck_data.get('position')
            if position:
                all_x.append(position[0])
                all_y.append(position[1])
    
    # Crear mapa de densidad
    if all_x and all_y:
        # Guardar el resultado de kdeplot para usarlo con colorbar
        kde_plot = sns.kdeplot(x=all_x, y=all_y, cmap="viridis", fill=True, alpha=0.7, levels=20)
        
        # Añadir barra de color usando el resultado de kdeplot
        cbar = plt.colorbar(kde_plot.collections[0], label='Densidad')
        
        # Añadir también contornos para mejor visualización
        sns.kdeplot(x=all_x, y=all_y, cmap=None, linewidths=0.5, alpha=0.5, levels=10)
    
    plt.title('Mapa de Densidad de Posiciones de Patos', fontsize=15)
    plt.xlabel('X (píxeles)')
    plt.ylabel('Y (píxeles)')
    plt.grid(alpha=0.3)
    
    # Guardar mapa de densidad corregido
    output_file = os.path.join(visualizations_folder, 'mapa_densidad_posiciones_corregido.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Mapa de densidad corregido guardado en: {output_file}")
    return True

def fix_movement_analysis_code():
    """
    Genera una versión corregida del código para movement_analysis.py
    """
    fix_code = """# Código corregido para el error en la función analyze_duck_movements
# Reemplaza el bloque que genera el mapa de densidad (líneas aproximadas 271-284) con esto:

    # 7. Visualización de densidad de patos por zonas
    # Crear mapa de calor de densidad usando KDE para todas las posiciones
    plt.figure(figsize=(14, 10))
    
    # Extraer todas las posiciones x, y
    all_x = []
    all_y = []
    
    for duck_id, trajectory in duck_trajectories.items():
        if len(trajectory) > 0:
            all_x.extend(trajectory[:, 0])
            all_y.extend(trajectory[:, 1])
    
    # Crear mapa de densidad
    if all_x and all_y:
        # Guardar el resultado de kdeplot para usarlo con colorbar
        kde_plot = sns.kdeplot(x=all_x, y=all_y, cmap="viridis", fill=True, alpha=0.7, levels=20)
        
        # Añadir barra de color usando el resultado de kdeplot
        if kde_plot.collections:
            plt.colorbar(kde_plot.collections[0], label='Densidad')
        
        # Dibujar contornos
        sns.kdeplot(x=all_x, y=all_y, cmap=None, linewidths=0.5, alpha=0.5, levels=10)
    
    plt.title('Mapa de Densidad de Posiciones de Patos', fontsize=15)
    plt.xlabel('X (píxeles)')
    plt.ylabel('Y (píxeles)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'mapa_densidad_posiciones.png'), dpi=300)
    plt.close()
"""
    
    print("\nPara corregir el error en movement_analysis.py, edita el archivo y:")
    print("1. Busca la sección que genera el mapa de densidad (alrededor de la línea 271)")
    print("2. Reemplaza ese bloque de código con el siguiente:\n")
    print(fix_code)
    print("\nAlternativamente, puedes ejecutar la función fix_density_plot() que acabo de crear")
    print("para generar un nuevo mapa de densidad corregido sin modificar el script original.")

if __name__ == "__main__":
    # Carpeta con las visualizaciones
    visualizations_folder = "/home/alfonso/Duck-Tracker/batch_output/visualizations"
    
    # Corregir el mapa de densidad
    fixed = fix_density_plot(visualizations_folder)
    
    if fixed:
        print("\n¡Corrección aplicada con éxito!")
    else:
        print("\nHubo problemas al aplicar la corrección.")
    
    # Mostrar cómo corregir el código fuente
    fix_movement_analysis_code()