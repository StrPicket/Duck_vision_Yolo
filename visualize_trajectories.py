import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import colorsys

def visualize_trajectories(data_file, output_folder):
    """
    Visualiza las trayectorias de los patos con colores mejorados
    
    Args:
        data_file: Ruta al archivo JSON con los datos combinados
        output_folder: Carpeta donde guardar las visualizaciones
    """
    # Crear carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    # Cargar datos
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Extraer datos de frames
    frames_data = data.get('frames', {})
    print(f"Frames totales cargados: {len(frames_data)}")
    
    # Preparar estructuras para almacenar trayectorias
    duck_trajectories = {}  # Diccionario para almacenar trayectorias por ID de pato
    duck_colors = {}        # Diccionario para almacenar color de cada pato (del dataset)
    
    # Procesar cada frame para extraer trayectorias
    for frame_idx, frame_data in sorted(frames_data.items(), key=lambda x: int(x[0])):
        frame_number = frame_data.get('frame_number', int(frame_idx))
        positions = frame_data.get('positions', {})
        
        for duck_id, duck_data in positions.items():
            if duck_id not in duck_trajectories:
                duck_trajectories[duck_id] = []
                duck_colors[duck_id] = duck_data.get('color', 'yellow')
            
            position = duck_data.get('position')
            if position:
                # Almacenar (x, y, frame) para el gráfico 3D
                duck_trajectories[duck_id].append((position[0], position[1], frame_number))
    
    print(f"Patos detectados: {len(duck_trajectories)}")
    
    # Convertir trayectorias a arrays numpy para graficar
    for duck_id, trajectory in duck_trajectories.items():
        duck_trajectories[duck_id] = np.array(trajectory)
    
    # Generar colores vibrantes para cada pato
    # En lugar de usar solo amarillo/negro, usar un arcoíris de colores
    num_ducks = len(duck_trajectories)
    vibrant_colors = {}
    
    # Crear colores separados por el espectro de color HSV
    for i, duck_id in enumerate(duck_trajectories.keys()):
        # Generar un color del espectro HSV, evitando los colores muy claros o oscuros
        hue = i / num_ducks
        saturation = 0.8
        value = 0.9
        rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)
        vibrant_colors[duck_id] = rgb_color
    
    # Crear gráfico 2D de trayectorias con colores vibrantes
    plt.figure(figsize=(15, 10))
    
    # Establecer un estilo de fondo oscuro para resaltar colores
    plt.style.use('dark_background')
    
    # Dibujar cuadrícula
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    for duck_id, trajectory in duck_trajectories.items():
        if len(trajectory) > 1:
            # Usar colores vibrantes
            duck_color = vibrant_colors[duck_id]
            
            # Dibujar trayectoria con línea más gruesa para mejor visibilidad
            plt.plot(trajectory[:, 0], trajectory[:, 1], 
                    color=duck_color,
                    alpha=0.8, linewidth=2.5)
            
            # Marcar posición inicial con círculo
            plt.scatter(trajectory[0, 0], trajectory[0, 1], 
                      color=duck_color, marker='o', 
                      s=100, alpha=0.7)
            
            # Marcar posición final con estrella
            plt.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                      color=duck_color, marker='*', 
                      s=200, alpha=0.9)
            
            # Añadir etiqueta de ID en posición final
            plt.annotate(duck_id, (trajectory[-1, 0], trajectory[-1, 1]), 
                        color='white', fontsize=12, fontweight='bold',
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc=duck_color, alpha=0.7))
    
    plt.title('Trayectorias de Patos', fontsize=18, fontweight='bold')
    plt.xlabel('X (pixels)', fontsize=14)
    plt.ylabel('Y (pixels)', fontsize=14)
    
    # Guardar gráfico de trayectorias
    plt.savefig(os.path.join(output_folder, 'trayectorias_2d_colores.png'), dpi=300, bbox_inches='tight')
    print(f"Gráfico de trayectorias guardado en: {os.path.join(output_folder, 'trayectorias_2d_colores.png')}")
    
    # Crear gráfico 3D de trayectorias con colores vibrantes
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Fondo negro para resaltar colores
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # Graficar trayectorias 3D
    for duck_id, trajectory in duck_trajectories.items():
        if len(trajectory) > 1:  # Solo graficar si hay más de un punto
            # Usar color vibrante
            duck_color = vibrant_colors[duck_id]
            
            # Línea más gruesa para mejor visibilidad
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                    color=duck_color,
                    alpha=0.8, linewidth=3, label=duck_id)
            
            # Marcar posición inicial y final con esferas más grandes
            ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                      color=duck_color,
                      marker='o', s=150, alpha=0.8)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                      color=duck_color,
                      marker='*', s=250, alpha=0.9)
            
            # Añadir etiqueta en 3D
            ax.text(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                   duck_id, color='white', fontsize=12, fontweight='bold',
                   backgroundcolor=duck_color)
    
    # Configurar gráfico 3D
    ax.set_xlabel('X (pixels)', fontsize=14, color='white')
    ax.set_ylabel('Y (pixels)', fontsize=14, color='white')
    ax.set_zlabel('Frame', fontsize=14, color='white')
    ax.set_title('Trayectorias 3D de Patos', fontsize=18, fontweight='bold', color='white')
    
    # Añadir iluminación para efecto 3D mejorado
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    ax.xaxis._axinfo["grid"]['color'] = (1,1,1,0.3)
    ax.yaxis._axinfo["grid"]['color'] = (1,1,1,0.3)
    ax.zaxis._axinfo["grid"]['color'] = (1,1,1,0.3)
    
    # Establecer punto de vista para mejor visualización
    ax.view_init(elev=30, azim=45)
    
    # Guardar gráfico 3D mejorado
    plt.savefig(os.path.join(output_folder, 'trayectorias_3d_colores.png'), dpi=300, bbox_inches='tight')
    print(f"Gráfico 3D guardado en: {os.path.join(output_folder, 'trayectorias_3d_colores.png')}")
    
    # Crear mapa de densidad 2D con colores para cada pato
    plt.figure(figsize=(15, 10))
    plt.style.use('dark_background')
    
    # Fondo oscuro para el mapa de densidad
    plt.gca().set_facecolor('black')
    
    # Dibujar cuadrícula
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Crear arrays para el mapa de calor
    all_points_x = []
    all_points_y = []
    
    for duck_id, trajectory in duck_trajectories.items():
        if len(trajectory) > 0:
            # Extraer coordenadas x, y
            all_points_x.extend(trajectory[:, 0])
            all_points_y.extend(trajectory[:, 1])
    
    # Crear mapa de calor usando KDE (Kernel Density Estimation)
    if all_points_x and all_points_y:
        sns.kdeplot(x=all_points_x, y=all_points_y, cmap="plasma", fill=True, alpha=0.5, levels=20)
        
        # Superponer trayectorias con colores vibrantes
        for duck_id, trajectory in duck_trajectories.items():
            if len(trajectory) > 1:
                duck_color = vibrant_colors[duck_id]
                plt.plot(trajectory[:, 0], trajectory[:, 1], 
                        color=duck_color,
                        alpha=0.8, linewidth=2.5)
                
                # Marcar puntos final e inicial
                plt.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                          color=duck_color, marker='*', 
                          s=200, alpha=0.9)
                
                # Añadir etiqueta
                plt.annotate(duck_id, (trajectory[-1, 0], trajectory[-1, 1]), 
                            color='white', fontsize=12, fontweight='bold',
                            xytext=(5, 5), textcoords='offset points',
                            bbox=dict(boxstyle="round,pad=0.3", fc=duck_color, alpha=0.7))
    
    plt.title('Mapa de Densidad y Trayectorias de Patos', fontsize=18, fontweight='bold')
    plt.xlabel('X (pixels)', fontsize=14)
    plt.ylabel('Y (pixels)', fontsize=14)
    
    # Guardar mapa de densidad con trayectorias
    plt.savefig(os.path.join(output_folder, 'mapa_densidad_trayectorias_colores.png'), dpi=300, bbox_inches='tight')
    print(f"Mapa de densidad con trayectorias guardado en: {os.path.join(output_folder, 'mapa_densidad_trayectorias_colores.png')}")
    
    print("\n¡Visualizaciones mejoradas completas!")

if __name__ == "__main__":
    # Ruta al archivo de datos combinados
    data_file = "/home/alfonso/Duck-Tracker/batch_output/merged_results/merged_tracking_data.json"
    
    # Carpeta para guardar las visualizaciones
    output_folder = "/home/alfonso/Duck-Tracker/batch_output/visualizations"
    
    # Generar visualizaciones
    visualize_trajectories(data_file, output_folder)