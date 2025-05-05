import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import colorsys

def create_trajectory_animation(data_file, output_folder):
    """
    Crea una animación 3D de las trayectorias de los patos con colores vibrantes
    y velocidad reducida para mejor visualización
    
    Args:
        data_file: Ruta al archivo JSON con los datos combinados
        output_folder: Carpeta donde guardar la animación
    """
    # Crear carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    # Cargar datos
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Extraer datos de frames
    frames_data = data.get('frames', {})
    print(f"Frames totales cargados: {len(frames_data)}")
    
    # Ordenar frames por número
    sorted_frames = sorted([(int(k), v) for k, v in frames_data.items()], key=lambda x: x[0])
    
    # Crear lista de frames
    frame_numbers = [item[0] for item in sorted_frames]
    
    # Preparar estructuras para almacenar trayectorias
    duck_positions = {}  # Diccionario para almacenar posiciones por frame
    duck_colors = {}     # Diccionario para almacenar color original de cada pato
    
    # Procesar cada frame para extraer posiciones
    for frame_idx, frame_data in sorted_frames:
        frame_number = frame_data.get('frame_number', frame_idx)
        positions = frame_data.get('positions', {})
        
        duck_positions[frame_number] = {}
        
        for duck_id, duck_data in positions.items():
            if duck_id not in duck_colors:
                duck_colors[duck_id] = duck_data.get('color', 'yellow')
            
            position = duck_data.get('position')
            if position:
                duck_positions[frame_number][duck_id] = position
    
    # Determinar el rango de frames
    min_frame = min(frame_numbers)
    max_frame = max(frame_numbers)
    
    # Crear diccionario para almacenar trayectorias acumuladas
    accumulated_trajectories = {}
    
    # Generar colores vibrantes para cada pato
    all_duck_ids = set()
    for frame_data in duck_positions.values():
        all_duck_ids.update(frame_data.keys())
    
    num_ducks = len(all_duck_ids)
    vibrant_colors = {}
    
    # Crear colores separados por el espectro de color HSV para cada pato
    for i, duck_id in enumerate(sorted(all_duck_ids)):
        # Generar un color del espectro HSV, evitando los colores muy claros o oscuros
        hue = i / num_ducks
        saturation = 0.9  # Alta saturación para colores más vibrantes
        value = 0.95      # Alto valor para colores más brillantes
        rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)
        vibrant_colors[duck_id] = rgb_color
    
    # Crear figura 3D para la animación con fondo negro
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Configurar fondo negro para mejor contraste
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # Función para inicializar la animación
    def init():
        ax.clear()
        ax.set_xlabel('X (pixels)', color='white', fontsize=12)
        ax.set_ylabel('Y (pixels)', color='white', fontsize=12)
        ax.set_zlabel('Frame', color='white', fontsize=12)
        ax.set_title('Animación de Trayectorias 3D de Patos', fontsize=18, color='white', fontweight='bold')
        
        # Configurar color de los ejes y la cuadrícula
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        
        ax.xaxis._axinfo["grid"]['color'] = (1,1,1,0.3)
        ax.yaxis._axinfo["grid"]['color'] = (1,1,1,0.3)
        ax.zaxis._axinfo["grid"]['color'] = (1,1,1,0.3)
        
        # Establecer límites fijos para los ejes
        ax.set_xlim(0, 640)  # Ancho típico de frame
        ax.set_ylim(0, 384)  # Alto típico de frame
        ax.set_zlim(min_frame, max_frame)
        
        # Ajustar color de las etiquetas de los ejes
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')
        
        return []
    
    # Función de animación para actualizar cada frame
    def animate(i):
        # Limpiar eje
        ax.clear()
        
        # Obtener el frame actual para la animación con progresión más lenta
        # Reducir el salto entre frames para que la animación sea más lenta
        current_frame = min_frame + i * 5  # Saltos de 5 frames en lugar de 10
        if current_frame > max_frame:
            current_frame = max_frame
            
        # Mostrar progreso
        if i % 10 == 0:
            print(f"Animando frame {current_frame}/{max_frame}")
        
        # Actualizar trayectorias acumuladas hasta el frame actual
        for frame_number in range(min_frame, current_frame + 1):
            if frame_number in duck_positions:
                for duck_id, position in duck_positions[frame_number].items():
                    if duck_id not in accumulated_trajectories:
                        accumulated_trajectories[duck_id] = []
                    
                    # Añadir posición a la trayectoria acumulada
                    accumulated_trajectories[duck_id].append((position[0], position[1], frame_number))
        
        # Dibujar trayectorias acumuladas con colores vibrantes
        for duck_id, trajectory in accumulated_trajectories.items():
            if len(trajectory) > 1:
                # Convertir a array numpy
                trajectory_array = np.array(trajectory)
                
                # Usar color vibrante asignado a este pato
                duck_color = vibrant_colors.get(duck_id, (1, 1, 1))  # Blanco como fallback
                
                # Graficar trayectoria con línea más gruesa
                ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2],
                        color=duck_color,
                        alpha=0.8, linewidth=3)
                
                # Marcar posición actual con esfera más grande
                last_pos = trajectory_array[-1]
                ax.scatter(last_pos[0], last_pos[1], last_pos[2],
                         color=duck_color,
                         marker='o', s=100, alpha=0.9)
                
                # Añadir etiqueta de ID
                ax.text(last_pos[0], last_pos[1], last_pos[2], 
                       duck_id, color='white', fontsize=10, fontweight='bold',
                       backgroundcolor=duck_color)
        
        # Configurar vista con estilo mejorado
        ax.set_xlabel('X (pixels)', color='white', fontsize=12)
        ax.set_ylabel('Y (pixels)', color='white', fontsize=12)
        ax.set_zlabel('Frame', color='white', fontsize=12)
        ax.set_title(f'Trayectorias de Patos (Frame {current_frame}/{max_frame})', 
                    fontsize=18, color='white', fontweight='bold')
        
        # Configurar color de los ejes y la cuadrícula
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        
        ax.xaxis._axinfo["grid"]['color'] = (1,1,1,0.3)
        ax.yaxis._axinfo["grid"]['color'] = (1,1,1,0.3)
        ax.zaxis._axinfo["grid"]['color'] = (1,1,1,0.3)
        
        # Ajustar color de las etiquetas de los ejes
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')
        
        # Establecer límites fijos
        ax.set_xlim(0, 640)
        ax.set_ylim(0, 384)
        ax.set_zlim(min_frame, max_frame)
        
        # Rotar vista más lentamente para mejor visualización
        ax.view_init(elev=30, azim=i/2)  # Rotación más lenta (dividido por 2)
        
        return []
    
    # Crear animación con más frames para que sea más lenta
    # Aumentar el número de frames para que la animación sea más larga
    num_frames = min(200, (max_frame - min_frame) // 5)  # Más frames y saltos más pequeños
    
    print(f"Creando animación con {num_frames} frames...")
    
    # Aumentar el intervalo entre frames para velocidad más lenta
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=200, blit=True)
    
    # Guardar animación con FPS más bajo para hacerla más lenta
    output_path = os.path.join(output_folder, 'animacion_trayectorias_lenta.mp4')
    anim.save(output_path, writer='ffmpeg', fps=8, dpi=200)  # Reducido de 15 a 8 FPS
    
    print(f"Animación lenta guardada en: {output_path}")
    plt.close()
    
    # También crear un gif más lento
    output_path_gif = os.path.join(output_folder, 'animacion_trayectorias_lenta.gif')
    anim.save(output_path_gif, writer='pillow', fps=5, dpi=150)  # Reducido de 10 a 5 FPS
    
    print(f"GIF lento guardado en: {output_path_gif}")
    print("\n¡Animación con colores vibrantes y velocidad reducida completada!")

if __name__ == "__main__":
    # Ruta al archivo de datos combinados
    data_file = "/home/alfonso/Duck-Tracker/batch_output/merged_results/merged_tracking_data.json"
    
    # Carpeta para guardar las visualizaciones
    output_folder = "/home/alfonso/Duck-Tracker/batch_output/visualizations"
    
    # Generar animación
    create_trajectory_animation(data_file, output_folder)