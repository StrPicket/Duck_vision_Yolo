import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import colorsys
from scipy.ndimage import gaussian_filter1d
import math
import plotly.graph_objects as go
import shutil

def process_cameraman_visualizations(cameraman_visualizations_folder):
    """
    Procesa las visualizaciones del movimiento del camarógrafo para incluirlas en el informe
    
    Args:
        cameraman_visualizations_folder: Carpeta con las visualizaciones del camarógrafo
        
    Returns:
        Lista de tuplas con (nombre_archivo, título, descripción) para cada visualización
    """
    # Verificar si la carpeta existe
    if not os.path.exists(cameraman_visualizations_folder):
        print(f"¡Advertencia! La carpeta {cameraman_visualizations_folder} no existe.")
        return []
    
    # Mapeo de archivos a títulos y descripciones
    visualization_info = {
        'cameraman_3d_trajectory.png': (
            'Trayectoria 3D del Camarógrafo', 
            'Visualización tridimensional del desplazamiento del camarógrafo en el espacio, ' +
            'mostrando la evolución de su posición y orientación a lo largo del tiempo.'
        ),
        'cameraman_xy_trajectory.png': (
            'Vista Superior del Movimiento (XY)', 
            'Proyección en el plano horizontal (plano XY) del movimiento del camarógrafo, ' +
            'mostrando su desplazamiento desde una vista cenital.'
        ),
        'cameraman_xz_trajectory.png': (
            'Vista Lateral del Movimiento (XZ)', 
            'Proyección en el plano vertical (plano XZ) del movimiento del camarógrafo, ' +
            'revelando cambios de altura y profundidad durante la grabación.'
        ),
        'cameraman_rotation.png': (
            'Rotación del Camarógrafo', 
            'Gráfica que muestra la evolución de los ángulos de rotación (roll, pitch, yaw) ' +
            'del camarógrafo a lo largo del tiempo, revelando cómo cambia la orientación de la cámara.'
        ),
        'cameraman_displacement.png': (
            'Desplazamiento del Camarógrafo', 
            'Gráfica que muestra la magnitud del desplazamiento del camarógrafo desde su ' +
            'posición inicial a lo largo del tiempo, cuantificando la distancia recorrida.'
        ),
        'cameraman_animation.mp4': (
            'Animación del Movimiento del Camarógrafo',
            'Animación 3D que muestra el movimiento y rotación del camarógrafo ' +
            'a lo largo del tiempo, permitiendo visualizar su comportamiento de forma dinámica.'
        )
    }
    
    # Lista para almacenar las visualizaciones encontradas
    cameraman_visualizations = []
    
    # Comprobar específicamente si el archivo de animación existe en la carpeta
    animation_file = 'cameraman_animation.mp4'
    animation_path = os.path.join(cameraman_visualizations_folder, animation_file)
    
    # Buscar archivos en la carpeta
    for filename in os.listdir(cameraman_visualizations_folder):
        if filename in visualization_info:
            title, description = visualization_info[filename]
            cameraman_visualizations.append((filename, title, description))
    
    # Verificar si hay algún archivo de video o animación adicional que no esté en la lista predefinida
    for filename in os.listdir(cameraman_visualizations_folder):
        if filename.endswith(('.mp4', '.gif', '.avi')) and filename not in visualization_info:
            # Si encontramos algún archivo de video no mapeado, lo añadimos con un título genérico
            title = f"Animación: {os.path.splitext(filename)[0].replace('_', ' ').title()}"
            description = "Animación que muestra el movimiento del camarógrafo durante la grabación."
            cameraman_visualizations.append((filename, title, description))
            print(f"Encontrado archivo de animación adicional: {filename}")
    
    # Si no se encuentra ninguna animación, buscar en ubicaciones alternativas
    if not any(filename.endswith('.mp4') for filename, _, _ in cameraman_visualizations):
        print("No se encontró ningún archivo de animación del camarógrafo. Buscando en ubicaciones alternativas...")
        
        # Construir posibles rutas alternativas para buscar el archivo de animación
        base_dir = os.path.dirname(cameraman_visualizations_folder)
        parent_dir = os.path.dirname(base_dir)
        
        alt_paths = [
            os.path.join(base_dir, animation_file),
            os.path.join(parent_dir, animation_file),
            os.path.join(parent_dir, "cameraman_visualizations", animation_file),
            os.path.join(os.path.dirname(parent_dir), "cameraman_visualizations", animation_file)
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                print(f"¡Encontrado archivo de animación en ubicación alternativa: {alt_path}!")
                # Copiar el archivo a la carpeta de visualizaciones del camarógrafo
                import shutil
                try:
                    shutil.copy2(alt_path, os.path.join(cameraman_visualizations_folder, animation_file))
                    title, description = visualization_info[animation_file]
                    cameraman_visualizations.append((animation_file, title, description))
                    print(f"Archivo de animación copiado a {cameraman_visualizations_folder}")
                    break
                except Exception as e:
                    print(f"Error al copiar el archivo de animación: {e}")
    
    print(f"Encontradas {len(cameraman_visualizations)} visualizaciones del camarógrafo")
    return cameraman_visualizations

def create_duck_3d_model(data_file, output_folder):
    """
    Crea un modelo 3D profesional de los patitos con métricas detalladas
    
    Args:
        data_file: Ruta al archivo JSON con los datos combinados
        output_folder: Carpeta donde guardar el modelo
    """
    # Crear carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    print("Cargando datos para crear modelo 3D profesional...")
    
    # Cargar datos
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Extraer datos de frames
    frames_data = data.get('frames', {})
    print(f"Frames totales cargados: {len(frames_data)}")
    
    # Preparar estructuras para almacenar trayectorias
    duck_trajectories = {}  # Diccionario para almacenar trayectorias por ID de pato
    duck_colors = {}        # Diccionario para almacenar color de cada pato
    duck_velocities = {}    # Almacenar velocidades
    duck_accelerations = {} # Almacenar aceleraciones
    duck_metrics = {}       # Almacenar métricas calculadas
    
    # Procesar cada frame para extraer trayectorias
    for frame_idx, frame_data in sorted(frames_data.items(), key=lambda x: int(x[0])):
        frame_number = frame_data.get('frame_number', int(frame_idx))
        positions = frame_data.get('positions', {})
        
        for duck_id, duck_data in positions.items():
            if duck_id not in duck_trajectories:
                duck_trajectories[duck_id] = []
                duck_velocities[duck_id] = []
                duck_accelerations[duck_id] = []
                duck_colors[duck_id] = duck_data.get('color', 'yellow')
            
            position = duck_data.get('position')
            velocity = duck_data.get('velocity', (0, 0))
            
            if position:
                # Almacenar (x, y, frame) para el modelo 3D
                duck_trajectories[duck_id].append((position[0], position[1], frame_number))
                
                # Almacenar velocidad si existe
                if velocity:
                    vx, vy = velocity
                    duck_velocities[duck_id].append((vx, vy, frame_number))
    
    print(f"Patos detectados: {len(duck_trajectories)}")
    
    # Calcular aceleraciones para cada pato (derivada de velocidad)
    for duck_id, velocities in duck_velocities.items():
        if len(velocities) > 1:
            for i in range(1, len(velocities)):
                vx_prev, vy_prev, frame_prev = velocities[i-1]
                vx_curr, vy_curr, frame_curr = velocities[i]
                
                # Calcular cambio en velocidad
                frame_diff = frame_curr - frame_prev
                if frame_diff > 0:  # Evitar división por cero
                    ax = (vx_curr - vx_prev) / frame_diff
                    ay = (vy_curr - vy_prev) / frame_diff
                    
                    duck_accelerations[duck_id].append((ax, ay, frame_curr))
    
    # Convertir trayectorias a arrays numpy para cálculos
    trajectory_arrays = {}
    for duck_id, trajectory in duck_trajectories.items():
        if len(trajectory) > 1:
            trajectory_arrays[duck_id] = np.array(trajectory)
    
    # Calcular métricas importantes para cada pato
    for duck_id, trajectory_array in trajectory_arrays.items():
        # Calcular distancia total recorrida
        distances = np.sqrt(np.sum(np.diff(trajectory_array[:, :2], axis=0)**2, axis=1))
        total_distance = np.sum(distances)
        
        # Calcular velocidad promedio
        time_span = trajectory_array[-1, 2] - trajectory_array[0, 2]
        avg_speed = total_distance / time_span if time_span > 0 else 0
        
        # Calcular velocidad máxima
        if len(distances) > 0:
            max_speed = np.max(distances)
        else:
            max_speed = 0
            
        # Calcular aceleración media si hay datos de aceleración
        if duck_id in duck_accelerations and len(duck_accelerations[duck_id]) > 0:
            acc_array = np.array(duck_accelerations[duck_id])
            avg_acceleration = np.mean(np.sqrt(acc_array[:, 0]**2 + acc_array[:, 1]**2))
        else:
            avg_acceleration = 0
            
        # Calcular rango de movimiento
        x_range = np.max(trajectory_array[:, 0]) - np.min(trajectory_array[:, 0])
        y_range = np.max(trajectory_array[:, 1]) - np.min(trajectory_array[:, 1])
        
        # Guardar métricas
        duck_metrics[duck_id] = {
            'total_distance': total_distance,
            'avg_speed': avg_speed,
            'max_speed': max_speed,
            'avg_acceleration': avg_acceleration,
            'x_range': x_range,
            'y_range': y_range,
            'start_frame': trajectory_array[0, 2],
            'end_frame': trajectory_array[-1, 2],
            'duration': trajectory_array[-1, 2] - trajectory_array[0, 2],
            'points': len(trajectory_array)
        }
    
    # Generar colores vibrantes para cada pato
    num_ducks = len(duck_trajectories)
    vibrant_colors = {}
    
    # Crear colores separados por el espectro de color HSV
    for i, duck_id in enumerate(duck_trajectories.keys()):
        # Generar un color del espectro HSV
        hue = i / num_ducks
        saturation = 0.9
        value = 0.95
        rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)
        # Convertir a formato hexadecimal para Plotly
        hex_color = f'#{int(rgb_color[0]*255):02x}{int(rgb_color[1]*255):02x}{int(rgb_color[2]*255):02x}'
        vibrant_colors[duck_id] = hex_color
    
    # Crear figura 3D 
    fig = go.Figure()
    
    # Función para crear un modelo 3D mejorado del pato
    def create_enhanced_duck_model(trajectory_points, color, duck_id, metrics):
        # Convertir a array numpy para manipulación
        points = np.array(trajectory_points)
        
        # Suavizar trayectoria para modelo más realista
        if len(points) > 5:
            points_smoothed = np.copy(points)
            points_smoothed[:, 0] = gaussian_filter1d(points[:, 0], sigma=1.5)
            points_smoothed[:, 1] = gaussian_filter1d(points[:, 1], sigma=1.5)
        else:
            points_smoothed = points
        
        # Calcular promedio de posiciones para el centro del patito
        center_x = np.mean(points_smoothed[:, 0])
        center_y = np.mean(points_smoothed[:, 1])
        center_z = np.mean(points_smoothed[:, 2])
        
        # Determinar dirección de movimiento para orientar el pato
        if len(points_smoothed) > 1:
            # Usar los últimos puntos para determinar dirección
            last_points = points_smoothed[-5:] if len(points_smoothed) > 5 else points_smoothed
            avg_dir_x = np.mean(np.diff(last_points[:, 0]))
            avg_dir_y = np.mean(np.diff(last_points[:, 1]))
            
            # Normalizar dirección
            dir_len = np.sqrt(avg_dir_x**2 + avg_dir_y**2)
            if dir_len > 0:
                avg_dir_x = avg_dir_x / dir_len
                avg_dir_y = avg_dir_y / dir_len
            else:
                # Si no hay dirección clara, usar valor por defecto
                avg_dir_x, avg_dir_y = 1, 0
        else:
            avg_dir_x, avg_dir_y = 1, 0
        
        # Ajustar tamaño del pato según métricas
        base_size = 15
        # Escalar según distancia recorrida o duración
        if metrics['total_distance'] > 0:
            scale_factor = min(1.5, max(0.5, math.log10(1 + metrics['total_distance']) / 2))
            body_size = base_size * scale_factor
        else:
            body_size = base_size
        
        # Crear "cuerpo" del pato como una elipsoide orientada según la dirección
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        
        # Crear cuerpo elipsoide
        x = body_size * np.outer(np.cos(u), np.sin(v))
        y = body_size * np.outer(np.sin(u), np.sin(v))
        z = body_size * 0.7 * np.outer(np.ones(np.size(u)), np.cos(v))  # Aplanar verticalmente
        
        # Rotar según la dirección de movimiento
        angle = np.arctan2(avg_dir_y, avg_dir_x)
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        # Aplicar rotación y traslación
        for i in range(len(x)):
            for j in range(len(x[i])):
                point = np.array([x[i, j], y[i, j], z[i, j]])
                rotated_point = rot_matrix @ point
                x[i, j] = rotated_point[0] + center_x
                y[i, j] = rotated_point[1] + center_y
                z[i, j] = rotated_point[2] + center_z
        
        # Crear superficie del cuerpo
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            opacity=0.8,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            name=f'Cuerpo de {duck_id}'
        ))
        
        # Crear "cabeza" - otra elipsoide más pequeña
        head_size = body_size * 0.6
        head_dist = body_size * 0.8  # Distancia de la cabeza al cuerpo
        
        # Posición de la cabeza según dirección
        head_x = center_x + head_dist * avg_dir_x
        head_y = center_y + head_dist * avg_dir_y
        
        # Crear cabeza elipsoide
        x_head = head_size * np.outer(np.cos(u), np.sin(v))
        y_head = head_size * np.outer(np.sin(u), np.sin(v))
        z_head = head_size * 0.8 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Aplicar rotación y traslación
        for i in range(len(x_head)):
            for j in range(len(x_head[i])):
                point = np.array([x_head[i, j], y_head[i, j], z_head[i, j]])
                rotated_point = rot_matrix @ point
                x_head[i, j] = rotated_point[0] + head_x
                y_head[i, j] = rotated_point[1] + head_y
                z_head[i, j] = rotated_point[2] + center_z
        
        # Crear superficie de la cabeza
        fig.add_trace(go.Surface(
            x=x_head, y=y_head, z=z_head,
            opacity=0.9,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            name=f'Cabeza de {duck_id}'
        ))
        
        # Crear "pico" - una pirámide
        beak_size = head_size * 0.5
        # Posición del pico (adelante de la cabeza)
        beak_x = head_x + head_size * avg_dir_x * 0.8
        beak_y = head_y + head_size * avg_dir_y * 0.8
        
        # Determinar color del pico según tipo de pato
        if duck_colors[duck_id] == 'yellow':
            beak_color = '#FF8000'  # Naranja para patos amarillos
        else:
            beak_color = '#FF2000'  # Rojo para patos negros
        
        # Agregar pico triangular
        tip_x = beak_x + beak_size * avg_dir_x * 1.2
        tip_y = beak_y + beak_size * avg_dir_y * 1.2
        
        # Base del pico (un cuadrado perpendicular a la dirección)
        perp_dir_x, perp_dir_y = -avg_dir_y, avg_dir_x  # Vector perpendicular
        
        base_points = [
            [beak_x + perp_dir_x * beak_size*0.4, beak_y + perp_dir_y * beak_size*0.4, center_z + beak_size*0.2],
            [beak_x - perp_dir_x * beak_size*0.4, beak_y - perp_dir_y * beak_size*0.4, center_z + beak_size*0.2],
            [beak_x - perp_dir_x * beak_size*0.4, beak_y - perp_dir_y * beak_size*0.4, center_z - beak_size*0.2],
            [beak_x + perp_dir_x * beak_size*0.4, beak_y + perp_dir_y * beak_size*0.4, center_z - beak_size*0.2]
        ]
        
        # Triángulos del pico (uno por cada cara)
        for i in range(4):
            j = (i + 1) % 4
            fig.add_trace(go.Mesh3d(
                x=[tip_x, base_points[i][0], base_points[j][0]],
                y=[tip_y, base_points[i][1], base_points[j][1]],
                z=[center_z, base_points[i][2], base_points[j][2]],
                color=beak_color,
                opacity=0.9,
                name=f'Pico de {duck_id}'
            ))
        
        # Añadir trayectoria como línea
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='lines',
            line=dict(color=color, width=3),
            opacity=0.7,
            name=f'Trayectoria de {duck_id}'
        ))
        
        # Añadir puntos de inicio y fin
        fig.add_trace(go.Scatter3d(
            x=[points[0, 0]], y=[points[0, 1]], z=[points[0, 2]],
            mode='markers',
            marker=dict(color=color, size=8, symbol='circle'),
            name=f'Inicio de {duck_id}'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[points[-1, 0]], y=[points[-1, 1]], z=[points[-1, 2]],
            mode='markers',
            marker=dict(color=color, size=10, symbol='diamond'),
            name=f'Fin de {duck_id}'
        ))
        
        # Añadir etiqueta en 3D con métricas resumidas
        metrics_text = (
            f"{duck_id}<br>"
            f"Dist: {metrics['total_distance']:.1f} px<br>"
            f"Vel: {metrics['avg_speed']:.2f} px/f"
        )
        
        fig.add_trace(go.Scatter3d(
            x=[center_x], y=[center_y], z=[center_z + body_size + 10],
            mode='text',
            text=[metrics_text],
            textfont=dict(color='white', size=12, family='Arial Bold'),
            name=f'Etiqueta de {duck_id}'
        ))
    
    # Crear modelos 3D para cada pato
    for duck_id, trajectory in duck_trajectories.items():
        if len(trajectory) > 1:  # Solo crear modelo si hay suficientes puntos
            create_enhanced_duck_model(
                trajectory, 
                vibrant_colors[duck_id], 
                duck_id, 
                duck_metrics[duck_id]
            )
    
    # Crear tabla de métricas
    metrics_table = create_metrics_table(duck_metrics, duck_colors, vibrant_colors)
    
    # Configurar la escena 3D
    fig.update_layout(
        scene=dict(
            xaxis_title='X (píxeles)',
            yaxis_title='Y (píxeles)',
            zaxis_title='Frame',
            aspectmode='data'
        ),
        title={
            'text': "Modelo 3D Profesional de Patos",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        width=1000,
        height=800,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.5)"
        ),
        margin=dict(l=0, r=0, b=0, t=60)
    )
    
    # Guardar como HTML interactivo
    output_file = os.path.join(output_folder, 'modelo_3d_patos_profesional.html')
    fig.write_html(output_file, include_plotlyjs='cdn', full_html=True)
    print(f"Modelo 3D profesional interactivo guardado en: {output_file}")
    
    # Guardar también una versión estática como imagen de alta resolución
    static_output = os.path.join(output_folder, 'modelo_3d_patos_profesional.png')
    fig.write_image(static_output, width=1600, height=900, scale=2)
    print(f"Imagen estática del modelo guardada en: {static_output}")
    
    # Guardar métricas detalladas en CSV
    metrics_csv = os.path.join(output_folder, 'metricas_detalladas_patos.csv')
    metrics_df = pd.DataFrame.from_dict(duck_metrics, orient='index')
    metrics_df.index.name = 'duck_id'
    metrics_df.reset_index(inplace=True)
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Métricas detalladas guardadas en: {metrics_csv}")
    
    return output_file, duck_metrics, vibrant_colors


def create_metrics_table(duck_metrics, duck_colors, vibrant_colors):
    """
    Crea una tabla HTML con métricas detalladas de los patos
    
    Args:
        duck_metrics: Diccionario con métricas por pato
        duck_colors: Colores originales de los patos
        vibrant_colors: Colores usados en la visualización
        
    Returns:
        Tabla HTML con las métricas
    """
    metrics_html = """
    <table style="width:100%; border-collapse: collapse; margin-top: 20px; font-family: Arial;">
        <tr style="background-color: #3498db; color: white; text-align: center; font-weight: bold;">
            <th style="padding: 10px; border: 1px solid #ddd;">Pato ID</th>
            <th style="padding: 10px; border: 1px solid #ddd;">Color</th>
            <th style="padding: 10px; border: 1px solid #ddd;">Distancia Total (px)</th>
            <th style="padding: 10px; border: 1px solid #ddd;">Velocidad Media (px/f)</th>
            <th style="padding: 10px; border: 1px solid #ddd;">Velocidad Máx. (px/f)</th>
            <th style="padding: 10px; border: 1px solid #ddd;">Aceleración (px/f²)</th>
            <th style="padding: 10px; border: 1px solid #ddd;">Duración (frames)</th>
            <th style="padding: 10px; border: 1px solid #ddd;">Rango X (px)</th>
            <th style="padding: 10px; border: 1px solid #ddd;">Rango Y (px)</th>
        </tr>
    """
    
    # Ordenar patos por distancia total
    sorted_ducks = sorted(duck_metrics.keys(), key=lambda x: duck_metrics[x]['total_distance'], reverse=True)
    
    # Alternar colores de fila
    for i, duck_id in enumerate(sorted_ducks):
        metrics = duck_metrics[duck_id]
        bg_color = "#f2f2f2" if i % 2 == 0 else "white"
        color_dot = f'<span style="color:{vibrant_colors[duck_id]}">●</span>'
        
        metrics_html += f"""
        <tr style="background-color: {bg_color};">
            <td style="padding: 8px; border: 1px solid #ddd;"><b>{color_dot} {duck_id}</b></td>
            <td style="padding: 8px; border: 1px solid #ddd;">{duck_colors[duck_id]}</td>
            <td style="padding: 8px; border: 1px solid #ddd;">{metrics['total_distance']:.1f}</td>
            <td style="padding: 8px; border: 1px solid #ddd;">{metrics['avg_speed']:.2f}</td>
            <td style="padding: 8px; border: 1px solid #ddd;">{metrics['max_speed']:.2f}</td>
            <td style="padding: 8px; border: 1px solid #ddd;">{metrics['avg_acceleration']:.3f}</td>
            <td style="padding: 8px; border: 1px solid #ddd;">{metrics['duration']:.0f}</td>
            <td style="padding: 8px; border: 1px solid #ddd;">{metrics['x_range']:.1f}</td>
            <td style="padding: 8px; border: 1px solid #ddd;">{metrics['y_range']:.1f}</td>
        </tr>
        """
    
    metrics_html += "</table>"
    
    return metrics_html


def generate_additional_visualizations(data_file, output_folder):
    """
    Genera visualizaciones adicionales para enriquecer el informe
    
    Args:
        data_file: Ruta al archivo JSON con los datos combinados
        output_folder: Carpeta donde guardar las visualizaciones
    """
    # Crear carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    # Cargar datos para obtener información general
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Extraer datos de frames
    frames_data = data.get('frames', {})
    
    # Preparar estructuras para almacenar trayectorias
    duck_trajectories = {}
    duck_colors = {}
    
    # Procesar cada frame para extraer trayectorias
    for frame_idx, frame_data in sorted(frames_data.items(), key=lambda x: int(x[0])):
        frame_number = int(frame_data.get('frame_number', frame_idx))
        positions = frame_data.get('positions', {})
        
        for duck_id, duck_data in positions.items():
            if duck_id not in duck_trajectories:
                duck_trajectories[duck_id] = []
                duck_colors[duck_id] = duck_data.get('color', 'yellow')
            
            position = duck_data.get('position')
            if position:
                duck_trajectories[duck_id].append({
                    'x': position[0],
                    'y': position[1],
                    'frame': frame_number
                })
    
    # Calcular estadísticas de movimiento
    duck_stats = []
    for duck_id, trajectory in duck_trajectories.items():
        if len(trajectory) > 1:
            # Convertir a arrays numpy para cálculos
            positions = np.array([[point['x'], point['y']] for point in trajectory])
            frames = np.array([point['frame'] for point in trajectory])
            
            # Calcular distancias entre puntos consecutivos
            deltas = positions[1:] - positions[:-1]
            distances = np.sqrt(np.sum(deltas**2, axis=1))
            total_distance = np.sum(distances)
            
            # Calcular velocidades
            frame_deltas = frames[1:] - frames[:-1]
            speeds = distances / np.maximum(frame_deltas, 1)  # Evitar división por cero
            avg_speed = np.mean(speeds)
            max_speed = np.max(speeds)
            
            # Calcular ángulos de movimiento
            angles = np.arctan2(deltas[:, 1], deltas[:, 0]) * 180 / np.pi
            
            # Añadir estadísticas
            duck_stats.append({
                'duck_id': duck_id,
                'color': duck_colors[duck_id],
                'total_distance': total_distance,
                'average_speed': avg_speed,
                'max_speed': max_speed,
                'num_frames': len(trajectory),
                'duration': frames[-1] - frames[0] + 1,
                'angles': angles
            })
    
    # 1. Gráfico de distancia total por pato
    if duck_stats:
        plt.figure(figsize=(12, 8))
        df = pd.DataFrame(duck_stats)
        df = df.sort_values('total_distance', ascending=False)
        
        sns.set_style("whitegrid")
        ax = sns.barplot(x='duck_id', y='total_distance', data=df, palette='viridis')
        plt.title('Distancia Total Recorrida por Pato', fontsize=16)
        plt.xlabel('ID del Pato', fontsize=14)
        plt.ylabel('Distancia Total (píxeles)', fontsize=14)
        plt.xticks(rotation=45)
        
        # Añadir valores en las barras
        for i, v in enumerate(df['total_distance']):
            ax.text(i, v + 5, f"{v:.1f}", ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'distancia_total_por_pato.png'), dpi=300)
        plt.close()
        
        # 2. Gráfico de velocidad promedio por pato
        plt.figure(figsize=(12, 8))
        df = df.sort_values('average_speed', ascending=False)
        
        ax = sns.barplot(x='duck_id', y='average_speed', data=df, palette='plasma')
        plt.title('Velocidad Promedio por Pato', fontsize=16)
        plt.xlabel('ID del Pato', fontsize=14)
        plt.ylabel('Velocidad Promedio (píxeles/frame)', fontsize=14)
        plt.xticks(rotation=45)
        
        # Añadir valores en las barras
        for i, v in enumerate(df['average_speed']):
            ax.text(i, v + 0.05, f"{v:.2f}", ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'velocidad_promedio_por_pato.png'), dpi=300)
        plt.close()
        
        # 3. Gráfico de dispersión velocidad vs distancia
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='total_distance', y='average_speed', 
                       size='num_frames', hue='duck_id',
                       sizes=(100, 500), palette='viridis',
                       data=df)
        
        plt.title('Relación entre Velocidad Promedio y Distancia Total', fontsize=16)
        plt.xlabel('Distancia Total (píxeles)', fontsize=14)
        plt.ylabel('Velocidad Promedio (píxeles/frame)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Añadir etiquetas para cada punto
        for i, row in df.iterrows():
            plt.text(row['total_distance'] + 10, row['average_speed'],
                    row['duck_id'], fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'velocidad_vs_distancia.png'), dpi=300)
        plt.close()
        
        # 4. Gráfico de duración por pato
        plt.figure(figsize=(12, 8))
        df = df.sort_values('duration', ascending=False)
        
        ax = sns.barplot(x='duck_id', y='duration', data=df, palette='magma')
        plt.title('Duración de Aparición en Video por Pato', fontsize=16)
        plt.xlabel('ID del Pato', fontsize=14)
        plt.ylabel('Duración (frames)', fontsize=14)
        plt.xticks(rotation=45)
        
        # Añadir valores en las barras
        for i, v in enumerate(df['duration']):
            ax.text(i, v + 5, f"{int(v)}", ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'duracion_por_pato.png'), dpi=300)
        plt.close()
        
        # 5. Diagrama de rosa de direcciones de movimiento
        plt.figure(figsize=(10, 10), facecolor='white')
        
        # Convertir todos los ángulos a un solo array
        all_angles = []
        for stats in duck_stats:
            all_angles.extend(stats['angles'])
        
        # Crear diagrama de rosa
        ax = plt.subplot(111, projection='polar')
        bins = 16  # Divisiones para las direcciones
        
        # Histograma circular
        heights, edges = np.histogram(all_angles, bins=np.linspace(-180, 180, bins+1))
        width = 2 * np.pi / bins
        bars = ax.bar(np.deg2rad(edges[:-1]), heights, width=width, bottom=0.0)
        
        # Colorear barras según altura
        cm = plt.cm.plasma
        max_height = max(heights)
        for i, bar in enumerate(bars):
            bar.set_facecolor(cm(heights[i]/max_height))
            bar.set_alpha(0.8)
        
        # Configurar gráfico
        ax.set_theta_zero_location('N')  # 0 grados en el Norte
        ax.set_theta_direction(-1)  # Sentido horario
        ax.set_title('Distribución de Direcciones de Movimiento', fontsize=16, pad=20)
        
        # Etiquetas cardinales
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        angles = np.linspace(0, 2*np.pi, len(directions), endpoint=False)
        ax.set_xticks(angles)
        ax.set_xticklabels(directions, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'direcciones_movimiento.png'), dpi=300)
        plt.close()
        
        # 6. Matriz de correlación entre métricas
        plt.figure(figsize=(10, 8))
        
        # Seleccionar columnas numéricas
        numeric_cols = ['total_distance', 'average_speed', 'max_speed', 'num_frames', 'duration']
        corr_matrix = df[numeric_cols].corr()
        
        # Crear mapa de calor
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlación entre Métricas de Movimiento', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'correlacion_metricas.png'), dpi=300)
        plt.close()
        
        # Guardar estadísticas en CSV
        stats_file = os.path.join(output_folder, 'estadisticas_patos.csv')
        df[['duck_id', 'color', 'total_distance', 'average_speed', 'max_speed', 'num_frames', 'duration']].to_csv(stats_file, index=False)
        
        print(f"Visualizaciones adicionales generadas en: {output_folder}")
        
        # Crear resumen global
        resumen_file = os.path.join(output_folder, 'resumen_global.txt')
        with open(resumen_file, 'w') as f:
            f.write(f"RESUMEN ESTADÍSTICO GLOBAL\n")
            f.write(f"=========================\n\n")
            f.write(f"Total de patos analizados: {len(duck_stats)}\n")
            f.write(f"Distancia total promedio: {df['total_distance'].mean():.2f} píxeles\n")
            f.write(f"Velocidad promedio global: {df['average_speed'].mean():.2f} píxeles/frame\n")
            f.write(f"Velocidad máxima registrada: {df['max_speed'].max():.2f} píxeles/frame (Pato {df.loc[df['max_speed'].idxmax(), 'duck_id']})\n")
            f.write(f"Duración promedio en video: {df['duration'].mean():.1f} frames\n\n")
            
            # Pato más activo
            most_active = df.loc[df['total_distance'].idxmax()]
            f.write(f"Pato más activo: {most_active['duck_id']} (Distancia: {most_active['total_distance']:.2f} píxeles)\n")
            
            # Pato más rápido
            fastest = df.loc[df['average_speed'].idxmax()]
            f.write(f"Pato más rápido: {fastest['duck_id']} (Velocidad: {fastest['average_speed']:.2f} píxeles/frame)\n")
            
            # Pato con mayor duración
            longest = df.loc[df['duration'].idxmax()]
            f.write(f"Pato con mayor tiempo en video: {longest['duck_id']} ({int(longest['duration'])} frames)\n")
        
        print(f"Resumen global guardado en: {resumen_file}")
    
    return stats_file

def create_enhanced_html_report(data_file, visualizations_folder, output_folder, 
                                code_files=None, animation_files=None, include_3d_model=True,
                                cameraman_visualizations_folder=None):
    """
    Crea un informe HTML mejorado con todas las visualizaciones, análisis,
    código fuente, animaciones y modelos 3D
    
    Args:
        data_file: Ruta al archivo JSON con los datos combinados
        visualizations_folder: Carpeta con las visualizaciones generadas
        output_folder: Carpeta donde guardar el informe
        code_files: Lista de archivos Python para incluir como código fuente
        animation_files: Lista de archivos de animación (MP4, GIF)
        include_3d_model: Si se debe incluir el modelo 3D en el informe
        cameraman_visualizations_folder: Carpeta con visualizaciones del camarógrafo
    """
    # Crear carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    # Valores por defecto
    if code_files is None:
        code_files = []
    
    if animation_files is None:
        animation_files = [
            os.path.join(visualizations_folder, 'animacion_trayectorias_lenta.mp4'),
            os.path.join(visualizations_folder, 'animacion_trayectorias_lenta.gif')
        ]
    
    # Cargar datos para obtener información general
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Extraer información general
    total_frames = len(data.get('frames', {}))
    first_frame_positions = data.get('frames', {}).get('0', {}).get('positions', {})
    total_ducks = len(first_frame_positions)
    
    # Generar modelo 3D si es solicitado
    model_3d_path = None
    duck_metrics = None
    vibrant_colors = None
    duck_colors = {}
    if include_3d_model:
        model_3d_path, duck_metrics, vibrant_colors = create_duck_3d_model(data_file, visualizations_folder)
        # Añadir el archivo del modelo 3D a los archivos de animación
        if model_3d_path and os.path.exists(model_3d_path):
            animation_files.append(model_3d_path)
            
        # Extraer colores originales de los patos de los datos
        frames_data = data.get('frames', {})
        for frame_idx, frame_data in sorted(frames_data.items(), key=lambda x: int(x[0])):
            positions = frame_data.get('positions', {})
            for duck_id, duck_info in positions.items():
                if duck_id not in duck_colors:
                    duck_colors[duck_id] = duck_info.get('color', 'yellow')
    
    # Procesar visualizaciones del camarógrafo si se proporcionan
    cameraman_visualizations = []
    if cameraman_visualizations_folder and os.path.exists(cameraman_visualizations_folder):
        cameraman_visualizations = process_cameraman_visualizations(cameraman_visualizations_folder)
        
        # Crear una carpeta para las visualizaciones del camarógrafo dentro de la carpeta de salida del informe
        informe_cameraman_folder = os.path.join(output_folder, 'cameraman_visualizations')
        os.makedirs(informe_cameraman_folder, exist_ok=True)
        
        # Copiar las visualizaciones del camarógrafo a la carpeta del informe
        for filename, _, _ in cameraman_visualizations:
            source_path = os.path.join(cameraman_visualizations_folder, filename)
            target_path = os.path.join(informe_cameraman_folder, filename)
            if os.path.exists(source_path):
                shutil.copy2(source_path, target_path)
                print(f"Copiado: {filename} a {informe_cameraman_folder}")
            else:
                print(f"Advertencia: No se pudo encontrar {source_path}")
    
    # Función para calcular estadísticas adicionales
    def calculate_additional_stats(data):
        frames_data = data.get('frames', {})
        
        # Estructuras para almacenar datos por pato
        duck_data = {}
        
        # Procesar cada frame
        for frame_idx, frame_data in sorted(frames_data.items(), key=lambda x: int(x[0])):
            frame_number = int(frame_data.get('frame_number', frame_idx))
            positions = frame_data.get('positions', {})
            
            for duck_id, duck_info in positions.items():
                if duck_id not in duck_data:
                    duck_data[duck_id] = {
                        'positions': [],
                        'frames': [],
                        'color': duck_info.get('color', 'yellow')
                    }
                
                position = duck_info.get('position')
                if position:
                    duck_data[duck_id]['positions'].append(position)
                    duck_data[duck_id]['frames'].append(frame_number)
        
        # Calcular métricas adicionales
        stats = []
        for duck_id, info in duck_data.items():
            positions = np.array(info['positions'])
            frames = np.array(info['frames'])
            
            if len(positions) > 1:
                # Calcular distancias entre puntos consecutivos
                deltas = positions[1:] - positions[:-1]
                distances = np.sqrt(deltas[:, 0]**2 + deltas[:, 1]**2)
                total_distance = np.sum(distances)
                
                # Calcular velocidades (distancia / cambio de frame)
                frame_deltas = frames[1:] - frames[:-1]
                speeds = distances / np.maximum(frame_deltas, 1)  # Evitar división por cero
                avg_speed = np.mean(speeds)
                max_speed = np.max(speeds)
                
                # Calcular aceleración
                if len(speeds) > 1:
                    accelerations = speeds[1:] - speeds[:-1]
                    avg_acceleration = np.mean(np.abs(accelerations))
                    max_acceleration = np.max(np.abs(accelerations))
                else:
                    avg_acceleration = 0
                    max_acceleration = 0
                
                # Duración
                duration = frames[-1] - frames[0] + 1
                
                # Calcular dirección predominante
                if len(deltas) > 0:
                    angles = np.arctan2(deltas[:, 1], deltas[:, 0]) * 180 / np.pi
                    # Convertir a 8 direcciones cardinales
                    directions = np.round(angles / 45) % 8
                    dir_counts = np.bincount(directions.astype(int), minlength=8)
                    main_dir_idx = np.argmax(dir_counts)
                    dir_names = ['Este', 'Noreste', 'Norte', 'Noroeste', 
                                 'Oeste', 'Suroeste', 'Sur', 'Sureste']
                    main_direction = dir_names[main_dir_idx]
                    dir_percentage = dir_counts[main_dir_idx] / len(deltas) * 100
                else:
                    main_direction = "N/A"
                    dir_percentage = 0
                
                # Puntos inicial y final
                start_pos = positions[0]
                end_pos = positions[-1]
                
                # Distancia en línea recta entre inicio y fin
                direct_distance = np.sqrt((end_pos[0]-start_pos[0])**2 + (end_pos[1]-start_pos[1])**2)
                
                # Ratio de eficiencia (distancia directa / distancia total)
                efficiency = direct_distance / total_distance if total_distance > 0 else 0
                
                # Almacenar estadísticas
                stats.append({
                    'duck_id': duck_id,
                    'color': info['color'],
                    'num_frames': len(frames),
                    'total_distance': total_distance,
                    'average_speed': avg_speed,
                    'max_speed': max_speed,
                    'avg_acceleration': avg_acceleration,
                    'max_acceleration': max_acceleration,
                    'duration': duration,
                    'main_direction': main_direction,
                    'dir_percentage': dir_percentage,
                    'efficiency': efficiency,
                    'start_x': start_pos[0],
                    'start_y': start_pos[1],
                    'end_x': end_pos[0],
                    'end_y': end_pos[1]
                })
        
        return pd.DataFrame(stats) if stats else None
    
    # Calcular estadísticas detalladas
    stats_df = calculate_additional_stats(data)
    
    # Leer CSV de estadísticas si existe, de lo contrario usar las calculadas
    stats_file = os.path.join(visualizations_folder, 'estadisticas_patos.csv')
    if os.path.exists(stats_file):
        csv_stats_df = pd.read_csv(stats_file)
        # Combinar con estadísticas calculadas si es necesario
        if stats_df is not None:
            # Usar columnas de ambos dataframes sin duplicados
            stats_df = pd.merge(stats_df, csv_stats_df, on='duck_id', how='outer', suffixes=('', '_csv'))
    
    # Leer archivos de código fuente
    code_snippets = {}
    for code_file in code_files:
        if os.path.exists(code_file):
            with open(code_file, 'r') as f:
                code_content = f.read()
                code_snippets[os.path.basename(code_file)] = code_content
    
    # Lista extendida de imágenes de visualización con descripciones detalladas
    visualization_files = [
        ('trayectorias_3d_colores.png', 'Trayectorias 3D en Espacio-Tiempo', 
         'Visualización tridimensional de las trayectorias de los patos a lo largo del tiempo, mostrando ' + 
         'la evolución espacial y temporal del movimiento de cada individuo.'),
        
        ('trayectorias_2d_colores.png', 'Trayectorias 2D en el Plano', 
         'Proyección bidimensional de las trayectorias completas de los patos, permitiendo analizar ' + 
         'los patrones de movimiento y las áreas más transitadas en el espacio de la escena.'),
        
        ('mapa_densidad_trayectorias_colores.png', 'Mapa de Densidad con Trayectorias', 
         'Combinación de un mapa de calor que muestra la densidad de posiciones con las trayectorias ' + 
         'individuales superpuestas, revelando tanto el comportamiento individual como colectivo.'),
        
        ('distancia_total_por_pato.png', 'Distancia Total Recorrida por Pato', 
         'Comparativa de la distancia total recorrida por cada pato durante toda la grabación, ' + 
         'identificando los individuos más activos y los más sedentarios.'),
        
        ('velocidad_promedio_por_pato.png', 'Velocidad Promedio por Pato', 
         'Análisis comparativo de la velocidad promedio de desplazamiento de cada pato, ' + 
         'permitiendo identificar diferencias en la movilidad individual.'),
        
        ('velocidad_vs_distancia.png', 'Relación entre Velocidad y Distancia', 
         'Gráfico de dispersión que muestra la correlación entre la velocidad promedio y la ' + 
         'distancia total recorrida para cada pato, revelando patrones de comportamiento.'),
        
        ('duracion_por_pato.png', 'Tiempo de Aparición en Video', 
         'Duración total de la presencia de cada pato en la grabación, mostrando ' + 
         'cuáles permanecieron visibles durante más tiempo.'),
        
        ('evolucion_velocidad.png', 'Evolución Temporal de la Velocidad', 
         'Análisis de cómo cambia la velocidad de los patos a lo largo del tiempo, ' + 
         'revelando patrones de aceleración, desaceleración y posibles eventos de interés.'),
        
        ('mapa_calor_cuadricula.png', 'Mapa de Calor en Cuadrícula', 
         'Representación de la densidad de presencia de patos en una cuadrícula que divide ' + 
         'el espacio de la escena, identificando zonas de alta concentración.'),
        
        ('direcciones_movimiento.png', 'Distribución de Direcciones', 
         'Análisis de las direcciones predominantes de movimiento, mostrando si hay ' + 
         'tendencias direccionales específicas en el comportamiento de los patos.'),
        
        ('correlacion_metricas.png', 'Correlación entre Métricas de Movimiento', 
         'Matriz de correlación entre diferentes métricas de movimiento, revelando ' + 
         'relaciones entre variables como velocidad, aceleración, distancia y duración.'),
         
        ('modelo_3d_patos_profesional.png', 'Modelo 3D de Patos', 
         'Visualización tridimensional avanzada que muestra modelos 3D de los patos con sus trayectorias ' + 
         'completas, permitiendo analizar el comportamiento desde múltiples ángulos.')
    ]
    
    # Función para destacar código Python
    def format_python_code(code):
        formatter = HtmlFormatter(style='monokai', linenos=True, cssclass='codehilite')
        highlighted = highlight(code, PythonLexer(), formatter)
        css = formatter.get_style_defs('.codehilite')
        return highlighted, css
    
    # Obtener CSS para el código destacado
    code_css = ""
    if code_snippets:
        _, code_css = format_python_code(next(iter(code_snippets.values())))
    
    # Preparar contenido HTML del informe
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Análisis Avanzado de Trayectorias de Patos</title>
        <style>
            :root {{
                --primary-color: #3498db;
                --secondary-color: #2c3e50;
                --accent-color: #e74c3c;
                --light-bg: #f9f9f9;
                --dark-bg: #34495e;
                --text-color: #333;
                --light-text: #ecf0f1;
                --border-radius: 8px;
                --box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: var(--text-color);
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: var(--light-bg);
            }}
            
            h1, h2, h3, h4 {{
                color: var(--secondary-color);
                margin-top: 1.5em;
            }}
            
            h1 {{
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 2px solid var(--primary-color);
                padding-bottom: 15px;
                font-size: 2.5em;
            }}
            
            h2 {{
                border-left: 5px solid var(--primary-color);
                padding-left: 15px;
                font-size: 1.8em;
                margin-top: 2em;
            }}
            
            h3 {{
                font-size: 1.4em;
                border-bottom: 1px dashed var(--primary-color);
                padding-bottom: 5px;
            }}
            
            /* Tarjetas para visualizaciones */
            .visualization-card {{
                margin: 30px 0;
                background: white;
                border-radius: var(--border-radius);
                box-shadow: var(--box-shadow);
                padding: 20px;
                transition: transform 0.3s ease;
            }}
            
            .visualization-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            }}
            
            .visualization-card img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 0 auto;
                border-radius: 5px;
            }}
            
            .card-caption {{
                text-align: center;
                margin-top: 15px;
                font-weight: bold;
                color: var(--secondary-color);
            }}
            
            .card-description {{
                text-align: justify;
                margin-top: 10px;
                color: #555;
            }}
            
            /* Tabla de estadísticas */
            .stats-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                box-shadow: var(--box-shadow);
                border-radius: var(--border-radius);
                overflow: hidden;
            }}
            
            .stats-table th, .stats-table td {{
                border: 1px solid #ddd;
                padding: 12px 15px;
                text-align: left;
            }}
            
            .stats-table tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            
            .stats-table th {{
                background-color: var(--primary-color);
                color: white;
                text-transform: uppercase;
                font-size: 0.9em;
                letter-spacing: 1px;
            }}
            
            .stats-table tr:hover {{
                background-color: #e6f7ff;
            }}
            
            /* Sección de código */
            .code-section {{
                margin: 40px 0;
                background: var(--dark-bg);
                border-radius: var(--border-radius);
                padding: 20px;
                color: var(--light-text);
            }}
            
            .code-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 1px solid rgba(255,255,255,0.2);
            }}
            
            .code-title {{
                font-family: 'Consolas', monospace;
                font-weight: bold;
                color: #e74c3c;
            }}
            
            /* Animaciones */
            .animation-section {{
                margin: 40px 0;
                text-align: center;
            }}
            
            .animation-container {{
                background: white;
                border-radius: var(--border-radius);
                box-shadow: var(--box-shadow);
                padding: 20px;
                margin: 20px 0;
            }}
            
            .animation-container video, .animation-container img {{
                max-width: 100%;
                border-radius: 5px;
                margin: 10px auto;
                display: block;
            }}
            
            /* Resumen y métricas */
            .metrics-summary {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                margin: 30px 0;
            }}
            
            .metric-card {{
                background: white;
                border-radius: var(--border-radius);
                box-shadow: var(--box-shadow);
                padding: 20px;
                margin: 10px 0;
                width: calc(25% - 20px);
                text-align: center;
                transition: transform 0.3s ease;
            }}
            
            .metric-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            }}
            
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: var(--primary-color);
                margin: 10px 0;
            }}
            
            .metric-label {{
                font-size: 0.9em;
                color: #777;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            
            /* Apartado de navegación */
            .toc {{
                background: white;
                border-radius: var(--border-radius);
                box-shadow: var(--box-shadow);
                padding: 20px;
                margin: 30px 0;
            }}
            
            .toc ul {{
                list-style-type: none;
                padding-left: 0;
            }}
            
            .toc ul ul {{
                padding-left: 20px;
            }}
            
            .toc li {{
                margin: 8px 0;
            }}
            
            .toc a {{
                color: var(--primary-color);
                text-decoration: none;
                transition: color 0.3s ease;
            }}
            
            .toc a:hover {{
                color: var(--accent-color);
                text-decoration: underline;
            }}
            
            /* Pie de página */
            .footer {{
                margin-top: 50px;
                text-align: center;
                font-size: 0.9em;
                color: #7f8c8d;
                border-top: 1px solid #ddd;
                padding-top: 20px;
            }}
            
            /* Media queries para responsive */
            @media (max-width: 768px) {{
                .metric-card {{
                    width: calc(50% - 20px);
                }}
            }}
            
            @media (max-width: 480px) {{
                .metric-card {{
                    width: 100%;
                }}
            }}
            
            /* CSS para código Python destacado */
            {code_css}
            
            /* Gráficos interactivos */
            .interactive-section {{
                background: white;
                border-radius: var(--border-radius);
                box-shadow: var(--box-shadow);
                padding: 20px;
                margin: 30px 0;
            }}
            
            .tab-container {{
                margin-top: 20px;
            }}
            
            .tab-buttons {{
                display: flex;
                overflow-x: auto;
                border-bottom: 1px solid #ddd;
            }}
            
            .tab-btn {{
                padding: 10px 20px;
                background: none;
                border: none;
                cursor: pointer;
                font-size: 1em;
                color: #555;
                position: relative;
            }}
            
            .tab-btn.active {{
                color: var(--primary-color);
                font-weight: bold;
            }}
            
            .tab-btn.active::after {{
                content: '';
                position: absolute;
                bottom: -1px;
                left: 0;
                width: 100%;
                height: 3px;
                background-color: var(--primary-color);
            }}
            
            .tab-content {{
                padding: 20px 0;
                display: none;
            }}
            
            .tab-content.active {{
                display: block;
            }}
            
            /* Modelo 3D */
            .model-3d-section {{
                margin: 40px 0;
                text-align: center;
            }}
            
            .model-3d-container {{
                background: white;
                border-radius: var(--border-radius);
                box-shadow: var(--box-shadow);
                padding: 20px;
                margin: 20px 0;
            }}
            
            .model-3d-iframe {{
                width: 100%;
                height: 600px;
                border: none;
                border-radius: 5px;
            }}
            
            .model-3d-link {{
                display: inline-block;
                margin-top: 15px;
                background-color: var(--primary-color);
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                text-decoration: none;
                font-weight: bold;
                transition: background-color 0.3s ease;
            }}
            
            .model-3d-link:hover {{
                background-color: var(--secondary-color);
                text-decoration: none;
            }}
        </style>
    </head>
    <body>
        <h1>Análisis Avanzado de Trayectorias de Patos</h1>
        
        <div class="toc">
            <h3>Contenido</h3>
            <ul>
                <li><a href="#resumen">1. Resumen Ejecutivo</a></li>
                <li><a href="#visualizaciones">2. Visualizaciones de Trayectorias</a>
                    <ul>
                        <li><a href="#trayectorias-2d">2.1 Trayectorias 2D</a></li>
                        <li><a href="#trayectorias-3d">2.2 Trayectorias 3D</a></li>
                        <li><a href="#mapas-calor">2.3 Mapas de Calor y Densidad</a></li>
                    </ul>
                </li>
                <li><a href="#animaciones">3. Animaciones del Movimiento</a></li>
                <li><a href="#modelo-3d">4. Modelo 3D Interactivo</a></li>
                <li><a href="#estadisticas">5. Análisis Estadístico</a>
                    <ul>
                        <li><a href="#metricas-individuales">5.1 Métricas por Individuo</a></li>
                        <li><a href="#patrones-globales">5.2 Patrones Globales</a></li>
                    </ul>
                </li>
                <li><a href="#codigo">6. Código Fuente</a></li>
                <li><a href="#conclusiones">7. Conclusiones</a></li>
                """
    
    # Añadir enlace a la sección del camarógrafo si hay visualizaciones disponibles
    if cameraman_visualizations:
        html_content += """
                <li><a href="#cameraman">8. Análisis del Movimiento del Camarógrafo</a></li>
                <li><a href="#visualizaciones-interactivas">9. Visualizaciones Interactivas</a></li>
        """
    else:
        html_content += """
                <li><a href="#visualizaciones-interactivas">8. Visualizaciones Interactivas</a></li>
        """
    
    html_content += """
            </ul>
        </div>
        
        <section id="resumen">
            <h2>1. Resumen Ejecutivo</h2>
            
            <div class="metrics-summary">
                <div class="metric-card">
                    <div class="metric-label">Total de Frames</div>
                    <div class="metric-value">2083</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Total de Patos</div>
                    <div class="metric-value">7</div>
                </div>
    """
    
    # Añadir métricas adicionales si hay estadísticas
    if stats_df is not None:
        avg_distance = stats_df['total_distance'].mean()
        avg_speed = stats_df['average_speed'].mean()
        
        html_content += f"""
                <div class="metric-card">
                    <div class="metric-label">Distancia Promedio</div>
                    <div class="metric-value">{avg_distance:.1f}</div>
                    <div>píxeles</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Velocidad Promedio</div>
                    <div class="metric-value">{avg_speed:.2f}</div>
                    <div>píxeles/frame</div>
                </div>
        """
    
    html_content += """
            </div>
            
            <p>Este informe presenta un análisis completo del movimiento de patos capturados mediante el sistema de seguimiento automático Duck-Tracker. Las visualizaciones y análisis estadísticos revelan patrones de desplazamiento, áreas de concentración y diferencias de comportamiento entre individuos.</p>
        </section>
        
        <section id="visualizaciones">
            <h2>2. Visualizaciones de Trayectorias</h2>
    """
    
    # Añadir visualizaciones 2D
    html_content += """
            <section id="trayectorias-2d">
                <h3>2.1 Trayectorias 2D</h3>
                <p>Las siguientes visualizaciones muestran las trayectorias de los patos proyectadas en el plano bidimensional, permitiendo analizar los patrones de movimiento horizontal.</p>
    """
    
    # Añadir visualizaciones que existen
    for filename, caption, description in visualization_files:
        if "2d" in filename.lower() and os.path.exists(os.path.join(visualizations_folder, filename)):
            html_content += f"""
                <div class="visualization-card">
                    <img src="../visualizations/{filename}" alt="{caption}">
                    <div class="card-caption">{caption}</div>
                    <div class="card-description">{description}</div>
                </div>
            """
    
    # Añadir visualizaciones 3D
    html_content += """
            </section>
            
            <section id="trayectorias-3d">
                <h3>2.2 Trayectorias 3D</h3>
                <p>Las visualizaciones tridimensionales incorporan el tiempo como tercera dimensión, mostrando la evolución temporal del movimiento y facilitando la identificación de patrones de comportamiento a lo largo del video.</p>
    """
    
    for filename, caption, description in visualization_files:
        if "3d" in filename.lower() and os.path.exists(os.path.join(visualizations_folder, filename)) and not "modelo_3d" in filename.lower():
            html_content += f"""
                <div class="visualization-card">
                    <img src="../visualizations/{filename}" alt="{caption}">
                    <div class="card-caption">{caption}</div>
                    <div class="card-description">{description}</div>
                </div>
            """
    
    # Añadir mapas de calor
    html_content += """
            </section>
            
            <section id="mapas-calor">
                <h3>2.3 Mapas de Calor y Densidad</h3>
                <p>Los mapas de calor y densidad revelan las áreas de mayor concentración y actividad de los patos, identificando zonas de interés y patrones de uso del espacio.</p>
    """
    
    for filename, caption, description in visualization_files:
        if ("calor" in filename.lower() or "densidad" in filename.lower()) and os.path.exists(os.path.join(visualizations_folder, filename)):
            html_content += f"""
                <div class="visualization-card">
                    <img src="../visualizations/{filename}" alt="{caption}">
                    <div class="card-caption">{caption}</div>
                    <div class="card-description">{description}</div>
                </div>
            """
    
    html_content += """
            </section>
        </section>
        
        <section id="animaciones">
            <h2>3. Animaciones del Movimiento</h2>
            <p>Las siguientes animaciones muestran la evolución temporal de las trayectorias de los patos, permitiendo apreciar el movimiento dinámico y los patrones de comportamiento a lo largo del tiempo.</p>
    """
    
    # Añadir animaciones
    for anim_file in animation_files:
        if os.path.exists(anim_file) and not "modelo_3d" in anim_file.lower():
            file_basename = os.path.basename(anim_file)
            extension = os.path.splitext(file_basename)[1].lower()
            
            if extension == '.mp4':
                html_content += f"""
                <div class="animation-container">
                    <h3>Animación de Trayectorias (MP4)</h3>
                    <video width="800" height="600" controls>
                        <source src="../visualizations/{file_basename}" type="video/mp4">
                        Tu navegador no soporta el tag de video.
                    </video>
                    <p class="card-description">Animación 3D que muestra la evolución de las trayectorias de los patos a lo largo del tiempo. La vista rota para facilitar la apreciación de la estructura tridimensional del movimiento.</p>
                </div>
                """
            elif extension == '.gif':
                html_content += f"""
                <div class="animation-container">
                    <h3>Animación de Trayectorias (GIF)</h3>
                    <img src="../visualizations/{file_basename}" alt="Animación GIF de trayectorias">
                    <p class="card-description">Versión GIF de la animación de trayectorias, optimizada para compatibilidad web y fácil compartición.</p>
                </div>
                """
    
    # Añadir sección de modelo 3D si está disponible
    if include_3d_model and model_3d_path and os.path.exists(model_3d_path):
        model_3d_basename = os.path.basename(model_3d_path)
        html_content += f"""
        </section>
        
        <section id="modelo-3d">
            <h2>4. Modelo 3D Interactivo</h2>
            <p>El siguiente modelo 3D permite una exploración interactiva de las trayectorias de los patos, con representaciones tridimensionales de cada individuo según su comportamiento.</p>
            
            <div class="model-3d-container">
                <h3>Exploración Interactiva en 3D</h3>
                <iframe class="model-3d-iframe" src="../visualizations/{model_3d_basename}" allowfullscreen></iframe>
                <p class="card-description">Modelo 3D interactivo que permite rotar, hacer zoom y explorar desde cualquier ángulo las trayectorias y representaciones de los patos. Cada pato tiene un tamaño proporcional a su actividad y una orientación basada en su dirección de movimiento predominante.</p>
                <a href="../visualizations/{model_3d_basename}" target="_blank" class="model-3d-link">Abrir en Ventana Completa</a>
            </div>
            
            <div class="visualization-card">
                <img src="../visualizations/modelo_3d_patos_profesional.png" alt="Vista estática del modelo 3D">
                <div class="card-caption">Vista Previa del Modelo 3D</div>
                <div class="card-description">Captura estática del modelo 3D mostrando la vista general de todos los patos y sus trayectorias en el espacio tridimensional.</div>
            </div>
        """
        
        # Añadir tabla de métricas 3D si está disponible
        if duck_metrics and vibrant_colors:
            metrics_table = create_metrics_table(duck_metrics, duck_colors, vibrant_colors)
            html_content += f"""
            <div class="visualization-card">
                <h3>Métricas Detalladas del Modelo 3D</h3>
                {metrics_table}
                <div class="card-description">Tabla de métricas detalladas utilizadas para generar el modelo 3D, mostrando los parámetros de movimiento de cada pato que determinan su representación visual.</div>
            </div>
            """
    
    html_content += """
        </section>
        
        <section id="estadisticas">
            <h2>5. Análisis Estadístico</h2>
            
            <section id="metricas-individuales">
                <h3>5.1 Métricas por Individuo</h3>
                <p>La siguiente tabla muestra las métricas detalladas de movimiento para cada pato detectado, permitiendo comparar el comportamiento individual.</p>
    """
    
    # Añadir tabla de estadísticas si está disponible
    if stats_df is not None:
        # Ordenar por distancia total
        stats_df_sorted = stats_df.sort_values('total_distance', ascending=False)
        
        # Seleccionar columnas relevantes y formatear
        display_columns = ['duck_id', 'color', 'num_frames', 'total_distance', 
                          'average_speed', 'max_speed', 'duration', 'efficiency']
        
        # Verificar qué columnas están disponibles
        available_columns = [col for col in display_columns if col in stats_df_sorted.columns]
        
        # Crear tabla HTML
        html_content += """
                <table class="stats-table">
                    <tr>
        """
        
        # Encabezados de columna
        column_headers = {
            'duck_id': 'ID Pato', 
            'color': 'Color Original', 
            'num_frames': 'Frames', 
            'total_distance': 'Distancia Total (px)', 
            'average_speed': 'Vel. Promedio (px/frame)', 
            'max_speed': 'Vel. Máxima (px/frame)', 
            'duration': 'Duración (frames)',
            'efficiency': 'Eficiencia (%)',
            'main_direction': 'Dir. Principal',
            'dir_percentage': 'Dir. Principal (%)'
        }
        
        for col in available_columns:
            header = column_headers.get(col, col.replace('_', ' ').title())
            html_content += f"<th>{header}</th>"
        
        html_content += """
                    </tr>
        """
        
        # Filas de datos
        for _, row in stats_df_sorted.iterrows():
            html_content += "<tr>"
            for col in available_columns:
                value = row[col]
                # Formatear valores numéricos
                if col in ['total_distance', 'average_speed', 'max_speed']:
                    formatted_value = f"{value:.2f}"
                elif col in ['efficiency', 'dir_percentage']:
                    formatted_value = f"{value*100:.1f}%" if col == 'efficiency' and value <= 1 else f"{value:.1f}%"
                elif col in ['num_frames', 'duration']:
                    formatted_value = f"{int(value)}"
                else:
                    formatted_value = str(value)
                    
                html_content += f"<td>{formatted_value}</td>"
            html_content += "</tr>"
        
        html_content += """
                </table>
        """
    
    # Añadir gráficos estadísticos
    html_content += """
            </section>
            
            <section id="patrones-globales">
                <h3>5.2 Patrones Globales</h3>
                <p>Los siguientes gráficos muestran análisis estadísticos agregados de los patrones de movimiento, revelando tendencias generales y correlaciones entre diferentes métricas.</p>
    """
    
    # Añadir visualizaciones estadísticas
    for filename, caption, description in visualization_files:
        if any(keyword in filename.lower() for keyword in ['velocidad', 'distancia', 'correlacion', 'direcciones']) and os.path.exists(os.path.join(visualizations_folder, filename)):
            html_content += f"""
                <div class="visualization-card">
                    <img src="../visualizations/{filename}" alt="{caption}">
                    <div class="card-caption">{caption}</div>
                    <div class="card-description">{description}</div>
                </div>
            """
    
    html_content += """
            </section>
        </section>
        
        <section id="codigo">
            <h2>6. Código Fuente</h2>
            <p>A continuación se muestra el código fuente utilizado para generar las visualizaciones y análisis de este informe.</p>
    """
    
    # Añadir código fuente con formato destacado
    for filename, code in code_snippets.items():
        highlighted_code, _ = format_python_code(code)
        
        html_content += f"""
            <div class="code-section">
                <div class="code-header">
                    <div class="code-title">{filename}</div>
                </div>
                {highlighted_code}
            </div>
        """
    
    html_content += """
        </section>
        
        <section id="conclusiones">
            <h2>7. Conclusiones</h2>
            <p>El análisis detallado de las trayectorias de los patos revela patrones de comportamiento significativos:</p>
            <ul>
    """
    
    # Generar conclusiones basadas en los datos
    if stats_df is not None:
        # Pato con mayor distancia
        max_dist_duck = stats_df.loc[stats_df['total_distance'].idxmax()]
        html_content += f"""
                <li>El pato <strong>{max_dist_duck['duck_id']}</strong> mostró la mayor actividad, recorriendo una distancia total de <strong>{max_dist_duck['total_distance']:.2f}</strong> píxeles.</li>
        """
        
        # Pato más rápido
        max_speed_duck = stats_df.loc[stats_df['average_speed'].idxmax()]
        html_content += f"""
                <li>El pato <strong>{max_speed_duck['duck_id']}</strong> fue el más rápido, con una velocidad promedio de <strong>{max_speed_duck['average_speed']:.2f}</strong> píxeles/frame.</li>
        """
        
        # Dirección predominante global si está disponible
        if 'main_direction' in stats_df.columns:
            direction_counts = stats_df['main_direction'].value_counts()
            if not direction_counts.empty:
                main_dir = direction_counts.index[0]
                dir_percentage = direction_counts[0] / len(stats_df) * 100
                html_content += f"""
                    <li>La dirección predominante de movimiento fue <strong>{main_dir}</strong>, observada en el <strong>{dir_percentage:.1f}%</strong> de los patos.</li>
            """
    
    html_content += """
                <li>Las visualizaciones de mapas de calor muestran zonas claramente definidas de alta densidad, sugiriendo áreas de interés o recursos dentro del espacio monitoreado.</li>
                <li>La animación tridimensional revela interacciones temporales entre individuos, mostrando posibles comportamientos sociales o respuestas a estímulos externos.</li>
                <li>El modelo 3D interactivo permite identificar comportamientos distintivos de cada pato, facilitando la clasificación de patrones de comportamiento.</li>
            </ul>
            
            <p>Estos patrones de movimiento proporcionan información valiosa sobre el comportamiento de los patos en entornos controlados, contribuyendo a una mejor comprensión de su ecología y comportamiento social.</p>
        </section>
        
        """
    
    # Añadir sección de análisis del camarógrafo si hay visualizaciones disponibles
    if cameraman_visualizations:
        html_content += """
        <section id="cameraman">
            <h2>8. Análisis del Movimiento del Camarógrafo</h2>
            <p>Esta sección presenta el análisis del movimiento y rotación del camarógrafo durante la grabación, permitiendo entender cómo el movimiento de la cámara afecta a las trayectorias de los patos.</p>
            
            <div class="visualization-card">
                <h3>Movimiento del Camarógrafo en el Espacio</h3>
                <p>Las siguientes visualizaciones muestran el desplazamiento y rotación del camarógrafo durante la grabación, permitiendo correlacionar su movimiento con el comportamiento de los patos.</p>
            </div>
        """
        
        # Buscar si existe el archivo de animación
        animation_found = False
        animation_filename = None
        animation_title = None
        animation_description = None
        
        for filename, title, description in cameraman_visualizations:
            if filename.endswith('.mp4'):
                animation_found = True
                animation_filename = filename
                animation_title = title
                animation_description = description
                break
        
        # Si hay animación, ponerla primero como elemento destacado
        if animation_found:
            html_content += f"""
            <div class="animation-container" style="background-color: #f0f8ff; border: 1px solid #3498db; border-radius: 10px; box-shadow: 0 4px 12px rgba(52, 152, 219, 0.2); margin: 30px 0;">
                <h3 style="color: #3498db; text-align: center; margin-bottom: 20px;">{animation_title}</h3>
                <div style="display: flex; justify-content: center;">
                    <video width="800" height="600" controls style="border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                        <source src="cameraman_visualizations/{animation_filename}" type="video/mp4">
                        Tu navegador no soporta el tag de video.
                    </video>
                </div>
                <p class="card-description" style="text-align: center; margin-top: 15px; padding: 0 30px; font-style: italic;">{animation_description}</p>
            </div>
            """
        
        # Añadir las visualizaciones estáticas
        for filename, title, description in cameraman_visualizations:
            # Omitir el archivo de video que ya se mostró
            if filename.endswith('.mp4'):
                continue
                
            # Determinar si es una imagen o video
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                html_content += f"""
                <div class="visualization-card">
                    <img src="cameraman_visualizations/{filename}" alt="{title}">
                    <div class="card-caption">{title}</div>
                    <div class="card-description">{description}</div>
                </div>
                """
        
        # Añadir análisis e interpretación
        html_content += """
        <div class="visualization-card">
            <h3>Interpretación del Movimiento del Camarógrafo</h3>
            <p>El análisis del movimiento del camarógrafo revela información importante sobre la dinámica de la grabación:</p>
            <ul>
                <li>El desplazamiento del camarógrafo muestra patrones que pueden influir en el comportamiento de los patos observados.</li>
                <li>La rotación de la cámara proporciona contexto sobre los cambios de perspectiva durante la grabación.</li>
                <li>La correlación entre el movimiento del camarógrafo y las trayectorias de los patos puede ayudar a identificar reacciones a la presencia humana.</li>
                <li>El análisis conjunto de ambos movimientos permite una interpretación más precisa de los patrones de comportamiento natural versus comportamiento inducido.</li>
            </ul>
        </div>
        </section>
        """
        
        # Actualizar la numeración de la sección "Visualizaciones Interactivas"
        html_content += """
        <section>
            <h2>9. Visualizaciones Interactivas</h2>
            <p>A continuación se presentan visualizaciones interactivas que permiten explorar los datos de diferentes maneras.</p>
        """
    else:
        # Si no hay visualizaciones del camarógrafo, mantener la numeración original
        html_content += """
        <section>
            <h2>8. Visualizaciones Interactivas</h2>
            <p>A continuación se presentan visualizaciones interactivas que permiten explorar los datos de diferentes maneras.</p>
        """
            
    html_content += """
            <div class="interactive-section">
                <h3>Selección de Visualizaciones</h3>
                
                <div class="tab-container">
                    <div class="tab-buttons">
                        <button class="tab-btn active" onclick="openTab(event, 'tab-trayectorias')">Trayectorias</button>
                        <button class="tab-btn" onclick="openTab(event, 'tab-velocidades')">Velocidades</button>
                        <button class="tab-btn" onclick="openTab(event, 'tab-densidad')">Mapas de Calor</button>
                        <button class="tab-btn" onclick="openTab(event, 'tab-modelo3d')">Modelo 3D</button>
    """
    
    # Añadir pestaña del camarógrafo si hay visualizaciones disponibles
    if cameraman_visualizations:
        html_content += """
                        <button class="tab-btn" onclick="openTab(event, 'tab-cameraman')">Camarógrafo</button>
        """
    
    html_content += """
                    </div>
                    
                    <div id="tab-trayectorias" class="tab-content active">
                        <h4>Trayectorias de Patos</h4>
                        <p>Visualización de las trayectorias completas de todos los patos detectados.</p>
                        <img src="../visualizations/trayectorias_2d_colores.png" alt="Trayectorias 2D" style="width:100%; max-width:800px;">
                    </div>
                    
                    <div id="tab-velocidades" class="tab-content">
                        <h4>Análisis de Velocidades</h4>
                        <p>Comparativa de velocidades promedio y máximas por pato.</p>
                        <img src="../visualizations/velocidad_promedio_por_pato.png" alt="Velocidades" style="width:100%; max-width:800px;">
                    </div>
                    
                    <div id="tab-densidad" class="tab-content">
                        <h4>Mapas de Calor</h4>
                        <p>Visualización de áreas de mayor concentración de patos.</p>
                        <img src="../visualizations/mapa_densidad_trayectorias_colores.png" alt="Mapa de Calor" style="width:100%; max-width:800px;">
                    </div>
                    
                    <div id="tab-modelo3d" class="tab-content">
                        <h4>Modelo 3D Interactivo</h4>
                        <p>Exploración tridimensional de las trayectorias con modelos 3D.</p>
                        <img src="../visualizations/modelo_3d_patos_profesional.png" alt="Modelo 3D" style="width:100%; max-width:800px;">
                        <p><a href="../visualizations/modelo_3d_patos_profesional.html" target="_blank" class="model-3d-link">Explorar Modelo 3D Interactivo</a></p>
                    </div>
                    """
    
    # Añadir contenido de la pestaña del camarógrafo si hay visualizaciones disponibles
    if cameraman_visualizations:
        # Buscar la visualización 3D del camarógrafo
        cam_3d_vis = next((f for f, _, _ in cameraman_visualizations if "3d" in f.lower()), None)
        # Si no hay visualización 3D, usar la primera disponible
        if not cam_3d_vis and cameraman_visualizations:
            cam_3d_vis = cameraman_visualizations[0][0]
            
        html_content += f"""
                    <div id="tab-cameraman" class="tab-content">
                        <h4>Movimiento del Camarógrafo</h4>
                        <p>Análisis del desplazamiento y rotación del camarógrafo durante la grabación.</p>
                        """
        
        if cam_3d_vis:
            html_content += f"""
                        <img src="cameraman_visualizations/{cam_3d_vis}" alt="Movimiento del Camarógrafo" style="width:100%; max-width:800px;">
            """
            
        html_content += """
                        <p>El movimiento del camarógrafo puede influir en el comportamiento de los patos, proporcionando contexto importante para el análisis.</p>
                        <p><a href="#cameraman" class="model-3d-link">Ver Análisis Completo del Camarógrafo</a></p>
                    </div>
        """
    
    html_content += """
                </div>
            </div>
        </section>

<div class="footer">
    <p>Informe generado con el sistema Duck-Tracker | Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
    <p>Desarrollado para análisis avanzado de patrones de movimiento en patos</p>
</div>

<script>
    function openTab(evt, tabName) {{
        // Ocultar todos los contenidos de pestañas
        var tabContents = document.getElementsByClassName("tab-content");
        for (var i = 0; i < tabContents.length; i++) {{
            tabContents[i].style.display = "none";
            tabContents[i].className = tabContents[i].className.replace(" active", "");
        }}
        
        // Desactivar todos los botones
        var tabButtons = document.getElementsByClassName("tab-btn");
        for (var i = 0; i < tabButtons.length; i++) {{
            tabButtons[i].className = tabButtons[i].className.replace(" active", "");
        }}
        
        // Mostrar el contenido de la pestaña actual y activar el botón
        document.getElementById(tabName).style.display = "block";
        document.getElementById(tabName).className += " active";
        evt.currentTarget.className += " active";
    }}
</script>
</body>
</html>
    """
    
    # Guardar el archivo HTML
    output_file = os.path.join(output_folder, 'informe_avanzado_patos.html')
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Informe HTML avanzado generado en: {output_file}")
    
    return output_file

if __name__ == "__main__":
    import os
    import shutil
    
    # Obtener la ruta base del proyecto
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Rutas para archivos y carpetas (usando rutas relativas desde el directorio base)
    # Si no existe el archivo merged_tracking_data.json, podemos usar cualquier archivo JSON del proyecto como ejemplo
    data_file = os.path.join(base_dir, "batch_output/merged_results/merged_tracking_data.json")
    
    # Verificar si el archivo existe, si no, crear un archivo de ejemplo
    if not os.path.exists(data_file):
        print(f"Aviso: No se encontró el archivo de datos fusionados en {data_file}")
        print("Creando un archivo de ejemplo para la demostración...")
        
        # Crear las carpetas necesarias
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        
        # Crear un archivo JSON de ejemplo
        import json
        example_data = {
            "frames": {
                "0": {
                    "frame_number": 0,
                    "positions": {
                        "duck1": {"position": [100, 150], "color": "yellow"},
                        "duck2": {"position": [200, 250], "color": "black"}
                    }
                },
                "1": {
                    "frame_number": 1,
                    "positions": {
                        "duck1": {"position": [105, 155], "color": "yellow"},
                        "duck2": {"position": [195, 245], "color": "black"}
                    }
                },
                # Añadir más frames de ejemplo para tener suficientes datos para visualizaciones
                "2": {
                    "frame_number": 2,
                    "positions": {
                        "duck1": {"position": [110, 160], "color": "yellow"},
                        "duck2": {"position": [190, 240], "color": "black"}
                    }
                },
                "3": {
                    "frame_number": 3,
                    "positions": {
                        "duck1": {"position": [115, 165], "color": "yellow"},
                        "duck2": {"position": [185, 235], "color": "black"}
                    }
                }
            }
        }
        with open(data_file, 'w') as f:
            json.dump(example_data, f, indent=2)
        print(f"Archivo de ejemplo creado en {data_file}")
    
    # Carpeta con las visualizaciones
    visualizations_folder = os.path.join(base_dir, "batch_output/visualizations")
    os.makedirs(visualizations_folder, exist_ok=True)
    
    # Carpeta con las visualizaciones del camarógrafo
    cameraman_visualizations_folder = os.path.join(base_dir, "batch_output/cameraman_visualizations")
    
    # Verificar si la carpeta existe
    if not os.path.exists(cameraman_visualizations_folder):
        print(f"Aviso: No se encontró la carpeta de visualizaciones del camarógrafo en {cameraman_visualizations_folder}")
        print("Buscando las visualizaciones en rutas alternativas...")
        
        # Buscar archivos de visualización del camarógrafo en carpetas alternativas
        alt_paths = [
            os.path.join(base_dir, "cameraman_visualizations"),
            os.path.join(os.path.dirname(base_dir), "cameraman_visualizations"),
            os.path.join(base_dir, "Duck-Tracker-Project-main/batch_output/cameraman_visualizations"),
            os.path.join(os.path.dirname(os.path.dirname(base_dir)), "Duck-Tracker-Project-main/batch_output/cameraman_visualizations")
        ]
        
        for path in alt_paths:
            if os.path.exists(path):
                cameraman_visualizations_folder = path
                print(f"Encontradas visualizaciones del camarógrafo en: {path}")
                break
        else:
            print("No se encontraron visualizaciones del camarógrafo en ninguna ubicación alternativa.")
    
    # Carpeta para guardar el informe
    output_folder = os.path.join(base_dir, "batch_output/informe")
    os.makedirs(output_folder, exist_ok=True)
    
    # Lista de archivos de código a incluir
    code_files = [
        os.path.join(base_dir, "visualize_trajectories.py"),
        os.path.join(base_dir, "create_animation.py"),
        os.path.join(base_dir, "cameraman_movement.py"),
        os.path.join(base_dir, "visualize_cameraman.py")
    ]
    
    # Filtrar solo los archivos de código que existen
    code_files = [f for f in code_files if os.path.exists(f)]
    
    try:
        # Generar visualizaciones adicionales
        print("Paso 1: Generando visualizaciones adicionales...")
        generate_additional_visualizations(data_file, visualizations_folder)
    except Exception as e:
        print(f"Error al generar visualizaciones adicionales: {e}")
        print("Continuando con el resto del proceso...")
    
    try:
        # Generar modelo 3D
        print("\nPaso 2: Creando modelo 3D de patos...")
        model_3d_path, _, _ = create_duck_3d_model(data_file, visualizations_folder)
    except Exception as e:
        print(f"Error al crear el modelo 3D: {e}")
        print("Continuando con el resto del proceso...")
        model_3d_path = None
    
    # Generar informe HTML mejorado
    print("\nPaso 3: Generando informe HTML avanzado con visualizaciones, modelo 3D y análisis del camarógrafo...")
    try:
        report_file = create_enhanced_html_report(
            data_file, 
            visualizations_folder, 
            output_folder,
            code_files=code_files,
            include_3d_model=(model_3d_path is not None),
            cameraman_visualizations_folder=cameraman_visualizations_folder
        )
        
        print(f"\n¡Informe avanzado generado con éxito en {report_file}!")
        print("Incluye:")
        print("  - Visualizaciones 2D y 3D estáticas")
        print("  - Animaciones de movimiento")
        if model_3d_path:
            print("  - Modelo 3D interactivo de patos")
        print("  - Análisis estadísticos detallados")
        if os.path.exists(cameraman_visualizations_folder):
            print("  - Análisis del movimiento del camarógrafo")
        print("  - Código fuente y documentación")
        print("\nModo de uso:")
        print("  1. Abre el archivo HTML en tu navegador para ver el informe completo")
        print("  2. Utiliza las pestañas interactivas para explorar diferentes visualizaciones")
        if model_3d_path:
            print("  3. El modelo 3D permite rotación, zoom y exploración completa de las trayectorias")
        print("  4. Las métricas detalladas permiten comparar el comportamiento de cada pato")
        if os.path.exists(cameraman_visualizations_folder):
            print("  5. El análisis del camarógrafo muestra cómo su movimiento puede influir en las trayectorias observadas")
    except Exception as e:
        print(f"Error al generar el informe HTML: {e}")
        import traceback
        traceback.print_exc()