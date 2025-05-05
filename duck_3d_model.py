import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import colorsys
from scipy.ndimage import gaussian_filter1d
import math

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
    
    print("\n¡Modelo 3D profesional de patos creado con éxito!")
    
    return output_file


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

if __name__ == "__main__":
    # Ruta al archivo de datos combinados
    data_file = "/home/alfonso/Duck-Tracker/batch_output/merged_results/merged_tracking_data.json"
    
    # Carpeta para guardar las visualizaciones
    output_folder = "/home/alfonso/Duck-Tracker-Project/batch_output/visualizations"
    
    # Generar modelo 3D profesional
    model_file = create_duck_3d_model(data_file, output_folder)
    
    print(f"\nPuedes abrir el archivo HTML en tu navegador para interactuar con el modelo 3D profesional: {model_file}")
    print("\nEl modelo incluye:")
    print("  - Modelos 3D realistas de cada pato con escala proporcional a su movimiento")
    print("  - Métricas precisas de distancia, velocidad y aceleración")
    print("  - Tabla detallada con todas las estadísticas")
    print("\nEstos datos te permiten evaluar el comportamiento de los patos de forma precisa.")

    