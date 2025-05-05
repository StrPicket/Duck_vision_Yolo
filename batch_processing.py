import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import json
import os
import gc
import time
import argparse
from tqdm import tqdm

# Importar la clase DuckTrackerAdvanced desde tu script original
# Asumiendo que está en un archivo llamado duck_tracker.py en el mismo directorio
from duck_test1 import DuckTrackerAdvanced

def process_video_batch(video_path, output_dir, start_frame, end_frame, model_path=None):
    """
    Procesa un lote específico de frames de un video
    
    Args:
        video_path: Ruta al archivo de video
        output_dir: Directorio donde guardar los resultados
        start_frame: Índice del frame inicial (inclusivo)
        end_frame: Índice del frame final (inclusivo)
        model_path: Ruta al modelo YOLOv8 (opcional)
    """
    # Usar la ruta de modelo predeterminada si no se proporciona
    if model_path is None:
        model_path = r'/home/alfonso/Duck-Tracker/best.pt'
    
    # Crear directorio de salida con información del lote
    batch_dir = os.path.join(output_dir, f"batch_{start_frame}_{end_frame}")
    os.makedirs(batch_dir, exist_ok=True)
    
    # Crear subdirectorios para frames
    processed_frames_dir = os.path.join(batch_dir, "processed_frames")
    os.makedirs(processed_frames_dir, exist_ok=True)
    
    original_frames_dir = os.path.join(batch_dir, "original_frames")
    os.makedirs(original_frames_dir, exist_ok=True)
    
    # Inicializar el rastreador con un número limitado de frames para controlar el uso de memoria
    tracker = DuckTrackerAdvanced(
        model_path=model_path,
        min_confidence=0.3,
        min_distance=20,
        max_frames=(end_frame - start_frame + 1),  # Solo procesar el rango especificado
        max_ducks=7,  # Limitar a 7 patos
        max_tracking_history=20,  # Reducido para ahorrar memoria
        max_distance_threshold=50,
        track_timeout=20,
        reid_threshold=0.3
    )
    
    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")
    
    # Obtener información del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Configurar el video de salida
    output_video_path = os.path.join(batch_dir, f"output_video_{start_frame}_{end_frame}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width*2, height))
    
    print(f"\nProcesando lote de video: frames {start_frame} a {end_frame}")
    
    # Saltar al frame inicial
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_number = start_frame
    batch_frame_number = 0  # Contador de frames local para este lote
    
    # Procesar frames en el lote
    with tqdm(total=(end_frame - start_frame + 1), desc="Procesando frames") as pbar:
        while cap.isOpened() and frame_number <= end_frame:
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Guardar frame original
                original_frame_path = os.path.join(original_frames_dir, f"original_{frame_number:04d}.png")
                cv2.imwrite(original_frame_path, frame)
                
                # Procesar frame
                tracker.process_frame(frame)
                
                # Crear visualización
                plt.figure(figsize=(15, 7))
                
                # Imagen original con detecciones
                ax1 = plt.subplot(121)
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tracker.draw_grid(ax1, frame.shape)
                
                # Dibujar patos perdidos
                for lost_id, lost_data in tracker.lost_ducks.items():
                    if lost_data['trajectory']:
                        last_pos = lost_data['trajectory'][-1]
                        color_info = tracker.color_map[lost_data['color']]
                        
                        # Dibujar trayectoria
                        trajectory = np.array(lost_data['trajectory'])
                        plt.plot(trajectory[:, 0], trajectory[:, 1], '--', 
                                color='gray', alpha=0.3, linewidth=1)
                        
                        # Dibujar predicción de Kalman
                        kf = lost_data['kalman_filter']
                        predicted_pos = kf.predict()
                        plt.plot(predicted_pos[0], predicted_pos[1], 'r+', markersize=8, alpha=0.3)
                
                # Dibujar patos activos
                for duck_id, data in tracker.duck_positions.items():
                    x1, y1, x2, y2 = data['bbox']
                    x, y = data['position']
                    confidence = data['confidence']
                    duck_color = data['color']
                    velocity = data['velocity']
                    grid_pos = data['grid_position']
                    tracking_score = tracker.tracked_ducks[duck_id]['tracking_score']
                    
                    color_info = tracker.color_map[duck_color]
                    
                    # Dibujar bounding box
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       fill=False, 
                                       color=color_info['rgb'],
                                       alpha=tracking_score,
                                       linewidth=2)
                    plt.gca().add_patch(rect)
                    
                    # Dibujar trayectoria
                    if 'trajectory' in data and len(data['trajectory']) > 1:
                        trajectory = np.array(data['trajectory'])
                        num_points = len(trajectory)
                        for i in range(num_points - 1):
                            alpha = (i + 1) / num_points
                            plt.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], '--', 
                                    color=color_info['rgb'], alpha=alpha * 0.8, linewidth=1)
                    
                    # Dibujar punto central
                    plt.plot(x, y, color_info['plot'] + 'o', markersize=8)
                    
                    # Dibujar posición predicha por Kalman
                    kf = tracker.tracked_ducks[duck_id]['kalman_filter']
                    predicted_pos = kf.predict()
                    plt.plot(predicted_pos[0], predicted_pos[1], 'r+', markersize=8, alpha=0.6)
                    
                    # Añadir etiqueta
                    info_text = f'{duck_id} ({confidence:.2f})\n{duck_color}\nGrid: {grid_pos}\nTrack: {tracking_score:.2f}'
                    if velocity:
                        info_text += f'\nv: ({velocity[0]:.1f}, {velocity[1]:.1f})'
                    
                    plt.text(x1, y1-10, info_text,
                            color=color_info['text'], 
                            fontsize=8,
                            bbox=dict(facecolor=color_info['rgb'], alpha=0.7))
                
                plt.title('Detecciones y Predicciones')
                plt.axis('off')
                
                # Vista 2D con trayectorias
                ax2 = plt.subplot(122)
                plt.imshow(np.zeros_like(frame[:,:,0]), cmap='gray')
                tracker.draw_grid(ax2, frame.shape)
                
                # Dibujar patos perdidos
                for lost_id, lost_data in tracker.lost_ducks.items():
                    if lost_data['trajectory']:
                        last_pos = lost_data['trajectory'][-1]
                        color_info = tracker.color_map[lost_data['color']]
                        
                        # Dibujar trayectoria
                        trajectory = np.array(lost_data['trajectory'])
                        plt.plot(trajectory[:, 0], trajectory[:, 1], '--', 
                                color='gray', alpha=0.3, linewidth=1)
                        
                        # Dibujar predicción de Kalman
                        kf = lost_data['kalman_filter']
                        predicted_pos = kf.predict()
                        plt.plot([last_pos[0], predicted_pos[0]], 
                                [last_pos[1], predicted_pos[1]], 
                                'r--', alpha=0.3)
                        plt.scatter(predicted_pos[0], predicted_pos[1], 
                                  c='r', s=50, alpha=0.3)
                
                # Dibujar patos activos
                for duck_id, data in tracker.duck_positions.items():
                    x, y = data['position']
                    duck_color = data['color']
                    color_info = tracker.color_map[duck_color]
                    tracking_score = tracker.tracked_ducks[duck_id]['tracking_score']
                    
                    # Dibujar trayectoria
                    if 'trajectory' in data and len(data['trajectory']) > 1:
                        trajectory = np.array(data['trajectory'])
                        num_points = len(trajectory)
                        for i in range(num_points - 1):
                            alpha = (i + 1) / num_points * tracking_score
                            plt.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], '-', 
                                    color=color_info['rgb'], alpha=alpha)
                    
                    # Posición actual
                    plt.scatter(x, y, c=[color_info['rgb']], s=100, alpha=tracking_score)
                    
                    # Posición predicha
                    kf = tracker.tracked_ducks[duck_id]['kalman_filter']
                    predicted_pos = kf.predict()
                    plt.plot([x, predicted_pos[0]], [y, predicted_pos[1]], 'r--', alpha=0.4)
                    plt.scatter(predicted_pos[0], predicted_pos[1], c='r', s=50, alpha=0.4)
                    
                    # Etiqueta con ID e info
                    info_text = f"{duck_id}\n({duck_color})"
                    if data['velocity']:
                        vx, vy = data['velocity']
                        speed = np.sqrt(vx**2 + vy**2)
                        info_text += f"\nv: {speed:.1f}"
                    info_text += f"\ntrack: {tracking_score:.2f}"
                    
                    plt.annotate(info_text, 
                                (x, y),
                                xytext=(5, 5), 
                                textcoords='offset points',
                                color='white', 
                                fontsize=8)
                
                plt.title('Vista 2D con Trayectorias y Predicciones')
                plt.axis('off')
                plt.tight_layout()
                
                # Guardar frame procesado
                processed_frame_path = os.path.join(processed_frames_dir, f"processed_{frame_number:04d}.png")
                plt.savefig(processed_frame_path)
                
                # Crear frame combinado para video
                processed_frame = cv2.imread(processed_frame_path)
                if processed_frame is not None:
                    # Redimensionar si es necesario
                    processed_frame = cv2.resize(processed_frame, (width, height))
                    # Combinar frames lado a lado
                    combined_frame = np.hstack((frame, processed_frame))
                    out.write(combined_frame)
                
                # Cerrar figura para liberar memoria
                plt.close('all')
                
                # Guardar datos del frame
                tracker.all_frames_data[batch_frame_number] = {
                    'positions': tracker.duck_positions.copy(),
                    'frame_number': frame_number,
                    'batch_frame_number': batch_frame_number
                }
                
                frame_number += 1
                batch_frame_number += 1
                pbar.update(1)
                
                # Limpieza parcial cada 50 frames para evitar acumulación de memoria
                if batch_frame_number % 50 == 0:
                    gc.collect()
                    
            except Exception as e:
                print(f"Error procesando frame {frame_number}: {str(e)}")
                frame_number += 1
                batch_frame_number += 1
                pbar.update(1)
                continue
    
    # Liberar recursos
    cap.release()
    out.release()
    
    # Guardar datos de tracking para este lote
    tracker.save_all_data(os.path.join(batch_dir, f"tracking_data_{start_frame}_{end_frame}.json"))
    
    print(f"\n¡Procesamiento de lote completado!")
    print(f"Video procesado guardado como: {output_video_path}")
    print(f"Datos guardados en: {batch_dir}")
    
    # Limpieza final para liberar memoria antes de pasar al siguiente lote
    plt.close('all')
    del tracker
    gc.collect()
    
    return batch_dir, frame_number - 1  # Devolver el último frame procesado

def merge_batch_results(output_dir, batch_dirs, total_frames):
    """
    Combinar resultados de múltiples lotes
    
    Args:
        output_dir: Directorio principal de salida
        batch_dirs: Lista de rutas de directorios de lotes
        total_frames: Número total de frames procesados
    """
    print("\nCombinando resultados de lotes...")
    
    # Crear directorio de salida combinado
    merged_dir = os.path.join(output_dir, "merged_results")
    os.makedirs(merged_dir, exist_ok=True)
    
    # Combinar datos de tracking
    all_frames_data = {}
    
    for batch_dir in batch_dirs:
        # Encontrar el archivo de datos de tracking
        data_files = [f for f in os.listdir(batch_dir) if f.startswith("tracking_data_")]
        if not data_files:
            continue
        
        # Cargar datos del lote
        with open(os.path.join(batch_dir, data_files[0]), 'r') as f:
            batch_data = json.load(f)
        
        # Extraer datos de frames
        frames_data = batch_data.get('frames', {})
        for frame_idx, frame_data in frames_data.items():
            # Obtener el número de frame real de los datos
            actual_frame_number = frame_data.get('frame_number', int(frame_idx))
            all_frames_data[actual_frame_number] = frame_data
    
    # Guardar datos combinados
    merged_data = {
        'total_frames': total_frames,
        'grid_size': (8, 8),  # Predeterminado de tu script
        'frames': all_frames_data
    }
    
    with open(os.path.join(merged_dir, "merged_tracking_data.json"), 'w') as f:
        json.dump(merged_data, f, indent=4)
    
    print(f"\nDatos combinados guardados en: {merged_dir}")
    print("Nota: Para crear un video completo, puedes usar ffmpeg para concatenar los videos de lotes.")

def main():
    parser = argparse.ArgumentParser(description='Procesar video en lotes')
    parser.add_argument('--video', type=str, default='/home/alfonso/Duck-Tracker/assets/DuckVideo.mp4',
                      help='Ruta al video de entrada')
    parser.add_argument('--model', type=str, default='/home/alfonso/Duck-Tracker/best.pt',
                      help='Ruta al modelo YOLOv8')
    parser.add_argument('--output', type=str, default='batch_output',
                      help='Directorio de salida')
    parser.add_argument('--batch_size', type=int, default=200,
                      help='Número de frames por lote')
    parser.add_argument('--total_frames', type=int, default=2083,
                      help='Total de frames a procesar')
    parser.add_argument('--start_frame', type=int, default=0,
                      help='Frame desde el cual comenzar el procesamiento')
    parser.add_argument('--pause_between_batches', type=int, default=5,
                      help='Tiempo de pausa entre lotes (segundos)')
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    os.makedirs(args.output, exist_ok=True)
    
    start_frame = args.start_frame
    batch_dirs = []
    last_processed_frame = start_frame - 1
    
    # Procesar cada lote
    while last_processed_frame < args.total_frames - 1:
        start_frame = last_processed_frame + 1
        end_frame = min(start_frame + args.batch_size - 1, args.total_frames - 1)
        
        batch_idx = len(batch_dirs) + 1
        total_batches = (args.total_frames - start_frame + args.batch_size) // args.batch_size
        
        print(f"\n===== Procesando Lote {batch_idx}/{total_batches} (Frames {start_frame}-{end_frame}) =====")
        
        try:
            batch_dir, last_processed_frame = process_video_batch(
                args.video,
                args.output,
                start_frame,
                end_frame,
                args.model
            )
            
            batch_dirs.append(batch_dir)
            
            # Limpieza completa entre lotes
            plt.close('all')
            gc.collect()
            
            # Pausa entre lotes para permitir que el sistema se recupere
            if last_processed_frame < args.total_frames - 1:
                print(f"Pausa de {args.pause_between_batches} segundos antes del siguiente lote...")
                time.sleep(args.pause_between_batches)
                
        except Exception as e:
            print(f"Error procesando lote {batch_idx}: {str(e)}")
            print(f"Intentando continuar con el siguiente lote...")
            last_processed_frame = end_frame  # Asumir que procesamos hasta el final y seguir adelante
    
    # Combinar resultados
    if batch_dirs:
        merge_batch_results(args.output, batch_dirs, args.total_frames)
        print("\n¡Todos los lotes procesados correctamente!")
    else:
        print("\nNo se procesaron lotes correctamente.")

if __name__ == "__main__":
    main()