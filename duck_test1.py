import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional, Union
import json
import os
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from datetime import datetime
from pathlib import Path
import glob
from tqdm import tqdm
from collections import deque

class KalmanFilter:
    def __init__(self, dt=1.0):
        """
        Implementación simple de Filtro de Kalman para tracking 2D
        
        Args:
            dt: Intervalo de tiempo entre frames
        """
        self.dt = dt
        
        # Estado [x, y, vx, vy]
        self.state = np.zeros(4)
        
        # Matriz de transición de estado
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Matriz de medición
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Covarianza del proceso
        self.Q = np.eye(4) * 0.1
        
        # Covarianza de la medición
        self.R = np.eye(2) * 1.0
        
        # Covarianza del error
        self.P = np.eye(4) * 1000
        
    def predict(self):
        """Predice el siguiente estado"""
        self.state = self.A @ self.state
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.H @ self.state
    
    def update(self, measurement):
        """
        Actualiza el estado con una nueva medición
        
        Args:
            measurement: Array [x, y] con la posición medida
        """
        # Ganancia de Kalman
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Actualizar estado
        y = measurement - (self.H @ self.state)
        self.state = self.state + K @ y
        
        # Actualizar covarianza
        self.P = (np.eye(4) - K @ self.H) @ self.P

class DuckTrackerAdvanced:
    def __init__(self, model_path: str = r'/home/strpicket/Duck-Tracker-Project/best.pt',
                 min_confidence: float = 0.3,
                 min_distance: int = 20,
                 max_frames: int = 2083,
                 max_ducks: int = 7,
                 max_tracking_history: int = 30,
                 max_distance_threshold: int = 50,
                 track_timeout: int = 20,  # Aumentado para mayor estabilidad
                 reid_threshold: float = 0.3):  # Umbral más bajo para facilitar la reasignación
        """
        Initialize the advanced duck tracker with trajectory-based ID tracking.
        
        Args:
            model_path: Path to trained YOLOv8 model
            min_confidence: Minimum confidence for valid detection
            min_distance: Minimum pixel distance between detections
            max_frames: Maximum frames to process
            max_ducks: Maximum number of ducks to track
            max_tracking_history: Number of frames to keep in tracking history
            max_distance_threshold: Maximum distance to associate detections between frames
            track_timeout: Frames before considering a track lost
            reid_threshold: Threshold for re-identification confidence
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
            
        self.model = YOLO(model_path)
        self.duck_positions = {}
        self.duck_history = {}
        self.min_confidence = min_confidence
        self.min_distance = min_distance
        self.frame_count = 0
        self.max_frames = max_frames
        self.max_ducks = max_ducks
        self.grid_size = (8, 8)
        self.all_frames_data = {}
        
        # Tracking parameters
        self.max_tracking_history = max_tracking_history
        self.max_distance_threshold = max_distance_threshold
        self.track_timeout = track_timeout
        self.reid_threshold = reid_threshold
        
        # Nueva estructura para usar IDs fijos
        self.available_ids = [f"duck_{i+1}" for i in range(self.max_ducks)]  # Preasignar IDs del 1 al 7
        self.id_in_use = {id: False for id in self.available_ids}  # Seguimiento de qué IDs están en uso
        
        self.tracked_ducks = {}  # Store tracking information for each duck
        self.next_duck_id = 1    # Counter for assigning new duck IDs (se mantiene para compatibilidad)
        self.lost_ducks = {}     # Store information about lost ducks
        
        # Nuevos parámetros para estabilidad de IDs
        self.id_persistency = 10  # Frames a mantener un ID incluso si no se detecta
        self.color_stability = {}  # Registro histórico de colores para cada posición en la imagen
        
        self.color_map = {
            'yellow': {'plot': 'y', 'text': 'black', 'rgb': (1, 1, 0)},
            'black': {'plot': 'k', 'text': 'white', 'rgb': (0, 0, 0)}
        }

    def initialize_new_duck(self, position: Tuple[int, int], color: str) -> str:
        """
        Initialize tracking for a new duck with Kalman filter.
        Ahora usa IDs preasignados y evita crear nuevos IDs más allá del máximo.
        
        Args:
            position: Initial position (x, y)
            color: Duck color
            
        Returns:
            Duck ID string
        """
        # Buscar un ID disponible
        for duck_id in self.available_ids:
            if not self.id_in_use[duck_id]:
                self.id_in_use[duck_id] = True
                
                # Inicializar Kalman filter
                kf = KalmanFilter()
                kf.state[:2] = position
                
                self.tracked_ducks[duck_id] = {
                    'positions': deque(maxlen=self.max_tracking_history),
                    'color': color,
                    'last_seen': self.frame_count,
                    'trajectory': [],
                    'kalman_filter': kf,
                    'tracking_score': 1.0,  # Iniciar con confianza máxima
                    'persistence_count': self.id_persistency  # Nuevo contador de persistencia
                }
                self.tracked_ducks[duck_id]['positions'].append(position)
                self.tracked_ducks[duck_id]['trajectory'].append(position)
                
                return duck_id
        
        # Si llegamos aquí, no hay IDs disponibles (no debería ocurrir con el nuevo sistema)
        # Elegimos el ID con el valor de tracking_score más bajo
        min_score = float('inf')
        replaced_id = None
        for duck_id, data in self.tracked_ducks.items():
            if data['tracking_score'] < min_score:
                min_score = data['tracking_score']
                replaced_id = duck_id
        
        if replaced_id:
            # Reutilizar este ID para el nuevo pato
            kf = KalmanFilter()
            kf.state[:2] = position
            
            self.tracked_ducks[replaced_id] = {
                'positions': deque(maxlen=self.max_tracking_history),
                'color': color,
                'last_seen': self.frame_count,
                'trajectory': [],
                'kalman_filter': kf,
                'tracking_score': 0.7,  # Un poco reducido para nuevo reemplazo
                'persistence_count': self.id_persistency  # Nuevo contador de persistencia
            }
            self.tracked_ducks[replaced_id]['positions'].append(position)
            self.tracked_ducks[replaced_id]['trajectory'].append(position)
            
            return replaced_id
            
        # En caso extremo, devolvemos duck_1 (nunca debe llegar aquí)
        return "duck_1"

    def update_duck_tracking(self, duck_id: str, position: Tuple[int, int], color: str, confidence: float):
        """
        Update tracking information for an existing duck.
        
        Args:
            duck_id: Duck identifier
            position: New position (x, y)
            color: Duck color
            confidence: Detection confidence
        """
        if duck_id not in self.tracked_ducks:
            return self.initialize_new_duck(position, color)
        
        track_data = self.tracked_ducks[duck_id]
        
        # Actualizar Kalman filter
        kf = track_data['kalman_filter']
        measurement = np.array(position)
        kf.update(measurement)
        
        # Actualizar posición y trayectoria
        track_data['positions'].append(position)
        track_data['trajectory'].append(position)
        track_data['last_seen'] = self.frame_count
        
        # Mantener estabilidad de color (no cambiar color fácilmente)
        if track_data['color'] == color or confidence > 0.7:
            track_data['color'] = color
        
        # Restablecer contador de persistencia
        track_data['persistence_count'] = self.id_persistency
        
        # Actualizar score de tracking - más peso a la confianza actual para estabilidad
        track_data['tracking_score'] = min(1.0, 
            track_data['tracking_score'] * 0.8 + confidence * 0.2)  # Actualización más suave
        
        return duck_id

    def calculate_trajectory_similarity(self, track1: List[Tuple[int, int]], track2: List[Tuple[int, int]]) -> float:
        """
        Calcula la similitud entre dos trayectorias usando DTW (Dynamic Time Warping).
        
        Args:
            track1: Primera trayectoria
            track2: Segunda trayectoria
            
        Returns:
            Score de similitud entre 0 y 1
        """
        if not track1 or not track2:
            return 0.0
            
        # Convertir a arrays numpy
        t1 = np.array(track1)
        t2 = np.array(track2)
        
        # Calcular matriz de distancias
        distances = cdist(t1, t2)
        
        # DTW simple
        n, m = distances.shape
        dtw_matrix = np.zeros((n+1, m+1))
        dtw_matrix.fill(np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = distances[i-1, j-1]
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],    # inserción
                    dtw_matrix[i, j-1],    # eliminación
                    dtw_matrix[i-1, j-1]   # coincidencia
                )
        
        # Normalizar por la longitud de la trayectoria más larga
        max_length = max(n, m)
        similarity = 1.0 - (dtw_matrix[n, m] / (max_length * self.max_distance_threshold))
        return max(0.0, min(1.0, similarity))

    def find_lost_duck_match(self, detection: Dict, current_pos: Tuple[int, int]) -> Optional[str]:
        """
        Intenta encontrar una coincidencia entre una detección y un pato perdido.
        Mejorado con criterios más flexibles para reasignación.
        
        Args:
            detection: Detección actual
            current_pos: Posición actual del pato
            
        Returns:
            ID del pato perdido si se encuentra una coincidencia, None en caso contrario
        """
        best_match = None
        best_score = self.reid_threshold  # Umbral reducido para facilitar reidentificación
        
        for lost_id, lost_data in self.lost_ducks.items():
            # El color ya no es requisito obligatorio, pero sí un factor importante
            color_match = 1.0 if lost_data['color'] == detection['color'] else 0.5
            
            # Calcular similitud de trayectoria
            trajectory_similarity = self.calculate_trajectory_similarity(
                lost_data['trajectory'],
                [current_pos]
            )
            
            # Calcular similitud de posición predicha
            kf = lost_data['kalman_filter']
            predicted_pos = kf.predict()
            distance = np.linalg.norm(np.array(current_pos) - predicted_pos)
            position_similarity = 1.0 - min(1.0, distance / self.max_distance_threshold)
            
            # Verificar tiempo desde última vista - penalización más gradual
            frames_lost = self.frame_count - lost_data['last_seen']
            time_penalty = max(0, 1 - (frames_lost / (self.track_timeout * 2)))
            
            # Score combinado con penalización por tiempo y más peso a posición
            combined_score = (0.3 * trajectory_similarity + 
                            0.4 * position_similarity + 
                            0.2 * time_penalty +
                            0.1 * color_match)
            
            if combined_score > best_score:
                best_score = combined_score
                best_match = lost_id
        
        return best_match

    def update_lost_ducks(self):
        """
        Actualiza la lista de patos perdidos y mantiene IDs consistentes.
        Ahora incluye persistencia de IDs por un número de frames.
        """
        # Primero decrementar los contadores de persistencia para todos los patos
        for duck_id, track_data in self.tracked_ducks.items():
            frames_since_last_seen = self.frame_count - track_data['last_seen']
            if frames_since_last_seen > 0:
                if 'persistence_count' in track_data:
                    track_data['persistence_count'] -= 1
                else:
                    track_data['persistence_count'] = self.id_persistency - 1
                
                # Actualizamos el tracking score para reflejar que no ha sido visto
                track_data['tracking_score'] *= 0.95
        
        # Mover a lost_ducks solo si se agota la persistencia y excede timeout
        for duck_id in list(self.tracked_ducks.keys()):
            track_data = self.tracked_ducks[duck_id]
            frames_since_last_seen = self.frame_count - track_data['last_seen']
            
            if (frames_since_last_seen > self.track_timeout and 
                track_data.get('persistence_count', 0) <= 0):
                # Mover a lost_ducks
                self.lost_ducks[duck_id] = {
                    'color': track_data['color'],
                    'trajectory': list(track_data['trajectory']),
                    'kalman_filter': track_data['kalman_filter'],
                    'last_seen': track_data['last_seen'],
                    'tracking_score': track_data['tracking_score']
                }
                # Liberar el ID para reutilizar
                self.id_in_use[duck_id] = False
                # Eliminar de tracked_ducks
                del self.tracked_ducks[duck_id]
        
        # Limpiar lost_ducks antiguos pero preservando información para reutilización posterior
        for duck_id in list(self.lost_ducks.keys()):
            frames_since_last_seen = self.frame_count - self.lost_ducks[duck_id]['last_seen']
            if frames_since_last_seen > self.track_timeout * 4:  # Mayor duración para recuperación
                # Solo liberamos completamente si ha pasado mucho tiempo
                del self.lost_ducks[duck_id]

    def match_ducks_to_detections(self, detections: List[Dict]) -> Dict[str, Dict]:
        """
        Match current detections to existing tracked ducks using Hungarian algorithm.
        Mejorado para mantener IDs consistentes y evitar reasignaciones innecesarias.
        
        Args:
            detections: List of current frame detections
            
        Returns:
            Dictionary mapping duck IDs to detection data
        """
        if not detections:
            return {}

        # Limitar detecciones a max_ducks
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:self.max_ducks]

        # Actualizar lista de patos perdidos
        self.update_lost_ducks()

        # Predecir nuevas posiciones para todos los tracks activos
        active_tracks = []
        predicted_positions = []
        
        for duck_id, track_data in self.tracked_ducks.items():
            frames_since_last_seen = self.frame_count - track_data['last_seen']
            # Incluir patos vistos recientemente o con persistencia
            if frames_since_last_seen <= self.track_timeout or track_data.get('persistence_count', 0) > 0:
                active_tracks.append(duck_id)
                kf = track_data['kalman_filter']
                predicted_pos = kf.predict()
                predicted_positions.append(predicted_pos)

        # Si no hay tracks activos, intentar re-identificar con patos perdidos primero
        matched_ducks = {}
        used_detections = set()
        
        # Primero intentar re-identificar patos perdidos
        for i, detection in enumerate(detections):
            center = self.calculate_center_point(detection['bbox'])
            lost_id = self.find_lost_duck_match(detection, center)
            
            if lost_id:
                matched_ducks[lost_id] = detection
                # Restaurar el pato perdido con tracking más estable
                self.tracked_ducks[lost_id] = self.lost_ducks[lost_id]
                self.tracked_ducks[lost_id]['persistence_count'] = self.id_persistency
                self.id_in_use[lost_id] = True
                del self.lost_ducks[lost_id]
                used_detections.add(i)

        # Si aún no tenemos suficientes patos activos, proceder con el matching normal
        if active_tracks and len(matched_ducks) < min(len(detections), self.max_ducks):
            # Preparar detecciones no usadas
            remaining_detections = [det for i, det in enumerate(detections) if i not in used_detections]
            detection_positions = [self.calculate_center_point(det['bbox']) for det in remaining_detections]
            
            if detection_positions:
                # Construir matriz de costos
                cost_matrix = np.zeros((len(active_tracks), len(detection_positions)))
                
                for i, pred_pos in enumerate(predicted_positions):
                    for j, det_pos in enumerate(detection_positions):
                        # Distancia euclidiana entre predicción y detección
                        distance = np.linalg.norm(pred_pos - det_pos)
                        # Penalizar cambios de color pero no prohibirlos
                        color_penalty = 10 if self.tracked_ducks[active_tracks[i]]['color'] != remaining_detections[j]['color'] else 0
                        cost_matrix[i, j] = distance + color_penalty

                # Aplicar algoritmo Húngaro para asignación óptima
                track_indices, detection_indices = linear_sum_assignment(cost_matrix)
                
                # Procesar asignaciones válidas - umbral más permisivo
                for track_idx, det_idx in zip(track_indices, detection_indices):
                    # Aumentamos el umbral para evitar perder tracks fácilmente
                    if cost_matrix[track_idx, det_idx] <= self.max_distance_threshold * 1.5:
                        duck_id = active_tracks[track_idx]
                        detection = remaining_detections[det_idx]
                        center = self.calculate_center_point(detection['bbox'])
                        
                        self.update_duck_tracking(duck_id, center, detection['color'], detection['confidence'])
                        matched_ducks[duck_id] = detection
                        used_detections.add(det_idx)

        # Solo crear nuevos IDs si no hemos alcanzado el límite de patos y hay detecciones sin asignar
        available_slots = self.max_ducks - len(self.tracked_ducks)
        if available_slots > 0:
            # Solo procesamos detecciones no utilizadas
            remaining = [i for i, det in enumerate(detections) if i not in used_detections]
            for i in remaining:
                if len(self.tracked_ducks) < self.max_ducks:
                    detection = detections[i]
                    center = self.calculate_center_point(detection['bbox'])
                    duck_id = self.initialize_new_duck(center, detection['color'])
                    matched_ducks[duck_id] = detection
        
        return matched_ducks

    def process_frame(self, frame: np.ndarray) -> None:
        """
        Process a frame and update duck positions with consistent IDs.
        """
        detections = self.detect_ducks(frame)
        self.duck_positions.clear()
        
        # Match detections to existing tracks
        matched_ducks = self.match_ducks_to_detections(detections)
        
        # Update tracking information
        for duck_id, detection in matched_ducks.items():
            center = self.calculate_center_point(detection['bbox'])
            grid_pos = self.calculate_grid_position(center, frame.shape)
            
            # Update tracking
            self.update_duck_tracking(duck_id, center, detection['color'], detection['confidence'])
            
            # Calculate velocity from trajectory
            velocity = self.calculate_velocity(duck_id)
            
            self.duck_positions[duck_id] = {
                'position': center,
                'grid_position': grid_pos,
                'confidence': detection['confidence'],
                'bbox': detection['bbox'],
                'color': detection['color'],
                'velocity': velocity,
                'trajectory': list(self.tracked_ducks[duck_id]['positions'])
            }
        
        self.frame_count += 1

    def calculate_velocity(self, duck_id: str) -> Optional[Tuple[float, float]]:
        """
        Calculate velocity based on recent trajectory.
        """
        if duck_id not in self.tracked_ducks or len(self.tracked_ducks[duck_id]['positions']) < 2:
            return None
            
        positions = self.tracked_ducks[duck_id]['positions']
        if len(positions) < 2:
            return None
            
        last_pos = positions[-1]
        prev_pos = positions[-2]
        
        vx = last_pos[0] - prev_pos[0]
        vy = last_pos[1] - prev_pos[1]
        
        return (vx, vy)

    def visualize_2d(self, frame: np.ndarray) -> None:
        """
        Visualiza los patitos con trayectorias y predicciones de Kalman.
        """
        plt.figure(figsize=(15, 7))
        
        # Original image with detections
        ax1 = plt.subplot(121)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.draw_grid(ax1, frame.shape)
        
        # Dibujar patos perdidos primero (en gris)
        for lost_id, lost_data in self.lost_ducks.items():
            if lost_data['trajectory']:
                last_pos = lost_data['trajectory'][-1]
                color_info = self.color_map[lost_data['color']]
                
                # Dibujar trayectoria en gris
                trajectory = np.array(lost_data['trajectory'])
                plt.plot(trajectory[:, 0], trajectory[:, 1], '--', 
                        color='gray', alpha=0.3, linewidth=1)
                
                # Dibujar predicción de Kalman
                kf = lost_data['kalman_filter']
                predicted_pos = kf.predict()
                plt.plot(predicted_pos[0], predicted_pos[1], 'r+', markersize=8, alpha=0.3)
                
                # Etiqueta de pato perdido
                plt.text(last_pos[0], last_pos[1]-10, 
                        f'{lost_id} (perdido)',
                        color='gray',
                        fontsize=8,
                        bbox=dict(facecolor='gray', alpha=0.3))
        
        # Dibujar patos activos
        for duck_id, data in self.duck_positions.items():
            x1, y1, x2, y2 = data['bbox']
            x, y = data['position']
            confidence = data['confidence']
            duck_color = data['color']
            velocity = data['velocity']
            grid_pos = data['grid_position']
            tracking_score = self.tracked_ducks[duck_id]['tracking_score']
            
            color_info = self.color_map[duck_color]
            
            # Draw bounding box with alpha based on tracking score
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, 
                               color=color_info['rgb'],
                               alpha=tracking_score,
                               linewidth=2)
            plt.gca().add_patch(rect)
            
            # Draw trajectory with fade effect
            if 'trajectory' in data and len(data['trajectory']) > 1:
                trajectory = np.array(data['trajectory'])
                num_points = len(trajectory)
                for i in range(num_points - 1):
                    alpha = (i + 1) / num_points  # Más opaco para puntos más recientes
                    plt.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], '--', 
                            color=color_info['rgb'], alpha=alpha * 0.8, linewidth=1)
            
            # Draw center point
            plt.plot(x, y, color_info['plot'] + 'o', markersize=8)
            
            # Draw predicted position from Kalman filter
            kf = self.tracked_ducks[duck_id]['kalman_filter']
            predicted_pos = kf.predict()
            plt.plot(predicted_pos[0], predicted_pos[1], 'r+', markersize=8, alpha=0.6)
            
            # Add label with consistent ID and tracking info
            info_text = f'{duck_id} ({confidence:.2f})\n{duck_color}\nGrid: {grid_pos}\nTrack: {tracking_score:.2f}'
            if velocity:
                info_text += f'\nv: ({velocity[0]:.1f}, {velocity[1]:.1f})'
            
            plt.text(x1, y1-10, info_text,
                    color=color_info['text'], 
                    fontsize=8,
                    bbox=dict(facecolor=color_info['rgb'], alpha=0.7))
        
        plt.title('Detecciones y Predicciones')
        plt.axis('off')
        
        # 2D view with trajectories
        ax2 = plt.subplot(122)
        plt.imshow(np.zeros_like(frame[:,:,0]), cmap='gray')
        self.draw_grid(ax2, frame.shape)
        
        # Dibujar patos perdidos primero
        for lost_id, lost_data in self.lost_ducks.items():
            if lost_data['trajectory']:
                last_pos = lost_data['trajectory'][-1]
                color_info = self.color_map[lost_data['color']]
                
                # Dibujar trayectoria en gris
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
        for duck_id, data in self.duck_positions.items():
            x, y = data['position']
            duck_color = data['color']
            color_info = self.color_map[duck_color]
            tracking_score = self.tracked_ducks[duck_id]['tracking_score']
            
            # Draw complete trajectory with fade effect
            if 'trajectory' in data and len(data['trajectory']) > 1:
                trajectory = np.array(data['trajectory'])
                num_points = len(trajectory)
                for i in range(num_points - 1):
                    alpha = (i + 1) / num_points * tracking_score
                    plt.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], '-', 
                            color=color_info['rgb'], alpha=alpha)
            
            # Current position
            plt.scatter(x, y, c=[color_info['rgb']], s=100, alpha=tracking_score)
            
            # Predicted position
            kf = self.tracked_ducks[duck_id]['kalman_filter']
            predicted_pos = kf.predict()
            plt.plot([x, predicted_pos[0]], [y, predicted_pos[1]], 'r--', alpha=0.4)
            plt.scatter(predicted_pos[0], predicted_pos[1], c='r', s=50, alpha=0.4)
            
            # Label with ID and info
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
        plt.show()

    def get_duck_color(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """
        Determina el color del pato basado en su región de interés.
        """
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        yellow_ratio = np.sum(yellow_mask > 0) / (yellow_mask.shape[0] * yellow_mask.shape[1])
        
        return 'yellow' if yellow_ratio > 0.15 else 'black'

    def calculate_grid_position(self, point: Tuple[int, int], frame_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calcula la posición en la cuadrícula basada en las coordenadas de la imagen.
        
        Args:
            point: Coordenadas (x, y) en la imagen
            frame_shape: Dimensiones del frame (height, width)
            
        Returns:
            Tupla con la posición en la cuadrícula (grid_x, grid_y)
        """
        x, y = point
        height, width = frame_shape[:2]
        
        grid_x = int((x / width) * self.grid_size[0])
        grid_y = int((y / height) * self.grid_size[1])
        
        return (grid_x, grid_y)

    def calculate_center_point(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Calcula el punto central de un bounding box.
        """
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def detect_ducks(self, frame: np.ndarray) -> List[Dict]:
        """
        Detecta patitos en el frame usando YOLOv8.
        
        Args:
            frame: Frame de video a procesar
            
        Returns:
            Lista de detecciones de patitos
        """
        results = self.model(frame)
        detections = []
        
        for result in results:
            boxes = result.boxes
            # Convertir todos los boxes a detecciones
            all_detections = []
            for box in boxes:
                if box.cls == 0 and box.conf[0] > self.min_confidence:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    all_detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence
                    })
            
            # Ordenar por confianza y tomar solo los 7 mejores
            all_detections = sorted(all_detections, key=lambda x: x['confidence'], reverse=True)[:self.max_ducks]
            
            # Filtrar detecciones cercanas y asignar colores
            filtered_detections = []
            centers = []
            
            for detection in all_detections:
                center = self.calculate_center_point(detection['bbox'])
                
                if not centers:
                    detection['color'] = self.get_duck_color(frame, detection['bbox'])
                    filtered_detections.append(detection)
                    centers.append(center)
                    continue
                
                # Verificar distancia mínima con otros centros
                distances = cdist([center], centers)[0]
                if np.min(distances) > self.min_distance:
                    detection['color'] = self.get_duck_color(frame, detection['bbox'])
                    filtered_detections.append(detection)
                    centers.append(center)
                else:
                    # Si está cerca, verificar si tienen colores diferentes
                    nearest_idx = np.argmin(distances)
                    current_color = self.get_duck_color(frame, detection['bbox'])
                    if current_color != filtered_detections[nearest_idx]['color']:
                        detection['color'] = current_color
                        if len(filtered_detections) < self.max_ducks:
                            filtered_detections.append(detection)
                            centers.append(center)
            
            detections = filtered_detections
        
        return detections

    def draw_grid(self, ax, shape: Tuple[int, int]):
        """
        Dibuja una cuadrícula en el plot.
        
        Args:
            ax: Eje de matplotlib donde dibujar
            shape: Dimensiones del frame
        """
        height, width = shape[:2]
        
        # Líneas verticales
        for i in range(self.grid_size[0] + 1):
            x = (width / self.grid_size[0]) * i
            ax.plot([x, x], [0, height], 'w-', alpha=0.3)
            
        # Líneas horizontales
        for i in range(self.grid_size[1] + 1):
            y = (height / self.grid_size[1]) * i
            ax.plot([0, width], [y, y], 'w-', alpha=0.3)

    def save_positions(self, filename: str = None) -> None:
        """
        Guarda las posiciones y datos adicionales de los patitos en un archivo JSON.
        
        Args:
            filename: Nombre del archivo donde guardar las posiciones
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"duck_positions_{timestamp}.json"
        
        data_to_save = {
            'frame_count': self.frame_count,
            'grid_size': self.grid_size,
            'ducks': self.duck_positions,
            'history': self.duck_history
        }
        
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=4)

    def process_video(self, video_path: str, output_dir: str = "output", save_frames: bool = True, display: bool = False, save_originals: bool = True) -> None:
        """
        Procesa un video completo y guarda los resultados.
        
        Args:
            video_path: Ruta al archivo de video
            output_dir: Directorio donde guardar los resultados
            save_frames: Si se deben guardar las imágenes de cada frame
            display: Si se debe mostrar el video mientras se procesa
            save_originals: Si se deben guardar los frames originales sin procesar
        """
        # Crear directorio de salida si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Crear subdirectorios para frames
        if save_frames:
            processed_frames_dir = os.path.join(output_dir, "processed_frames")
            os.makedirs(processed_frames_dir, exist_ok=True)
            
            if save_originals:
                original_frames_dir = os.path.join(output_dir, "original_frames")
                os.makedirs(original_frames_dir, exist_ok=True)
        
        # Abrir el video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        # Obtener información del video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Configurar el video de salida
        output_video_path = os.path.join(output_dir, "output_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width*2, height))
        
        print(f"\nProcesando video: {video_path}")
        print(f"Dimensiones: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Frames totales: {total_frames}")
        print(f"Frames a procesar: {min(total_frames, self.max_frames)}")
        print("Procesando frames... (esto puede tomar un tiempo)")
        
        frame_number = 0
        
        # Barra de progreso
        with tqdm(total=min(total_frames, self.max_frames), desc="Procesando frames") as pbar:
            while cap.isOpened() and frame_number < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Guardar frame original si está habilitado
                if save_frames and save_originals:
                    original_frame_path = os.path.join(original_frames_dir, f"original_{frame_number:04d}.png")
                    cv2.imwrite(original_frame_path, frame)
                
                # Procesar el frame
                self.process_frame(frame)
                
                # Crear visualización sin mostrar
                plt.figure(figsize=(15, 7))
                
                # Original image with detections
                ax1 = plt.subplot(121)
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.draw_grid(ax1, frame.shape)
                
                # Dibujar patos perdidos primero (en gris)
                for lost_id, lost_data in self.lost_ducks.items():
                    if lost_data['trajectory']:
                        last_pos = lost_data['trajectory'][-1]
                        color_info = self.color_map[lost_data['color']]
                        
                        # Dibujar trayectoria en gris
                        trajectory = np.array(lost_data['trajectory'])
                        plt.plot(trajectory[:, 0], trajectory[:, 1], '--', 
                                color='gray', alpha=0.3, linewidth=1)
                        
                        # Dibujar predicción de Kalman
                        kf = lost_data['kalman_filter']
                        predicted_pos = kf.predict()
                        plt.plot(predicted_pos[0], predicted_pos[1], 'r+', markersize=8, alpha=0.3)
                
                # Dibujar patos activos
                for duck_id, data in self.duck_positions.items():
                    x1, y1, x2, y2 = data['bbox']
                    x, y = data['position']
                    confidence = data['confidence']
                    duck_color = data['color']
                    velocity = data['velocity']
                    grid_pos = data['grid_position']
                    tracking_score = self.tracked_ducks[duck_id]['tracking_score']
                    
                    color_info = self.color_map[duck_color]
                    
                    # Draw bounding box with alpha based on tracking score
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       fill=False, 
                                       color=color_info['rgb'],
                                       alpha=tracking_score,
                                       linewidth=2)
                    plt.gca().add_patch(rect)
                    
                    # Draw trajectory with fade effect
                    if 'trajectory' in data and len(data['trajectory']) > 1:
                        trajectory = np.array(data['trajectory'])
                        num_points = len(trajectory)
                        for i in range(num_points - 1):
                            alpha = (i + 1) / num_points
                            plt.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], '--', 
                                    color=color_info['rgb'], alpha=alpha * 0.8, linewidth=1)
                    
                    # Draw center point
                    plt.plot(x, y, color_info['plot'] + 'o', markersize=8)
                    
                    # Draw predicted position from Kalman filter
                    kf = self.tracked_ducks[duck_id]['kalman_filter']
                    predicted_pos = kf.predict()
                    plt.plot(predicted_pos[0], predicted_pos[1], 'r+', markersize=8, alpha=0.6)
                    
                    # Add label with consistent ID and tracking info
                    info_text = f'{duck_id} ({confidence:.2f})\n{duck_color}\nGrid: {grid_pos}\nTrack: {tracking_score:.2f}'
                    if velocity:
                        info_text += f'\nv: ({velocity[0]:.1f}, {velocity[1]:.1f})'
                    
                    plt.text(x1, y1-10, info_text,
                            color=color_info['text'], 
                            fontsize=8,
                            bbox=dict(facecolor=color_info['rgb'], alpha=0.7))
                
                plt.title('Detecciones y Predicciones')
                plt.axis('off')
                
                # 2D view with trajectories
                ax2 = plt.subplot(122)
                plt.imshow(np.zeros_like(frame[:,:,0]), cmap='gray')
                self.draw_grid(ax2, frame.shape)
                
                # Dibujar patos perdidos primero
                for lost_id, lost_data in self.lost_ducks.items():
                    if lost_data['trajectory']:
                        last_pos = lost_data['trajectory'][-1]
                        color_info = self.color_map[lost_data['color']]
                        
                        # Dibujar trayectoria en gris
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
                for duck_id, data in self.duck_positions.items():
                    x, y = data['position']
                    duck_color = data['color']
                    color_info = self.color_map[duck_color]
                    tracking_score = self.tracked_ducks[duck_id]['tracking_score']
                    
                    # Draw complete trajectory with fade effect
                    if 'trajectory' in data and len(data['trajectory']) > 1:
                        trajectory = np.array(data['trajectory'])
                        num_points = len(trajectory)
                        for i in range(num_points - 1):
                            alpha = (i + 1) / num_points * tracking_score
                            plt.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], '-', 
                                    color=color_info['rgb'], alpha=alpha)
                    
                    # Current position
                    plt.scatter(x, y, c=[color_info['rgb']], s=100, alpha=tracking_score)
                    
                    # Predicted position
                    kf = self.tracked_ducks[duck_id]['kalman_filter']
                    predicted_pos = kf.predict()
                    plt.plot([x, predicted_pos[0]], [y, predicted_pos[1]], 'r--', alpha=0.4)
                    plt.scatter(predicted_pos[0], predicted_pos[1], c='r', s=50, alpha=0.4)
                    
                    # Label with ID and info
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
                
                # Guardar frame procesado en la carpeta correspondiente
                if save_frames:
                    processed_frame_path = os.path.join(processed_frames_dir, f"processed_{frame_number:04d}.png")
                    plt.savefig(processed_frame_path)
                else:
                    # Uso temporal si no se guardan los frames
                    frame_output_path = os.path.join(output_dir, f"temp_frame_{frame_number:04d}.png")
                    plt.savefig(frame_output_path)
                
                # Crear el frame combinado para el video
                if save_frames:
                    processed_frame = cv2.imread(processed_frame_path)
                else:
                    processed_frame = cv2.imread(frame_output_path)
                
                if processed_frame is not None:
                    # Redimensionar si es necesario
                    processed_frame = cv2.resize(processed_frame, (width, height))
                    # Combinar frames lado a lado
                    combined_frame = np.hstack((frame, processed_frame))
                    out.write(combined_frame)
                
                # Cerrar la figura para liberar memoria
                plt.close()
                
                # Eliminar el archivo temporal si no se guardan los frames
                if not save_frames and os.path.exists(frame_output_path):
                    os.remove(frame_output_path)
                
                # Guardar datos del frame actual
                self.all_frames_data[frame_number] = {
                    'positions': self.duck_positions.copy(),
                    'timestamp': datetime.now().isoformat(),
                    'frame_number': frame_number
                }
                
                frame_number += 1
                pbar.update(1)
        
        # Liberar recursos
        cap.release()
        out.release()
        
        # Guardar todos los datos en un archivo JSON
        self.save_all_data(os.path.join(output_dir, "tracking_data.json"))
        
        # Mensaje de completado
        print(f"\n¡Procesamiento completado!")
        print(f"Video procesado guardado como: {output_video_path}")
        
        if save_frames:
            print(f"Frames procesados guardados en: {processed_frames_dir}")
            if save_originals:
                print(f"Frames originales guardados en: {original_frames_dir}")
                
        print(f"Datos guardados en: {os.path.join(output_dir, 'tracking_data.json')}")
        
        # Mostrar el video final
        if display:
            print("\nReproduciendo video procesado...")
            os.system(f"xdg-open {output_video_path}")

    def process_image_sequence(self, image_dir: str, output_dir: str = "output") -> None:
        """
        Procesa una secuencia de imágenes en un directorio.
        
        Args:
            image_dir: Directorio que contiene las imágenes
            output_dir: Directorio donde guardar los resultados
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Obtener lista de imágenes
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) + 
                           glob.glob(os.path.join(image_dir, "*.png")))
        
        for frame_number, image_path in enumerate(image_files[:self.max_frames]):
            print(f"Procesando imagen {frame_number + 1}/{min(len(image_files), self.max_frames)}")
            
            # Leer y procesar la imagen
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"No se pudo leer la imagen: {image_path}")
                continue
                
            # Procesar el frame
            self.process_frame(frame)
            
            # Guardar datos del frame actual
            self.all_frames_data[frame_number] = {
                'positions': self.duck_positions.copy(),
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path
            }
            
            # Visualizar y guardar
            self.visualize_2d(frame)
            plt.savefig(os.path.join(output_dir, f"frame_{frame_number:04d}.png"))
            plt.close()
        
        # Guardar todos los datos en un archivo JSON
        self.save_all_data(os.path.join(output_dir, "tracking_data.json"))

    def save_all_data(self, filename: str) -> None:
        """
        Guarda todos los datos de tracking en un archivo JSON.
        
        Args:
            filename: Nombre del archivo donde guardar los datos
        """
        data_to_save = {
            'total_frames': len(self.all_frames_data),
            'grid_size': self.grid_size,
            'frames': self.all_frames_data,
            'tracking_params': {
                'min_confidence': self.min_confidence,
                'min_distance': self.min_distance
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=4)

    def visualize_trajectories(self, output_path: str = None) -> None:
        """
        Genera una visualización separada de las trayectorias de todos los patos.
        
        Args:
            output_path: Ruta donde guardar la visualización. Si es None, se muestra en pantalla.
        """
        plt.figure(figsize=(12, 8))
        
        # Configurar el espacio de visualización
        plt.style.use('dark_background')
        plt.grid(True, alpha=0.2)
        
        # Dibujar trayectorias para patos activos
        for duck_id, track_data in self.tracked_ducks.items():
            if len(track_data['trajectory']) > 1:
                trajectory = np.array(track_data['trajectory'])
                color_info = self.color_map[track_data['color']]
                
                # Dibujar línea de trayectoria con efecto de degradado
                points = np.array(track_data['trajectory'])
                segments = np.array([[points[i], points[i+1]] for i in range(len(points)-1)])
                
                # Crear colección de segmentos de línea con colores graduales
                for i, segment in enumerate(segments):
                    alpha = (i + 1) / len(segments)  # Aumenta la opacidad hacia el final
                    plt.plot(segment[:, 0], segment[:, 1], '-',
                            color=color_info['rgb'],
                            alpha=alpha,
                            linewidth=2)
                
                # Marcar inicio y fin de la trayectoria
                plt.scatter(trajectory[0, 0], trajectory[0, 1], 
                          c=[color_info['rgb']], marker='o', s=100, 
                          label=f'{duck_id} (inicio)', alpha=0.5)
                plt.scatter(trajectory[-1, 0], trajectory[-1, 1], 
                          c=[color_info['rgb']], marker='*', s=200, 
                          label=f'{duck_id} (actual)')
                
                # Añadir flecha de dirección
                if len(trajectory) > 1:
                    dx = trajectory[-1, 0] - trajectory[-2, 0]
                    dy = trajectory[-1, 1] - trajectory[-2, 1]
                    plt.arrow(trajectory[-1, 0], trajectory[-1, 1], dx, dy,
                            color=color_info['rgb'], width=0.5,
                            head_width=10, head_length=10, alpha=0.7)
        
        # Dibujar trayectorias para patos perdidos
        for lost_id, lost_data in self.lost_ducks.items():
            if len(lost_data['trajectory']) > 1:
                trajectory = np.array(lost_data['trajectory'])
                
                # Dibujar trayectoria en gris
                plt.plot(trajectory[:, 0], trajectory[:, 1], '--',
                        color='gray', alpha=0.3, linewidth=1,
                        label=f'{lost_id} (perdido)')
        
        plt.title('Trayectorias de los Patos', pad=20)
        plt.xlabel('Posición X')
        plt.ylabel('Posición Y')
        
        # Ajustar leyenda
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

def main():
    # Inicializar el rastreador con el modelo específico
    model_path = r'/home/strpicket/Duck-Tracker-Project/best.pt'
    tracker = DuckTrackerAdvanced(
        model_path,
        min_confidence=0.3,
        min_distance=20,
        max_frames=2083,
        max_ducks=7,  # Limitar a 7 patos
        max_tracking_history=30,  # Number of frames to keep in history
        max_distance_threshold=50,  # Maximum distance for tracking association
        track_timeout=20,  # Frames before considering a track lost
        reid_threshold=0.3  # Threshold for re-identification confidence
    )
    
    # Procesar video (ajusta la ruta a tu video)
    video_path = input("Ingresa la ruta al video (o presiona Enter para usar 'assets/ducks_video.mp4'): ").strip()
    if not video_path:
        video_path = r'/home/strpicket/Duck-Tracker-Project/assets/DuckVideo.mp4'
    
    if os.path.exists(video_path):
        tracker.process_video(
            video_path,
            output_dir="output_video3",
            save_frames=True,
            display=True,  # Mostrar el video mientras se procesa
            save_originals=True  # Guardar frames originales
        )
    else:
        print(f"Error: No se encontró el video en: {video_path}")
        print("Asegúrate de que el archivo existe y la ruta es correcta.")

    # Guardar datos de tracking
    tracker.save_all_data('tracking_data.json')
    
    # Generar visualización de trayectorias
    tracker.visualize_trajectories('trayectorias_patos.png')
    print("Visualización de trayectorias guardada en 'trayectorias_patos.png'")

if __name__ == "__main__":
    main()