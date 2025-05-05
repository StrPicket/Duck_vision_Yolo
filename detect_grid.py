import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch

class DuckDetectorGrid:
    def __init__(self, model_path: str = "best.pt", conf_threshold: float = 0.25):
        """
        Inicializar detector de patos con cuadrícula
    
    Args:
            model_path: Ruta al modelo YOLOv8 entrenado
            conf_threshold: Umbral de confianza para detecciones
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.grid_size = (8, 8)  # Tamaño de la cuadrícula
        self.max_ducks = 7  # Máximo número de patos a detectar
        self.min_distance = 20  # Distancia mínima entre patos
        
        # Color verde para todas las detecciones
        self.box_color = (0, 255, 0)  # BGR
        self.text_color = (255, 255, 255)  # Texto blanco
        
        # Historial de trayectorias
        self.trajectories = {}
        self.next_id = 1
        self.max_trajectory_points = 30
        
        # Parámetros para manejo de patos perdidos
        self.lost_ducks = {}
        self.max_frames_lost = 10
        
    def get_grid_position(self, point: Tuple[float, float], frame_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Obtener posición en la cuadrícula
        """
        x, y = point
        h, w = frame_shape[:2]
        grid_x = int((x / w) * self.grid_size[0])
        grid_y = int((y / h) * self.grid_size[1])
        return (grid_x, grid_y)
    
    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calcular IoU entre dos cajas
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
    
    def predict_next_position(self, trajectory: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Predecir siguiente posición basada en la trayectoria
        """
        if len(trajectory) < 2:
            return trajectory[-1]
        
        # Calcular velocidad promedio de los últimos puntos
        velocities = []
        for i in range(1, min(5, len(trajectory))):
            vx = trajectory[-i][0] - trajectory[-i-1][0]
            vy = trajectory[-i][1] - trajectory[-i-1][1]
            velocities.append((vx, vy))
        
        avg_vx = sum(v[0] for v in velocities) / len(velocities)
        avg_vy = sum(v[1] for v in velocities) / len(velocities)
        
        last_x, last_y = trajectory[-1]
        return (last_x + avg_vx, last_y + avg_vy)
    
    def detect_and_track(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Detectar y trackear patos en el frame
        """
        # Ejecutar detección
        results = self.model(frame)[0]
        detections = []
        
        # Procesar detecciones
        for box in results.boxes:
            if box.conf[0] < self.conf_threshold:
                continue
                
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            grid_pos = self.get_grid_position(center, frame.shape)
            
            detections.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': conf,
                'center': center,
                'grid_position': grid_pos
            })
        
        # Asociar detecciones con trayectorias existentes
        if self.trajectories:
            # Crear matrices de costo
            cost_matrix = np.zeros((len(self.trajectories), len(detections)))
            trajectory_ids = list(self.trajectories.keys())
            
            for i, track_id in enumerate(trajectory_ids):
                trajectory = self.trajectories[track_id]
                predicted_pos = self.predict_next_position(trajectory['points'])
                
                for j, detection in enumerate(detections):
                    # Distancia entre predicción y detección
                    distance = np.sqrt(
                        (predicted_pos[0] - detection['center'][0])**2 +
                        (predicted_pos[1] - detection['center'][1])**2
                    )
                    
                    # Penalizar distancias grandes
                    if distance > self.min_distance * 3:
                        cost_matrix[i, j] = 1000
                    else:
                        cost_matrix[i, j] = distance
            
            # Asignar detecciones a trayectorias
            if cost_matrix.size > 0:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # Actualizar trayectorias asignadas
                assigned_detections = set()
                for i, j in zip(row_ind, col_ind):
                    if cost_matrix[i, j] < 1000:
                        track_id = trajectory_ids[i]
                        detection = detections[j]
                        
                        self.trajectories[track_id]['points'].append(detection['center'])
                        if len(self.trajectories[track_id]['points']) > self.max_trajectory_points:
                            self.trajectories[track_id]['points'].pop(0)
                            
                        self.trajectories[track_id]['last_detection'] = detection
                        self.trajectories[track_id]['frames_lost'] = 0
                        assigned_detections.add(j)
                
                # Crear nuevas trayectorias para detecciones no asignadas
                for i, detection in enumerate(detections):
                    if i not in assigned_detections and len(self.trajectories) < self.max_ducks:
                        self.trajectories[self.next_id] = {
                            'points': [detection['center']],
                            'last_detection': detection,
                            'frames_lost': 0
                        }
                        self.next_id += 1
        else:
            # Inicializar primeras trayectorias
            for detection in detections[:self.max_ducks]:
                self.trajectories[self.next_id] = {
                    'points': [detection['center']],
                    'last_detection': detection,
                    'frames_lost': 0
                }
                self.next_id += 1
        
        # Actualizar patos perdidos y eliminar trayectorias viejas
        for track_id in list(self.trajectories.keys()):
            self.trajectories[track_id]['frames_lost'] += 1
            if self.trajectories[track_id]['frames_lost'] > self.max_frames_lost:
                # Mover a patos perdidos antes de eliminar
                self.lost_ducks[track_id] = self.trajectories[track_id]
                del self.trajectories[track_id]
        
        # Dibujar resultados
        vis_frame = frame.copy()
        self.draw_detections(vis_frame)
        
        return list(self.trajectories.values()), vis_frame
    
    def draw_detections(self, frame: np.ndarray) -> None:
        """
        Dibujar detecciones y trayectorias
        """
        # Dibujar cuadrícula
        h, w = frame.shape[:2]
        for i in range(1, self.grid_size[0]):
            x = int(w * i / self.grid_size[0])
            cv2.line(frame, (x, 0), (x, h), (128, 128, 128), 1)
        for i in range(1, self.grid_size[1]):
            y = int(h * i / self.grid_size[1])
            cv2.line(frame, (0, y), (w, y), (128, 128, 128), 1)
        
        # Dibujar trayectorias activas
        for track_id, trajectory in self.trajectories.items():
            # Dibujar trayectoria
            points = np.array(trajectory['points'], dtype=np.int32)
            for i in range(len(points) - 1):
                # Aumentar opacidad para puntos más recientes
                alpha = (i + 1) / len(points)
                color = (int(self.box_color[0] * alpha),
                        int(self.box_color[1] * alpha),
                        int(self.box_color[2] * alpha))
                cv2.line(frame,
                        tuple(points[i]),
                        tuple(points[i + 1]),
                        color, 2)
            
            # Dibujar caja de detección actual
            if 'last_detection' in trajectory:
                det = trajectory['last_detection']
                x1, y1, x2, y2 = det['bbox']
                
                # Dibujar caja verde
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, 2)
                
                # Dibujar información
                text = f"Duck {track_id} ({det['confidence']:.2f})"
                text += f" Grid: {det['grid_position']}"
                
                cv2.putText(frame, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           self.text_color, 2)
                
        # Dibujar predicciones para patos perdidos
        for track_id, trajectory in self.lost_ducks.items():
            if len(trajectory['points']) >= 2:
                predicted_pos = self.predict_next_position(trajectory['points'])
                cv2.circle(frame,
                          (int(predicted_pos[0]), int(predicted_pos[1])),
                          5, (0, 0, 255), -1)  # Punto rojo para predicciones