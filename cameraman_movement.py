import os
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import argparse

def load_camera_data(intrinsics_file, essential_matrices_file):
    """
    Load camera intrinsics and essential matrices
    """
    try:
        K = np.load(intrinsics_file)
        essential_matrices = np.load(essential_matrices_file)
        return K, essential_matrices
    except Exception as e:
        print(f"Error loading camera data: {e}")
        return None, None

def decompose_essential_matrix(E, K):
    """
    Decompose essential matrix to get rotation and translation
    """
    # SVD of essential matrix
    U, S, Vt = np.linalg.svd(E)
    
    # Define W matrix
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    
    # Calculate rotation matrices (there are two possible solutions)
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    
    # Ensure rotation matrices are valid by enforcing det(R) = 1
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
    
    # Calculate translation vectors
    t1 = U[:, 2]
    t2 = -U[:, 2]
    
    # We get 4 possible combinations: (R1, t1), (R1, t2), (R2, t1), (R2, t2)
    # In practice, you'd need to check which one makes sense in your context
    # For simplicity, we'll return just the first one
    return R1, t1

def extract_cameraman_movement(essential_matrices, K):
    """
    Extract cameraman movement (rotation and translation) from essential matrices
    """
    movements = []
    cumulative_position = np.array([0.0, 0.0, 0.0])  # Starting position
    cumulative_rotation = np.eye(3)  # Starting rotation (identity matrix)
    
    for i, E in enumerate(essential_matrices):
        # Skip invalid matrices
        if E is None or np.isnan(E).any():
            movements.append({
                'frame': i,
                'position': cumulative_position.tolist(),
                'rotation_euler': [0, 0, 0],
                'rotation_matrix': cumulative_rotation.tolist()
            })
            continue
        
        # Decompose essential matrix to get R, t
        R, t = decompose_essential_matrix(E, K)
        
        # Normalize translation vector (essential matrix only gives direction)
        t = t / np.linalg.norm(t) * 0.1  # Scale factor of 0.1 (adjust as needed)
        
        # Update cumulative position and rotation
        cumulative_rotation = R @ cumulative_rotation
        cumulative_position += t
        
        # Convert rotation matrix to Euler angles (in degrees)
        r = Rotation.from_matrix(cumulative_rotation)
        euler_angles = np.degrees(r.as_euler('xyz'))
        
        # Store frame, position and rotation
        movements.append({
            'frame': i,
            'position': cumulative_position.tolist(),
            'rotation_euler': euler_angles.tolist(),
            'rotation_matrix': cumulative_rotation.tolist()
        })
    
    return movements

def analyze_cameraman_movement(movements):
    """
    Analyze cameraman movement patterns and calculate statistics
    """
    if not movements:
        return {}
    
    frames = [m['frame'] for m in movements]
    positions = np.array([m['position'] for m in movements])
    rotations = np.array([m['rotation_euler'] for m in movements])
    
    # Calculate total displacement
    if len(positions) > 1:
        total_displacement = np.linalg.norm(positions[-1] - positions[0])
    else:
        total_displacement = 0
    
    # Calculate total rotation (sum of absolute changes in each angle)
    total_rotation = np.zeros(3)
    if len(rotations) > 1:
        for i in range(1, len(rotations)):
            total_rotation += np.abs(rotations[i] - rotations[i-1])
    
    # Calculate maximum displacement from origin
    max_displacement = np.max(np.linalg.norm(positions, axis=1))
    
    # Calculate average speed (displacement per frame)
    avg_speed = 0
    if len(positions) > 1:
        displacements = np.zeros(len(positions)-1)
        for i in range(len(positions)-1):
            displacements[i] = np.linalg.norm(positions[i+1] - positions[i])
        avg_speed = np.mean(displacements)
    
    # Calculate average angular speed (degrees per frame)
    avg_angular_speed = np.zeros(3)
    if len(rotations) > 1:
        angular_speeds = np.zeros((len(rotations)-1, 3))
        for i in range(len(rotations)-1):
            angular_speeds[i] = np.abs(rotations[i+1] - rotations[i])
        avg_angular_speed = np.mean(angular_speeds, axis=0)
    
    return {
        'total_displacement': float(total_displacement),
        'total_rotation': total_rotation.tolist(),
        'max_displacement': float(max_displacement),
        'avg_speed': float(avg_speed),
        'avg_angular_speed': avg_angular_speed.tolist()
    }

def generate_cameraman_data(intrinsics_file, essential_matrices_file, output_file):
    """
    Generate JSON file with cameraman movement data
    """
    # Load camera data
    K, essential_matrices = load_camera_data(intrinsics_file, essential_matrices_file)
    if K is None or essential_matrices is None:
        print("Failed to load camera data")
        return False
    
    # Extract movement data
    print(f"Processing {len(essential_matrices)} frames...")
    movements = extract_cameraman_movement(essential_matrices, K)
    
    # Analyze movement
    analysis = analyze_cameraman_movement(movements)
    
    # Create output data
    cameraman_data = {
        'intrinsics': K.tolist(),
        'movements': movements,
        'analysis': analysis
    }
    
    # Save to JSON file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(cameraman_data, f, indent=2)
    
    print(f"Cameraman movement data saved to {output_file}")
    print(f"\nSummary:")
    print(f"Total displacement: {analysis['total_displacement']:.2f} units")
    print(f"Total rotation (x,y,z): ({analysis['total_rotation'][0]:.2f}°, {analysis['total_rotation'][1]:.2f}°, {analysis['total_rotation'][2]:.2f}°)")
    print(f"Max displacement from origin: {analysis['max_displacement']:.2f} units")
    print(f"Average speed: {analysis['avg_speed']:.4f} units/frame")
    print(f"Average angular speed (x,y,z): ({analysis['avg_angular_speed'][0]:.2f}°, {analysis['avg_angular_speed'][1]:.2f}°, {analysis['avg_angular_speed'][2]:.2f}°)/frame")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze cameraman movement and generate JSON data")
    parser.add_argument('--intrinsics', default=r'/home/alfonso/Duck-Tracker-Project/camera_information/intrinsics.npy', 
                        help="Path to camera intrinsics file")
    parser.add_argument('--essential', default=r'/home/alfonso/Duck-Tracker-Project/camera_information/essential_matrices.npy', 
                        help="Path to essential matrices file")
    parser.add_argument('--output', default="batch_output/cameraman_data/cameraman_movement.json", 
                        help="Output JSON file path")
    
    args = parser.parse_args()
    
    generate_cameraman_data(args.intrinsics, args.essential, args.output)