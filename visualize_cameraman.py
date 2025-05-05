import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import argparse

def load_cameraman_data(json_file):
    """
    Load cameraman movement data from JSON file
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading cameraman data: {e}")
        return None

def plot_3d_trajectory(movements, output_folder):
    """
    Create a 3D visualization of the cameraman's trajectory
    """
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare data
    positions = np.array([m['position'] for m in movements])
    frames = np.array([m['frame'] for m in movements])
    
    # Create a colormap based on frame number
    norm = plt.Normalize(frames.min(), frames.max())
    cmap = plt.cm.plasma
    colors = cmap(norm(frames))
    
    # Create the trajectory plot
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            color='blue', linewidth=2, alpha=0.7)
    
    # Plot each point with color based on frame number
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                       c=frames, cmap=cmap, s=50, alpha=0.8)
    
    # Mark start and end points
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
              color='green', marker='o', s=150, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
              color='red', marker='*', s=200, label='End')
    
    # Plot arrows to show the camera's orientation at key points
    rotations = [np.array(m['rotation_matrix']) for m in movements]
    num_points = len(positions)
    arrow_points = list(range(0, num_points, max(1, num_points // 10)))  # Show ~10 arrows
    arrow_length = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)) * 0.05
    
    for i in arrow_points:
        pos = positions[i]
        rot = np.array(rotations[i])
        
        # Create arrows in the direction of the camera's axes
        ax.quiver(pos[0], pos[1], pos[2],
                 rot[0, 0] * arrow_length, rot[0, 1] * arrow_length, rot[0, 2] * arrow_length,
                 color='red', alpha=0.7)
        ax.quiver(pos[0], pos[1], pos[2],
                 rot[1, 0] * arrow_length, rot[1, 1] * arrow_length, rot[1, 2] * arrow_length,
                 color='green', alpha=0.7)
        ax.quiver(pos[0], pos[1], pos[2],
                 rot[2, 0] * arrow_length, rot[2, 1] * arrow_length, rot[2, 2] * arrow_length,
                 color='blue', alpha=0.7)
    
    # Add colorbar to show frame progression
    cbar = plt.colorbar(scatter)
    cbar.set_label('Frame Number')
    
    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Cameraman 3D Trajectory')
    
    # Set equal aspect ratio
    max_range = np.array([
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.legend()
    
    # Save the figure
    plt.tight_layout()
    output_path = os.path.join(output_folder, 'cameraman_3d_trajectory.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"3D trajectory plot saved to: {output_path}")
    plt.close()

def plot_2d_trajectories(movements, output_folder):
    """
    Create 2D visualizations of cameraman's movement in different planes
    """
    positions = np.array([m['position'] for m in movements])
    frames = np.array([m['frame'] for m in movements])
    rotations = np.array([m['rotation_euler'] for m in movements])
    
    # Create a colormap for frame progression
    norm = plt.Normalize(frames.min(), frames.max())
    cmap = plt.cm.plasma
    
    # === XY Plane (Top View) ===
    plt.figure(figsize=(14, 10))
    plt.scatter(positions[:, 0], positions[:, 1], c=frames, cmap=cmap, s=50, alpha=0.8)
    plt.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.7)
    plt.scatter(positions[0, 0], positions[0, 1], color='green', marker='o', s=150, label='Start')
    plt.scatter(positions[-1, 0], positions[-1, 1], color='red', marker='*', s=200, label='End')
    
    plt.colorbar(label='Frame Number')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Cameraman Movement (Top View)')
    plt.grid(alpha=0.3)
    plt.legend()
    
    output_path = os.path.join(output_folder, 'cameraman_xy_trajectory.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"XY trajectory plot saved to: {output_path}")
    plt.close()
    
    # === XZ Plane (Side View) ===
    plt.figure(figsize=(14, 10))
    plt.scatter(positions[:, 0], positions[:, 2], c=frames, cmap=cmap, s=50, alpha=0.8)
    plt.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2, alpha=0.7)
    plt.scatter(positions[0, 0], positions[0, 2], color='green', marker='o', s=150, label='Start')
    plt.scatter(positions[-1, 0], positions[-1, 2], color='red', marker='*', s=200, label='End')
    
    plt.colorbar(label='Frame Number')
    plt.xlabel('X Position')
    plt.ylabel('Z Position')
    plt.title('Cameraman Movement (Side View)')
    plt.grid(alpha=0.3)
    plt.legend()
    
    output_path = os.path.join(output_folder, 'cameraman_xz_trajectory.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"XZ trajectory plot saved to: {output_path}")
    plt.close()
    
    # === Rotation Over Time (Euler Angles) ===
    plt.figure(figsize=(14, 10))
    plt.plot(frames, rotations[:, 0], 'r-', linewidth=2, label='X Rotation (Roll)')
    plt.plot(frames, rotations[:, 1], 'g-', linewidth=2, label='Y Rotation (Pitch)')
    plt.plot(frames, rotations[:, 2], 'b-', linewidth=2, label='Z Rotation (Yaw)')
    
    plt.xlabel('Frame Number')
    plt.ylabel('Rotation (degrees)')
    plt.title('Cameraman Rotation Over Time')
    plt.grid(alpha=0.3)
    plt.legend()
    
    output_path = os.path.join(output_folder, 'cameraman_rotation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Rotation plot saved to: {output_path}")
    plt.close()
    
    # === Displacement Over Time ===
    distances = np.zeros(len(positions))
    for i in range(1, len(positions)):
        distances[i] = np.linalg.norm(positions[i] - positions[0])
    
    plt.figure(figsize=(14, 10))
    plt.plot(frames, distances, 'b-', linewidth=2)
    plt.scatter(frames, distances, c=frames, cmap=cmap, s=50, alpha=0.8)
    
    plt.xlabel('Frame Number')
    plt.ylabel('Distance from Starting Position')
    plt.title('Cameraman Displacement Over Time')
    plt.grid(alpha=0.3)
    plt.colorbar(label='Frame Number')
    
    output_path = os.path.join(output_folder, 'cameraman_displacement.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Displacement plot saved to: {output_path}")
    plt.close()

def create_camera_movement_animation(movements, output_folder):
    """
    Create an animation of the camera movement in 3D
    """
    positions = np.array([m['position'] for m in movements])
    rotations = [np.array(m['rotation_matrix']) for m in movements]
    
    # Set up the figure and axes
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set initial view
    ax.view_init(elev=30, azim=45)
    
    # Calculate the bounds for consistent axis limits
    max_range = np.array([
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set labels
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Cameraman Movement Animation')
    
    # Calculate arrow scale based on plot size
    arrow_length = max_range * 0.2
    
    # Initialize plot elements
    trajectory_line, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.7)
    camera_point = ax.scatter([], [], [], color='red', s=100)
    
    # Arrows for camera orientation
    arrow_x = ax.quiver([], [], [], [], [], [], color='red', alpha=0.7)
    arrow_y = ax.quiver([], [], [], [], [], [], color='green', alpha=0.7)
    arrow_z = ax.quiver([], [], [], [], [], [], color='blue', alpha=0.7)
    
    # Frame text
    frame_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes)
    
    def init():
        trajectory_line.set_data([], [])
        trajectory_line.set_3d_properties([])
        camera_point._offsets3d = ([], [], [])
        frame_text.set_text('')
        
        # Return all artists
        return trajectory_line, camera_point, arrow_x, arrow_y, arrow_z, frame_text
    
    def animate(i):
        # Update trajectory line
        trajectory_line.set_data(positions[:i+1, 0], positions[:i+1, 1])
        trajectory_line.set_3d_properties(positions[:i+1, 2])
        
        # Update camera position
        camera_point._offsets3d = ([positions[i, 0]], [positions[i, 1]], [positions[i, 2]])
        
        # Update camera orientation arrows
        rot = rotations[i]
        ax.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                 rot[0, 0] * arrow_length, rot[0, 1] * arrow_length, rot[0, 2] * arrow_length,
                 color='red', alpha=0.7)
        ax.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                 rot[1, 0] * arrow_length, rot[1, 1] * arrow_length, rot[1, 2] * arrow_length,
                 color='green', alpha=0.7)
        ax.quiver(positions[i, 0], positions[i, 1], positions[i, 2],
                 rot[2, 0] * arrow_length, rot[2, 1] * arrow_length, rot[2, 2] * arrow_length,
                 color='blue', alpha=0.7)
        
        # Update frame text
        frame_text.set_text(f'Frame: {i}')
        
        return trajectory_line, camera_point, arrow_x, arrow_y, arrow_z, frame_text
    
    # Create animation (using a reduced number of frames for efficiency)
    frames = min(len(movements), 100)
    step = max(1, len(movements) // frames)
    selected_frames = range(0, len(movements), step)
    
    anim = animation.FuncAnimation(fig, animate, frames=selected_frames,
                                  init_func=init, blit=False, repeat=True)
    
    # Save animation
    output_path = os.path.join(output_folder, 'cameraman_animation.mp4')
    anim.save(output_path, writer='ffmpeg', fps=10, dpi=200)
    print(f"Animation saved to: {output_path}")
    plt.close()

def visualize_cameraman_data(json_file, output_folder):
    """
    Main function to visualize cameraman movement data
    """
    # Load data
    data = load_cameraman_data(json_file)
    if data is None:
        print("Failed to load cameraman data")
        return False
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract movement data
    movements = data['movements']
    analysis = data['analysis']
    
    print(f"Loaded data for {len(movements)} frames")
    print(f"\nMovement Analysis:")
    print(f"Total displacement: {analysis['total_displacement']:.2f} units")
    print(f"Total rotation (x,y,z): ({analysis['total_rotation'][0]:.2f}°, {analysis['total_rotation'][1]:.2f}°, {analysis['total_rotation'][2]:.2f}°)")
    print(f"Average speed: {analysis['avg_speed']:.4f} units/frame")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_3d_trajectory(movements, output_folder)
    plot_2d_trajectories(movements, output_folder)
    
    # Create animation if ffmpeg is available
    try:
        create_camera_movement_animation(movements, output_folder)
    except Exception as e:
        print(f"Could not create animation: {e}")
        print("Make sure ffmpeg is installed for animations.")
    
    print("\nVisualization complete!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize cameraman movement data")
    parser.add_argument('--input', default="batch_output/cameraman_data/cameraman_movement.json", 
                        help="Input JSON file path")
    parser.add_argument('--output', default="batch_output/cameraman_visualizations", 
                        help="Output folder for visualizations")
    
    args = parser.parse_args()
    
    visualize_cameraman_data(args.input, args.output)