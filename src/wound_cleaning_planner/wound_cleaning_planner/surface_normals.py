import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import svd

class SurfaceNormalEstimator:
    def __init__(self):
        pass

    def calculate_surface_normals(self, points_3d, radius=0.02, min_points=10):

        if len(points_3d) < min_points:
            print(f"Insufficient points in point cloud ({len(points_3d)} < {min_points}) for normal estimation")
            return np.zeros((len(points_3d), 3))
        
        point_cloud = np.array(points_3d)
        tree = KDTree(point_cloud)
        
        normals = []
        previous_normal = None
        
        # Reference direction
        reference_direction = np.array([0, 0, 1])
        
        for i, point in enumerate(points_3d):
            # Find neighboring points
            indices = tree.query_ball_point(point, radius)
            
            if len(indices) < min_points:
                _, indices = tree.query(point, k=min_points)
            
            if len(indices) < min_points:
                # Use previous normal or default normal vector
                if previous_normal is not None:
                    normals.append(previous_normal.copy())
                else:
                    normals.append(reference_direction.copy())
                continue
            
            neighbors = point_cloud[indices]
            
            # Calculate centroid and covariance matrix
            centroid = np.mean(neighbors, axis=0)
            centered_neighbors = neighbors - centroid
            cov_matrix = np.cov(centered_neighbors.T)
            
            # PCA calculation
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Get candidate normal vector (eigenvector corresponding to minimum eigenvalue)
            candidate_normal = eigenvectors[:, np.argmin(eigenvalues)]
            
            # Direction consistency processing
            if i == 0:
                if np.dot(candidate_normal, reference_direction) < 0:
                    candidate_normal = -candidate_normal
            else:
                if previous_normal is not None:
                    if np.dot(candidate_normal, previous_normal) < 0:
                        candidate_normal = -candidate_normal
                
                if np.dot(candidate_normal, reference_direction) < -0.5:
                    candidate_normal = -candidate_normal
            
            # Smoothing processing
            if previous_normal is not None and i > 0:
                smoothing_weight = 0.3
                smoothed_normal = (1 - smoothing_weight) * candidate_normal + smoothing_weight * previous_normal
                smoothed_normal = smoothed_normal / (np.linalg.norm(smoothed_normal) + 1e-6)
                normals.append(smoothed_normal)
                previous_normal = smoothed_normal
            else:
                normals.append(candidate_normal)
                previous_normal = candidate_normal
        
        return np.array(normals)

    def unify_normal_directions(self, normals):
       
        unified_normals = []
        reference_direction = np.array([0, 0, 1])
        
        for normal in normals:
            if np.dot(normal, reference_direction) < 0:
                normal = -normal
            unified_normals.append(normal)
        
        return np.array(unified_normals)

    def generate_consistent_orientations(self, positions, normals):
        
        orientations = []
        prev_x_axis = None
        
        for i, (position, normal) in enumerate(zip(positions, normals)):
            # Normalized normal vector as z-axis
            z_axis = normal / (np.linalg.norm(normal) + 1e-6)
            
            if i == 0:
                # Initial point: use direction from start to end
                if len(positions) > 1:
                    initial_direction = positions[-1] - positions[0]
                    initial_direction = initial_direction / (np.linalg.norm(initial_direction) + 1e-6)
                else:
                    initial_direction = np.array([1, 0, 0])
                
                # Project initial direction onto plane perpendicular to normal vector
                x_axis = initial_direction - np.dot(initial_direction, z_axis) * z_axis
                x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)
                
            else:
                # Subsequent points: prioritize continuity with previous point
                if prev_x_axis is not None:
                    # Project previous x-axis to vertical plane of current normal vector
                    x_axis = prev_x_axis - np.dot(prev_x_axis, z_axis) * z_axis
                    x_axis_norm = np.linalg.norm(x_axis)
                    
                    if x_axis_norm > 0.1:
                        x_axis = x_axis / x_axis_norm
                    else:
                        # Use path direction
                        if i < len(positions) - 1:
                            path_direction = positions[i+1] - positions[i-1]
                        else:
                            path_direction = positions[i] - positions[i-1]
                        
                        path_direction = path_direction / (np.linalg.norm(path_direction) + 1e-6)
                        x_axis = path_direction - np.dot(path_direction, z_axis) * z_axis
                        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)
                else:
                    # Use path direction
                    if i < len(positions) - 1:
                        path_direction = positions[i+1] - positions[i]
                    else:
                        path_direction = positions[i] - positions[i-1]
                    
                    path_direction = path_direction / (np.linalg.norm(path_direction) + 1e-6)
                    x_axis = path_direction - np.dot(path_direction, z_axis) * z_axis
                    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)
            
            # Calculate y-axis and re-orthogonalize x-axis
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-6)
            
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)
            
            R = np.column_stack([x_axis, y_axis, z_axis])
            
            # Check rotation matrix validity
            if np.abs(np.linalg.det(R) - 1.0) > 0.01:
                print(f"Warning: Invalid rotation matrix at point {i}, det = {np.linalg.det(R)}")
                R = np.eye(3)
            
            orientations.append(R)
            prev_x_axis = x_axis
        
        return orientations

    def smooth_orientations(self, orientations, smoothing_factor=0.3):
   
        if len(orientations) <= 2:
            return orientations
        
        smoothed = [orientations[0]]  # First point remains unchanged
        
        for i in range(1, len(orientations)-1):
            current = orientations[i]
            prev_smooth = smoothed[i-1]
            
            # Use matrix weighted average then re-orthogonalize
            blended = (1-smoothing_factor) * current + smoothing_factor * prev_smooth
            
            # Re-orthogonalization
            U, _, Vt = np.linalg.svd(blended)
            smoothed_rotation = U @ Vt
            
            # Ensure determinant is 1 (right-handed coordinate system)
            if np.linalg.det(smoothed_rotation) < 0:
                U[:, -1] *= -1
                smoothed_rotation = U @ Vt
            
            smoothed.append(smoothed_rotation)
        
        smoothed.append(orientations[-1])  # Last point remains unchanged
        return smoothed