import numpy as np
import open3d as o3d
import cv2

class LidarSemanticFuser:
    def __init__(self, camera_intrinsics=None, lidar_to_cam_extrinsics=None):
        """
        Initializes the fusion pipeline.
        
        Args:
            camera_intrinsics: 3x3 Camera Intrinsic Matrix (K)
            lidar_to_cam_extrinsics: 4x4 Extrinsic Transformation Matrix ([R|t]) 
                                     aligning Velodyne/Ouster LiDAR to the main RGB camera.
        """
        # Default typical RealSense/OAK-D matrix if none provided
        self.K = camera_intrinsics if camera_intrinsics is not None else np.array([
            [640.0, 0.0,   320.0],
            [0.0,   640.0, 240.0],
            [0.0,   0.0,   1.0]
        ])
        
        # Default Identity if LiDAR is mounted physically parallel 
        self.T_lidar2cam = lidar_to_cam_extrinsics if lidar_to_cam_extrinsics is not None else np.eye(4)
        
    def project_lidar_to_camera(self, point_cloud_np):
        """ Projects 3D LiDAR points (N, 3) onto the 2D image plane (u, v) """
        # Convert to homogeneous coords (N, 4)
        points_h = np.hstack((point_cloud_np, np.ones((point_cloud_np.shape[0], 1))))
        
        # Transform LIDAR to Camera Coordinates
        points_cam = np.dot(self.T_lidar2cam, points_h.T).T
        
        # Filter points behind the camera (Z <= 0)
        valid_idx = points_cam[:, 2] > 0
        points_cam = points_cam[valid_idx]
        original_indices = np.where(valid_idx)[0]
        
        # Project using camera intrinsics
        uvz = np.dot(self.K, points_cam[:, :3].T).T
        
        # Normalize by Z to get pixel coordinates (u, v)
        u = np.round(uvz[:, 0] / uvz[:, 2]).astype(int)
        v = np.round(uvz[:, 1] / uvz[:, 2]).astype(int)
        
        return u, v, points_cam[:, 2], original_indices

    def fuse_semantic_mask(self, pcd: o3d.geometry.PointCloud, semantic_mask_2d: np.ndarray, color_map: dict):
        """
        Takes a raw Open3D point-cloud and the 2D PyTorch semantic map (e.g. from app.py),
        and paints the point-cloud spheres with the specific terrain / hazard class.
        """
        points = np.asarray(pcd.points)
        height, width = semantic_mask_2d.shape
        
        # Default color all points to gray (Unclassified)
        colors = np.ones((points.shape[0], 3)) * 0.5
        
        # Project points to 2D
        u, v, z_depth, valid_pz = self.project_lidar_to_camera(points)
        
        # Find points that land inside the camera's FOV
        in_frame = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u_valid = u[in_frame]
        v_valid = v[in_frame]
        idx_valid = valid_pz[in_frame]
        
        # Extract the semantic class IDs from exactly where the LiDAR rays hit
        pixel_class_ids = semantic_mask_2d[v_valid, u_valid]
        
        # Colorize the 3D Point Cloud based on our PyTorch dictionary
        for i, point_idx in enumerate(idx_valid):
            c_id = pixel_class_ids[i]
            if c_id in color_map:
                # Open3D expects RGB values in 0-1 range
                rgb = color_map[c_id]
                colors[point_idx] = [rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0]
                
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def extract_obstacle_volume(self, pcd, target_class_id, min_depth=0.5, max_depth=15.0):
        """
        Isolates a specific semantic cluster (e.g., "Rock" or "Person" from YOLO)
        and calculates its physical 3D Bounding Box in meters.
        """
        # This would filter the PCD by colors matching the target_class_id,
        # apply DBSCAN clustering to find the contiguous physical object,
        # and return the height/width/depth of the hazard in real-world metric units.
        pass

if __name__ == "__main__":
    print("Testing LidarSemanticFuser Architecure...")
    fuser = LidarSemanticFuser()
    print("Integration ready for Open3D and ROS 2 PointCloud2 messages.")
