import numpy as np

class Observation(object):
    """Storage for both visual and low-dimensional observations."""

    def __init__(self,
                 left_shoulder_rgb: np.ndarray,
                 left_shoulder_depth: np.ndarray,
                 left_shoulder_mask: np.ndarray,
                 left_shoulder_intrinsic_matrix: np.ndarray,
                 left_shoulder_extrinsic_matrix: np.ndarray,
                 right_shoulder_rgb: np.ndarray,
                 right_shoulder_depth: np.ndarray,
                 right_shoulder_mask: np.ndarray,
                 right_shoulder_intrinsic_matrix: np.ndarray,
                 right_shoulder_extrinsic_matrix: np.ndarray,
                 wrist_rgb: np.ndarray,
                 wrist_depth: np.ndarray,
                 wrist_mask: np.ndarray,
                 wrist_intrinsic_matrix: np.ndarray,
                 wrist_extrinsic_matrix: np.ndarray,
                 front_rgb: np.ndarray,
                 front_depth: np.ndarray,
                 front_mask: np.ndarray,
                 front_intrinsic_matrix: np.ndarray,
                 front_extrinsic_matrix: np.ndarray,
                 joint_velocities: np.ndarray,
                 joint_positions: np.ndarray,
                 joint_forces: np.ndarray,
                 gripper_open: float,
                 gripper_pose: np.ndarray,
                 gripper_matrix: np.ndarray,
                 gripper_joint_positions: np.ndarray,
                 gripper_touch_forces: np.ndarray,
                 wrist_camera_matrix: np.ndarray,
                 task_low_dim_state: np.ndarray,
                 joints: list):
        self.left_shoulder_rgb = left_shoulder_rgb
        self.left_shoulder_depth = left_shoulder_depth
        self.left_shoulder_mask = left_shoulder_mask
        self.left_shoulder_intrinsic_matrix = left_shoulder_intrinsic_matrix
        self.left_shoulder_extrinsic_matrix = left_shoulder_extrinsic_matrix
        self.right_shoulder_rgb = right_shoulder_rgb
        self.right_shoulder_depth = right_shoulder_depth
        self.right_shoulder_mask = right_shoulder_mask
        self.right_shoulder_intrinsic_matrix = right_shoulder_intrinsic_matrix
        self.right_shoulder_extrinsic_matrix = right_shoulder_extrinsic_matrix
        self.wrist_rgb = wrist_rgb
        self.wrist_depth = wrist_depth
        self.wrist_mask = wrist_mask
        self.wrist_intrinsic_matrix = wrist_intrinsic_matrix
        self.wrist_extrinsic_matrix = wrist_extrinsic_matrix
        self.front_rgb = front_rgb
        self.front_depth = front_depth
        self.front_mask = front_mask
        self.front_intrinsic_matrix = front_intrinsic_matrix
        self.front_extrinsic_matrix = front_extrinsic_matrix
        self.joint_velocities = joint_velocities
        self.joint_positions = joint_positions
        self.joint_forces = joint_forces
        self.gripper_open = gripper_open
        self.gripper_pose = gripper_pose
        self.gripper_matrix = gripper_matrix
        self.gripper_joint_positions = gripper_joint_positions
        self.gripper_touch_forces = gripper_touch_forces
        self.wrist_camera_matrix = wrist_camera_matrix
        self.task_low_dim_state = task_low_dim_state
        self.joints = joints

    def get_low_dim_data(self) -> np.ndarray:
        """Gets a 1D array of all the low-dimensional obseervations.

        :return: 1D array of observations.
        """
        low_dim_data = [] if self.gripper_open is None else [[self.gripper_open]]
        for data in [self.joint_velocities, self.joint_positions,
                     self.joint_forces,
                     self.gripper_pose, self.gripper_joint_positions,
                     self.gripper_touch_forces, self.task_low_dim_state]:
            if data is not None:
                low_dim_data.append(data)
        return np.concatenate(low_dim_data) if len(low_dim_data) > 0 else np.array([])
