import scipy.io
import gtsam
import numpy as np
from gtsam.utils.plot import plot_trajectory, plot_pose3, set_axes_equal, plot_pose3_on_axes
import matplotlib.pyplot as plt

# Step 1: Load the data from hw4_data.mat
mat = scipy.io.loadmat('hw4_data.mat')
dposes = mat['dpose'][0]  # Noisy odometry measurements (1*49 cell, each cell is 4x4 double)
traj3 = mat['traj3'][0]  # Initial trajectory estimate (1*50 cell, each cell is 4x4 double)
poses3_gt = mat['poses3_gt'][0]  # Ground truth poses (1*50 cell, each cell is 4x4 double)
print(traj3[0].dtype)  # Should show float64

# Step 2: Convert transformation matrices to gtsam.Pose3 objects
initial_trajectory = gtsam.Values()
ground_truth_trajectory = gtsam.Values()

# Helper function to convert 4x4 matrix to gtsam.Pose3
def mat_to_pose3(matrix):
    R = gtsam.Rot3(matrix[:3, :3])  # Rotation part
    t = gtsam.Point3(matrix[:3, 3])  # Translation part
    return gtsam.Pose3(R, t)

# Convert initial trajectory (traj3)
for i, T in enumerate(traj3):
    pose = mat_to_pose3(T)
    initial_trajectory.insert(gtsam.symbol('x', i), pose)

# Convert ground truth trajectory (poses3_gt)
for i, T in enumerate(poses3_gt):
    pose = mat_to_pose3(T)
    ground_truth_trajectory.insert(gtsam.symbol('x', i), pose)



# Step 3: Plot the initial trajectory and the ground truth trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the initial trajectory
for key in initial_trajectory.keys():
    pose = initial_trajectory.atPose3(key)
    plot_pose3_on_axes(axes=ax, pose=pose, axis_length=0.5)

# Plot the ground truth trajectory
for key in ground_truth_trajectory.keys():
    pose = ground_truth_trajectory.atPose3(key)
    plot_pose3_on_axes(axes=ax, pose=pose, axis_length=0.5)

# set_axes_equal(fignum=0)
ax.set_title("Initial vs Ground Truth")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# PLOT INITIAL TRAJECTORY
xs_init, ys_init, zs_init = [], [], []
for key in initial_trajectory.keys():
    pose = initial_trajectory.atPose3(key)
    # Draw the coordinate frame for each pose
    plot_pose3_on_axes(ax, pose=pose, axis_length=0.5)
    # Save the translation for line-plotting
    t = pose.translation()
    xs_init.append(t[0])
    ys_init.append(t[1])
    zs_init.append(t[2])

# Connect initial trajectory poses with a blue line
ax.plot(xs_init, ys_init, zs_init, color='blue', label='Initial Trajectory')

# PLOT GROUND TRUTH TRAJECTORY
xs_gt, ys_gt, zs_gt = [], [], []
for key in ground_truth_trajectory.keys():
    pose = ground_truth_trajectory.atPose3(key)
    # Draw the coordinate frame for each pose
    plot_pose3_on_axes(ax, pose=pose, axis_length=0.5)
    # Save the translation for line-plotting
    t = pose.translation()
    xs_gt.append(t[0])
    ys_gt.append(t[1])
    zs_gt.append(t[2])

# Connect ground truth poses with a red line
ax.plot(xs_gt, ys_gt, zs_gt, color='red', label='Ground Truth Trajectory')

ax.set_title("Initial vs Ground Truth")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()