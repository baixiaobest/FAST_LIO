std_msgs/Header header
bool update_valid
float64[3] position_stddev
float64[3] rotation_stddev
uint32 downsampled_points
uint32 matched_points
float64 mean_abs_residual
float64[3] lidar_position_correction
float64 lidar_rotation_correction
# IMU-propagated orientation immediately before the LiDAR update, xyzw.
float64[4] predicted_orientation_xyzw
