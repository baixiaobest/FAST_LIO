# LiDAR measurement information at the final valid filter linearization.
# Upper triangle, row-major, of H_pose^T H_pose / point_variance.
# Columns: world position xyz, then IMU-frame rotation xyz.
std_msgs/Header header
bool update_valid
uint32 matched_points
float64[21] upper_triangle
