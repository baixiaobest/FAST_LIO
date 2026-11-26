# One learned velocity observation, including discarded observations.
# header.stamp is the original LowState observation timestamp.
std_msgs/Header header
builtin_interfaces/Time arrival_stamp
builtin_interfaces/Time processing_stamp
# Observation timestamp after FAST-LIO's LiDAR/IMU time offset.
builtin_interfaces/Time fusion_stamp
# 0 accepted, 1 late, 2 invalid or duplicate, 3 numerical failure.
uint8 status
float64[3] learned_velocity
float64[3] predicted_velocity
float64[3] residual
float64[3] velocity_before
float64[3] velocity_after
float64 nis
# State order: position, rotation, LiDAR extrinsics, velocity, gyro bias,
# accelerometer bias, gravity. Gain is row-major [23,3].
float64[23] state_correction
float64[69] kalman_gain
float64[23] covariance_diagonal
float64[3] gyro_bias
float64[3] accel_bias
