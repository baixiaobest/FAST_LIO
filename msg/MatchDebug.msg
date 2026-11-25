# One processed scan. All coordinates use header.frame_id (FAST-LIO world frame).
std_msgs/Header header
bool update_valid
geometry_msgs/Point32[] downsampled_returns
# Each index refers to downsampled_returns. Neighbors are flattened in groups of five.
uint32[] matched_indices
geometry_msgs/Point32[] map_neighbors
float32[] signed_residuals
# One world-frame unit plane normal per matched_indices entry.
geometry_msgs/Vector3[] match_normals
