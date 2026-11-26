#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#undef ROOT_DIR
#define ROOT_DIR "/tmp/"
#include "IMU_Processing.hpp"

namespace {

sensor_msgs::msg::Imu::ConstSharedPtr imu_at(double seconds) {
  auto imu = std::make_shared<sensor_msgs::msg::Imu>();
  imu->header.stamp = get_ros_time(seconds);
  imu->linear_acceleration.z = 9.81;
  return imu;
}

MeasureGroup scan(double begin, double end, const std::vector<double> &imu_times) {
  MeasureGroup group;
  group.lidar_beg_time = begin;
  group.lidar_end_time = end;
  for (double time : imu_times) group.imu.push_back(imu_at(time));
  PointType first, middle, last;
  first.x = middle.x = last.x = 2.0;
  first.y = middle.y = last.y = 0.0;
  first.z = middle.z = last.z = 0.0;
  first.curvature = 0.0;
  middle.curvature = 500.0 * (end - begin);
  last.curvature = 1000.0 * (end - begin);
  group.lidar->push_back(first);
  group.lidar->push_back(middle);
  group.lidar->push_back(last);
  return group;
}

}  // namespace

TEST(LearnedVelocityTiming, SplitsImuIntervalAndRetainsFutureSample) {
  learned_velocity::Filter filter;
  double limits[23] = {};
  filter.init_dyn_share(get_f, df_dx, df_dw,
      [](state_ikfom &, esekfom::dyn_share_datastruct<double> &) {}, 3, limits);
  ImuProcess process;
  process.set_acc_cov(V3D::Constant(0.1));
  process.set_gyr_cov(V3D::Constant(0.1));
  process.set_acc_bias_cov(V3D::Constant(0.001));
  process.set_gyr_bias_cov(V3D::Constant(0.001));
  std::deque<learned_velocity::Observation> queue;
  std::vector<double> accepted;
  process.configure_learned_velocity(V3D::Zero(), M3D::Identity(), V3D::Constant(0.1),
      &queue, [&](const learned_velocity::Result &result, uint8_t status) {
        if (status == 0) accepted.push_back(result.observation.time);
      });
  PointCloudXYZI::Ptr output(new PointCloudXYZI());
  const auto initialization = scan(1.0, 1.1,
      {1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09});
  process.Process(initialization, filter, output);
  learned_velocity::Observation between, future;
  between.time = between.original_time = 1.125;
  between.velocity << 0.4, 0.0, 0.0;
  future.time = future.original_time = 1.5;
  future.velocity << 0.6, 0.0, 0.0;
  queue.push_back(between);
  queue.push_back(future);
  const auto next = scan(1.1, 1.16, {1.11, 1.12, 1.13, 1.14, 1.15});
  process.Process(next, filter, output);
  ASSERT_EQ(accepted.size(), 1u);
  EXPECT_NEAR(accepted.front(), 1.125, 1e-12);
  ASSERT_EQ(queue.size(), 1u);
  EXPECT_NEAR(queue.front().time, 1.5, 1e-12);
  EXPECT_GT(filter.get_x().vel.x(), 0.0);
  EXPECT_NEAR(process.last_processed_time(), 1.16, 1e-12);
  EXPECT_EQ(output->size(), 3u);

  // Deskew must use the corrected intermediate path, including the velocity
  // update between the first and last returns of this synthetic scan.
  learned_velocity::Filter baseline_filter;
  baseline_filter.init_dyn_share(get_f, df_dx, df_dw,
      [](state_ikfom &, esekfom::dyn_share_datastruct<double> &) {}, 3, limits);
  ImuProcess baseline_process;
  baseline_process.set_acc_cov(V3D::Constant(0.1));
  baseline_process.set_gyr_cov(V3D::Constant(0.1));
  baseline_process.set_acc_bias_cov(V3D::Constant(0.001));
  baseline_process.set_gyr_bias_cov(V3D::Constant(0.001));
  PointCloudXYZI::Ptr baseline_output(new PointCloudXYZI());
  baseline_process.Process(initialization, baseline_filter, baseline_output);
  baseline_process.Process(next, baseline_filter, baseline_output);
  ASSERT_EQ(baseline_output->size(), output->size());
  EXPECT_GT(std::abs(output->points[0].x - baseline_output->points[0].x), 1e-4);
  EXPECT_GT(std::abs(output->points[1].x - baseline_output->points[1].x), 1e-4);
}
