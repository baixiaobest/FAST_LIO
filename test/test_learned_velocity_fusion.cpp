#include <gtest/gtest.h>
#include "learned_velocity_fusion.hpp"

TEST(LearnedVelocityFusion, JacobianMatchesManifoldFiniteDifference) {
  state_ikfom state;
  state.build_S2_state();
  state.build_SO3_state();
  state.build_vect_state();
  state.pos << 1.0, -2.0, 0.4;
  state.vel << 0.7, -0.3, 0.2;
  state.bg << 0.02, -0.01, 0.03;
  state.rot = SO3(Eigen::AngleAxisd(0.35, Eigen::Vector3d(0.2, 0.4, 0.7).normalized()));
  const Eigen::Vector3d r_IB(0.02557, 0.0, -0.04232);
  const Eigen::Vector3d gyro(0.3, -0.4, 0.2);
  const Eigen::Matrix3d R_BI = Eigen::AngleAxisd(-0.1, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const auto analytic = learned_velocity::jacobian(state, r_IB, R_BI);
  const double eps = 1e-7;
  for (int column : {0, 3, 4, 5, 12, 13, 14, 15, 16, 17}) {
    learned_velocity::Vector23 delta = learned_velocity::Vector23::Zero();
    delta[column] = eps;
    auto plus = state;
    plus.boxplus(delta);
    delta[column] = -eps;
    auto minus = state;
    minus.boxplus(delta);
    const Eigen::Vector3d numeric =
        (learned_velocity::predicted_velocity(plus, gyro, r_IB, R_BI) -
         learned_velocity::predicted_velocity(minus, gyro, r_IB, R_BI)) / (2 * eps);
    EXPECT_LT((numeric - analytic.col(column)).norm(), 1e-6) << column;
  }
}

TEST(LearnedVelocityFusion, CorrectsFullStateAndKeepsCovarianceSymmetric) {
  learned_velocity::Filter filter;
  double limits[23] = {};
  filter.init_dyn_share(get_f, df_dx, df_dw,
      [](state_ikfom &, esekfom::dyn_share_datastruct<double> &) {}, 3, limits);
  auto state = filter.get_x();
  state.vel << 0.2, 0.0, 0.0;
  filter.change_x(state);
  auto P = filter.get_P();
  P.setIdentity();
  P(0, 12) = P(12, 0) = 0.3;
  filter.change_P(P);
  learned_velocity::Observation observation;
  observation.velocity << 0.6, 0.0, 0.0;
  learned_velocity::Result result;
  ASSERT_TRUE(learned_velocity::update(filter, observation, Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity(),
      Eigen::Vector3d::Constant(0.5), result));
  EXPECT_GT(filter.get_x().vel.x(), 0.2);
  EXPECT_GT(filter.get_x().pos.x(), 0.0);
  EXPECT_LT((filter.get_P() - filter.get_P().transpose()).norm(), 1e-10);
  EXPECT_GT(filter.get_P().diagonal().minCoeff(), 0.0);
  EXPECT_GT(result.nis, 0.0);
}
