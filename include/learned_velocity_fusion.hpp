#pragma once

#include <Eigen/Dense>
#include <cmath>
#include "use-ikfom.hpp"

namespace learned_velocity {

using Filter = esekfom::esekf<state_ikfom, 12, input_ikfom>;
using Vector23 = Eigen::Matrix<double, 23, 1>;
using Matrix23 = Eigen::Matrix<double, 23, 23>;
using Jacobian = Eigen::Matrix<double, 3, 23>;
using Gain = Eigen::Matrix<double, 23, 3>;

struct Observation {
  double time = 0.0;             // IMU time shifted into the LiDAR clock.
  double original_time = 0.0;    // Header stamp in the bridge's IMU clock.
  double arrival_time = 0.0;
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
};

struct Result {
  Observation observation;
  bool accepted = false;
  Eigen::Vector3d predicted = Eigen::Vector3d::Zero();
  Eigen::Vector3d residual = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity_before = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity_after = Eigen::Vector3d::Zero();
  Vector23 correction = Vector23::Zero();
  Gain gain = Gain::Zero();
  Vector23 covariance_diagonal = Vector23::Zero();
  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
  double nis = 0.0;
};

inline Eigen::Matrix3d hat(const Eigen::Vector3d &v) {
  Eigen::Matrix3d m;
  m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return m;
}

// R_BI maps an IMU-frame vector into the learned measurement frame. r_IB is
// the vector from the IMU origin to the learned-velocity origin in IMU axes.
inline Eigen::Vector3d predicted_velocity(const state_ikfom &x,
    const Eigen::Vector3d &gyro, const Eigen::Vector3d &r_IB,
    const Eigen::Matrix3d &R_BI) {
  const Eigen::Vector3d body_velocity = x.rot.toRotationMatrix().transpose() * x.vel;
  return R_BI * (body_velocity + (gyro - x.bg).cross(r_IB));
}

inline Jacobian jacobian(const state_ikfom &x, const Eigen::Vector3d &r_IB,
    const Eigen::Matrix3d &R_BI) {
  Jacobian H = Jacobian::Zero();
  const Eigen::Matrix3d R_IW = x.rot.toRotationMatrix().transpose();
  H.block<3, 3>(0, 3) = R_BI * hat(R_IW * x.vel);
  H.block<3, 3>(0, 12) = R_BI * R_IW;
  H.block<3, 3>(0, 15) = R_BI * hat(r_IB);
  return H;
}

inline bool update(Filter &filter, const Observation &observation,
    const Eigen::Vector3d &gyro, const Eigen::Vector3d &r_IB,
    const Eigen::Matrix3d &R_BI, const Eigen::Vector3d &sigma,
    Result &result) {
  result.observation = observation;
  state_ikfom before = filter.get_x();
  result.velocity_before = before.rot.toRotationMatrix().transpose() * before.vel;
  result.predicted = predicted_velocity(before, gyro, r_IB, R_BI);
  result.residual = observation.velocity - result.predicted;
  const Jacobian H = jacobian(before, r_IB, R_BI);
  const Matrix23 P = filter.get_P();
  const Eigen::Matrix3d R = sigma.array().square().matrix().asDiagonal();
  const Eigen::Matrix3d S = H * P * H.transpose() + R;
  Eigen::LDLT<Eigen::Matrix3d> solver(S);
  if (solver.info() != Eigen::Success || (solver.vectorD().array() <= 0).any()) return false;
  const Gain K = solver.solve((P * H.transpose()).transpose()).transpose();
  const Vector23 dx = K * result.residual;
  if (!K.allFinite() || !dx.allFinite()) return false;
  const Eigen::Vector3d normalized_residual = solver.solve(result.residual);
  result.nis = result.residual.dot(normalized_residual);
  state_ikfom after = before;
  after.boxplus(dx);
  const Matrix23 I_KH = Matrix23::Identity() - K * H;
  Matrix23 updated_P = I_KH * P * I_KH.transpose() + K * R * K.transpose();

  // Move covariance from the old tangent space into the tangent space at the
  // corrected IKFoM state, using the same retraction Jacobians as its IEKF.
  Matrix23 transport = Matrix23::Identity();
  for (const auto &entry : after.SO3_state) {
    const int index = entry.first;
    vect3 segment;
    segment = dx.segment<3>(index);
    transport.block<3, 3>(index, index) = MTK::A_matrix(segment).transpose();
  }
  for (const auto &entry : after.S2_state) {
    const int index = entry.first;
    Eigen::Matrix<double, 2, 3> Nx;
    Eigen::Matrix<double, 3, 2> Mx;
    after.S2_Nx_yy(Nx, index);
    before.S2_Mx(Mx, dx.segment<2>(index), index);
    transport.block<2, 2>(index, index) = Nx * Mx;
  }
  updated_P = transport * updated_P * transport.transpose();
  updated_P = 0.5 * (updated_P + updated_P.transpose()).eval();
  if (!updated_P.allFinite()) return false;
  filter.change_x(after);
  filter.change_P(updated_P);
  result.accepted = true;
  result.correction = dx;
  result.gain = K;
  result.velocity_after = after.rot.toRotationMatrix().transpose() * after.vel;
  result.covariance_diagonal = updated_P.diagonal();
  result.gyro_bias = after.bg;
  result.accel_bias = after.ba;
  return true;
}

}  // namespace learned_velocity
