#include "mpc_controller.h"

#include <cmath>
#include <Eigen/QR>
#include <boost/bind.hpp>
#include <pluginlib/class_list_macros.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <mbf_msgs/ExePathResult.h>

PLUGINLIB_EXPORT_CLASS(mpc_ros::MPCController, mbf_costmap_core::CostmapController)

namespace mpc_ros {

MPCController::MPCController()
    : initialized_(false), cancel_(false), path_set_(false),
      w_(0.0), throttle_(0.0), speed_(0.0),
      waypoints_dist_(-1.0), downsampling_(1) {}

void MPCController::reconfigureCB(MPCControllerConfig& config, uint32_t /*level*/)
{
    config_ = config;
    applyConfig(config_);
}

void MPCController::applyConfig(const MPCControllerConfig& cfg)
{
    mpc_params_["STEPS"]     = cfg.mpc_steps;
    mpc_params_["REF_CTE"]   = cfg.mpc_ref_cte;
    mpc_params_["REF_ETHETA"]= cfg.mpc_ref_etheta;
    mpc_params_["REF_V"]     = cfg.mpc_ref_vel;
    mpc_params_["W_CTE"]     = cfg.mpc_w_cte;
    mpc_params_["W_EPSI"]    = cfg.mpc_w_etheta;
    mpc_params_["W_V"]       = cfg.mpc_w_vel;
    mpc_params_["W_ANGVEL"]  = cfg.mpc_w_angvel;
    mpc_params_["W_A"]       = cfg.mpc_w_accel;
    mpc_params_["W_DANGVEL"] = cfg.mpc_w_angvel_d;
    mpc_params_["W_DA"]      = cfg.mpc_w_accel_d;
    mpc_params_["ANGVEL"]    = cfg.mpc_max_angvel;
    mpc_params_["MAXTHR"]    = cfg.mpc_max_throttle;
    mpc_params_["BOUND"]     = cfg.mpc_bound_value;
    mpc_.LoadParams(mpc_params_);
}

void MPCController::initialize(std::string name, TF* /*tf*/,
                                costmap_2d::Costmap2DROS* costmap_ros)
{
    if (initialized_) return;

    costmap_ros_ = costmap_ros;
    ros::NodeHandle pnh("~/" + name);

    // Fixed params (not in dynamic reconfigure)
    pnh.param("controller_freq", controller_freq_, 10);
    pnh.param<std::string>("car_frame", car_frame_, "base_footprint");
    dt_ = 1.0 / controller_freq_;
    mpc_params_["DT"] = dt_;

    // Set up dynamic reconfigure — this also reads initial values from param server
    dynamic_recfg_ = std::make_unique<dynamic_reconfigure::Server<MPCControllerConfig>>(pnh);
    dynamic_recfg_->setCallback(boost::bind(&MPCController::reconfigureCB, this, _1, _2));

    pub_odompath_ = nh_.advertise<nav_msgs::Path>("mpc_reference", 1);
    pub_mpctraj_  = nh_.advertise<nav_msgs::Path>("mpc_trajectory", 1);

    initialized_ = true;
    ROS_INFO("MPCController initialized as '%s'", name.c_str());
}

bool MPCController::setPlan(const std::vector<geometry_msgs::PoseStamped>& plan)
{
    if (!initialized_) {
        ROS_ERROR("MPCController: initialize() must be called before setPlan()");
        return false;
    }

    cancel_ = false;
    path_set_ = false;
    waypoints_dist_ = -1.0;
    w_ = 0.0; throttle_ = 0.0; speed_ = 0.0;

    if (plan.empty()) return false;

    goal_pose_ = plan.back();

    const double path_length = config_.path_length;

    // Estimate waypoint spacing and downsampling factor
    if (plan.size() >= 2) {
        double dx = plan[1].pose.position.x - plan[0].pose.position.x;
        double dy = plan[1].pose.position.y - plan[0].pose.position.y;
        waypoints_dist_ = std::sqrt(dx * dx + dy * dy);
        downsampling_ = (waypoints_dist_ > 0.0)
            ? std::max(1, static_cast<int>(path_length / 10.0 / waypoints_dist_))
            : 1;
    }

    // Cut to path_length and downsample
    nav_msgs::Path sampled;
    sampled.header = plan.front().header;
    double total_length = 0.0;
    int sampling = downsampling_;

    for (size_t i = 0; i < plan.size(); ++i) {
        if (total_length > path_length) break;
        if (sampling >= downsampling_) {
            sampled.poses.push_back(plan[i]);
            sampling = 0;
        }
        total_length += (waypoints_dist_ > 0.0 ? waypoints_dist_ : 0.05);
        ++sampling;
    }

    // Need at least 4 points for a degree-3 polyfit
    if (sampled.poses.size() < 4) {
        sampled.poses.clear();
        size_t limit = std::min(plan.size(), static_cast<size_t>(50));
        for (size_t i = 0; i < limit; ++i)
            sampled.poses.push_back(plan[i]);
    }

    if (sampled.poses.size() < 4) {
        ROS_WARN("MPCController: plan too short (%zu poses)", sampled.poses.size());
        return false;
    }

    stored_path_ = sampled;
    path_set_ = true;
    pub_odompath_.publish(stored_path_);
    return true;
}

uint32_t MPCController::computeVelocityCommands(const geometry_msgs::PoseStamped& pose,
                                                 const geometry_msgs::TwistStamped& velocity,
                                                 geometry_msgs::TwistStamped& cmd_vel,
                                                 std::string& message)
{
    if (!initialized_) {
        message = "not initialized";
        return mbf_msgs::ExePathResult::NOT_INITIALIZED;
    }
    if (!path_set_) {
        message = "no plan set";
        return mbf_msgs::ExePathResult::INVALID_PATH;
    }
    if (cancel_) {
        cmd_vel.twist.linear.x  = 0.0;
        cmd_vel.twist.angular.z = 0.0;
        message = "cancelled";
        return mbf_msgs::ExePathResult::CANCELED;
    }

    current_pose_ = pose;

    const double px    = pose.pose.position.x;
    const double py    = pose.pose.position.y;
    const double theta = tf2::getYaw(pose.pose.orientation);
    const double v     = velocity.twist.linear.x;

    const double costheta = std::cos(theta);
    const double sintheta = std::sin(theta);

    const int N = static_cast<int>(stored_path_.poses.size());
    Eigen::VectorXd x_veh(N), y_veh(N);
    for (int i = 0; i < N; ++i) {
        const double dx = stored_path_.poses[i].pose.position.x - px;
        const double dy = stored_path_.poses[i].pose.position.y - py;
        x_veh[i] =  dx * costheta + dy * sintheta;
        y_veh[i] = -dx * sintheta + dy * costheta;
    }

    auto coeffs = polyfit(x_veh, y_veh, 3);
    const double cte = polyeval(coeffs, 0.0);

    // Heading error from path tangent direction
    double etheta = std::atan(coeffs[1]);
    const int N_sample = std::max(1, static_cast<int>(N * 0.3));
    double gx = 0.0, gy = 0.0;
    for (int i = 1; i < N_sample; ++i) {
        gx += stored_path_.poses[i].pose.position.x - stored_path_.poses[i - 1].pose.position.x;
        gy += stored_path_.poses[i].pose.position.y - stored_path_.poses[i - 1].pose.position.y;
    }
    const double PI = 3.141592;
    const double traj_deg  = std::atan2(gy, gx);
    double temp_theta = theta;
    if (temp_theta <= -PI + traj_deg) temp_theta += 2.0 * PI;
    if ((gx != 0.0 || gy != 0.0) && temp_theta - traj_deg < 1.8 * PI)
        etheta = temp_theta - traj_deg;
    else
        etheta = 0.0;

    Eigen::VectorXd state(6);
    if (config_.delay_mode) {
        const double px_act     = v * dt_;
        const double py_act     = 0.0;
        const double theta_act  = w_ * dt_;
        const double v_act      = v + throttle_ * dt_;
        const double cte_act    = cte + v * std::sin(etheta) * dt_;
        const double etheta_act = etheta - theta_act;
        state << px_act, py_act, theta_act, v_act, cte_act, etheta_act;
    } else {
        state << 0.0, 0.0, 0.0, v, cte, etheta;
    }

    std::vector<double> mpc_results = mpc_.Solve(state, coeffs);
    w_        = mpc_results[0];
    throttle_ = mpc_results[1];
    speed_    = std::max(0.0, std::min(v + throttle_ * dt_, config_.max_speed));

    cmd_vel.header          = pose.header;
    cmd_vel.twist.linear.x  = speed_;
    cmd_vel.twist.angular.z = w_;

    // Publish predicted trajectory in car frame
    nav_msgs::Path mpc_traj;
    mpc_traj.header.frame_id = car_frame_;
    mpc_traj.header.stamp    = ros::Time::now();
    for (size_t i = 0; i < mpc_.mpc_x.size(); ++i) {
        geometry_msgs::PoseStamped p;
        p.header = mpc_traj.header;
        p.pose.position.x = mpc_.mpc_x[i];
        p.pose.position.y = mpc_.mpc_y[i];
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, mpc_.mpc_theta[i]);
        tf2::convert(q, p.pose.orientation);
        mpc_traj.poses.push_back(p);
    }
    pub_mpctraj_.publish(mpc_traj);

    if (config_.debug_info) {
        ROS_INFO("MPCController: cte=%.3f etheta=%.3f speed=%.3f w=%.3f",
                 cte, etheta, speed_, w_);
    }

    message = "success";
    return mbf_msgs::ExePathResult::SUCCESS;
}

bool MPCController::isGoalReached(double xy_tolerance, double yaw_tolerance)
{
    if (!path_set_) return false;

    const double dx = current_pose_.pose.position.x - goal_pose_.pose.position.x;
    const double dy = current_pose_.pose.position.y - goal_pose_.pose.position.y;
    const double dist = std::sqrt(dx * dx + dy * dy);

    double yaw_diff = std::fabs(tf2::getYaw(current_pose_.pose.orientation) -
                                tf2::getYaw(goal_pose_.pose.orientation));
    if (yaw_diff > M_PI) yaw_diff = std::fabs(yaw_diff - 2.0 * M_PI);

    return dist <= xy_tolerance && yaw_diff <= yaw_tolerance;
}

bool MPCController::cancel()
{
    cancel_ = true;
    return true;
}

double MPCController::polyeval(Eigen::VectorXd coeffs, double x)
{
    double result = 0.0;
    for (int i = 0; i < coeffs.size(); ++i)
        result += coeffs[i] * std::pow(x, i);
    return result;
}

Eigen::VectorXd MPCController::polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order)
{
    assert(xvals.size() == yvals.size());
    assert(order >= 1 && order <= xvals.size() - 1);
    Eigen::MatrixXd A(xvals.size(), order + 1);
    for (int i = 0; i < xvals.size(); ++i) A(i, 0) = 1.0;
    for (int j = 0; j < xvals.size(); ++j)
        for (int i = 0; i < order; ++i)
            A(j, i + 1) = A(j, i) * xvals(j);
    return A.householderQr().solve(yvals);
}

}  // namespace mpc_ros
