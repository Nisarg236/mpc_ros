#ifndef MPC_CONTROLLER_H
#define MPC_CONTROLLER_H

#include <atomic>
#include <map>
#include <memory>
#include <string>

#include <ros/ros.h>
#include <mbf_costmap_core/costmap_controller.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <tf2_ros/buffer.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Path.h>
#include <Eigen/Core>

#include "navMpc.h"
#include "mpc_ros/MPCControllerConfig.h"

namespace mpc_ros {

class MPCController : public mbf_costmap_core::CostmapController
{
public:
    MPCController();
    ~MPCController() override = default;

    void initialize(std::string name, TF* tf, costmap_2d::Costmap2DROS* costmap_ros) override;
    bool setPlan(const std::vector<geometry_msgs::PoseStamped>& plan) override;
    uint32_t computeVelocityCommands(const geometry_msgs::PoseStamped& pose,
                                     const geometry_msgs::TwistStamped& velocity,
                                     geometry_msgs::TwistStamped& cmd_vel,
                                     std::string& message) override;
    bool isGoalReached(double xy_tolerance, double yaw_tolerance) override;
    bool cancel() override;

private:
    void reconfigureCB(MPCControllerConfig& config, uint32_t level);
    void applyConfig(const MPCControllerConfig& config);

    double polyeval(Eigen::VectorXd coeffs, double x);
    Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order);

    bool initialized_;
    std::atomic<bool> cancel_;

    ros::NodeHandle nh_;
    ros::Publisher pub_odompath_;
    ros::Publisher pub_mpctraj_;

    costmap_2d::Costmap2DROS* costmap_ros_;

    std::unique_ptr<dynamic_reconfigure::Server<MPCControllerConfig>> dynamic_recfg_;
    MPCControllerConfig config_;

    MPC mpc_;
    std::map<std::string, double> mpc_params_;

    nav_msgs::Path stored_path_;
    geometry_msgs::PoseStamped goal_pose_;
    geometry_msgs::PoseStamped current_pose_;
    bool path_set_;

    double w_;
    double throttle_;
    double speed_;
    double waypoints_dist_;
    int downsampling_;

    // Fixed params (not reconfigurable at runtime)
    double dt_;
    int controller_freq_;
    std::string car_frame_;
};

}  // namespace mpc_ros

#endif  // MPC_CONTROLLER_H
