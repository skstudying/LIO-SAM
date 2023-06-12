#include "utility.h"

#include <geometry_msgs/PoseWithCovarianceStamped.h>

class globalLocalization : public ParamServer
{

private:
public:

    // ROS topic publisher.
    ros::Publisher kPubGlobalMap;
    ros::Publisher kPubOdom;
    ros::Publisher kPubSubmap;
    
    // ROS topic subscrber.
    ros::Subscriber kSubInitialPose;
    ros::Subscriber kSubPoseSubmap;
    ros::Subscriber kSubSubmap;

    // For gloabal localiztion.
    pcl::PointCloud<PointType>::Ptr kGlobalMapCloud;
    pcl::PointCloud<PointType>::Ptr kSubmapCloud;
    Eigen::Affine3f kPoseSubmap;
    Eigen::Affine3f kTransMapping2Localization;

    // Filter.
    pcl::VoxelGrid<PointType> kDownSizeFilter;

    // System lock.
    std::mutex kMtx;

    // For publisher timestamp sourece.
    ros::Time kTimestampSubmap;
    
    globalLocalization()
    {
        kPubGlobalMap = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/localization/global_map", 1);
        kPubOdom = nh.advertise<nav_msgs::Odometry>("lio_sam/localization/odom", 1);
        kPubSubmap = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/localization/subamp", 1);
        
        kSubInitialPose = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>("initialpose", 1, &globalLocalization::initialPoseHandler, this, ros::TransportHints().tcpNoDelay());
        kSubPoseSubmap = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry", 1, &globalLocalization::poseSubmapHandler, this, ros::TransportHints().tcpNoDelay());
        kSubSubmap = nh.subscribe<sensor_msgs::PointCloud2>("lio_sam/mapping/submap", 1, &globalLocalization::submapHandler, this, ros::TransportHints().tcpNoDelay());

        kTransMapping2Localization = Eigen::Affine3f::Identity();
        kDownSizeFilter.setLeafSize(kLocalizationLeafSize, kLocalizationLeafSize, kLocalizationLeafSize);

        allocateMemory();

        // Load maps for localization.
        loadFeatureMaps();
    }

    void allocateMemory()
    {
        kGlobalMapCloud.reset(new pcl::PointCloud<PointType>());
        kSubmapCloud.reset(new pcl::PointCloud<PointType>());
    }

    void poseSubmapHandler(const nav_msgs::Odometry::ConstPtr& msg)
    {
        kPoseSubmap = odom2affine(*msg);

        // Publish updated localization odom as well.
        Eigen::Affine3f odom = kTransMapping2Localization * kPoseSubmap;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles (odom, x, y, z, roll, pitch, yaw);
        nav_msgs::Odometry nav_odom;
        nav_odom.header.stamp = msg->header.stamp;
        nav_odom.header.frame_id = mapFrame;
        nav_odom.child_frame_id = "localization_odom";
        nav_odom.pose.pose.position.x = x;
        nav_odom.pose.pose.position.y = y;
        nav_odom.pose.pose.position.z = z;
        nav_odom.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        kPubOdom.publish(nav_odom);
    }

    void submapHandler(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        std::lock_guard<std::mutex> lock(kMtx);
        kTimestampSubmap = msg->header.stamp;
        // Update submap.
        pcl::fromROSMsg(*msg, *kSubmapCloud);
    }

    void initialPoseHandler(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg) 
    {
        // Get height from keyFrames.
        float height = 0;

        // Pre-processing of initial pose.
        float yaw = tf::getYaw(msg->pose.pose.orientation);

        // Modify tf:odom2lidar.
        float init_pose[6];
        init_pose[0] = 0;
        init_pose[1] = 0;
        init_pose[2] = yaw;
        init_pose[3] = msg->pose.pose.position.x; 
        init_pose[4] = msg->pose.pose.position.y;
        init_pose[5] = height; 

        kTransMapping2Localization = trans2Affine3f(init_pose) * kPoseSubmap.inverse() ;        
        LOG(INFO) << "Initial pose is set at x: " << init_pose[3] << ", y: " << init_pose[4] << ", z: " << init_pose[5] << ", yaw: " << init_pose[2] << ".";
    }
    
    // Utility functions. 
    Eigen::Affine3f odom2affine(nav_msgs::Odometry odom)
    {
        double x, y, z, roll, pitch, yaw;
        x = odom.pose.pose.position.x;
        y = odom.pose.pose.position.y;
        z = odom.pose.pose.position.z;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        return pcl::getTransformation(x, y, z, roll, pitch, yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transform_in[])
    {
        return pcl::getTransformation(transform_in[3], transform_in[4], transform_in[5], transform_in[0], transform_in[1], transform_in[2]);
    }

    void loadFeatureMaps()
    {
        /*
        // Load 3D and 6D pose keyframes.
        pcl::PointCloud<PointType>::Ptr traj;
        traj.reset(new pcl::PointCloud<PointType>());
        assert(("Running localization mode. Trajectory PCD doesn't exist." && pcl::io::loadPCDFile(kTrajPCDDir, *traj) >= 0));
        // *cloudKeyPoses3D = *traj;
        pcl::PointCloud<PointTypePose>::Ptr trans;
        trans.reset(new pcl::PointCloud<PointTypePose>());
        assert(("Running localization mode. Transformation PCD doesn't exist." && pcl::io::loadPCDFile(kTransPCDDir, *trans) >= 0));
        // *cloudKeyPoses6D = *trans;
        assert(("Number of position keyframe != number of pose keyframe." && cloudKeyPoses3D->size() == cloudKeyPoses6D->size()));
        // Go through corner and surface pcd files.
        boost::filesystem::path path_corner(kCornerPCDFolder);
        vector<pair<double, size_t>> ind_corner_pcd;
        vector <string> corner_pcds;
        size_t count = 0;
        // For corner.
        for (auto i = boost::filesystem::directory_iterator(path_corner); i != boost::filesystem::directory_iterator(); ++i) {
            ind_corner_pcd.push_back(make_pair(stod(i->path().filename().string()), count));
            corner_pcds.push_back(i->path().filename().string());
            ++count;
        }
        assert(("Number of corner keyframe != number of keyframe." && corner_pcds.size() == cloudKeyPoses6D->size()));
        // For surface.
        boost::filesystem::path path_surface(kSurfacePCDFolder);
        vector<pair<double, size_t>> ind_surface_pcd;
        vector <string> surface_pcds;
        count = 0;
        for (auto i = boost::filesystem::directory_iterator(path_surface); i != boost::filesystem::directory_iterator(); ++i) {
            ind_surface_pcd.push_back(make_pair(stod(i->path().filename().string()), count));
            surface_pcds.push_back(i->path().filename().string());
            ++count;
        }
        assert(("Number of corner keyframe != number of keyframe." && surface_pcds.size() == cloudKeyPoses6D->size()));
        // Sorted according to timestamp.
        sort(ind_corner_pcd.begin(), ind_corner_pcd.end());
        sort(ind_surface_pcd.begin(), ind_surface_pcd.end());

        pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
        for (size_t i = 0; i < ind_corner_pcd.size(); i++) {
            assert(("Timestamp of corner feature map mismatched." && fabs(stod(corner_pcds[ind_corner_pcd[i].second]) - cloudKeyPoses6D->points[i].time) < 1e-6));
            // pcl::PointCloud<PointType>::Ptr corner_keyframe(new pcl::PointCloud<PointType>());
            // corner_keyframe.reset(new pcl::PointCloud<PointType>());
            pcl::io::loadPCDFile(kCornerPCDFolder + "/" + corner_pcds[ind_corner_pcd[i].second], *cloud);
            // cornerCloudKeyFrames.push_back(cloud);
            assert(("Timestamp of surface feature map mismatched." && fabs(stod(surface_pcds[ind_surface_pcd[i].second]) - cloudKeyPoses6D->points[i].time) < 1e-6));
            // pcl::PointCloud<PointType>::Ptr surface_keyframe(new pcl::PointCloud<PointType>());
            // surface_keyframe.reset(new pcl::PointCloud<PointType>());
            pcl::io::loadPCDFile(kSurfacePCDFolder + "/" + surface_pcds[ind_surface_pcd[i].second], *cloud);
            // surfCloudKeyFrames.push_back(cloud);
            // ROS_INFO("Load feature %s and %s at keyframe %f", corner_pcds[ind_corner_pcd[i].second].c_str(), surface_pcds[ind_surface_pcd[i].second].c_str(), cloudKeyPoses6D->points[i].time);
        }
        */
        // Load global map.
        pcl::PointCloud<PointType>::Ptr global_map(new pcl::PointCloud<PointType>());
        // global_map.reset(new pcl::PointCloud<PointType>());
        pcl::io::loadPCDFile(kGlobalMapPCDDir, *global_map);
        kDownSizeFilter.setInputCloud(global_map);
        kDownSizeFilter.filter(*kGlobalMapCloud);
    }

    void submap2MapOptimizationThread()
    {
        if (!kLocalizationMode) {
            LOG(ERROR) << "Runing localization node but flag localization_mode is disable. Break localoization loop.";
            return;
        }

        static double last_timestamp = 0.0;
        double current_timestamp = 0.0;
        
        ros::Rate rate(kLocalizationICPFrequency);
        while (ros::ok()) {
            rate.sleep();
            current_timestamp = kTimestampSubmap.toSec();
            if (fabs(current_timestamp - last_timestamp) < std::numeric_limits<double>::epsilon())
                continue;
            last_timestamp = current_timestamp;
            std::lock_guard<std::mutex> lock(kMtx);
            if (kSubmapCloud->size() > 0 && kGlobalMapCloud->size() > 0) {
                // Start cloud registation.
                pcl::console::TicToc time;
                time.tic();
                pcl::IterativeClosestPoint<PointType, PointType> icp;
                icp.setMaxCorrespondenceDistance(kICPOptCorDistance);
                icp.setMaximumIterations(kICPOptIterTimes);
                icp.setTransformationEpsilon(1e-6);
                icp.setEuclideanFitnessEpsilon(1e-6);
                icp.setInputSource(kSubmapCloud);
                icp.setInputTarget(kGlobalMapCloud);
                pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
                icp.align(*unused_result, kTransMapping2Localization.matrix());
                if (icp.hasConverged() && icp.getFitnessScore() < kLocalizationICPThresh)
                    kTransMapping2Localization = icp.getFinalTransformation();
                else
                    LOG(WARNING) << "Bad ICP registration result. The fitness score is " << icp.getFitnessScore() << ", and the duration is " << time.toc() * 0.001 << ".";
            }
            kMtx.unlock();
        }
    }

    void publishInfoThread()
    {
        ros::Rate rate(5);
        while (ros::ok()) {
            rate.sleep();
            // Publish global map.
            publishCloud(kPubGlobalMap, kGlobalMapCloud, ros::Time::now(), mapFrame);
            // Publish submap map based on global map coordinate.
            pcl::PointCloud<PointType>::Ptr submap_on_global_frame(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*kSubmapCloud, *submap_on_global_frame, kTransMapping2Localization);
            publishCloud(kPubSubmap, submap_on_global_frame, ros::Time::now(), mapFrame);
            // tf of map and odom.
            static tf::TransformBroadcaster tf;
            static tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
            tf.sendTransform(tf::StampedTransform(map_to_odom, ros::Time::now(), mapFrame, odometryFrame));
        }
    }

};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");
    ros::NodeHandle nh("~");

    // glog parameter.
    string glog_directory;
    int glog_stderrthreshold;
    nh.getParam(nh.getNamespace() + "/glog_directory", glog_directory);
    nh.getParam(nh.getNamespace() + "/glog_stderrthreshold", glog_stderrthreshold);
    FLAGS_log_dir = glog_directory;
    FLAGS_stderrthreshold = glog_stderrthreshold;
    google::InitGoogleLogging(argv[0]);

    globalLocalization GL;
    LOG(INFO) << "globalLocalization starts." << " Log to " << glog_directory << ".";
    
    std::thread submap2MapOptimizationThread(&globalLocalization::submap2MapOptimizationThread, &GL);
    std::thread publishInfoThread(&globalLocalization::publishInfoThread, &GL);

    ros::spin();
    submap2MapOptimizationThread.join();
    publishInfoThread.join();

    google::ShutdownGoogleLogging();
    return 0;
}