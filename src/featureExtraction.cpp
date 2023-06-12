#include "utility.h"
#include "lio_sam/cloud_info.h"

struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{
private:
    std::mutex kCloudInfoMtx;

public:

    ros::Subscriber subLaserCloudInfo;

    ros::Publisher pubLaserCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubSurfacePoints;
    ros::Publisher kPubSharpCornerPoints;
    ros::Publisher kPubFlatSurfacePoints;

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;
    pcl::PointCloud<PointType>::Ptr kNonFeatureCloud;
    pcl::PointCloud<PointType>::Ptr kSharpCornerCloud;
    pcl::PointCloud<PointType>::Ptr kFlatSurfaceCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;

    lio_sam::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;
    queue<lio_sam::cloud_infoConstPtr> kCloudInfoBuff;

    std::vector<smoothness_t> cloudSmoothness;
    std::vector<int> kPointCornerMarks;
    float *cloudCurvature;
    int *cloudNeighborPicked;
    int *cloudLabel;

    vector<int> kPointFeatures;

    FeatureExtraction()
    {
        subLaserCloudInfo = nh.subscribe<lio_sam::cloud_info>("lio_sam/deskew/cloud_info", 1, &FeatureExtraction::laserCloudInfoLoader, this, ros::TransportHints().tcpNoDelay());

        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/feature/cloud_info", 1);
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_corner", 1);
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_surface", 1);
        kPubSharpCornerPoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_sharp_corner", 1);
        kPubFlatSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/feature/cloud_flat_surface", 1);
        
        initializationValue();
    }

    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);
        // Change to aloam parameters = 0.2,0.2,0.2
        downSizeFilter.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize); 

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());
        kNonFeatureCloud.reset(new pcl::PointCloud<PointType>());
        kSharpCornerCloud.reset(new pcl::PointCloud<PointType>());
        kFlatSurfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];
    }

    void laserCloudInfoLoader(const lio_sam::cloud_infoConstPtr& msgIn)
    {
        std::lock_guard<std::mutex> lock(kCloudInfoMtx);
        kCloudInfoBuff.push(msgIn);
    }

    bool plane_judge(const std::vector<PointType> &point_list, const int plane_threshold)
    {
        int num = point_list.size();
        float cx = 0;
        float cy = 0;
        float cz = 0;
        for (int j = 0; j < num; j++) {
            cx += point_list[j].x;
            cy += point_list[j].y;
            cz += point_list[j].z;
        }
        cx /= num;
        cy /= num;
        cz /= num;
        // mean square error
        float a11 = 0;
        float a12 = 0;
        float a13 = 0;
        float a22 = 0;
        float a23 = 0;
        float a33 = 0;
        for (int j = 0; j < num; j++) {
            float ax = point_list[j].x - cx;
            float ay = point_list[j].y - cy;
            float az = point_list[j].z - cz;

            a11 += ax * ax;
            a12 += ax * ay;
            a13 += ax * az;
            a22 += ay * ay;
            a23 += ay * az;
            a33 += az * az;
        }
        a11 /= num;
        a12 /= num;
        a13 /= num;
        a22 /= num;
        a23 /= num;
        a33 /= num;

        Eigen::Matrix<double, 3, 3> _mat_a1;
        _mat_a1.setZero();
        Eigen::Matrix<double, 3, 1> _mat_d1;
        _mat_d1.setZero();
        Eigen::Matrix<double, 3, 3> _mat_v1;
        _mat_v1.setZero();

        _mat_a1(0, 0) = a11;
        _mat_a1(0, 1) = a12;
        _mat_a1(0, 2) = a13;
        _mat_a1(1, 0) = a12;
        _mat_a1(1, 1) = a22;
        _mat_a1(1, 2) = a23;
        _mat_a1(2, 0) = a13;
        _mat_a1(2, 1) = a23;
        _mat_a1(2, 2) = a33;

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(_mat_a1, Eigen::ComputeThinU | Eigen::ComputeThinV);
        _mat_d1 = svd.singularValues();
        _mat_v1 = svd.matrixU();
        if (_mat_d1(0, 0) < plane_threshold * _mat_d1(1, 0))
            return true;
        else
            return false;
    }

    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++)
        {
            float diffRange = cloudInfo.pointRange[i-5] + cloudInfo.pointRange[i-4]
                            + cloudInfo.pointRange[i-3] + cloudInfo.pointRange[i-2]
                            + cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i] * 10
                            + cloudInfo.pointRange[i+1] + cloudInfo.pointRange[i+2]
                            + cloudInfo.pointRange[i+3] + cloudInfo.pointRange[i+4]
                            + cloudInfo.pointRange[i+5];            

            cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;

            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            // cloudSmoothness for sorting
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points
            float depth1 = cloudInfo.pointRange[i];
            float depth2 = cloudInfo.pointRange[i+1];
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i+1] - cloudInfo.pointColInd[i]));

            if (columnDiff < 10){
                // 10 pixel diff in range image
                if (depth1 - depth2 > 0.3){
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam
            float diff1 = std::abs(float(cloudInfo.pointRange[i-1] - cloudInfo.pointRange[i]));
            float diff2 = std::abs(float(cloudInfo.pointRange[i+1] - cloudInfo.pointRange[i]));

            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    void extractStrongFeatures_step1()
    {
        int th_break_corner_dis = 1;
        float th_lidar_nearest_dis = 1.0;
        int cloud_feature_flag[20000];
        kPointFeatures.assign(extractedCloud->size(), 0);
        for (int i = 0; i < N_SCAN; ++i) {
            if (cloudInfo.endRingIndex[i] - cloudInfo.startRingIndex[i] < 6)
                continue;
            int sp = cloudInfo.startRingIndex[i];
            int ep = cloudInfo.endRingIndex[i];
            PointType point;
            pcl::PointCloud<PointType>::Ptr laser_cloud(new pcl::PointCloud<PointType>());
            int cloudSize = ep - sp + 1;
            laser_cloud->reserve(cloudSize);

            for (int k = ep; k >= sp; k--) {
                int ind = cloudSmoothness[k].ind;
                point.x = extractedCloud->points[ind].x;
                point.y = extractedCloud->points[ind].y;
                point.z = extractedCloud->points[ind].z;
                point.intensity = extractedCloud->points[ind].intensity;
                if (!isfinite(point.x) || !isfinite(point.y) || !isfinite(point.z))
                    continue;
                laser_cloud->push_back(point);
                cloud_feature_flag[ind] = 0;
            }
            cloudSize = laser_cloud->size();

            //--------------------------------------------------- break points ---------------------------------------------
            for (int i = 5; i < cloudSize - 5; i++) {
                float diff_left[2];
                float diff_right[2];

                for (int count = 1; count < 3; count++) {
                    float diff_x1 = laser_cloud->points[i + count].x - laser_cloud->points[i].x;
                    float diff_y1 = laser_cloud->points[i + count].y - laser_cloud->points[i].y;
                    float diff_z1 = laser_cloud->points[i + count].z - laser_cloud->points[i].z;
                    diff_right[count - 1] = sqrt(diff_x1 * diff_x1 + diff_y1 * diff_y1 + diff_z1 * diff_z1);

                    float diff_x2 = laser_cloud->points[i - count].x - laser_cloud->points[i].x;
                    float diff_y2 = laser_cloud->points[i - count].y - laser_cloud->points[i].y;
                    float diff_z2 = laser_cloud->points[i - count].z - laser_cloud->points[i].z;
                    diff_left[count - 1] = sqrt(diff_x2 * diff_x2 + diff_y2 * diff_y2 + diff_z2 * diff_z2);
                }

                float depth_right = sqrt(laser_cloud->points[i + 1].x * laser_cloud->points[i + 1].x +
                laser_cloud->points[i + 1].y * laser_cloud->points[i + 1].y +
                laser_cloud->points[i + 1].z * laser_cloud->points[i + 1].z);
                float depth_left = sqrt(laser_cloud->points[i - 1].x * laser_cloud->points[i - 1].x +
                laser_cloud->points[i - 1].y * laser_cloud->points[i - 1].y +
                laser_cloud->points[i - 1].z * laser_cloud->points[i - 1].z);

                if (fabs(diff_right[0] - diff_left[0]) > th_break_corner_dis) {
                    if (diff_right[0] > diff_left[0]) {

                        Eigen::Vector3d surf_vector = Eigen::Vector3d(laser_cloud->points[i - 1].x - laser_cloud->points[i].x,
                                                                        laser_cloud->points[i - 1].y - laser_cloud->points[i].y,
                                                                        laser_cloud->points[i - 1].z - laser_cloud->points[i].z);
                        Eigen::Vector3d lidar_vector = Eigen::Vector3d(laser_cloud->points[i].x,
                                                                        laser_cloud->points[i].y,
                                                                        laser_cloud->points[i].z);
                        // double left_surf_dis = surf_vector.norm();
                        // calculate the angle between the laser direction and the surface
                        double cc = fabs(surf_vector.dot(lidar_vector) / (surf_vector.norm() * lidar_vector.norm()));

                        std::vector<PointType> left_list;
                        double min_dis = 10000;
                        double max_dis = 0;
                        for (int j = 0; j < 4; j++) { // TODO: change the plane window size and add thin rod support
                            left_list.push_back(laser_cloud->points[i - j]);
                            Eigen::Vector3d temp_vector = Eigen::Vector3d(laser_cloud->points[i - j].x - laser_cloud->points[i - j - 1].x,
                                                                            laser_cloud->points[i - j].y - laser_cloud->points[i - j - 1].y,
                                                                            laser_cloud->points[i - j].z - laser_cloud->points[i - j - 1].z);

                            if (j == 3)
                                break;
                            double temp_dis = temp_vector.norm();
                            if (temp_dis < min_dis)
                                min_dis = temp_dis;
                            if (temp_dis > max_dis)
                                max_dis = temp_dis;
                        }
                        // bool left_is_plane = plane_judge(left_list,0.3);

                        if (cc < 0.93) { //(max_dis < 2*min_dis) && left_surf_dis < 0.05 * depth  && left_is_plane &&
                            if (depth_right > depth_left)
                                cloud_feature_flag[i] = 100;
                            else if (depth_right == 0)
                                cloud_feature_flag[i] = 100;
                        }
                    } else {

                        Eigen::Vector3d surf_vector = Eigen::Vector3d(laser_cloud->points[i + 1].x - laser_cloud->points[i].x,
                                                                        laser_cloud->points[i + 1].y - laser_cloud->points[i].y,
                                                                        laser_cloud->points[i + 1].z - laser_cloud->points[i].z);
                        Eigen::Vector3d lidar_vector = Eigen::Vector3d(laser_cloud->points[i].x,
                                                                        laser_cloud->points[i].y,
                                                                        laser_cloud->points[i].z);

                        // calculate the angle between the laser direction and the surface
                        double cc = fabs(surf_vector.dot(lidar_vector) / (surf_vector.norm() * lidar_vector.norm()));

                        std::vector<PointType> right_list;
                        double min_dis = 10000;
                        double max_dis = 0;
                        for (int j = 0; j < 4; j++) { // TODO: change the plane window size and add thin rod support
                            right_list.push_back(laser_cloud->points[i - j]);
                            Eigen::Vector3d temp_vector = Eigen::Vector3d(laser_cloud->points[i + j].x - laser_cloud->points[i + j + 1].x,
                                                                            laser_cloud->points[i + j].y - laser_cloud->points[i + j + 1].y,
                                                                            laser_cloud->points[i + j].z - laser_cloud->points[i + j + 1].z);

                            if (j == 3)
                                break;
                            double temp_dis = temp_vector.norm();
                            if (temp_dis < min_dis)
                                min_dis = temp_dis;
                            if (temp_dis > max_dis)
                                max_dis = temp_dis;
                        }
                        // bool right_is_plane = plane_judge(right_list,0.3);

                        if (cc < 0.93)
                        { // right_is_plane  && (max_dis < 2*min_dis) && right_surf_dis < 0.05 * depth &&

                            if (depth_right < depth_left)
                                cloud_feature_flag[i] = 100;
                            else if (depth_left == 0)
                                    cloud_feature_flag[i] = 100;
                        }
                    }
                }

                // break points select
                if (cloud_feature_flag[i] == 100) {
                    std::vector<Eigen::Vector3d> front_norms;
                    Eigen::Vector3d norm_front(0, 0, 0);
                    Eigen::Vector3d norm_back(0, 0, 0);

                    for (int k = 1; k < 4; k++) {

                        float temp_depth = sqrt(laser_cloud->points[i - k].x * laser_cloud->points[i - k].x +
                                                laser_cloud->points[i - k].y * laser_cloud->points[i - k].y +
                                                laser_cloud->points[i - k].z * laser_cloud->points[i - k].z);

                        if (temp_depth < 1)
                            continue;

                        Eigen::Vector3d tmp = Eigen::Vector3d(laser_cloud->points[i - k].x - laser_cloud->points[i].x,
                                                                laser_cloud->points[i - k].y - laser_cloud->points[i].y,
                                                                laser_cloud->points[i - k].z - laser_cloud->points[i].z);
                        tmp.normalize();
                        front_norms.push_back(tmp);
                        norm_front += (k / 6.0) * tmp;
                    }
                    std::vector<Eigen::Vector3d> back_norms;
                    for (int k = 1; k < 4; k++) {

                        float temp_depth = sqrt(laser_cloud->points[i - k].x * laser_cloud->points[i - k].x +
                                                laser_cloud->points[i - k].y * laser_cloud->points[i - k].y +
                                                laser_cloud->points[i - k].z * laser_cloud->points[i - k].z);

                        if (temp_depth < 1)
                            continue;

                        Eigen::Vector3d tmp = Eigen::Vector3d(laser_cloud->points[i + k].x - laser_cloud->points[i].x,
                                                                laser_cloud->points[i + k].y - laser_cloud->points[i].y,
                                                                laser_cloud->points[i + k].z - laser_cloud->points[i].z);
                        tmp.normalize();
                        back_norms.push_back(tmp);
                        norm_back += (k / 6.0) * tmp;
                    }
                    double cc = fabs(norm_front.dot(norm_back) / (norm_front.norm() * norm_back.norm()));
                    if (cc > 0.93)
                        cloud_feature_flag[i] = 101;
                }
            }

            pcl::PointCloud<PointType>::Ptr laser_cloud_corner(new pcl::PointCloud<PointType>());
            std::vector<int> points_less_sharp_ori;

            for (int i = 5; i < cloudSize - 5; i++) {
                Eigen::Vector3d left_pt = Eigen::Vector3d(laser_cloud->points[i - 1].x,
                                                            laser_cloud->points[i - 1].y,
                                                            laser_cloud->points[i - 1].z);
                Eigen::Vector3d right_pt = Eigen::Vector3d(laser_cloud->points[i + 1].x,
                                                            laser_cloud->points[i + 1].y,
                                                            laser_cloud->points[i + 1].z);

                float dis = laser_cloud->points[i].x * laser_cloud->points[i].x +
                            laser_cloud->points[i].y * laser_cloud->points[i].y +
                            laser_cloud->points[i].z * laser_cloud->points[i].z;

                double clr = fabs(left_pt.dot(right_pt) / (left_pt.norm() * right_pt.norm()));

                if (clr < 0.999)
                    cloud_feature_flag[i] = 200;

                if (dis < th_lidar_nearest_dis * th_lidar_nearest_dis)
                    continue;

                if (cloud_feature_flag[i] == 100 || cloud_feature_flag[i] == 200) {
                    points_less_sharp_ori.push_back(i);
                    laser_cloud_corner->push_back(laser_cloud->points[i]);
                }
            }

            for (size_t i = 0; i < laser_cloud_corner->points.size(); i++) {
                int ind_ori = ep + points_less_sharp_ori[i];
                int index = cloudSmoothness[ind_ori].ind;
                kPointCornerMarks.push_back(index);
            }
            *cornerCloud += *laser_cloud_corner;
        }
    }

    void extractStrongFeatures_step2()
    {
        int cloud_size = extractedCloud->points.size();
        pcl::KdTreeFLANN<PointType>::Ptr kd_tree_cloud;
        kd_tree_cloud.reset(new pcl::KdTreeFLANN<PointType>);
        kd_tree_cloud->setInputCloud(extractedCloud);

        std::vector<int> point_search_ind;
        std::vector<float> point_search_sq_dis;

        int num_near = 10;
        int stride = 1;
        int interval = 4;

        for (int i = 5; i < cloud_size - 5; i = i + stride) {
            
            if (fabs(kPointFeatures[i]- 1.0) < 1e-5)
                continue;

            double thre1d = 0.5;
            double thre2d = 0.8;
            double thre3d = 0.5;
            double thre3d2 = 0.13;

            double disti = sqrt(extractedCloud->points[i].x * extractedCloud->points[i].x +
                                extractedCloud->points[i].y * extractedCloud->points[i].y +
                                extractedCloud->points[i].z * extractedCloud->points[i].z);

            if (disti < 30.0) {
                thre1d = 0.5;
                thre2d = 0.8;
                thre3d2 = 0.07;
                stride = 14;
                interval = 4;
            }
            else if (disti < 60.0) {
                stride = 10;
                interval = 3;
            }
            else {
                stride = 1;
                interval = 0;
            }

            if (disti > 100.0) {
                num_near = 6;
                kPointFeatures[i] = 3.0;
                kNonFeatureCloud->points.push_back(extractedCloud->points[i]);
                continue;
            }
            else if (disti > 60.0)
                num_near = 8;
            else
                num_near = 10;

            kd_tree_cloud->nearestKSearch(extractedCloud->points[i], num_near, point_search_ind, point_search_sq_dis);

            if (point_search_sq_dis[num_near - 1] > 5.0 && disti < 90.0)
                continue;

            //PCA process to find 
            Eigen::Matrix<double, 3, 3> mat_a1;
            mat_a1.setZero();

            float cx = 0;
            float cy = 0;
            float cz = 0;
            for (int j = 0; j < num_near; j++) {
                cx += extractedCloud->points[point_search_ind[j]].x;
                cy += extractedCloud->points[point_search_ind[j]].y;
                cz += extractedCloud->points[point_search_ind[j]].z;
            }
            cx /= num_near;
            cy /= num_near;
            cz /= num_near;

            float a11 = 0;
            float a12 = 0;
            float a13 = 0;
            float a22 = 0;
            float a23 = 0;
            float a33 = 0;
            for (int j = 0; j < num_near; j++) {
                float ax = extractedCloud->points[point_search_ind[j]].x - cx;
                float ay = extractedCloud->points[point_search_ind[j]].y - cy;
                float az = extractedCloud->points[point_search_ind[j]].z - cz;

                a11 += ax * ax;
                a12 += ax * ay;
                a13 += ax * az;
                a22 += ay * ay;
                a23 += ay * az;
                a33 += az * az;
            }
            a11 /= num_near;
            a12 /= num_near;
            a13 /= num_near;
            a22 /= num_near;
            a23 /= num_near;
            a33 /= num_near;

            mat_a1(0, 0) = a11;
            mat_a1(0, 1) = a12;
            mat_a1(0, 2) = a13;
            mat_a1(1, 0) = a12;
            mat_a1(1, 1) = a22;
            mat_a1(1, 2) = a23;
            mat_a1(2, 0) = a13;
            mat_a1(2, 1) = a23;
            mat_a1(2, 2) = a33;

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(mat_a1);
            double a1d = (sqrt(saes.eigenvalues()[2]) - sqrt(saes.eigenvalues()[1])) / sqrt(saes.eigenvalues()[2]);
            double a2d = (sqrt(saes.eigenvalues()[1]) - sqrt(saes.eigenvalues()[0])) / sqrt(saes.eigenvalues()[2]);
            double a3d = sqrt(saes.eigenvalues()[0]) / sqrt(saes.eigenvalues()[2]);

            if (a2d > thre2d || (a3d < thre3d2 && a1d < thre1d)) {
                for (int k = 1; k < interval; k++) {
                    kPointFeatures[i - k]= 2.0;
                    surfaceCloud->points.push_back(extractedCloud->points[i - k]);
                    kPointFeatures[i + k] = 2.0;
                    surfaceCloud->points.push_back(extractedCloud->points[i + k]);
                }
                kPointFeatures[i] = 2.0;
                surfaceCloud->points.push_back(extractedCloud->points[i]);
            } else if (a3d > thre3d) {
                for (int k = 1; k < interval; k++) {
                    kPointFeatures[i - k] = 3.0;
                    kNonFeatureCloud->points.push_back(extractedCloud->points[i - k]);
                    kPointFeatures[i + k] = 3.0;
                    kNonFeatureCloud->points.push_back(extractedCloud->points[i + k]);
                }
                kPointFeatures[i] = 3.0;
                kNonFeatureCloud->points.push_back(extractedCloud->points[i]);
            }
        }
    }

    void extractStrongFeactures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();
        kSharpCornerCloud->clear();
        kFlatSurfaceCloud->clear();
        kNonFeatureCloud->clear();
        kPointCornerMarks.clear();

        extractStrongFeatures_step1();
        
        for (size_t i = 0; i < kPointCornerMarks.size(); ++i)
            kPointFeatures[kPointCornerMarks[i]] = 1.0;
        
        extractStrongFeatures_step2();
        int cloud_num = extractedCloud->points.size();
        for (int i = 0; i < cloud_num; ++i) {
            PointType point = extractedCloud->points[i];
            if (!isfinite(point.x) || !isfinite(point.y) || !isfinite(point.z))
                continue;
            float dis = point.x * point.x + point.y * point.y + point.z * point.z;
            if (kSelectionMode && kPointFeatures[i] > 9 && dis < 2500.0)
                kPointFeatures[i] = 0;
            else if (dis < 2500.0)
                kPointFeatures[i] = 0;
        }
        cornerCloud->clear();
        surfaceCloud->clear();
        kNonFeatureCloud->clear();
        for (size_t i = 0; i < extractedCloud->points.size(); ++i) {
            if (std::fabs(kPointFeatures[i] - 1.0) < 1e-5)
                cornerCloud->push_back(extractedCloud->points[i]);
            if (std::fabs(kPointFeatures[i] - 2.0) < 1e-5)
                surfaceCloud->push_back(extractedCloud->points[i]);
            if (std::fabs(kPointFeatures[i] - 3.0) < 1e-5)
                kNonFeatureCloud->push_back(extractedCloud->points[i]);
        }

        kFlatSurfaceCloud = surfaceCloud;
        kSharpCornerCloud = cornerCloud;
    }

    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();
        kSharpCornerCloud->clear();
        kFlatSurfaceCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        for (int i = 0; i < N_SCAN; i++)
            {
                surfaceCloudScan->clear();

                // li-cong.
                if (cloudInfo.endRingIndex[i] - cloudInfo.startRingIndex[i] < 6)
                    continue;

                for (int j = 0; j < 6; j++)
                {

                    int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
                    int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

                    if (sp >= ep)
                        continue;

                    std::sort(cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep, by_value());

                    int largestPickedNum = 0;

                    for (int k = ep; k >= sp; k--)
                    {
                        int ind = cloudSmoothness[k].ind;
                        if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                        {
                            largestPickedNum++;
                            if (largestPickedNum <= 2)
                            {
                                cloudLabel[ind] = 2;
                                kSharpCornerCloud->push_back(extractedCloud->points[ind]);
                                cornerCloud->push_back(extractedCloud->points[ind]);
                            }
                            else if (largestPickedNum <= 20)
                            {
                                cloudLabel[ind] = 1;
                                cornerCloud->push_back(extractedCloud->points[ind]);
                            }
                            else
                                break;

                            cloudNeighborPicked[ind] = 1;

                            // Change to aloam version of marking neighbors.
                            for (int l = 1; l <= 5; l++)
                            {
                                float diffX = extractedCloud->points[ind + l].x - extractedCloud->points[ind + l - 1].x;
                                float diffY = extractedCloud->points[ind + l].y - extractedCloud->points[ind + l - 1].y;
                                float diffZ = extractedCloud->points[ind + l].z - extractedCloud->points[ind + l - 1].z;
                                if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                    break;

                                cloudNeighborPicked[ind + l] = 1;
                            }
                            for (int l = -1; l >= -5; l--)
                            {
                                float diffX = extractedCloud->points[ind + l].x - extractedCloud->points[ind + l + 1].x;
                                float diffY = extractedCloud->points[ind + l].y - extractedCloud->points[ind + l + 1].y;
                                float diffZ = extractedCloud->points[ind + l].z - extractedCloud->points[ind + l + 1].z;
                                if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                    break;
                                cloudNeighborPicked[ind + l] = 1;
                            }
                        }
                    }
                    int smallestPickedNum = 0;

                    // Surface
                    for (int k = sp; k <= ep; k++)
                    {

                        int ind = cloudSmoothness[k].ind;
                        if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                        {

                            cloudLabel[ind] = -1;
                            kFlatSurfaceCloud->push_back(extractedCloud->points[k]);

                            smallestPickedNum++;
                            if (smallestPickedNum >= 4)
                            {
                                break;
                            }
                            cloudNeighborPicked[ind] = 1;

                            // Change to aloam version of marking neighbors.
                            for (int l = 1; l <= 5; l++)
                            {

                                float diffX = extractedCloud->points[ind + l].x - extractedCloud->points[ind + l - 1].x;
                                float diffY = extractedCloud->points[ind + l].y - extractedCloud->points[ind + l - 1].y;
                                float diffZ = extractedCloud->points[ind + l].z - extractedCloud->points[ind + l - 1].z;
                                if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                    break;

                                cloudNeighborPicked[ind + l] = 1;
                            }
                            for (int l = -1; l >= -5; l--)
                            {

                                float diffX = extractedCloud->points[ind + l].x - extractedCloud->points[ind + l + 1].x;
                                float diffY = extractedCloud->points[ind + l].y - extractedCloud->points[ind + l + 1].y;
                                float diffZ = extractedCloud->points[ind + l].z - extractedCloud->points[ind + l + 1].z;
                                if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                    break;

                                cloudNeighborPicked[ind + l] = 1;
                            }
                        }
                    }

                    for (int k = sp; k <= ep; k++)
                    {
                        if (cloudLabel[k] <= 0)
                        {
                            surfaceCloudScan->push_back(extractedCloud->points[k]);
                        }
                    }
                }

                surfaceCloudScanDS->clear();

                downSizeFilter.setInputCloud(surfaceCloudScan);
                downSizeFilter.filter(*surfaceCloudScanDS);

                *surfaceCloud += *surfaceCloudScanDS;
            }
        
    }

    void freeCloudInfoMemory()
    {
        cloudInfo.startRingIndex.clear();
        cloudInfo.endRingIndex.clear();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointRange.clear();
    }

    void publishFeatureCloud()
    {
        // free cloud info memory
        freeCloudInfoMemory();
        // Save newly extracted features.  Pack lidar feature into cloudInfo.
        cloudInfo.cloud_corner  = publishCloud(pubCornerPoints,  cornerCloud,  cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_sharp_corner = publishCloud(kPubSharpCornerPoints, kSharpCornerCloud, cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_flat_surface = publishCloud(kPubFlatSurfacePoints, kFlatSurfaceCloud, cloudHeader.stamp, lidarFrame);
        // publish to mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
    }
    void lasercloudHandler()
    {
        while (ros::ok()){
            if (kCloudInfoBuff.empty())
                continue;

            kCloudInfoMtx.lock();
            if (kCloudInfoBuff.size() > 8)
                LOG(WARNING) << "The cloudinfo message has been jamed in feature extraction (lasercloudHandler). Buffer size " << kCloudInfoBuff.size() << ".";
            lio_sam::cloud_infoConstPtr tmp_msg_info = kCloudInfoBuff.front();
            kCloudInfoBuff.pop();
            kCloudInfoMtx.unlock();

            // New cloud info and header.
            cloudInfo = *tmp_msg_info; 
            cloudHeader = tmp_msg_info->header;
            pcl::fromROSMsg(tmp_msg_info->cloud_deskewed, *extractedCloud);
            calculateSmoothness();

            markOccludedPoints();

            if (kUseStrongFeature)
                extractStrongFeactures();
            else
                extractFeatures();

            publishFeatureCloud();
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

    FeatureExtraction FE;
    LOG(INFO) << "featureExtraction starts." << " Log to " << glog_directory << ".";

    thread lasercloudThread(&FeatureExtraction::lasercloudHandler, &FE);
   
    ros::spin();
    lasercloudThread.join();

    google::ShutdownGoogleLogging();
    return 0;
}
