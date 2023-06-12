#include "utility.h"
#include "lio_sam/cloud_info.h"

#define GND_IMG_NX1 24
#define GND_IMG_NY1 20
#define GND_IMG_DX1 4
#define GND_IMG_DY1 4
#define GND_IMG_OFFX1 40
#define GND_IMG_OFFY1 40

struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (std::uint16_t, ring, ring) (float, time, time)
)

struct RSLidarPointXYZIRT
{
    PCL_ADD_POINT4D
    uint8_t intensity;
    uint16_t ring = 0;
    double timestamp = 0;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (RSLidarPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (std::uint8_t, intensity, intensity)
    (std::uint16_t, ring, ring) (double, timestamp, timestamp)
)

struct RSLidar2PointXYZIRT
{
    PCL_ADD_POINT4D
    float intensity;
    uint16_t ring = 0;
    double timestamp = 0;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (RSLidar2PointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (std::uint16_t, ring, ring) (double, timestamp, timestamp)
)

struct SClusterFeature {
    // Basic param.
    int pnum;
    float xmin;
    float xmax;
    float ymin;
    float ymax;
    float zmin;
    float zmax;
    float zmean;
    // PCA.
    float d0[3];
    float d1[3];
    float center[3];
    float obb[8];
    // Class.
    int cls;
};

struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (std::uint32_t, t, t) (std::uint16_t, reflectivity, reflectivity)
    (std::uint8_t, ring, ring) (std::uint16_t, noise, noise) (std::uint32_t, range, range)
)

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:
    std::mutex kImuMtx;
    std::mutex kOdomMtx;
    std::mutex kCloudMtx;

    ros::Subscriber subLaserCloud;
    ros::Publisher pubLaserCloud;
    ros::Publisher pubExtractedCloud;

    // Pointcloud segmentation publisher.
    ros::Publisher kPubObjectsCloud;
    ros::Publisher kPubBackgroundCloud;
    ros::Publisher kPubNewGround;

    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;
    queue<sensor_msgs::PointCloud2ConstPtr> kLaserCloudMsgBuff;

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;

    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    pcl::PointCloud<RSLidarPointXYZIRT>::Ptr kTmpRSLidarCloudIn;
    pcl::PointCloud<RSLidar2PointXYZIRT>::Ptr kTmpRSLidar2CloudIn;
    pcl::PointCloud<PointType>::Ptr fullCloud;
    pcl::PointCloud<PointType>::Ptr extractedCloud;
    
    // Segmentation point cloud
    pcl::PointCloud<PointType>::Ptr kSegGroundCloud;
    pcl::PointCloud<PointType>::Ptr kObjectCloud;
    pcl::PointCloud<PointType>::Ptr kBackGroundCloud;

    int deskewFlag;
    cv::Mat rangeMat;
    cv::Mat kSegMat;

    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    lio_sam::cloud_info cloudInfo;
    double timeScanCur;
    double timeScanEnd;
    std_msgs::Header cloudHeader;

    vector<int> columnIdnCountVec;
    vector<int> kPointLabels;

    // Transformation of cloud to base.
    Eigen::Affine3f kTransCloud2Base;

    pcl::KdTreeFLANN<PointType> kkdtreeSeg;

public:
    ImageProjection():
    deskewFlag(0)
    {
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudLoader, this, ros::TransportHints().tcpNoDelay());

        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("lio_sam/deskew/cloud_deskewed", 1);
        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/deskew/cloud_info", 1);

        // Init segmentation pointcloud publisher
        kPubNewGround = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/new_ground_cloud", 1);
        kPubObjectsCloud = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/objects_cloud", 1);
        kPubBackgroundCloud = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/background_cloud", 1);

        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

        if (kCloud2Base) {
            Eigen::Quaterniond eq = Eigen::Quaterniond(kExtRotCloud);
            tf::Quaternion quat(eq.x(), eq.y(), eq.z(), eq.w());
            double roll, pitch, yaw;
            tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
            kTransCloud2Base = pcl::getTransformation(kExtTransCloud.x(), kExtTransCloud.y(), kExtTransCloud.z(), roll, pitch, yaw);
        }
    }

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        kTmpRSLidarCloudIn.reset(new pcl::PointCloud<RSLidarPointXYZIRT>());
        kTmpRSLidar2CloudIn.reset(new pcl::PointCloud<RSLidar2PointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        // For pointcloud segmentation.
        kBackGroundCloud.reset(new pcl::PointCloud<PointType>());
        kSegGroundCloud.reset(new pcl::PointCloud<PointType>());
        kObjectCloud.reset(new pcl::PointCloud<PointType>()); 

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        // For pointcloud segmentation.
        kSegGroundCloud->clear();
        kBackGroundCloud->clear(); 
        kObjectCloud -> clear();
        // Reset range matrix for range image projection.
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        // Segmentation labeling mat.
        kSegMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F,cv::Scalar::all(0));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }

        columnIdnCountVec.assign(N_SCAN, 0);
        kPointLabels.assign(N_SCAN * Horizon_SCAN, 0);
    }

    ~ImageProjection(){}

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);

        std::lock_guard<std::mutex> lock1(kImuMtx);
        imuQueue.push_back(thisImu);
        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x << 
        //       ", y: " << thisImu.linear_acceleration.y << 
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x << 
        //       ", y: " << thisImu.angular_velocity.y << 
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }

    // Cassel, taking Imu prediction state from ImuP.
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)  
    {
        std::lock_guard<std::mutex> lock2(kOdomMtx);
        odomQueue.push_back(*odometryMsg);
    }

    void cloudLoader(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {
        std::lock_guard<std::mutex> lock(kCloudMtx);
        kLaserCloudMsgBuff.push(laserCloudMsg);
    }

    void cloudHandler()
    {
        while (ros::ok()){
            if (kLaserCloudMsgBuff.empty())
                continue;
            sensor_msgs::PointCloud2ConstPtr laserCloudMsg;
            kCloudMtx.lock();
            if (kLaserCloudMsgBuff.size() > 8)
                LOG(WARNING) << "The laserMsg message has been jamed in image projection (cloudHandler). Buffer size " << kLaserCloudMsgBuff.size() << ".";
            laserCloudMsg = kLaserCloudMsgBuff.front();
            kLaserCloudMsgBuff.pop();
            kCloudMtx.unlock();
            if (!cachePointCloud(laserCloudMsg)) continue;

            if (!deskewInfo()) continue;

            projectPointCloud();


            if (kSelectionMode) {
                kkdtreeSeg.setInputCloud(fullCloud);
                // Ground and background segments will be further used for pose estimation.
                if (kSelectGround)
                  selectGround();
                if (kSelectBackGround)
                  selectBackGround();
                // Static and dynamic objects will be removed.
                if (kSelectObjects)
                  selectObjects();
                removePointsUnSelected();
            }

            cloudExtraction();

            publishClouds();

            resetParameters();
        }
    }

    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // cache point cloud
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2)
            return false;

        // convert cloud
        currentCloudMsg = std::move(cloudQueue.front());
        cloudQueue.pop_front();
        if (sensor == SensorType::VELODYNE || sensor == SensorType::LIVOX)
        {
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
        }
        else if (sensor == SensorType::OUSTER)
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        else if (sensor == SensorType::RSLIDAR)
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *kTmpRSLidarCloudIn);
            laserCloudIn->points.resize(kTmpRSLidarCloudIn->size());
            laserCloudIn->is_dense = true;
            for (size_t i = 0; i < kTmpRSLidarCloudIn->size(); i++) {
                auto &src = kTmpRSLidarCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = float(src.timestamp - kTmpRSLidarCloudIn->points[0].timestamp);
            }
        }
        else if (sensor == SensorType::RSLIDAR2)
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *kTmpRSLidar2CloudIn);
            laserCloudIn->points.resize(kTmpRSLidar2CloudIn->size());
            laserCloudIn->is_dense = true;
            for (size_t i = 0; i < kTmpRSLidar2CloudIn->size(); i++) {
                auto &src = kTmpRSLidar2CloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = float(src.timestamp - kTmpRSLidar2CloudIn->points[0].timestamp);
            }
        }
        else {
            LOG(FATAL) << "Unknown sensor type: " << int(sensor) << ".";
            ros::shutdown();
        }

        // get timestamp
        cloudHeader = currentCloudMsg.header;
        timeScanCur = cloudHeader.stamp.toSec();
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

        //remove Nan
        vector<int> indices;
        pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);

        // check dense flag
        if (laserCloudIn->is_dense == false) {
            LOG(FATAL) << "Point cloud is not in dense format, please remove NaN points first!";
            ros::shutdown();
        }

        // check ring channel
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1) {
                LOG(FATAL) << "Point cloud ring channel not available, please configure your point cloud data!";
                ros::shutdown();
            }
        }

        // check point time
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t" || field.name == "timestamp")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                LOG(WARNING) << "Point cloud timestamp not available, deskew function disabled, system will drift significantly!";
        }

        return true;
    }

    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(kImuMtx);
        std::lock_guard<std::mutex> lock2(kOdomMtx);

        // make sure IMU data available for the scan
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd) {
            LOG(INFO) << "Waiting for IMU data...";
            return false;
        }

        imuDeskewInfo();

        odomDeskewInfo();

        return true;
    }

    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;

        while (!imuQueue.empty())
        {
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = thisImuMsg.header.stamp.toSec();

            // get roll, pitch, and yaw estimation for this scan
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);

            if (currentImuTime > timeScanEnd + 0.01)
                break;

            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;

        if (imuPointerCur <= 0)
            return;

        cloudInfo.imuAvailable = true;
    }

    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;

        //Cassel, align Imu odo with laser data
        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        // get start odometry at the beinning of the scan
        nav_msgs::Odometry startOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }

        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization. Packing inital guess for MO.
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;

        cloudInfo.odomAvailable = true;

        // get end odometry at the end of the scan
        odomDeskewFlag = false;

        if (odomQueue.back().header.stamp.toSec() < timeScanEnd)
            return;

        nav_msgs::Odometry endOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanEnd)
                continue;
            else
                break;
        }

        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }

    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        int imuPointerFront = 0;
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanEnd - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }

    PointType deskewPoint(PointType *point, double relTime)
    {
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;

        double pointTime = timeScanCur + relTime;

        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // transform points to start
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            if (kCloud2Base) {
                thisPoint.x = kTransCloud2Base(0,0) * laserCloudIn->points[i].x + kTransCloud2Base(0,1) * laserCloudIn->points[i].y + kTransCloud2Base(0,2) * laserCloudIn->points[i].z + kTransCloud2Base(0,3);
                thisPoint.y = kTransCloud2Base(1,0) * laserCloudIn->points[i].x + kTransCloud2Base(1,1) * laserCloudIn->points[i].y + kTransCloud2Base(1,2) * laserCloudIn->points[i].z + kTransCloud2Base(1,3);
                thisPoint.z = kTransCloud2Base(2,0) * laserCloudIn->points[i].x + kTransCloud2Base(2,1) * laserCloudIn->points[i].y + kTransCloud2Base(2,2) * laserCloudIn->points[i].z + kTransCloud2Base(2,3);
            } else {
                thisPoint.x = laserCloudIn->points[i].x;
                thisPoint.y = laserCloudIn->points[i].y;
                thisPoint.z = laserCloudIn->points[i].z;
            }
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            float range = pointDistance(thisPoint);
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;

            int rowIdn = laserCloudIn->points[i].ring;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            if (rowIdn % downsampleRate != 0)
                continue;

            int columnIdn = -1;
            if (sensor == SensorType::VELODYNE || sensor == SensorType::OUSTER || sensor == SensorType::RSLIDAR || sensor == SensorType::RSLIDAR2)
            {
                float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
                static float ang_res_x = 360.0/float(Horizon_SCAN);
                columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
                if (columnIdn >= Horizon_SCAN)
                    columnIdn -= Horizon_SCAN;
            }
            else if (sensor == SensorType::LIVOX)
            {
                columnIdn = columnIdnCountVec[rowIdn];
                columnIdnCountVec[rowIdn] += 1;
            }
            
            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;

            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);

            rangeMat.at<float>(rowIdn, columnIdn) = range;

            int index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }

    SClusterFeature FindACluster(int seed_id, int label_id, pcl::PointCloud<PointType>::Ptr cloud, float f_search_radius, float thr_height)
    {
        // Initialize seed.
        std::vector<int> seeds;
        seeds.push_back(seed_id);
        kPointLabels[seed_id] = label_id;

        SClusterFeature cf;
        cf.pnum = 1;
        cf.xmax = -2000;
        cf.xmin = 2000;
        cf.ymax = -2000;
        cf.ymin = 2000;
        cf.zmax = -2000;
        cf.zmin = 2000;
        cf.zmean = 0;

        // Increase searching area.
        while (seeds.size() > 0) {
            int sid = seeds[seeds.size() - 1];
            seeds.pop_back();

            PointType searchPoint;
            searchPoint.x = cloud->points[sid].x;
            searchPoint.y = cloud->points[sid].y;
            searchPoint.z = cloud->points[sid].z;

            // Feature statistics.
            if (searchPoint.x > cf.xmax) cf.xmax = searchPoint.x;
            if (searchPoint.x < cf.xmin) cf.xmin = searchPoint.x;

            if (searchPoint.y > cf.ymax) cf.ymax = searchPoint.y;
            if (searchPoint.y < cf.ymin) cf.ymin = searchPoint.y;

            if (searchPoint.z > cf.zmax) cf.zmax = searchPoint.z;
            if (searchPoint.z < cf.zmin) cf.zmin = searchPoint.z;

            cf.zmean += searchPoint.z;

            std::vector<float> k_dis;
            std::vector<int> k_inds;

            if (searchPoint.x < 44.8f)
                kkdtreeSeg.radiusSearch(searchPoint, f_search_radius, k_inds, k_dis);
            else
                kkdtreeSeg.radiusSearch(searchPoint, 2.0f * f_search_radius, k_inds, k_dis);

            for (size_t ii = 0; ii < k_inds.size(); ii++) {
                if (kPointLabels[k_inds[ii]] == 0) {
                    kPointLabels[k_inds[ii]] = label_id;
                    cf.pnum++;
                    // Filter point below than 60cm.
                    if (cloud->points[k_inds[ii]].z > thr_height) {
                      seeds.push_back(k_inds[ii]);
                    }
                }
            }
        }
        cf.zmean /= (cf.pnum + 0.000001);
        return cf;
    }

    void selectObjects()
    {
        float f_search_radius = 0.7;
        int pnum = fullCloud->points.size();
        // Start object's ID from 10.
        int label_id = 10;

        // Go through non-background points.
        for (int pid = 0; pid < pnum; pid++) {
            if (kPointLabels[pid] == 0) {
                // Height threshold.
                if (fullCloud->points[pid].z > 0.4) {
                    SClusterFeature cf = FindACluster(pid, label_id, fullCloud, f_search_radius, 0);
                    int is_bg = 0;

                    // Cluster.
                    float dx = cf.xmax - cf.xmin;
                    float dy = cf.ymax - cf.ymin;
                    // float dz = cf.zmax - cf.zmin;
                    float cx = 10000;
                    for (int ii = 0; ii < pnum; ii++) {
                        if (cx > fullCloud->points[pid].x) {
                          cx = fullCloud->points[pid].x;
                        }
                    }
                    // Large object.
                    if ((dx > 15) || (dy > 15) || ((dx > 10) && (dy > 10))) {
                        is_bg = 2;
                    }
                    // Long and too low.
                    else if (((dx > 6) || (dy > 6)) && (cf.zmean < 1.5)) {
                        is_bg = 3;
                    }
                    // Small and too high.
                    else if (((dx < 1.5) && (dy < 1.5)) && (cf.zmax > 2.5)) {
                        is_bg = 4;
                    }
                    // Few point in the group.
                    else if (cf.pnum < 5 || (cf.pnum < 10 && cx < 50)) {
                        is_bg = 5;
                    }
                    // Too high or too low.
                    else if ((cf.zmean > 3) || (cf.zmean < 0.3)) {
                        is_bg = 6;
                    }

                    if (is_bg > 0) {
                        for (int ii = 0; ii < pnum; ii++) {
                          if (kPointLabels[ii] == label_id) {
                            kPointLabels[ii] = is_bg;
                          }
                        }
                    } else {
                        label_id++;
                    }
                }
            }
        }
    }

    void selectBackGround()
    {
        float f_search_radius = 0.5;
        vector<int> background_labels;
        int point_num = fullCloud->points.size();
        background_labels.assign(point_num, 0);
        // Initiate seeds.
        std::vector<int> seeds;
        for (int pid = 0; pid < point_num; pid++) {
            // Save the tall point cloud for future clustering.
            if (fullCloud->points[pid].z > 4) {
                seeds.push_back(pid);
            } else {
                background_labels[pid] = 0;
            }
        }
        // Area growth.
        while (seeds.size() > 0) {
            int sid = seeds[seeds.size() - 1];
            seeds.pop_back();

            std::vector<float> k_dis;
            std::vector<int> k_inds;
            if (fullCloud->points[sid].x < 44.8)
                kkdtreeSeg.radiusSearch(sid, f_search_radius, k_inds, k_dis);
            else
                kkdtreeSeg.radiusSearch(sid, 1.5 * f_search_radius, k_inds, k_dis);

            for (size_t index = 0; index < k_inds.size(); index++) {
                if (background_labels[k_inds[index]] == 0) {
                    background_labels[k_inds[index]] = 1;
                    if (fullCloud->points[k_inds[index]].z > 0.2) {
                        seeds.push_back(k_inds[index]);
                    }
                }
            }
        }

        for (int index = 0; index < point_num; ++index) {
            if (kPointLabels[index] == 0 && background_labels[index] != 0) {
                kPointLabels[index] = 1;
            }
        }
    }

    void selectGround() {
        int point_num = fullCloud->points.size();
        float *p_gnd_img1 = (float *)calloc(GND_IMG_NX1 * GND_IMG_NY1, sizeof(float));
        int *tmp_label1 = (int *)calloc(point_num, sizeof(int));
        for (int ii = 0; ii < GND_IMG_NX1 * GND_IMG_NY1; ii++)
            p_gnd_img1[ii] = 100;

        // Find the lowest image.
        for (int pid = 0; pid < point_num; pid++) {
            int ix = (fullCloud->points[pid].x + GND_IMG_OFFX1) / (GND_IMG_DX1 + 0.000001);
            int iy = (fullCloud->points[pid].y + GND_IMG_OFFY1) / (GND_IMG_DY1 + 0.000001);
            if (ix < 0 || ix >= GND_IMG_NX1 || iy < 0 || iy >= GND_IMG_NY1) {
                tmp_label1[pid] = -1;
                continue;
            }

            int iid = ix + iy * GND_IMG_NX1;
            tmp_label1[pid] = iid;
            // Find the lowest height.
            if (p_gnd_img1[iid] > fullCloud->points[pid].z)
                p_gnd_img1[iid] = fullCloud->points[pid].z;
        }

        int pnum = 0;
        for (int pid = 0; pid < point_num; pid++) {
            if (tmp_label1[pid] >= 0) {
                if (p_gnd_img1[tmp_label1[pid]] + 0.5 > fullCloud->points[pid].z) {
                    kPointLabels[pid] = 10000;
                    pnum++;
                }
            }
        }

        free(p_gnd_img1);
        free(tmp_label1);

        // Limit height.
        for (int pid = 0; pid < point_num; pid++) {
            if (kPointLabels[pid] == 10000) {
                // For all cases.
                if (fullCloud->points[pid].z > 1)
                    kPointLabels[pid] = 0;
                else if (fullCloud->points[pid].x * fullCloud->points[pid].x + fullCloud->points[pid].y * fullCloud->points[pid].y < 225 && fullCloud->points[pid].z > 0.5)
                    kPointLabels[pid] = 0;
            } else if (fullCloud->points[pid].x * fullCloud->points[pid].x + fullCloud->points[pid].y * fullCloud->points[pid].y < 400 && fullCloud->points[pid].z < 0.2)
                    kPointLabels[pid] = 10000;
        }

        float z_mean = 0;
        int gnum = 0;
        for (int pid = 0; pid < point_num; pid++) {
            if (kPointLabels[pid] == 10000 && fullCloud->points[pid].x * fullCloud->points[pid].x + fullCloud->points[pid].y * fullCloud->points[pid].y < 400) {
                z_mean += fullCloud->points[pid].z;
                gnum++;
            }
        }
        z_mean /= (gnum + 0.0001);
        for (int pid = 0; pid < point_num; pid++) {
            if (kPointLabels[pid] == 10000) {
                if (fullCloud->points[pid].x * fullCloud->points[pid].x + fullCloud->points[pid].y * fullCloud->points[pid].y < 400 && fullCloud->points[pid].z > z_mean + 0.4)
                    kPointLabels[pid] = 0;
            }
        }

        gnum = 0;
        for (int pid = 0; pid < point_num; pid++) {
            if (kPointLabels[pid] == 10000)
                gnum++;
        }
    }

    void removePointsUnSelected()
    {
        for (int i = 0; i < N_SCAN; ++i) {
            for (int j = 0; j < Horizon_SCAN; ++j) {
                if (kPointLabels[j + i * Horizon_SCAN] == 10000) {
                    kSegMat.at<float>(i, j) = 1;
                    continue;
                } else if ((kPointLabels[j + i * Horizon_SCAN] >= 10) &&
                           (kSelectObjects == true)) {
                    kSegMat.at<float>(i, j) = 2;
                    continue;
                } else if ((kPointLabels[j + i * Horizon_SCAN] >= 1) &&
                           (kPointLabels[j + i * Horizon_SCAN] <= 10) &&
                           (kSelectBackGround == true)) {
                    kSegMat.at<float>(i, j) = 3;
                    continue;
                } else {
                    rangeMat.at<float>(i, j) = FLT_MAX;
                }
            }
        }
    }

    void cloudExtraction()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry
        for (int i = 0; i < N_SCAN; ++i) {
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (int j = 0; j < Horizon_SCAN; ++j) {
            if (kSegMat.at<float>(i, j) != 0) {
                    if (kSegMat.at<float>(i, j) == 1) {
                        kSegGroundCloud->push_back(
                            fullCloud->points[j + i * Horizon_SCAN]);
                    }
                    if (kSegMat.at<float>(i, j) == 2) {
                        kObjectCloud->push_back(
                            fullCloud->points[j + i * Horizon_SCAN]);
                    }
                    if (kSegMat.at<float>(i, j) == 3) {
                        kBackGroundCloud->push_back(
                            fullCloud->points[j + i * Horizon_SCAN]);
                    }
            }
            if (rangeMat.at<float>(i, j) != FLT_MAX) {
                    // mark the points' column index for marking occlusion later
                    cloudInfo.pointColInd[count] = j;
                    // save range info
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i, j);
                    // save extracted cloud
                    extractedCloud->push_back(
                        fullCloud->points[j + i * Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
            }
            }
            cloudInfo.endRingIndex[i] = count - 1 - 5;
        }
    }

    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed  = publishCloud(pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        pubLaserCloudInfo.publish(cloudInfo);

        // Publish segmenation ground cloud.
        if (kPubNewGround.getNumSubscribers() != 0)
            publishCloud(kPubNewGround, kSegGroundCloud, cloudHeader.stamp, lidarFrame);
        // Publish segmentation objects cloud.
        if (kPubObjectsCloud.getNumSubscribers() != 0)
            publishCloud(kPubObjectsCloud, kObjectCloud, cloudHeader.stamp, lidarFrame);
        // Publish segmentation background cloud.
        if (kPubBackgroundCloud.getNumSubscribers() != 0)
            publishCloud(kPubBackgroundCloud, kBackGroundCloud, cloudHeader.stamp, lidarFrame);
    }

    template<typename T>
    bool has_nan(T point) 
    {
        if (std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z))
            return true;
        else
            return false;
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

    ImageProjection IP;
    LOG(INFO) << "imageProjection starts." << " Log to " << glog_directory << ".";

    thread cloudThread(&ImageProjection::cloudHandler, &IP);

    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    cloudThread.join();

    google::ShutdownGoogleLogging();
    return 0;
}
