#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <chrono>

using namespace sensor_msgs;
using namespace message_filters;

void imageCallback(const ImageConstPtr& depth_msg, const ImageConstPtr& color_msg, ros::Publisher& depth_pub, ros::Publisher& color_pub)
{
    try
    {
        auto start = std::chrono::high_resolution_clock::now();
        // Convierte los mensajes de ROS a imágenes OpenCV
        cv_bridge::CvImageConstPtr depth_cv = cv_bridge::toCvShare(depth_msg);
        cv_bridge::CvImageConstPtr color_cv = cv_bridge::toCvShare(color_msg);

        // Aquí puedes procesar las imágenes de profundidad y color juntas o por separado según tus necesidades

        // Comprime las imágenes en formato PNG
        cv::Mat depth_image = depth_cv->image;
        cv::Mat color_image = color_cv->image;

        std::vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(4); // Nivel de compresión (0-9)

        std::vector<uchar> depth_data, color_data;
        cv::imencode(".png", depth_image, depth_data, compression_params);
        cv::imencode(".png", color_image, color_data, compression_params);

        // Publica las imágenes comprimidas de profundidad y color en tópicos separados
        sensor_msgs::CompressedImage compressed_depth_image, compressed_color_image;
        compressed_depth_image.format = "png";
        compressed_color_image.format = "png";
        compressed_depth_image.data = depth_data;
        compressed_color_image.data = color_data;
        // Obtén el tiempo actual después de realizar la tarea
        auto end = std::chrono::high_resolution_clock::now();

        // Calcula la diferencia de tiempo
        std::chrono::duration<double> duration = end - start;

        // Muestra la diferencia de tiempo en segundos
        std::cout << "Tiempo transcurrido: " << duration.count() << " segundos" << std::endl;
        depth_pub.publish(compressed_depth_image);
        color_pub.publish(compressed_color_image);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_compress_node");
    ros::NodeHandle nh;
    std::cout << "E" << std::endl;
    // Sincronización de las suscripciones a las imágenes de profundidad y color
    message_filters::Subscriber<Image> depth_image_sub(nh, "/l515/aligned_depth_to_color/image_raw", 1);
    message_filters::Subscriber<Image> color_image_sub(nh, "/l515/color/image_raw", 1);

    // Publicadores para las imágenes comprimidas de profundidad y color
    ros::Publisher depth_pub = nh.advertise<sensor_msgs::CompressedImage>("/depth/compressed", 1);
    ros::Publisher color_pub = nh.advertise<sensor_msgs::CompressedImage>("/rgb/compressed", 1);

    typedef sync_policies::ApproximateTime<Image, Image> MySyncPolicy;
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), depth_image_sub, color_image_sub);
    sync.registerCallback(boost::bind(&imageCallback, _1, _2, depth_pub, color_pub));



    ros::spin();

    return 0;
}