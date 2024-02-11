#!/usr/bin/env python3

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import *
import message_filters
import numpy as np
import time


class ImageCompressor():
    def __init__(self):
        self.color_publisher = rospy.Publisher("/rgb/compressed", CompressedImage, queue_size=3)
        self.depth_publisher = rospy.Publisher("/depth/compressed", CompressedImage, queue_size=3)

        self.color = []
        self.depth = []
        self.new_images = False

        self.read_queue = queue.Queue(1)

        self.cv_bridge = CvBridge()

        rgb_subscriber = message_filters.Subscriber("/xtion/rgb/image_raw", Image)
        depth_subscriber = message_filters.Subscriber("/xtion/depth/image_raw", Image)
        
        ts = message_filters.TimeSynchronizer([rgb_subscriber, depth_subscriber], 3)
        ts.registerCallback(self.get_images)

        rospy.Timer(rospy.Duration(0.04), self.republish_compressed_images)
        rospy.spin()

    def get_images(self, rgb, depth):
        self.read_queue.put([rgb, depth])

    def republish_compressed_images(self, event):
        print(time.time())
        data = self.read_queue.get()
        # # t1 = time.time()
        rgb_image = self.cv_bridge.imgmsg_to_cv2(data[0], data[0].encoding)
        depth_image = self.cv_bridge.imgmsg_to_cv2(data[1], data[1].encoding)
        # # print("CONVERT TIME", time.time() - t1)
        # # t2 = time.time()
        self.color = self.cv_bridge.cv2_to_compressed_imgmsg(rgb_image, dst_format = "jpeg")
        self.depth = self.cv_bridge.cv2_to_compressed_imgmsg(depth_image.astype(np.uint16), dst_format = "png")
        self.color_publisher.publish(self.color)
        self.depth_publisher.publish(self.depth)
        # # print("CPMRESS TIME", time.time() - t2)
        # # print("TOTAL:", time.time()-t1)
        # self.new_images = True

if __name__ == '__main__':
    rospy.init_node("image_compresser")
    rospy.loginfo("Image compresser node has been started")

    img = ImageCompressor()
    # rate = rospy.Rate(60) 
    # while not rospy.is_shutdown():
    #     # if img.new_images:
    #     #     print(time.time())
    #     #     img.color_publisher.publish(img.color)
    #     #     img.depth_publisher.publish(img.depth)
    #     #     img.new_images = False
    #     rate.sleep()
