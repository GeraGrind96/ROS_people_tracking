#!/usr/bin/env python3
import rospy
import math
import sys
from yolov8_data.msg import Object, ObjectsMSG
from yolov8_data.srv import *
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo as msg_CameraInfo
from sensor_msgs.msg import CompressedImage as msg_CompressedImage
from sensor_msgs.msg import PointCloud2 as msg_PointCloud2
from geometry_msgs.msg import *
from nav_msgs.msg import *
import numpy as np
import cv2
import queue
import yaml
import copy

sys.path.append('/home/gerardo/software/JointBDOE')
from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords, non_max_suppression
from utils.datasets import LoadImages
from models.experimental import attempt_load
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from ultralytics import YOLO

import time
import tf
import message_filters
import torch

class yolov8():
    def __init__(self):
        self.image_queue = queue.Queue(1)
        self.objects_publisher = rospy.Publisher("/perceived_people", ObjectsMSG, queue_size=10)
        self.objects_write = []
        self.objects_read = []
        self.camera_info = None

        self.depth_image = []
        self.color_image = []

        self.robot_world_transform_matrix = np.array([])
        self.robot_orientation = None
        self.camera_pose_respect_robot = np.array([[1, 0, 0, 0.21331892690256105],
                                                    [0, 1, 0, 0.004864029093594846],
                                                    [0, 0, 1, -0.9769708264898666],
                                                    [0,   0,   0,   1]])

        self.width = 640
        self.height = 480

        self.color_depth_ratio = None
        self.color_yolo_ratio_height = None
        self.color_yolo_ratio_width = None

        self.new_data = False

        self.device = select_device("0", batch_size=1)
        self.model = attempt_load("/home/gerardo/software/JointBDOE/runs/JointBDOE/coco_s_1024_e500_t020_w005/weights/best.pt", map_location=self.device)
        self.stride = int(self.model.stride.max())
        with open("/home/gerardo/software/JointBDOE/data/JointBDOE_weaklabel_coco.yaml") as f:
            self.data = yaml.safe_load(f)  # load data dict
        
################# SUBSCRIBER CALLBACKS #################

    def store_data(self, rgb, depth, odom):
        # print("STORING DATA")
        self.color_image = cv2.cvtColor(np.frombuffer(rgb.data, np.uint8).reshape(rgb.height, rgb.width, 4), cv2.COLOR_RGBA2RGB )
        self.depth_image = np.frombuffer(depth.data, np.float32).reshape(depth.height, depth.width, 1)

        euler_rotation = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w])
        self.robot_orientation = euler_rotation[2]
        self.robot_world_transform_matrix = np.array([[math.cos(euler_rotation[2]), -math.sin(euler_rotation[2]), 0, odom.pose.pose.position.x],
                        [math.sin(euler_rotation[2]), math.cos(euler_rotation[2]), 0, odom.pose.pose.position.y],
                        [0, 0, 1, odom.pose.pose.position.z],
                        [0,   0,   0,   1]])
        self.new_data = True
        
################# DATA OBTAINING #################

    def get_people_data(self, img):
        print("GET PEOPLE")
        img0 = copy.deepcopy(img)
        img = letterbox(img, 640, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        out_ori = self.model(img, augment=True, scales=[1])[0]
        out = non_max_suppression(out_ori, 0.3, 0.5, num_angles=self.data['num_angles'])

        bboxes = scale_coords(img.shape[2:], out[0][:, :4], img0.shape[:2]).cpu().numpy().astype(int)  # native-space pred
        scores = out[0][:, 4].cpu().numpy() 
        orientations = (out[0][:, 6:].cpu().numpy() * 360) - 180   # N*1, (0,1)*360 --> (0,360)
        aux_objects_write = []
        for index in range(len(bboxes)):
            act_object = Object()
            act_object.type = 0
            act_object.left = int(bboxes[index][0])
            act_object.top = int(bboxes[index][1])
            act_object.right = int(bboxes[index][2])
            act_object.bot = int(bboxes[index][3])
            act_object.score = scores[index]
            act_object.orientation = orientations[index]
            aux_objects_write.append(act_object)

        bytetrack_srv_proxy = rospy.ServiceProxy('bytetrack_srv', ObjectsSRV)
        try:
           aux_objects_read = bytetrack_srv_proxy(aux_objects_write).res
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        returned_bboxes = [[person.left if person.left > 0 else 0, person.top if person.top > 0 else 0, person.right if person.right < self.width else self.width - 1, person.bot if person.bot < self.height else self.height - 1] for person in aux_objects_read]
        returned_scores = [person.score for person in aux_objects_read]
        returned_orientations = [person.orientation for person in aux_objects_read]
        return returned_bboxes, returned_scores, returned_orientations
        # return bboxes, scores, orientations
       

    def get_people_pose(self, people_bboxes, depth_image):
        radius = 5
        color = (0, 0, 255)  # Color en formato BGR (azul)
        thickness = -1  # Relleno del círculo
        people_poses = []
        for person_bbox in people_bboxes:
            cv2.rectangle(self.color_image, (int(person_bbox[0]), int(person_bbox[1])), (int(person_bbox[2]), int(person_bbox[3])), (255, 0, 0), 2)
            x_range = int(person_bbox[0] + (person_bbox[2] - person_bbox[0]) / 2)
            y_range = int(person_bbox[1] + (person_bbox[3] - person_bbox[1]) / 5)
            # image_section = depth_image[int(person_bbox[3] / 5):int(person_bbox[3] / 4), x_range]
            # print(image_section)
            # # print(image_section)
            # if image_section.size > 0: 
            #     min_value = np.unravel_index(np.argmin(image_section), image_section.shape)  
            # else: 
            #     continue

            if math.isinf(depth_image[y_range][x_range]):
                image_section = depth_image[y_range, int(person_bbox[0]):int(person_bbox[2])]
                print("SECTION X", int(person_bbox[2] / 4), int(person_bbox[2] * 3 / 4))
                # print(image_section)
                if image_section.size > 0: 
                    print(image_section.shape)
                    min_value = np.unravel_index(np.argmin(image_section), image_section.shape)  
                    if not math.isinf(depth_image[y_range][min_value[0]]): 
                        print("Min value:", min_value)
                        from_robot_pose = self.depth_point_to_xyz([min_value[0] + person_bbox[0], y_range], depth_image[y_range][min_value[0] + person_bbox[0]])
                        cv2.circle(self.color_image, (min_value[0] + person_bbox[0], y_range), radius, color, thickness)
                    else:
                        print("PROJECTED POINT:", x_range, person_bbox[3])
                        cv2.circle(self.color_image, (x_range, person_bbox[3]), radius, color, thickness)
                        from_robot_pose = self.calculate_depth_with_projection([x_range, person_bbox[3]])
                else: 
                    print("PROJECTED POINT:", x_range, person_bbox[3])
                    cv2.circle(self.color_image, (x_range, person_bbox[3]), radius, color, thickness)
                    from_robot_pose = self.calculate_depth_with_projection([x_range, person_bbox[3]])
            else:
                print("DEPTH POINT:", x_range, y_range)
                cv2.circle(self.color_image, (x_range, y_range), radius, color, thickness)
                from_robot_pose = self.depth_point_to_xyz([x_range, y_range], depth_image[y_range][x_range])
            world_person_pose = self.transform_pose_to_world_reference(from_robot_pose)
            people_poses.append(world_person_pose)
        return people_poses
    
    def get_yolo_objects(self, event):
        if self.new_data:
            bboxes, scores, orientations = self.get_people_data(self.color_image)
            poses = self.get_people_pose(bboxes, self.depth_image)
            self.create_interface_data(bboxes, orientations, poses, scores)
            cv2.imshow("YOLO", self.color_image)
            cv2.waitKey(1)

################# DATA STRUCTURATION #################

    def create_interface_data(self, boxes, orientations, centers, scores):
        objects = ObjectsMSG()
        if len(boxes) == len(orientations) == len(centers) == len(scores):
            for index in range(len(boxes)):
                act_object = Object()
                act_object.type = 0
                act_object.left = boxes[index][0]
                act_object.top = boxes[index][1]
                act_object.right = boxes[index][2]
                act_object.bot = boxes[index][3]
                act_object.score = scores[index]
                # bbx_center_depth = [int((act_object.left + (act_object.right - act_object.left)/2)), int((act_object.top + (act_object.bot - act_object.top)/2))]
                act_object.pose = Pose()
                act_object.pose.position.x = centers[index][0] 
                act_object.pose.position.y = centers[index][1]

                act_object.pose.orientation = self.transform_orientation_to_world_reference(math.radians(orientations[index]))
                act_object.image = self.get_bbox_image_data(self.color_image, [act_object.left, act_object.top, act_object.right, act_object.bot])
                
                objects.objectsmsg.append(act_object)

            self.objects_publisher.publish(objects)

    def get_bbox_image_data(self, image, element_box):
        print(int(element_box[1]),int(element_box[3]), int(element_box[0]),int(element_box[2]))
        cropped_image = image[int(element_box[1]):int(element_box[3]), int(element_box[0]):int(element_box[2])]
        y, x, _ = cropped_image.shape
        cv2.imshow("image", cropped_image)
        return msg_Image(data=cropped_image.tobytes(), height=y, width=x)

################# TO WORLD TRANSFORMATIONS #################

    def transform_pose_to_world_reference(self, person_pose):
        person_world_position = np.dot(self.robot_world_transform_matrix, np.array([person_pose[0], -person_pose[1], 0, 1])) 
        return [person_world_position[0], person_world_position[1]]
    
    def transform_orientation_to_world_reference(self, orientation):
        theta_world = self.robot_orientation + orientation
        transformed_pose = tf.transformations.quaternion_from_euler(0, 0, -((math.pi)-np.arctan2(np.sin(theta_world), np.cos(theta_world)))) 
        return Quaternion(x=transformed_pose[0], y=transformed_pose[1], z=transformed_pose[2], w=transformed_pose[3])

################# IMAGE POINTS TO DEPTH #################

    def depth_point_to_xyz(self, pixel, depth):
        # angle_y = ((math.pi - 1.01)/2) + (pixel[1]*1.01/480)
        # angle_z = ((2*math.pi) - 0.785/2) + (pixel[0]*0.785/640)
        angle_y = ((math.pi - 0.785)/2) + (pixel[1]*0.785/480)
        angle_z = ((2*math.pi) - 1.01/2) + (pixel[0]*1.01/640)
        y_distance = depth / math.tan(angle_y)
        z_distance = depth * math.tan(angle_z)
        return depth[0], z_distance[0], y_distance[0]
    
    def calculate_depth_with_projection(self, projected_point):
        world_y = 579.65506 * 1.2 / (projected_point[1] - 243.0783)
        world_x = world_y * (projected_point[0] - 317.47191) / 577.55158
        return [world_y, world_x]

################# MAIN #################

if __name__ == '__main__':
    rospy.init_node("yolov8")
    rospy.loginfo("yolov8 node has been started")


    yolo = yolov8()
    rospy.wait_for_service('bytetrack_srv')

    rgb_subscriber = message_filters.Subscriber("/xtion/rgb/image_raw", msg_Image)
    depth_subscriber = message_filters.Subscriber("/xtion/depth/image_raw", msg_Image)
    odom_subscriber = message_filters.Subscriber("/odom", Odometry)
    
    ts = message_filters.TimeSynchronizer([rgb_subscriber, depth_subscriber, odom_subscriber], 5)
    ts.registerCallback(yolo.store_data)
    rospy.Timer(rospy.Duration(0.02), yolo.get_yolo_objects)
    rospy.spin()

    # rospy.logwarn("Warning test message")
    # rospy.logerr("Error test message")
    # rospy.loginfo("End of program")
