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
import pyrealsense2
from ultralytics import YOLO
import time
import tf
import message_filters

_OBJECT_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                 'sheep',
                 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                 'scissors',
                 'teddy bear', 'hair drier', 'toothbrush']

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

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

        self.width = 640
        self.height = 480

        self.color_depth_ratio = None
        self.color_yolo_ratio_height = None
        self.color_yolo_ratio_width = None

        self.new_data = False

        self.model = YOLO('yolov8n-pose.pt')

    def get_bounding_boxes(self, results):
        boxes = [] 
        for result in results:
            boxes_to_numpy = result.boxes.xyxyn.cpu().numpy()
            if len(boxes_to_numpy) > 0:
                for box in boxes_to_numpy:
                    box[np.arange(len(box)) % 2 == 0] *= self.width
                    box[np.arange(len(box)) % 2 != 0] *= self.height
                    box = box.astype(int)
                    boxes.append([box[0], box[1], box[2], box[3]])
                    cv2.rectangle(self.color_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)  # Masks object for segmentation masks outputs

        return boxes
    
    def get_person_interest_keypoints(self, results, depth_image):
        keypoints = []
        centers = []
        for result in results:
            keypoints_to_numpy = result.keypoints.xyn.cpu().numpy()
            if len(keypoints_to_numpy) > 0:
                for keypoint_set in keypoints_to_numpy:
                    if len(keypoint_set) > 0: 
                        # Getting points from hips
                        chosen_points = keypoint_set[[11, 12]]
                        chosen_points[:, 0] *= self.width
                        chosen_points[:, 1] *= self.height
                        if np.any(chosen_points[:, 0] >= self.width) or np.any(chosen_points[:, 1] >= self.height):
                            print("Predicted joint position out of image. Not considering")
                            continue

                        # Calculate neck point
                        x_avg = (keypoint_set[11, 0] + keypoint_set[12, 0]) / 2
                        y_avg = (keypoint_set[5, 1] + keypoint_set[6, 1]) / 2
                        neck_point = np.array([x_avg * self.width, y_avg * self.height]).astype(int)
                        complete_points = np.append(chosen_points, [neck_point], axis=0)
                        complete_points = complete_points.astype(int)

                        radius = 5
                        color = (0, 0, 255)  # Color en formato BGR (azul)
                        thickness = -1  # Relleno del círculo

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1
                        text_color = (255, 255, 255)  # Color en formato BGR (blanco)
                        text_thickness = 1
                        text_size, _ = cv2.getTextSize(str(4), font, font_scale, text_thickness)

                        # for i, point in enumerate(complete_points):
                        #     cv2.circle(self.color_image, (point[0], point[1]), radius, color, thickness)
                        #     cv2.putText(self.color_image, str(i), (point[0], point[1]), font, font_scale, color, text_thickness, cv2.LINE_AA)
                        keypoints.append(self.depth_point_to_xyz(complete_points, depth_image))
                        centers.append(neck_point)
        return centers, keypoints

    def calculate_orientation(self, people):
        people = np.asarray(people)
        orientations = []
        for person in people:
            left_v = person[1] - person[2]
            right_v = person[0] - person[2]

            normal = np.cross(left_v, right_v)

            vector_1 = [1,0]
            vector_2 = [normal[0], normal[1]]

            orientations.append(self.get_degrees_between_vectors(vector_1, vector_2))
        return orientations

    def get_degrees_between_vectors(self, v1, v2):
        uv1 = v1 / np.linalg.norm(v1)
        uv2 = v2 / np.linalg.norm(v2)

        # Vector extra que es el vector unitario 2 rotado -90º
        uv2_90 = [np.cos(-math.pi / 2) * uv2[0] - np.sin(-math.pi / 2) * uv2[1],
                  np.sin(-math.pi / 2) * uv2[0] + np.cos(-math.pi / 2) * uv2[1]]

        # Sacamos el producto de uv1 con uv2 y uv2_90
        dp = np.dot(uv1, uv2)
        dp_90 = np.dot(uv1, uv2_90)

        # Comprobamos si estamos en la zona dificil (zona mas alla de 180º)
        hard_side = True if dp_90 < 0 else False

        # Adaptamos el resultado
        if hard_side == False:
            ret = np.arccos(dp)
        else:
            # Zona dificil.
            ret = math.pi + (math.pi - np.arccos(dp))

        # Devolvemos en el formato indicado
        return ret

    def save_image(self, data):
        self.color_image = cv2.cvtColor(np.frombuffer(data.data, np.uint8).reshape(data.height, data.width, 4), cv2.COLOR_RGBA2RGB )
        
        # radius = 5
        # color = (0, 0, 255)  # Color en formato BGR (azul)
        # thickness = -1  # Relleno del círculo

        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 1
        # text_color = (255, 255, 255)  # Color en formato BGR (blanco)
        # text_thickness = 1
        # text_size, _ = cv2.getTextSize(str(4), font, font_scale, text_thickness)
        # keypoints = self.get_person_interest_keypoints(results)
        # orientation = self.calculate_orientation(keypoints[0])
        # for person_points in keypoints:
        #     for point in person_points:
        #         cv2.circle(self.color_image, (point[0], point[1]), radius, color, thickness)

        # cv2.rectangle(self.color_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)  # Masks object for segmentation masks outputs

    def save_depth(self, data):
        self.depth_image = np.frombuffer(data.data, np.float32).reshape(data.height, data.width, 1)

    def save_camera_info(self, data):
        self.camera_info = data

    def store_robot_pose(self, robot_pose):
        euler_rotation = tf.transformations.euler_from_quaternion([robot_pose.pose.pose.orientation.x, robot_pose.pose.pose.orientation.y, robot_pose.pose.pose.orientation.z, robot_pose.pose.pose.orientation.w])
        self.robot_orientation = euler_rotation[2]
        self.robot_world_transform_matrix = np.array([[math.cos(euler_rotation[2]), -math.sin(euler_rotation[2]), 0, robot_pose.pose.pose.position.x],
                        [math.sin(euler_rotation[2]), math.cos(euler_rotation[2]), 0, robot_pose.pose.pose.position.y],
                        [0, 0, 1, robot_pose.pose.pose.position.z],
                        [0,   0,   0,   1]])
        
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
    

    def get_yolo_objects(self, event):
        if self.new_data:
            if self.color_depth_ratio == None:
                self.color_depth_ratio_y = self.depth_image.shape[0] / self.color_image.shape[0]
                self.color_depth_ratio_x = self.depth_image.shape[1] / self.color_image.shape[1]
            results = self.model.predict(self.color_image, save=False)
            people_center, people_triangle = self.get_person_interest_keypoints(results, self.depth_image)
            people_orientation = self.calculate_orientation(people_triangle)
            people_bounding_boxes = self.get_bounding_boxes(results)
            self.create_interface_data(people_bounding_boxes, people_orientation, people_center)
            # self.color_image = self.display_data_tracks(self.color_image, self.objects_write, class_names=self.yolo_object_predictor.class_names)
            cv2.imshow("YOLO", self.color_image)
            cv2.imshow("DEPTH", self.depth_image.reshape(self.depth_image.shape[0], self.depth_image.shape[1], 1))
            cv2.waitKey(1)
    
    def create_interface_data(self, boxes, orientations, centers):
        objects = ObjectsMSG()
        if len(boxes) == len(orientations) == len(centers):
            for index in range(len(boxes)):
                act_object = Object()
                act_object.type = 0
                act_object.left = boxes[index][0]
                act_object.top = boxes[index][1]
                act_object.right = boxes[index][2]
                act_object.bot = boxes[index][3]
                act_object.score = 0.9
                # bbx_center_depth = [int((act_object.left + (act_object.right - act_object.left)/2)), int((act_object.top + (act_object.bot - act_object.top)/2))]
                person_pose = self.depth_point_to_xyz([[centers[index][0], centers[index][1]]], self.depth_image)
                world_person_pose = self.transform_pose_to_world_reference(person_pose)
                act_object.pose = Pose()
                act_object.pose.position.x = world_person_pose[0] 
                act_object.pose.position.y = world_person_pose[1]

                # if index < len(orientations):
                #     if not math.isnan(orientations[index])and not math.isinf(orientations[index]):
                #         yaw_to_euler = tf.transformations.quaternion_from_euler(0, 0, orientations[index])  
                #         act_object.pose.orientation = Quaternion(x=yaw_to_euler[0], y=yaw_to_euler[1], z=yaw_to_euler[2], w=yaw_to_euler[3])
                act_object.image = self.get_bbox_image_data(self.color_image, [act_object.left, act_object.top, act_object.right, act_object.bot])
                
                objects.objectsmsg.append(act_object)

            self.objects_publisher.publish(objects)

    def transform_pose_to_world_reference(self, person_pose):
        person_world_position = np.dot(self.robot_world_transform_matrix, np.array([person_pose[0][0], -person_pose[0][1], 0, 1])) 
        return [person_world_position[0], person_world_position[1]]

    def get_bbox_image_data(self, image, element_box):
        cropped_image = image[int(element_box[1]):int(element_box[3]), int(element_box[0]):int(element_box[2])]
        y, x, _ = cropped_image.shape
        return msg_Image(data=cropped_image.tobytes(), height=y, width=x)

    def depth_point_to_xyz(self, pixels, depth):
        depth_points = []
        for pixel in pixels:
            angle_y = ((math.pi - 1.01)/2) + (pixel[1]*1.01/640)
            angle_z = ((2*math.pi) - 0.785/2) + (pixel[0]*0.785/480)
            y_distance = depth[pixel[1]][pixel[0]] / math.tan(angle_y)
            z_distance = depth[pixel[1]][pixel[0]] * math.tan(angle_z)
            depth_points.append([depth[pixel[1]][pixel[0]][0], z_distance[0], y_distance[0]])
        return depth_points

    # def display_data_tracks(self, img, elements, class_names=None):
    #     num_rows, num_cols, _ = img.shape
    #     for i in elements:
    #         # if inds[i] == -1:
    #         #     continue
    #         x0 = int(i.left)
    #         y0 = int(i.top)
    #         x1 = int(i.right)
    #         y1 = int(i.bot)
    #         color = (_COLORS[i.type] * 255).astype(np.uint8).tolist()
    #         text = 'Class: {} - ID: {} - X: {} - Y: {}'.format(class_names[i.type], i.id, i.pose.position.x, i.pose.position.y)
    #         txt_color = (0, 0, 0) if np.mean(_COLORS[i.type]) > 0.5 else (255, 255, 255)
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
    #         cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    #         txt_bk_color = (_COLORS[i.type] * 255 * 0.7).astype(np.uint8).tolist()
    #         cv2.rectangle(
    #             img,
    #             (x0, y0 + 1),
    #             (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
    #             txt_bk_color,
    #             -1
    #         )
    #         cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    #     return img

if __name__ == '__main__':
    rospy.init_node("yolov8")
    rospy.loginfo("yolov8 node has been started")

    # rate = rospy.Rate(30)
    yolo = yolov8()
    # rospy.wait_for_service('bytetrack_srv')
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
