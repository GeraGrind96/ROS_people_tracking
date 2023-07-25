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
from geometry_msgs.msg import Pose, Twist
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2
import queue
import pyrealsense2

sys.path.append('/home/robocomp/software/TensorRT-For-YOLO-Series')
from utils.utils import BaseEngine

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
        # trt
        self.yolo_object_predictor = BaseEngine(engine_path='/home/gerardo/ros_projects/cohan_ws/src/yolov8/scripts/yolov8m.trt')
        self.image_queue = queue.Queue(1)
        self.objects_publisher = rospy.Publisher("/yolo_objects", ObjectsMSG, queue_size=10)
        self.objects_write = []
        self.objects_read = []
        self.camera_info = None

        self.depth_image = []
        self.color_image = []

        self.color_depth_ratio = None
        self.color_yolo_ratio_height = None
        self.color_yolo_ratio_width = None

    def save_image(self, data):
        self.color_image = cv2.cvtColor(np.frombuffer(data.data, np.uint8).reshape(data.height, data.width, 4), cv2.COLOR_RGBA2RGB )

    def save_depth(self, data):
        self.depth_image = np.frombuffer(data.data, np.float32).reshape(data.height, data.width, 1)

    def save_camera_info(self, data):
        self.camera_info = data
        
    def get_yolo_objects(self):
        if len(self.color_image) > 0 and len(self.depth_image) > 0:
            if self.color_depth_ratio == None:
                self.color_depth_ratio_y = self.depth_image.shape[0] / self.color_image.shape[0]
                self.color_depth_ratio_x = self.depth_image.shape[1] / self.color_image.shape[1]

        # if self.color_image != []:
            self.color_yolo_ratio_height = self.color_image.shape[0] / 640
            self.color_yolo_ratio_width = self.color_image.shape[1] / 640
            image_res = cv2.resize(self.color_image, (640, 640), interpolation = cv2.INTER_AREA)
            blob = self.pre_process(image_res, (640, 640))
            dets = self.yolov8_objects(blob)
            if dets is not None:
                self.create_interface_data(dets[:, :4], dets[:, 4], dets[:, 5])
                self.color_image = self.display_data_tracks(self.color_image, self.objects_write, class_names=self.yolo_object_predictor.class_names)
            cv2.imshow("YOLO", self.color_image)
            cv2.imshow("DEPTH", self.depth_image.reshape(self.depth_image.shape[0], self.depth_image.shape[1], 1))
            cv2.waitKey(1)

    def pre_process(self, image, input_size, swap=(2, 0, 1)):
        padded_img = np.ones((input_size[0], input_size[1], 3))
        img = np.array(image).astype(np.float32)
        padded_img[: int(img.shape[0]), : int(img.shape[1])] = img
        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img

    def yolov8_objects(self, blob):
        data = self.yolo_object_predictor.infer(blob)
        num, final_boxes, final_scores, final_cls_inds = data
        final_boxes = np.reshape(final_boxes, (-1, 4))
        dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1),
                               np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
        return dets  
    
    def create_interface_data(self, boxes, scores, cls_inds):
        aux_objects_write = []
        desired_inds = [i for i, cls in enumerate(cls_inds) if cls == 0]
                        #cls in self.classes]  # index of elements that match desired classes
        desired_scores = scores[desired_inds]
        desired_boxes = boxes[desired_inds]
        desired_clases = cls_inds[desired_inds]
        for index in range(len(desired_scores)):
            act_object = Object()
            act_object.type = int(desired_clases[index])
            act_object.left = int(desired_boxes[index][0] * self.color_yolo_ratio_width)
            act_object.top = int(desired_boxes[index][1] * self.color_yolo_ratio_height)
            act_object.right = int(desired_boxes[index][2] * self.color_yolo_ratio_width)
            act_object.bot = int(desired_boxes[index][3] * self.color_yolo_ratio_height)
            act_object.score = desired_scores[index]
            bbx_center_depth = [int((act_object.left + (act_object.right - act_object.left)/2) * self.color_depth_ratio_x), int((act_object.top + (act_object.bot - act_object.top)/2) * self.color_depth_ratio_y)]
            act_object.pose = Pose()
            act_object.pose.position.x, act_object.pose.position.y = self.depth_point_to_xy([bbx_center_depth[1], bbx_center_depth[0]], self.depth_image[bbx_center_depth[1]][bbx_center_depth[0]])
            act_object.image = self.get_bbox_image_data(self.color_image, [act_object.left, act_object.top, act_object.right, act_object.bot])
            aux_objects_write.append(act_object)
        # self.objects_write = aux_objects_write

        bytetrack_srv_proxy = rospy.ServiceProxy('bytetrack_srv', ObjectsSRV)
        try:
           aux_objects_write = bytetrack_srv_proxy(aux_objects_write)
           self.objects_write = aux_objects_write.res
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
        # swap
        self.objects_publisher.publish(self.objects_write)

    def get_bbox_image_data(self, image, element_box):
        cropped_image = image[int(element_box[1]):int(element_box[3]), int(element_box[0]):int(element_box[2])]
        y, x, _ = cropped_image.shape
        return msg_Image(data=cropped_image.tobytes(), height=y, width=x)

    def depth_point_to_xy(self, pixel, depth):
        angle_x = ((math.pi - 1.01)/2) + (pixel[1]*1.01/640)
        y_distance = depth / math.tan(angle_x)
        return depth, y_distance

    def display_data_tracks(self, img, elements, class_names=None):
        num_rows, num_cols, _ = img.shape
        for i in elements:
            # if inds[i] == -1:
            #     continue
            x0 = int(i.left)
            y0 = int(i.top)
            x1 = int(i.right)
            y1 = int(i.bot)
            color = (_COLORS[i.type] * 255).astype(np.uint8).tolist()
            text = 'Class: {} - ID: {} - X: {} - Y: {}'.format(class_names[i.type], i.id, i.pose.position.x, i.pose.position.y)
            txt_color = (0, 0, 0) if np.mean(_COLORS[i.type]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            txt_bk_color = (_COLORS[i.type] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

        return img

if __name__ == '__main__':
    rospy.init_node("yolov8")
    rospy.loginfo("yolov8 node has been started")

    rate = rospy.Rate(30)
    cvbridge = CvBridge()
    yolo = yolov8()
    rospy.wait_for_service('bytetrack_srv')
    rgb_subscriber = rospy.Subscriber("/xtion/rgb/image_raw", msg_Image, yolo.save_image)
    depth_subscriber = rospy.Subscriber("/xtion/depth/image_raw", msg_Image, yolo.save_depth)
    
    while not rospy.is_shutdown():        
        yolo.get_yolo_objects()
        try:
            rate.sleep()
        except KeyboardInterrupt:
           print("Shutting down")

    # rospy.logwarn("Warning test message")
    # rospy.logerr("Error test message")
    # rospy.loginfo("End of program")
