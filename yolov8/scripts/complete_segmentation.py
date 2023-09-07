#!/usr/bin/env python3
import rospy
import math
import sys
import os
from yolov8_data.msg import Object, ObjectsMSG
from yolov8_data.srv import *
from sensor_msgs.msg import *
from geometry_msgs.msg import *
from nav_msgs.msg import *
import numpy as np
import cupy as cp
from cv_bridge import CvBridge, CvBridgeError
import cv2
import queue
import yaml
import torch
import copy
import gc
from signal import signal, SIGINT
from collections import Counter
import argparse

import lap
from cython_bbox import bbox_overlaps as bbox_ious

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

package_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(package_folder + '/3rdparty/JointBDOE')
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

sys.path.append(package_folder + '/3rdparty/BOSCH-Age-and-Gender-Prediction/models')
from base_block import FeatClassifier, BaseClassifier
from resnet import resnet50
from collections import OrderedDict
import torchvision.transforms as T

class yolov8():
    def __init__(self, real_robot, compressed_image):
        self.image_queue = queue.Queue(1)
        self.objects_publisher = rospy.Publisher("/perceived_people", ObjectsMSG, queue_size=10)
        self.objects_write = []
        self.objects_read = []
        self.camera_info = None

        self.depth_image = []
        self.color_image = []     

        self.display = True
        self.real_robot = real_robot

        self.compressed_image = compressed_image
        if self.real_robot:
            self.width = 960
            self.height = 540
            self.vertical_FOV = 0.75
            self.horizontal_FOV = 1.22
        else:
            self.width = 640
            self.height = 480
            self.vertical_FOV = 0.785
            self.horizontal_FOV = 1.01

        self.color_depth_ratio = None
        self.color_yolo_ratio_height = None
        self.color_yolo_ratio_width = None

        self.new_data = False

        self.tf_listener = tf.TransformListener()

        self.cv_bridge = CvBridge()

        # CV text
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.7
        self.color = (255, 255, 0)
        self.thickness = 2

        # Models
        self.model_v8_pose = YOLO(package_folder +'/3rdparty/YOLO_models/yolov8n-pose.pt')
        self.model_v8_seg = YOLO(package_folder +'/3rdparty/YOLO_models/yolov8n-seg.engine')
        self.load_orientation_model()
        # self.init_person_classifier()

        if self.compressed_image:
            rgb_subscriber = message_filters.Subscriber("/rgb/compressed", CompressedImage)
            depth_subscriber = message_filters.Subscriber("/depth/compressed", CompressedImage)
        else:
            rgb_subscriber = message_filters.Subscriber("/xtion/rgb/image_raw", Image)
            depth_subscriber = message_filters.Subscriber("/xtion/depth/image_raw", Image)
        
        ts = message_filters.TimeSynchronizer([rgb_subscriber, depth_subscriber], 3)
        ts.registerCallback(self.store_data)
        rospy.spin()

################# SUBSCRIBER CALLBACKS #################

    # def store_data(self, rgb, depth, odom):
    def store_data(self, rgb, depth):
        if self.compressed_image:
            self.color_image = self.cv_bridge.compressed_imgmsg_to_cv2(rgb, "rgb8")
            self.depth_image = self.cv_bridge.compressed_imgmsg_to_cv2(depth)
        else:
            self.color_image = self.cv_bridge.imgmsg_to_cv2(depth, depth.encoding)
            self.depth_image = self.cv_bridge.imgmsg_to_cv2(depth, depth.encoding)
            
        self.get_yolo_objects()
            
################# DATA OBTAINING #################

    def get_people_data(self, img, depth):
        t1 = time.time()
        img0 = copy.deepcopy(img)
        img = letterbox(img, 640, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0       

        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Make inference with both models
        t2 = time.time()
        out_ori = self.model(img, augment=True, scales=[1])[0]
        t3 = time.time()  
        # out_v8_seg = self.model_v8_seg.predict(img0, classes=0, show_conf=True)
        out_v8_pose = self.model_v8_pose.predict(img0, classes=0, show_conf=True)
              
        # YOLO V8 data processing
        # bboxes, confidences, poses, masks = self.get_segmentator_data(out_v8_seg, img0, depth, robot_trans_matrix)
        bboxes, confidences, poses, masks = self.get_pose_data(out_v8_pose, img0, depth)

        # # Orientation model data processing

        out = non_max_suppression(out_ori, 0.3, 0.5, num_angles=self.data['num_angles'])
        orientation_bboxes = scale_coords(img.shape[2:], out[0][:, :4], img0.shape[:2]).cpu().numpy().astype(int)  # native-space pred
        orientations = (out[0][:, 6:].cpu().numpy() * 360) - 180   # N*1, (0,1)*360 --> (0,360)
        
        # Hungarian algorithm for matching people from segmentation model and orientation model

        matches = self.associate_orientation_with_segmentation(bboxes, orientation_bboxes)

        people_poses = []
        for i in range(len(matches)):
            pose_respect_to_camera = poses[matches[i][0]]
            orientation = tf.transformations.quaternion_from_euler(0, 0, math.radians(orientations[matches[i][1]][0]) - math.pi)    
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = rospy.Time(0)
            pose_stamped.header.frame_id = "base_footprint" 
            pose = Pose()
            pose.orientation = Quaternion(x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3])
            pose.position.x = pose_respect_to_camera[0]
            pose.position.y = -pose_respect_to_camera[1]
            pose.position.z = 0.0
            pose_stamped.pose = pose
            transformed_pose = self.tf_listener.transformPose("map", pose_stamped)
            people_poses.append(transformed_pose)
        if len(bboxes) == 0:
            return [], [], [], []
        return bboxes, confidences, people_poses, masks

    def get_pose_data(self, result, color_image, depth_image):
        pose_bboxes = []
        pose_poses = []
        pose_confidences = []
        pose_masks = []
        for result in result:
            if result.keypoints != None and result.boxes != None:
                boxes = result.boxes
                keypoints = result.keypoints.xy.cpu().numpy().astype(int)
                if len(keypoints) == len(boxes):
                    for i in range(len(keypoints)):
                        person_confidence = boxes[i].conf.cpu().numpy()[0]
                        if person_confidence > 0.8:
                            person_bbox = boxes[i].xyxy.cpu().numpy().astype(int)[0] 
                            if len(keypoints[i]) > 0: 
                                interesting_points_color = [keypoints[i][5], keypoints[i][6], keypoints[i][7], keypoints[i][8], keypoints[i][11], keypoints[i][12], keypoints[i][13], keypoints[i][14]]
                                interesting_points_pose = [keypoints[i][5], keypoints[i][6], keypoints[i][11], keypoints[i][12]]
                                # valid_pose = True
                                # for j, keypoint in enumerate(keypoints[i]):
                                #     cv2.circle(color_image, (keypoint[0], keypoint[1]), 2, (0, 255, 0), 1)
                                #     color_image = cv2.putText(color_image, str(j), (int(keypoint[0]), int(keypoint[1])), self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA)
                                #     keypoint[0] = keypoint[0] - 1 if keypoint[0] >= self.height else keypoint[0]
                                #     keypoint[1] = keypoint[1] - 1 if keypoint[1] >= self.width else keypoint[1]
                                #     if np.isinf(depth_image[keypoint[1], keypoint[0]]) or depth_image[keypoint[1], keypoint[0]] == 0:
                                #         valid_pose = False
                                #         break
                                # if not valid_pose:
                                #     print("REMOVED FOR NON VALID POSE")
                                #     continue
                                x_avg_up = (interesting_points_pose[0][0] + interesting_points_pose[1][0]) / 2
                                y_avg_up = (interesting_points_pose[0][1] + interesting_points_pose[1][1]) / 2
                                # x_avg_down = (interesting_points_pose[2][0] + interesting_points_pose[3][0]) / 2
                                # y_avg_down = (interesting_points_pose[2][1] + interesting_points_pose[3][1]) / 2
                                # # if x_avg < 40 or x_avg > self.width - 40:
                                # #     continue
                                neck_point = np.array([x_avg_up, y_avg_up]).astype(int)
                                # back_point = np.array([x_avg_down, y_avg_down]).astype(int)
                                # gender_pred, age_pred = self.get_pred_attributes(frame, person_bbox[0], person_bbox[1], person_bbox[2], person_bbox[3])
                                person_pose_up = self.get_neck_distance(neck_point, depth_image, interesting_points_pose[0], interesting_points_pose[1])
                                # person_pose_down = self.get_neck_distance(back_point, depth_image)
                                # print("PERSON POSE UP:", person_pose_up)
                                # print("PERSON POSE DOWN:", person_pose_down)
                                if np.isinf(person_pose_up[0]) or person_pose_up[0] == 0 or person_pose_up[0] > 5:
                                    # print("REMOVED FOR NON VALID POSE 2")
                                    continue                            
                                pose_poses.append(person_pose_up)
                                pose_bboxes.append(person_bbox)
                                pose_confidences.append(person_confidence)   
                                color_lines = [[interesting_points_color[0], interesting_points_color[1]], [interesting_points_color[3], interesting_points_color[1]], [interesting_points_color[0], interesting_points_color[2]], [interesting_points_color[4], interesting_points_color[6]], [interesting_points_color[5], interesting_points_color[7]]]
                                person_mask = self.get_most_common_color(color_lines, color_image)
                                # cv2.imshow("MASK", person_mask)
                                pose_masks.append(person_mask)
                        else:
                            # print("REMOVED FOR NOT ENOUGHT CONFIDENCE")
                            continue
        # cv2.imshow("SKELETONS", color_image)
        # cv2.waitKey(1)
        return pose_bboxes, pose_confidences, pose_poses, pose_masks

    def get_neck_distance(self, neck_point, depth_image, point_a, point_b):
        depth_mean = 0
        max_dist = 7
        counter = 0  
        range_vector_x = np.array([point_a[0], point_b[0]])
        range_vector_y = np.array([point_a[1], point_b[1]])
        max_value_x, min_value_x = np.max(range_vector_x), np.min(range_vector_x)
        max_value_y, min_value_y = np.max(range_vector_y), np.min(range_vector_y)
        last_depth_value = None
        for x in range(np.clip(neck_point[0] - max_dist, min_value_x, max_value_x), np.clip(neck_point[0] + max_dist + 1, min_value_x, max_value_x)):
            for y in range(np.clip(neck_point[1] - max_dist, min_value_y, max_value_y), np.clip(neck_point[1] + max_dist + 1, min_value_y, max_value_y)):
                depth_value = int(depth_image[y, x])
                if not np.isinf(depth_value) and depth_value != 0: 
                    if last_depth_value is None:
                        last_depth_value = depth_value
                        depth_mean += depth_value
                        counter += 1
                    elif abs(depth_value - last_depth_value) > 100:
                        if depth_value >= last_depth_value:
                            counter, depth_mean, last_depth_value = 1, depth_value, depth_value
                        else:
                            continue
                    else:
                        depth_mean += depth_value
                        counter += 1

        if counter > 0:            
            neck_point_3d = self.depth_point_to_xyz(neck_point, (depth_mean / (counter * 1000) ))
            return neck_point_3d
        else:
            return [np.inf, np.inf, np.inf]

    def get_segmentator_data(self, results, color_image, depth_image, robot_trans_matrix):
        segmentation_bboxes = []
        segmentation_poses = []
        segmentation_confidences = []
        segmentation_masks = []
        for result in results:
            if result.masks != None and result.boxes != None:
                masks = result.masks.xy
                boxes = result.boxes
                if len(masks) == len(boxes):
                    for i in range(len(boxes)):
                        # t1 = time.time()
                        # print("t1:", time.time() - t1)
                        # t2 = time.time()    
                        image_mask = np.zeros((self.height, self.width, 1), dtype=np.uint8)
                        act_mask = masks[i].astype(np.int32)
                        # print("t2:", time.time() - t2)
                        # t3 = time.time()
                        cv2.fillConvexPoly(image_mask, act_mask, (1, 1, 1))
                        # print("t3:", time.time() - t3)
                        t4 = time.time()
                        person_bbox = boxes[i].xyxy.cpu().numpy().astype(int)[0]
                        image_mask = image_mask[person_bbox[1]:person_bbox[3], person_bbox[0]:person_bbox[2]]
                        height, width, _ = image_mask.shape
                        depth_image_mask = depth_image[person_bbox[1]:person_bbox[3], person_bbox[0]:person_bbox[2]]
                        color_image_mask = color_image[person_bbox[1]:person_bbox[3], person_bbox[0]:person_bbox[2]]
                        person_pose = self.get_mask_distance(image_mask[:height // 5, :], depth_image_mask[:height // 5, :], robot_trans_matrix, person_bbox)
                        if np.isinf(person_pose[0]):
                            continue
                        segmentation_poses.append(person_pose)
                        segmentation_bboxes.append(person_bbox)
                        segmentation_confidences.append(boxes[i].conf.cpu().numpy()[0])
                        
                        
                        t5 = time.time()

                        person_mask = self.get_mask_with_modified_background(image_mask, person_bbox, color_image_mask)
                        segmentation_masks.append(person_mask)
        return segmentation_bboxes, segmentation_confidences, segmentation_poses, segmentation_masks

    def get_mask_with_modified_background(self, mask, bbox, image):
        masked_image = mask * image
        h, w, _ = masked_image.shape
        is_black_pixel = np.logical_and(masked_image[:, :, 0] == 0, masked_image[:, :, 1] == 0, masked_image[:, :, 2] == 0)
        masked_image[is_black_pixel] = [255, 255, 255]
        return masked_image

    def get_mask_with_pose(self, color, bbox, image):
        image_mask = np.zeros((bbox[2]-bbox[0], bbox[3]-bbox[1], 3), dtype=np.uint8)
        image_mask[:] = color
        return image_mask

    def get_most_common_color(self, lines, color_image):
        total_points = []
        for line in lines:
            init = line[0]
            end = line[1]
            init[0] = init[0] if init[0] < self.width else self.width - 1
            end[0] = end[0] if end[0] < self.width else self.width - 1
            init[1] = init[1] if init[1] < self.height else self.height - 1
            end[1] = end[1] if end[1] < self.height else self.height - 1
            puntos_en_linea = np.linspace(init, end, num=20, dtype=np.int32)
            total_points.extend(puntos_en_linea)
        total_points = np.stack(total_points)
        imagen = np.ones((10, len(total_points), 3), dtype=np.uint8) * 255
        for i, punto in enumerate(total_points):
            imagen[:, i] = tuple(color_image[punto[1], punto[0]])
        return imagen

    def get_mask_distance(self, mask, depth_image, robot_trans_matrix, bbox):
            
        segmentation_pixels = np.argwhere(np.all(mask == 1, axis=-1))
        segmentation_points = np.column_stack((segmentation_pixels[:, 1], segmentation_pixels[:, 0]))
        valid_points_mask = ~np.isinf(depth_image[segmentation_points[:, 1], segmentation_points[:, 0]])
        valid_points = segmentation_points[valid_points_mask.flatten()]
        if len(valid_points) > 0:
            mean_point = np.mean(valid_points, axis=0)
            depth_values = depth_image[valid_points[:, 1], valid_points[:, 0]]
            mean_depth = np.mean(depth_values)
            xyz_points = self.depth_point_to_xyz([mean_point[0] + bbox[0], mean_point[1]+ bbox[1]], mean_depth)
            world_pose = self.transform_pose_to_world_reference(xyz_points, robot_trans_matrix)
            return world_pose
        else:
            return [np.inf, np.inf]
        
    def associate_orientation_with_segmentation(self, seg_bboxes, ori_bboxes):
        dists = self.iou_distance(seg_bboxes, ori_bboxes)
        matches, unmatched_a, unmatched_b = self.linear_assignment(dists, 0.9)
        return matches

    def linear_assignment(self, cost_matrix, thresh):
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        matches, unmatched_a, unmatched_b = [], [], []
        cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
        matches = np.asarray(matches)
        return matches, unmatched_a, unmatched_b

    def iou_distance(self, atracks, btracks):
        if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
            atlbrs = atracks
            btlbrs = btracks
        else:
            atlbrs = [track.tlbr for track in atracks]
            btlbrs = [track.tlbr for track in btracks]
        _ious = self.ious(atlbrs, btlbrs)
        cost_matrix = 1 - _ious

        return cost_matrix

    def ious(self, atlbrs, btlbrs):
        ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
        if ious.size == 0:
            return ious

        ious = bbox_ious(
            np.ascontiguousarray(atlbrs, dtype=float),
            np.ascontiguousarray(btlbrs, dtype=float)
        )

        return ious

    def get_yolo_objects(self):
        depth_image = self.depth_image
        color_image = self.color_image
        bboxes, scores, poses, masks = self.get_people_data(color_image, depth_image)
        self.create_interface_data(bboxes, poses, scores, masks)

################# DATA STRUCTURATION #################

    def create_interface_data(self, boxes, centers, scores, masks):
        objects = ObjectsMSG()
        objects.header.stamp = rospy.Time.now()
        if len(boxes) == len(centers) == len(scores) == len(masks):
            for index in range(len(boxes)):
                act_object = Object()
                act_object.type = 0
                act_object.left = boxes[index][0]
                act_object.top = boxes[index][1]
                act_object.right = boxes[index][2]
                act_object.bot = boxes[index][3]
                act_object.score = scores[index]
                act_object.pose = centers[index]
                y, x, _ = masks[index].shape
                act_object.image = Image(data=masks[index].tobytes(), height=y, width=x)
                
                objects.objectsmsg.append(act_object)
            self.objects_publisher.publish(objects)

    def get_bbox_image_data(self, image, element_box):
        cropped_image = image[int(element_box[1]):int(element_box[3]), int(element_box[0]):int(element_box[2])]
        y, x, _ = cropped_image.shape
        return Image(data=cropped_image.tobytes(), height=y, width=x)

################# TO WORLD TRANSFORMATIONS #################

    def transform_pose_to_world_reference(self, person_pose, robot_trans_matrix):
        person_world_position = np.dot(robot_trans_matrix, np.array([person_pose[0], -person_pose[1], 0, 1]))
        person_pose = PoseStamped()
        person_pose.pose.position.x = person_pose[0]
        person_pose.pose.position.y = -person_pose[1]
        person_pose.pose.position.z = 0
        transformed_point = self.tf_listener.transformPoint("map", person_pose)
        return [round(person_world_position[0], 3), round(person_world_position[1], 3)]
    
################# IMAGE POINTS TO DEPTH #################

    def depth_point_to_xyz(self, pixel, depth):
        angle_y = ((math.pi - self.vertical_FOV)/2) + (pixel[1]*self.vertical_FOV/self.height)
        angle_z = ((2*math.pi) - self.horizontal_FOV/2) + (pixel[0]*self.horizontal_FOV/self.width)
        y_distance = depth / math.tan(angle_y)
        z_distance = depth * math.tan(angle_z)
        return depth, z_distance, y_distance
    
    def calculate_depth_with_projection(self, projected_point):
        world_y = 579.65506 * 1.2 / (projected_point[1] - 243.0783)
        world_x = world_y * (projected_point[0] - 317.47191) / 577.55158
        return [world_y, world_x]
    
################# PERSON ATTRIBUTES #################

    def init_person_classifier(self):
        self.age_range = [[10, 30], [30, 45], [45, 50], [50, 70]]
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.valid_tsfm = T.Compose([
            T.Resize((256, 192)),
            T.ToTensor(),
            normalize
        ])
        backbone = resnet50()
        classifier = BaseClassifier(nattr=35)
        self.age_classification_model = FeatClassifier(backbone, classifier)

        if torch.cuda.is_available():
            self.age_classification_model = torch.nn.DataParallel(self.age_classification_model).cuda()
        else:
            print("AGE CLASSIFICATION MODEL CAN'T BE EXECUTED WITH CUDA")

        self.load_age_predictor_state_dict(self.age_classification_model)

    def get_pred_attributes(self, frame, x1, y1, x2, y2):
        img = frame[y1:y2, x1:x2]
        img = Image.fromarray(img)
        img = self.valid_tsfm(img)
        valid_logits = self.age_classification_model(img.unsqueeze(0))
        valid_probs = torch.sigmoid(valid_logits)
        
        age_pred = self.age_range[torch.argmax(valid_probs[0][0:-1])]
        gender_pred = "M" if valid_probs[0][-1] > 0.5 else "F"

        return gender_pred, age_pred

    def load_age_predictor_state_dict(self, model):

        PATH_TO_AGE_GENDER_PREDICTOR_CHECKPOINT = package_folder + '/3rdparty/BOSCH-Age-and-Gender-Prediction/exp_result/PETA/PETA/img_model/ckpt_max.pth'

        loaded = torch.load(PATH_TO_AGE_GENDER_PREDICTOR_CHECKPOINT, map_location=torch.device("cuda:0"))

        if not torch.cuda.is_available():
            # remove `module.`
            new_state_dict = OrderedDict()
            for k, v in loaded['state_dicts'].items():
                name = k[7:] 
                new_state_dict[name] = v

            # load parameters
            model.load_state_dict(new_state_dict, strict=False)
        else:      
            model.load_state_dict(loaded['state_dicts'], strict=False)
        
        print("Load successful")
        model = model.eval()

################# PERSON ATTRIBUTES #################

    def load_orientation_model(self):
        self.device = select_device("0", batch_size=1)
        self.model = attempt_load(package_folder + "/3rdparty/JointBDOE/runs/JointBDOE/coco_s_1024_e500_t020_w005/weights/best.pt", map_location=self.device)
        self.stride = int(self.model.stride.max())
        with open(package_folder + "/3rdparty/JointBDOE/data/JointBDOE_weaklabel_coco.yaml") as f:
            self.data = yaml.safe_load(f)  # load data dict

def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    gc.collect()
    torch.cuda.empty_cache()
    exit(0)

################# MAIN #################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_robot', type=int)
    parser.add_argument('--compressed_image', type=int)
    args, unknown = parser.parse_known_args()
    real_robot = args.real_robot
    compressed_image = args.compressed_image

    if real_robot:
        real_robot = True
        print("Real robot setted")
    else:
        print("Real robot not setted")

    if compressed_image:
        compressed_image = True
        print("Using compressed image")
    else:
        print("Compressed image not used")

    rospy.init_node("yolov8")
    rospy.loginfo("yolov8 node has been started")
    signal(SIGINT, handler)
    yolo = yolov8(real_robot, compressed_image)


