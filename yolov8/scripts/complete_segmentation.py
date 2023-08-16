#!/usr/bin/env python3
import rospy
import math
import sys
import os
from yolov8_data.msg import Object, ObjectsMSG
from yolov8_data.srv import *
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo as msg_CameraInfo
from sensor_msgs.msg import CompressedImage as msg_CompressedImage
from sensor_msgs.msg import PointCloud2 as msg_PointCloud2
from geometry_msgs.msg import *
from nav_msgs.msg import *
import numpy as np
import cupy as cp
import cv2
import queue
import yaml
from PIL import Image
import torch
import copy
import gc
from signal import signal, SIGINT

import lap
from cython_bbox import bbox_overlaps as bbox_ious

torch.cuda.empty_cache() 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"

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

sys.path.append('/home/gerardo/software/BOSCH-Age-and-Gender-Prediction/models')
from base_block import FeatClassifier, BaseClassifier
from resnet import resnet50
from collections import OrderedDict
import torchvision.transforms as T

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

        # self.init_person_classifier()

        self.load_orientation_model()

        self.width = 640
        self.height = 480

        self.color_depth_ratio = None
        self.color_yolo_ratio_height = None
        self.color_yolo_ratio_width = None

        self.new_data = False

        self.yolo_model_name = 'yolov8n-seg.engine'

        self.model_v8 = YOLO(self.yolo_model_name)

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

    def get_people_data(self, img, depth, robot_trans_matrix, robot_orientation):
        t1 = time.time()
        img0 = copy.deepcopy(img)
        img = letterbox(img, 640, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        print("T1:", time.time() - t1)
        t2 = time.time()

        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Make inference with both models
        
        out_ori = self.model(img, augment=True, scales=[1])[0]
        out_v8 = self.model_v8.predict(img0, classes=0, show_conf=True)
        print("T2:", time.time() - t2)
        t3 = time.time()        
        # YOLO V8 data processing
        bboxes, confidences, poses, masks = self.get_segmentator_data(out_v8, img0, depth, robot_trans_matrix)
        print("T3:", time.time() - t3)
        t4 = time.time()    
        # print("T3 T4:", t4 - t3)    
        # Orientation model data processing

        out = non_max_suppression(out_ori, 0.3, 0.5, num_angles=self.data['num_angles'])
        orientation_bboxes = scale_coords(img.shape[2:], out[0][:, :4], img0.shape[:2]).cpu().numpy().astype(int)  # native-space pred
        orientations = (out[0][:, 6:].cpu().numpy() * 360) - 180   # N*1, (0,1)*360 --> (0,360)
        
        # Hungarian algorithm for matching people from segmentation model and orientation model

        matches = self.associate_orientation_with_segmentation(orientation_bboxes, bboxes)

        associated_orientations = []
        for i in range(len(matches)):
            for j in range(len(matches)):
                if i == matches[j][1]:
                    # transformed_pose = tf.transformations.quaternion_from_euler(0, 0, math.radians(orientations[matches[j][0]][0]) - math.pi) 
                    # transformed_pose_quaternion = Quaternion(x=transformed_pose[0], y=transformed_pose[1], z=transformed_pose[2], w=transformed_pose[3])
                    # associated_orientations.append(transformed_pose_quaternion)
                    associated_orientations.append(self.transform_orientation_to_world_reference(math.radians(orientations[matches[j][0]][0]), robot_orientation))
                    break
        print("T4:", time.time() - t4)
        if len(bboxes) == 0:
            return [], [], [], [], []
        return bboxes, confidences, associated_orientations, poses, masks

    def get_pose_data(self, result, depth_image, robot_trans_matrix, frame):
        pose_bboxes = []
        pose_poses = []
        pose_confidences = []
        for result in result:
            if result.keypoints != None and result.boxes != None:
                boxes = result.boxes
                keypoints = result.keypoints.xy.cpu().numpy().astype(int)
                if len(keypoints) == len(boxes):
                    for i in range(len(keypoints)):
                        person_bbox = boxes[i].xyxy.cpu().numpy().astype(int)[0] 
                        if len(keypoints[i]) > 0: 
                            x_avg = (keypoints[i][5, 0] + keypoints[i][6, 0]) / 2
                            y_avg = (keypoints[i][5, 1] + keypoints[i][6, 1]) / 2
                            if x_avg < 100 or x_avg > self.width - 100:
                                continue
                            neck_point = np.array([x_avg, y_avg]).astype(int)
                            gender_pred, age_pred = self.get_pred_attributes(frame, person_bbox[0], person_bbox[1], person_bbox[2], person_bbox[3])
                            person_pose = self.get_neck_distance(neck_point, depth_image, robot_trans_matrix)
                            pose_poses.append(person_pose)
                            pose_bboxes.append(person_bbox)
                            pose_confidences.append(boxes[i].conf.cpu().numpy()[0])   
        return pose_bboxes, pose_confidences, pose_poses


    def get_neck_distance(self, neck_point, depth_image, robot_trans_matrix):
        neck_point[0] = neck_point[0] - 1 if neck_point[0] >= self.height else neck_point[0]
        neck_point[1] = neck_point[1] - 1 if neck_point[1] >= self.width else neck_point[1]
        if not np.isinf(depth_image[neck_point[1], neck_point[0]]):
            neck_point_3d = self.depth_point_to_xyz(neck_point, depth_image[neck_point[1], neck_point[0]])
            world_neck_point_3d = self.transform_pose_to_world_reference(neck_point_3d, robot_trans_matrix)
            return world_neck_point_3d
        else:
            return [np.inf, np.inf]

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
                        print("IN LOOP")
                        # t1 = time.time()
                        # print("t1:", time.time() - t1)
                        # t2 = time.time()    
                        image_mask = np.zeros((480, 640, 1), dtype=np.uint8)
                        act_mask = masks[i].astype(np.int32)
                        # print("t2:", time.time() - t2)
                        # t3 = time.time()
                        cv2.fillConvexPoly(image_mask, act_mask, (1, 1, 1))
                        # print("t3:", time.time() - t3)
                        # t4 = time.time()
                        image_mask = cp.array(image_mask)
                        # rectangle_mask = np.zeros_like(image_mask)
                        # rectangle_mask[person_bbox[0]:person_bbox[2]+1,
                        #             person_bbox[1]:person_bbox[3]+1] = 1
                        # filtered_mask = image_mask & rectangle_mask
  
                        person_pose = self.get_mask_distance(image_mask, depth_image, robot_trans_matrix, [])
                        if np.isinf(person_pose[0]):
                            print("POSE INFINITA")
                            continue
                        segmentation_poses.append(person_pose)

                        person_bbox = boxes[i].xyxy.cpu().numpy().astype(int)[0]
                        segmentation_bboxes.append(person_bbox)
                        segmentation_confidences.append(boxes[i].conf.cpu().numpy()[0])
                        
                        # print("t4:", time.time() - t4)
                        # t5 = time.time()
                        color_image = cp.array(color_image)
                        person_mask = self.get_mask_with_modified_background(image_mask, person_bbox, color_image)
                        segmentation_masks.append(cp.asnumpy(person_mask))
                        # print("t5:", time.time() - t5)
                        # print("END LOOP")
        return segmentation_bboxes, segmentation_confidences, segmentation_poses, segmentation_masks

    def get_mask_with_modified_background(self, mask, bbox, image):
        masked_image = mask * image
        roi = masked_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        h, w, _ = roi.shape
        # black_pixels = np.where(np.all(roi == [0, 0, 0], axis=-1))

        background_color = roi[h // 2, w // 2]
        is_black_pixel = cp.logical_and(roi[:, :, 0] == 0, roi[:, :, 1] == 0, roi[:, :, 2] == 0)
    
        roi[is_black_pixel] = background_color

        # non_black_pixels = np.where(np.all(roi != [0, 0, 0], axis=-1))
        # non_black_values = roi[non_black_pixels]

        # print(non_black_values)

        # # Most common color
        # red_channel = non_black_values[:,0]
        # green_channel = non_black_values[:,1]
        # blue_channel = non_black_values[:,2]

        # # Calcula los histogramas de cada canal de color
        # red_histogram = np.histogram(red_channel, bins=np.arange(0, 256))
        # green_histogram = np.histogram(green_channel, bins=np.arange(0, 256))
        # blue_histogram = np.histogram(blue_channel, bins=np.arange(0, 256))

        # # Encuentra los valores de color más comunes en cada canal de color
        # most_common_red = np.argmax(red_histogram[0])
        # most_common_green = np.argmax(green_histogram[0])
        # most_common_blue = np.argmax(blue_histogram[0])

        # most_common_color = (most_common_red, most_common_green, most_common_blue)

        # print("Color RGB más común:", most_common_color)
        
        # Mean of segmentation colors
        # average_color = np.mean(non_black_values, axis=0)

        # T-shirt color
        # background_color = roi[int(h / 3), int(w / 2)]
        # roi[black_pixels] = background_color
        return roi

    def get_mask_distance(self, mask, depth_image, robot_trans_matrix, bbox):
        depth_image = cp.array(depth_image)
        segmentation_pixels = cp.argwhere(cp.all(mask == 1, axis=-1))
        segmentation_points = cp.column_stack((segmentation_pixels[:, 1], segmentation_pixels[:, 0]))
        valid_points_mask = ~cp.isinf(depth_image[segmentation_points[:, 1], segmentation_points[:, 0]])
        valid_points = cp.compress(valid_points_mask, segmentation_points, axis=0)
        mean_point = cp.asnumpy(cp.mean(valid_points, axis=0))
        if len(valid_points) > 0:
            depth_values = depth_image[valid_points[:, 1], valid_points[:, 0]]
            mean_depth = float(cp.asnumpy(cp.mean(depth_values)))
            print(mean_depth, mean_point)
            xyz_points = self.depth_point_to_xyz([mean_point[0], mean_point[1]], mean_depth)
            world_pose = self.transform_pose_to_world_reference(xyz_points, robot_trans_matrix)
            return cp.asnumpy(world_pose)
        else:
            return [np.inf, np.inf]
        
        # center_point = np.mean(first_20_rows, axis=0)
        # print(bbox)
        # x_central = bbox[0] + ((bbox[2] - bbox[0]) / 2)
        # y_central = bbox[1] + ((bbox[3] - bbox[1]) / 2) 
        # center_point = [x_central, y_central]
        # # print(center_point)
        # # center_depth = self.transform_pose_to_world_reference(self.depth_point_to_xyz(center_point, depth_image[int(center_point[1]), int(center_point[0])]), robot_trans_matrix)
        # center_depth = self.transform_pose_to_world_reference(self.depth_point_to_xyz(center_point, depth_image[int(center_point[1]), int(center_point[0])]), robot_trans_matrix)
        # return center_depth

    def associate_orientation_with_segmentation(self, seg_bboxes, ori_bboxes):
        dists = self.iou_distance(seg_bboxes, ori_bboxes)
        matches, unmatched_a, unmatched_b = self.linear_assignment(dists, 0.9)
        # print("NO MATCHES A", unmatched_a)
        # print("NO MATCHES b", unmatched_b)
        # print("MATCHES", matches)
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

    # def get_people_pose(self, people_bboxes, depth_image):
    #     radius = 5
    #     color = (0, 0, 255)  # Color en formato BGR (azul)
    #     thickness = -1  # Relleno del círculo
    #     people_poses = []
    #     for person_bbox in people_bboxes:
    #         cv2.rectangle(self.color_image, (int(person_bbox[0]), int(person_bbox[1])), (int(person_bbox[2]), int(person_bbox[3])), (255, 0, 0), 2)
    #         x_range = int(person_bbox[0] + (person_bbox[2] - person_bbox[0]) / 2)
    #         y_range = int(person_bbox[1] + (person_bbox[3] - person_bbox[1]) / 5)
    #         # image_section = depth_image[int(person_bbox[3] / 5):int(person_bbox[3] / 4), x_range]
    #         # print(image_section)
    #         # # print(image_section)
    #         # if image_section.size > 0: 
    #         #     min_value = np.unravel_index(np.argmin(image_section), image_section.shape)  
    #         # else: 
    #         #     continue

    #         if math.isinf(depth_image[y_range][x_range]):
    #             image_section = depth_image[y_range, int(person_bbox[0]):int(person_bbox[2])]
    #             print("SECTION X", int(person_bbox[2] / 4), int(person_bbox[2] * 3 / 4))
    #             # print(image_section)
    #             if image_section.size > 0: 
    #                 print(image_section.shape)
    #                 min_value = np.unravel_index(np.argmin(image_section), image_section.shape)  
    #                 if not math.isinf(depth_image[y_range][min_value[0]]): 
    #                     print("Min value:", min_value)
    #                     from_robot_pose = self.depth_point_to_xyz([min_value[0] + person_bbox[0], y_range], depth_image[y_range][min_value[0] + person_bbox[0]])
    #                     cv2.circle(self.color_image, (min_value[0] + person_bbox[0], y_range), radius, color, thickness)
    #                 else:
    #                     print("PROJECTED POINT:", x_range, person_bbox[3])
    #                     cv2.circle(self.color_image, (x_range, person_bbox[3]), radius, color, thickness)
    #                     from_robot_pose = self.calculate_depth_with_projection([x_range, person_bbox[3]])
    #             else: 
    #                 print("PROJECTED POINT:", x_range, person_bbox[3])
    #                 cv2.circle(self.color_image, (x_range, person_bbox[3]), radius, color, thickness)
    #                 from_robot_pose = self.calculate_depth_with_projection([x_range, person_bbox[3]])
    #         else:
    #             print("DEPTH POINT:", x_range, y_range)
    #             cv2.circle(self.color_image, (x_range, y_range), radius, color, thickness)
    #             from_robot_pose = self.depth_point_to_xyz([x_range, y_range], depth_image[y_range][x_range])
    #         world_person_pose = self.transform_pose_to_world_reference(from_robot_pose)
    #         people_poses.append(world_person_pose)
    #     return people_poses
    
    def get_yolo_objects(self, event):
        if self.new_data:
            init = time.time()
            depth_image = self.depth_image
            color_image = self.color_image
            robot_trans_matrix = self.robot_world_transform_matrix
            robot_orientation = self.robot_orientation
            bboxes, scores, orientations, poses, masks = self.get_people_data(color_image, depth_image, robot_trans_matrix, robot_orientation)
            self.create_interface_data(bboxes, orientations, poses, scores, masks)
            self.new_data = False
            print("EXPENDED TIME:", time.time() - init)

################# DATA STRUCTURATION #################

    def create_interface_data(self, boxes, orientations, centers, scores, masks):
        objects = ObjectsMSG()
        objects.header.stamp = rospy.Time.now()
        if len(boxes) == len(orientations) == len(centers) == len(scores) == len(masks):
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

                act_object.pose.orientation = orientations[index]
                y, x, _ = masks[index].shape
                act_object.image = msg_Image(data=masks[index].tobytes(), height=y, width=x)

                # output_folder = 'dataset_' + str(1)
                # os.makedirs(output_folder, exist_ok=True)
                # num_existing_files = len(os.listdir(output_folder))
                # output_path = os.path.join(output_folder, str(num_existing_files) + ".png")
                # cv2.imwrite(output_path, masks[index])
                
                objects.objectsmsg.append(act_object)
            self.objects_publisher.publish(objects)

    def get_bbox_image_data(self, image, element_box):
        cropped_image = image[int(element_box[1]):int(element_box[3]), int(element_box[0]):int(element_box[2])]
        y, x, _ = cropped_image.shape
        return msg_Image(data=cropped_image.tobytes(), height=y, width=x)

################# TO WORLD TRANSFORMATIONS #################

    def transform_pose_to_world_reference(self, person_pose, robot_trans_matrix):
        # print(person_pose)
        # person_world_position = np.dot(self.camera_pose_respect_robot, np.dot(robot_trans_matrix, np.array([person_pose[0], -person_pose[1], 0, 1])))
        person_world_position = np.dot(robot_trans_matrix, np.array([person_pose[0], -person_pose[1], 0, 1]))

        return [round(person_world_position[0], 3), round(person_world_position[1], 3)]
    
    def transform_orientation_to_world_reference(self, orientation, robot_orientation):
        theta_world = robot_orientation + orientation
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

        PATH_TO_AGE_GENDER_PREDICTOR_CHECKPOINT = '/home/gerardo/software/BOSCH-Age-and-Gender-Prediction/exp_result/PETA/PETA/img_model/ckpt_max.pth'

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
        self.model = attempt_load("/home/gerardo/software/JointBDOE/runs/JointBDOE/coco_s_1024_e500_t020_w005/weights/best.pt", map_location=self.device)
        self.stride = int(self.model.stride.max())
        with open("/home/gerardo/software/JointBDOE/data/JointBDOE_weaklabel_coco.yaml") as f:
            self.data = yaml.safe_load(f)  # load data dict

def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    gc.collect()
    torch.cuda.empty_cache()
    exit(0)

################# MAIN #################

if __name__ == '__main__':
    rospy.init_node("yolov8")
    rospy.loginfo("yolov8 node has been started")

    signal(SIGINT, handler)

    yolo = yolov8()
    # rospy.wait_for_service('bytetrack_srv')
    # rgb_subscriber = message_filters.Subscriber("/video_testing", msg_Image)
    rgb_subscriber = message_filters.Subscriber("/xtion/rgb/image_raw", msg_Image)
    depth_subscriber = message_filters.Subscriber("/xtion/depth/image_raw", msg_Image)
    odom_subscriber = message_filters.Subscriber("/camera_odom", Odometry)
    
    ts = message_filters.TimeSynchronizer([rgb_subscriber, depth_subscriber, odom_subscriber], 5)
    ts.registerCallback(yolo.store_data)
    rospy.Timer(rospy.Duration(0.033), yolo.get_yolo_objects)
    rospy.spin()

    # rospy.logwarn("Warning test message")
    # rospy.logerr("Error test message")
    # rospy.loginfo("End of program")
