#!/usr/bin/env python3

import rospy
from yolov8_data.msg import *
from yolov8_data.srv import *
from geometry_msgs.msg import *
from nav_msgs.msg import *
from people_msgs.msg import PositionMeasurementArray, People, PersonStamped
from std_msgs.msg import *
import sys
import numpy as np
import imagehash
from PIL import Image
import tf
import math
import message_filters
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/HashTrack')
from hash import HashTracker
from basetrack import TrackState
import lap
import time
import cv2

class bytetrack():
    def __init__(self) -> None:
        self.tracker = HashTracker(frame_rate=25)
        self.yolo_people = []
        # self.leg_detection_people = []
        # self.new_leg_detection_people = False
        self.followed_person_id = -1

        self.stop_speed_threshold = 0.2

        rospy.init_node("people_tracker")
        rospy.loginfo("People tracker node has been started")

        self.people_publisher = rospy.Publisher("/tracked_people", ObjectsMSG, queue_size=5)
        yolo_people_subscriber = rospy.Subscriber("/perceived_people", ObjectsMSG, self.set_new_yolo_people)
        self.tracks_publisher = rospy.Publisher("/tracks", PoseArray, queue_size = 3)
        # For also using leg detection
        # leg_detector_people_subscriber = rospy.Subscriber("/people", People, bytetracker.set_new_leg_detection_people)
        # yolo_people_subscriber = message_filters.Subscriber("/perceived_people", ObjectsMSG)
        # leg_detector_people_subscriber = message_filters.Subscriber("/people", People)
        # people_sync = message_filters.ApproximateTimeSynchronizer([yolo_people_subscriber, leg_detector_people_subscriber], queue_size=1, slop=0.1)
        # people_sync.registerCallback(bytetracker.set_new_people_data)

        rospy.Service("set_id", SetID, self.set_chosen_track)
        rospy.spin()
        self.tf_listener = tf.TransformListener()
        
    def set_new_yolo_people(self, data):
        self.yolo_people = data.objectsmsg
        self.execute_bytetrack()

    def set_chosen_track(self, req):
        ret = ""
        if req.id != self.followed_person_id:
            ret = "new"
        else:
            ret = "same"
        self.followed_person_id = req.id
        self.tracker.set_chosen_track(req.id)
        return SetIDResponse(ret)

    def set_new_people_data(self, yolo_data, leg_detection_data):
        # if len(yolo_data.objectsmsg) > 0:
        self.yolo_people = yolo_data.objectsmsg
        self.new_yolo_people = True
        # if len(leg_detection_data.people) > 0 :
        #     self.leg_detection_people = leg_detection_data
        #     # Transform legs position into the world frame
        #     header = Header(stamp = rospy.Time(0), frame_id = self.leg_detection_people.header.frame_id)
        #     for i in range(len(self.leg_detection_people.people)):
        #         self.leg_detection_people.people[i].position = self.transformate_legs_pose_to_world(self.leg_detection_people.people[i].position, header)
        #     self.new_leg_detection_people = True

    # def execute_bytetrack(self, event):
    #     # If exist data in both modalities, check if leg track associated to track still existing
        
    #     # if self.new_leg_detection_people and self.new_yolo_people:
    #     #     leg_detection_people = self.leg_detection_people
    #     #     yolo_people = self.yolo_people
        
    #     #     structured_data = self.read_visual_objects(yolo_people)
    #     #     returned_tracks = self.tracker.update(np.array(structured_data["scores"]),
    #     #                                                             np.array(structured_data["boxes"]),
    #     #                                                             np.array(structured_data["clases"]),
    #     #                                                             np.array(structured_data["image"]),
    #     #                                                             np.array(structured_data["hash"]),
    #     #                                                             np.array(structured_data["pose"]),
    #     #                                                             np.array(structured_data["orientation"]))
            
    #     #     # Associate legs to visual pose
    #     #     if len(returned_tracks) > 0:
    #     #         self.leg_with_image_pose_association(returned_tracks, leg_detection_people.people)
            
    #     #     objects = self.to_object_interface(returned_tracks)
    #     #     people_publisher.publish(objects)
    #     #     self.new_yolo_people = False
    #     #     self.new_leg_detection_people = False
        
    #     if self.new_yolo_people: 
    #         init = time.time()
    #         yolo_people = self.yolo_people
    #         structured_data = self.read_visual_objects(yolo_people)
    #         objects = self.to_object_interface(self.tracker.update(np.array(structured_data["scores"]),
    #                                                                 np.array(structured_data["boxes"]),
    #                                                                 np.array(structured_data["clases"]),
    #                                                                 np.array(structured_data["image"]),
    #                                                                 np.array(structured_data["hash"]),
    #                                                                 np.array(structured_data["pose"]),
    #                                                                 np.array(structured_data["orientation"])))
    #         self.people_publisher.publish(objects)
    #         self.new_yolo_people = False
    #         print("TIME EXPENDED:", time.time() - init)

    #     # elif self.new_leg_detection_people:
    #     #     leg_detection_people = self.leg_detection_people
    #     #     self.to_object_interface_legs(leg_detection_people.people)
    #     #     self.new_leg_detection_people = False

        
    #     # if self.new_yolo_people: 
    #     #     print("ONLY YOLO")
    #     #     yolo_people = self.yolo_people
    #     #     structured_data = self.read_visual_objects(yolo_people)
    #     #     objects = self.to_object_interface(self.tracker.update(np.array(structured_data["scores"]),
    #     #                                                             np.array(structured_data["boxes"]),
    #     #                                                             np.array(structured_data["clases"]),
    #     #                                                             np.array(structured_data["image"]),
    #     #                                                             np.array(structured_data["hash"]),
    #     #                                                             np.array(structured_data["pose"]),
    #     #                                                             np.array(structured_data["orientation"])))
    #     #     people_publisher.publish(objects)
    #     #     self.new_yolo_people = False

    def execute_bytetrack(self):        
        init = time.time()
        yolo_people = self.yolo_people
        structured_data = self.read_visual_objects(yolo_people)
        objects = self.to_object_interface(self.tracker.update(np.array(structured_data["scores"]),
                                                                np.array(structured_data["boxes"]),
                                                                np.array(structured_data["clases"]),
                                                                np.array(structured_data["image"]),
                                                                np.array(structured_data["hash"]),
                                                                np.array(structured_data["pose"]),
                                                                np.array(structured_data["orientation"])))
        self.people_publisher.publish(objects)

    ###############################################
    ################ LEG FUNCTIONS ################
    ###############################################

    def leg_with_image_pose_association(self, objects, leg_detections):
        person_poses = [[round(person.mean[0], 2) if person.kalman_initiated else round(person._pose[0], 2), round(person.mean[1], 2) if person.kalman_initiated else round(person._pose[1], 2)] for person in objects]
        leg_poses = [self.get_point_list_from_point(position.position) for position in leg_detections]
        person_legs_pose_comparison = self.calculate_person_legs_matrix(person_poses, leg_poses)
        if len(person_legs_pose_comparison) > 0:
            linear_assignment = self.linear_assignment(person_legs_pose_comparison, 1)[0]
            self.tracker.associate_leg_detector_with_track(linear_assignment, leg_detections)

    def transformate_legs_pose_to_world(self, robot_person_pose, header):
        person_pose = PointStamped(point = robot_person_pose, header = header)
        transformed_point = self.tf_listener.transformPoint("map", person_pose)
        return transformed_point.point
    
    def get_point_list_from_point(self, point):
        return [point.x, point.y]

    def calculate_person_legs_matrix(self, people, legs):
        array1 = np.array(people)
        array2 = np.array(legs)
        diff = array1[:, np.newaxis] - array2
        distances_array = np.linalg.norm(diff, axis=-1)
        return distances_array

    ###############################################
    ########### DATA STRUCTURE CONVERSION #########
    ###############################################

    def read_visual_objects(self, objects):
        return {
            "scores": [object.score for object in objects],
            "boxes": [[object.left, object.top, object.right, object.bot] for object in objects],
            "clases": [object.type for object in objects],
            "image": [object.image for object in objects],
            "hash": [self.get_color_histogram(object) for object in objects],
            "pose": [[object.pose.pose.position.x, object.pose.pose.position.y, object.pose.pose.position.z] for object in objects],
            "orientation" : [object.pose.pose.orientation for object in objects]
        }  
    
    def to_object_interface(self, objects):
        returned_tracks = []
        tracked_people = PoseArray()
        tracked_people.header.frame_id = "map"
        for track in objects:
            track_object = Object(
                id=int(track.track_id), score=float(track.score),
                left=int(track.bbox[0]), top=int(track.bbox[1]),
                right=int(track.bbox[2]),
                bot=int(track.bbox[3]), type=track.clase, image=track.image
            )
            speed = Twist()
            # print("ID", track.track_id) 
            if track.kalman_initiated:
                speed.linear.x = round(track.speed[0], 2)
                speed.linear.y = round(track.speed[1], 2)

                speed_module = math.sqrt(speed.linear.x ** 2 + speed.linear.y ** 2)
                # print("SPEED MODULE", speed_module) 
                if speed_module < self.stop_speed_threshold or np.isnan(speed.linear.x) or np.isnan(speed.linear.y):
                    speed.linear.x = 0
                    speed.linear.y = 0
                
                if speed_module >= self.stop_speed_threshold and self.followed_person_id != track_object.id and (track.state == TrackState.Lost or track.state == TrackState.Removed):
                    continue
                track_object.speed = speed

            pose = Pose()
            pose.position.x = round(track.mean[0], 2) if track.kalman_initiated and speed_module > self.stop_speed_threshold else round(track._pose[0], 2)
            pose.position.y = round(track.mean[1], 2) if track.kalman_initiated and speed_module > self.stop_speed_threshold else round(track._pose[1], 2)
            pose.position.z = round(track.pose_z, 2)
            # print("POSE", pose.position.x, pose.position.y) 
            if math.isnan(pose.position.x) or math.isinf(pose.position.x) or math.isnan(pose.position.y) or math.isinf(pose.position.y): 
                track_object.exist_position = False
            else:
                track_object.exist_position = True
            pose.orientation = track.orientation   
            tracked_people.poses.append(pose)
            pose_stamped = PoseStamped(pose = pose)
            track_object.pose = pose_stamped
            returned_tracks.append(track_object)
        self.tracks_publisher.publish(tracked_people)
        return ObjectsMSG(objectsmsg = returned_tracks)

    def to_object_interface_legs(self, leg_detections):
        returned_tracks = []
        for i, track in enumerate(leg_detections):
            track_object = Object()
            track_object.id = int(i)
            track_object.exist_position = True
            pose = Pose()
            pose.position.x = track.position.x
            pose.position.y = track.position.y
            qua_orientation = tf.transformations.quaternion_from_euler(0, 0, math.atan2(track.velocity.y, track.velocity.x)) 
            pose.orientation = Quaternion(x=qua_orientation[0], y=qua_orientation[1], z=qua_orientation[2], w=qua_orientation[3])
            track_object.pose = pose

            speed = Twist()
            speed.linear.x = track.velocity.x
            speed.linear.y = track.velocity.y
            track_object.speed = speed
            
            returned_tracks.append(track_object)
        tracked_people = ObjectsMSG(objectsmsg = returned_tracks)
        self.people_publisher.publish(tracked_people)

    ###############################################
    ########### IMAGE HASH FUNCTIONS ##############
    ###############################################

    def get_imagehash_from_roi(self, object):
        color = np.frombuffer(object.image.data, dtype=np.uint8).reshape(object.image.height, object.image.width, 3)
        image = Image.fromarray(color)
        return imagehash.colorhash(image)
    
    def get_color_histogram(self, object):
        color = np.frombuffer(object.image.data, dtype=np.uint8).reshape(object.image.height, object.image.width, 3)
        color =cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([color], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return self.normalize_histogram(hist)

    def normalize_histogram(self, hist):
        total_pixels = np.sum(hist)
        normalized_hist = hist / total_pixels
        return normalized_hist

    ###############################################
    ########### AUX FUNCTIONS ##############
    ###############################################

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

if __name__ == '__main__':
    bytetracker = bytetrack()

