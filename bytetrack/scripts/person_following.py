#!/usr/bin/env python3

import rospy
from cohan_msgs.msg import TrackedAgents, TrackedAgent, TrackedSegment, TrackedSegmentType, AgentType
from yolov8_data.msg import ObjectsMSG
from yolov8_data.srv import SetID
from sensor_msgs.msg import *
from geometry_msgs.msg import PoseStamped, Pose, Quaternion, PoseArray, PointStamped, Point
from nav_msgs.msg import *
from std_msgs.msg import *
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import math
import tf
import os
import time
from collections import deque
import argparse

class PersonFollowing():
    def __init__(self, real_robot):
        self.color_image = []
        self.new_image = False
        self.people = []
        self.new_people = False
        self.local_costmap = []
        self.timestamp = time.time()

        self.max_angle_to_back_point = math.pi / 3 
        self.number_of_possible_points = 8
        self.angle_threshold = math.pi / 2

        self.cv_bridge = CvBridge()

        self.generate_dataset = False

        self.robot_pose = None
        self.real_robot = real_robot

        self.person_id = -1

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1
        self.color = (255, 255, 0)
        self.thickness = 2

        self.tf_listener = tf.TransformListener()

        ################# FOR CALCULATING ROBOT-FOLLOWED PERSON RELATIVE SPEED #################
        self.speed_memory = deque(maxlen=3)
        self.last_relative_distance = None
        self.last_update_time = None
        self.max_walking_speed = 2 # max speed for setting thresholds to target point distance respect to the person

        self.max_possible_distance = 2.5
        self.min_possible_distance = 0.5

        ################# ROS functions #################

        rospy.Subscriber("tracked_people", ObjectsMSG, self.get_people_data)
        if self.real_robot:
            rospy.Subscriber("/base_odometry/odom", Odometry, self.get_robot_pose)
        else:
            rospy.Subscriber("/odom", Odometry, self.get_robot_pose)
        rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self.get_local_costmap)

        self.set_id_service = rospy.ServiceProxy('set_id', SetID)

        self.pose_publisher = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size = 1)
        self.person_pose_publisher = rospy.Publisher("/followed_person_pose", PoseStamped, queue_size = 1)
        self.behind_pose_publisher = rospy.Publisher("/point_behind_human", PoseArray, queue_size = 1)
        self.chosen_pose_publisher = rospy.Publisher("/chosen_track", Int32, queue_size = 1)

    def get_people_data(self, people):
        self.people = people.objectsmsg
        self.new_people = True

    def get_robot_pose(self, data):
        # print("ENTER")
        if self.real_robot:
            robot_pose = PoseStamped()
            robot_pose.header = data.header
            robot_pose.header.stamp = rospy.Time(0)
            robot_pose.header.frame_id = "base_footprint"

            # robot_pose.pose = data.pose.pose
            robot_pose.pose.orientation.w = 1
            self.tf_listener.waitForTransform("base_footprint", "map", rospy.Time(), rospy.Duration(0.5))
            transformed_point = self.tf_listener.transformPose("map", robot_pose)
            self.robot_pose = transformed_point.pose.position
        else:
            self.robot_pose = data.pose.pose.position
        
    def get_local_costmap(self, costmap):
        self.local_costmap = costmap

    ############ Target behind functions ############

    def get_points_behind_person(self, person_pose, person_orientation, robot_pose, points_radius):       
        act_local_grid = self.local_costmap 
        person_robot_vector = [robot_pose.x - person_pose.x, robot_pose.y - person_pose.y]
        opposite_angle = math.atan2(person_robot_vector[1], person_robot_vector[0])
        self.max_angle_to_back_point = math.pi / 4
        points_behind_person, angles_to_person = self.points_arc(person_pose, points_radius, opposite_angle, self.max_angle_to_back_point, self.number_of_possible_points, act_local_grid, True)
        return points_behind_person, angles_to_person 

    # Return possible poses points and angles to person
    def points_arc(self, person_pose, radio, central_angle, angular_range, point_number, local_grid, orientation_to_person=False):
        init_angle = central_angle - angular_range
        end_angle = central_angle + angular_range
        step = (2 * angular_range) / point_number
        points = []
        final_angles = []
        angles = np.arange(init_angle, end_angle + step, step)
        for i in range(len(angles)):
            if angles[i] > math.pi or angles[i] < -math.pi:
                angles[i] = - (angles[i] / abs(angles[i])) * (math.pi - abs(math.pi - abs(angles[i])))
            x_point = person_pose.x + radio * np.cos(angles[i])
            y_point = person_pose.y + radio * np.sin(angles[i])

            # # Check if pose is in available pose
            world_pose_x, world_pose_y = self.world_to_local_map_coords(x_point, y_point, local_grid)
            point_occupancy_value = self.get_occupancy_value(world_pose_x, world_pose_y, local_grid)

            if point_occupancy_value:
                points.append([x_point, y_point])
                final_angles.append(angles[i])
        
        if not orientation_to_person:
            return points, []
        return points, final_angles

    # Return relative person speed respect to robot
    def calculate_speed_between_robot_and_person(self, person_pose, robot_pose):
        distance_modulus = math.sqrt((person_pose.x - robot_pose.x) ** 2 + (person_pose.y - robot_pose.y) ** 2)
        if self.last_relative_distance == None:
            speed = 0
        else:
            speed = (distance_modulus - self.last_relative_distance) / (time.time() - self.last_update_time)
            self.speed_memory.append(speed)
            speed = np.clip(np.mean(self.speed_memory, axis=0), -self.max_walking_speed, self.max_walking_speed)
        self.last_relative_distance = distance_modulus
        self.last_update_time = time.time()
        return speed

    # Returns the target distance respect to the person pose
    def point_distance_by_speed(self, speed):
        b = math.log(self.min_possible_distance / self.max_possible_distance) / (self.max_walking_speed * 2)
        a = self.max_possible_distance / math.exp(-self.max_walking_speed * b)
        return round(a * math.exp(b * speed), self.max_walking_speed)

    def create_possible_pose(self, point, orientation):
        pose = Pose()
        pose.position.x = point[0]
        pose.position.y = point[1]
        pose.position.z = 0
        pose.orientation = orientation
        return pose

    def world_to_local_map_coords(self, world_x, world_y, local_map):
        try:
            # self.tf_listener.waitForTransform(local_map.header.frame_id, "base_link", rospy.Time(), rospy.Duration(1.0))
            point_stamped = PointStamped()
            point_stamped.header.stamp = rospy.Time(0)
            point_stamped.header.frame_id = "map"
            point_stamped.point = Point(x=world_x, y=world_y)
            transformed_point = self.tf_listener.transformPoint(local_map.header.frame_id, point_stamped)
            local_x = transformed_point.point.x
            local_y = transformed_point.point.y
            return local_x, local_y
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Transformación entre sistemas de referencia no disponible.")
            return None, None

    def get_occupancy_value(self, x, y, local_map):
        grid_x = int((x - local_map.info.origin.position.x) / local_map.info.resolution)
        grid_y = int((y - local_map.info.origin.position.y) / local_map.info.resolution)
        grid_width = local_map.info.width
        max_dist = 4
        for x in range(grid_x - max_dist, grid_x + max_dist + 1):
            for y in range(grid_y - max_dist, grid_y + max_dist + 1):
                grid_index = y * grid_width + x  
                if 0 <= grid_index < len(local_map.data):
                    occupancy_value = local_map.data[grid_index]
                    if occupancy_value < 0 or 50 < occupancy_value: 
                        return False
                else:
                    rospy.logwarn("Index out of map limits.")
                    return False 
        return True

    def publish_person_pose(self, event):
        robot_pose = self.robot_pose
        if self.person_id != -1:
            self.chosen_pose_publisher.publish(Int32(data=self.person_id))
            if self.local_costmap:
                for person in self.people:
                    if person.id == self.person_id:
                        speed = self.calculate_speed_between_robot_and_person(person.pose.pose.position, robot_pose)
                        point_radius = self.point_distance_by_speed(speed)
                        self.publish_goal(person, robot_pose, point_radius)
                        break

    def publish_goal(self, person, robot_pose, points_radius):
        quaternion = [person.pose.pose.orientation.x, person.pose.pose.orientation.y, person.pose.pose.orientation.z, person.pose.pose.orientation.w]
        points_behind_person, angles = self.get_points_behind_person(person.pose.pose.position, tf.transformations.euler_from_quaternion (quaternion)[2], robot_pose, points_radius)
        target_pose_stamped, person_pose_stamped = PoseStamped(), PoseStamped()
        target_pose_stamped.header.stamp = rospy.Time.now()
        target_pose_stamped.header.frame_id = "map" 
        person_pose_stamped.header = target_pose_stamped.header
        person_pose_stamped.pose = person.pose

        point_min_dist_to_robot = 9999
        nearest_pose = None

        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "map"
        if len(angles) > 0:
            for i, point in enumerate(points_behind_person):
                if angles[i] > 0:
                    opposite_angle = angles[i] - math.pi
                else:
                    opposite_angle = (angles[i] - math.pi) % math.pi
                yaw_to_quaternion = tf.transformations.quaternion_from_euler(0, 0, opposite_angle)       
                orientation = Quaternion(x=yaw_to_quaternion[0], y=yaw_to_quaternion[1], z=yaw_to_quaternion[2], w=yaw_to_quaternion[3])
                new_pose = self.create_possible_pose(point, orientation)
                point_dist_to_robot = math.sqrt((point[0] - robot_pose.x) ** 2 + (point[1] - robot_pose.y) ** 2)
                if point_dist_to_robot < point_min_dist_to_robot:
                    point_min_dist_to_robot = point_dist_to_robot
                    nearest_pose = new_pose
                pose_array.poses.append(new_pose)
        else:
            for point in points_behind_person:
                new_pose = self.create_possible_pose(point, person.pose.pose.orientation)
                point_dist_to_robot = math.sqrt((point[0] - robot_pose.x) ** 2 + (point[1] - robot_pose.y) ** 2)
                if point_dist_to_robot < point_min_dist_to_robot:
                    point_min_dist_to_robot = point_dist_to_robot
                    nearest_pose = new_pose
                pose_array.poses.append(new_pose)
        if nearest_pose != None:
            target_pose_stamped.pose = nearest_pose
            self.pose_publisher.publish(target_pose_stamped)

        self.behind_pose_publisher.publish(pose_array)
        # self.person_pose_publisher.publish(person_pose_stamped)
        
    ############ Plot people information ############

    def set_people_in_image(self, event):
        act_image = np.zeros((540, 960, 3), dtype=np.uint8)
        if self.new_people:
            act_people = self.people
            for person in act_people:
                # self.insert_image_to_dataset(act_image[int(person.top):int(person.bot), int(person.left):int(person.right)])

                if person.id == self.person_id:
                    cv2.rectangle(act_image, (int(person.left), int(person.top)), (int(person.right), int(person.bot)), (0, 0, 255), 2)
                else:
                    cv2.rectangle(act_image, (int(person.left), int(person.top)), (int(person.right), int(person.bot)), (255, 0, 0), 2)
                act_image = cv2.putText(act_image, str(person.id), (int(person.left + ((person.right - person.left)/ 2)), int(person.top + ((person.bot - person.top)/ 2))), self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA)
                # act_image = cv2.putText(act_image, str(person.id) + " " + str(person.score), (int(person.left + ((person.right - person.left)/ 2)), int(person.top + 25)), self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA)
            cv2.imshow("Robot Camera", act_image)
            cv2.waitKey(1)
            cv2.setMouseCallback("Robot Camera", self.select_person)
            self.new_people = False

    def insert_image_to_dataset(self, image):
        if self.generate_dataset:
            print("GENERATING DATASET")
            output_folder = 'dataset_' + str(self.timestamp)
            os.makedirs(output_folder, exist_ok=True)
            num_existing_files = len(os.listdir(output_folder))
            output_path = os.path.join(output_folder, str(num_existing_files) + ".png")
            cv2.imwrite(output_path, image)



    def select_person(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for person in self.people:
                if person.left < x and x < person.right and person.top < y and y < person.bot:
                    self.person_id = person.id
                    break
            self.set_id_service(self.person_id)
            self.timestamp = time.time()
            self.generate_dataset = True
        if event == cv2.EVENT_RBUTTONDOWN:
            self.person_id = -1
            self.set_id_service(self.person_id)
            self.generate_dataset = False

if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument('--real_robot', type=int)
    args, unknown = parser.parse_known_args()
    real_robot = args.real_robot

    set_real = False
    if real_robot:
        real_robot = True
        print("Real robot setted")
    else:
        print("Real robot not setted")

    rospy.init_node("person_following")
    rospy.loginfo("person_following node has been started")

    person_following = PersonFollowing(real_robot)

    rospy.Timer(rospy.Duration(0.033), person_following.set_people_in_image)
    rospy.Timer(rospy.Duration(0.1), person_following.publish_person_pose)
    rospy.spin()
