import rospy
from yolov8_data.msg import *
from geometry_msgs.msg import *
from nav_msgs.msg import *
import sys
import numpy as np
import imagehash
from PIL import Image
import tf
import math
import cv2

sys.path.append('/home/gerardo/software/ByteTrack')
from yolox.tracker.byte_tracker import BYTETracker

class bytetrack():
    def __init__(self) -> None:
        self.tracker = BYTETracker(frame_rate=30)
        # self.objects_data = ObjectsSRVResponse()
        self.tracked_objects_write, self.tracked_objects_read = [], []
        self.robot_world_transform_matrix = np.array([])
        self.robot_orientation = None
        self.act_people = []

    def store_robot_pose(self, robot_pose):
        euler_rotation = tf.transformations.euler_from_quaternion([robot_pose.pose.pose.orientation.x, robot_pose.pose.pose.orientation.y, robot_pose.pose.pose.orientation.z, robot_pose.pose.pose.orientation.w])
        self.robot_orientation = euler_rotation[2]
        self.robot_world_transform_matrix = np.array([[math.cos(euler_rotation[2]), -math.sin(euler_rotation[2]), 0, robot_pose.pose.pose.position.x],
                        [math.sin(euler_rotation[2]), math.cos(euler_rotation[2]), 0, robot_pose.pose.pose.position.y],
                        [0, 0, 1, robot_pose.pose.pose.position.z],
                        [0,   0,   0,   1]])

    def execute_bytetrack(self):
        data = self.read_visual_objects(self.act_people)
        self.tracked_objects_write = self.to_object_interface(self.tracker.update(np.array(data["scores"]),
                                                                np.array(data["boxes"]),
                                                                np.array(data["clases"]),
                                                                np.array(data["image"])))
        
    def set_new_people(self, data):
        self.act_people = data.objectsmsg
          
    def read_visual_objects(self, objects):
        return {
            "scores": [object.score for object in objects],
            "boxes": [[object.left, object.top, object.right, object.bot] for object in objects],
            "clases": [object.type for object in objects],
            "image": [object.image for object in objects],
            "hash": [self.get_imagehash_from_roi(object) for object in objects],
            "pose": [[object.pose.position.x, object.pose.position.y] for object in objects],
            "orientation" : [object.pose.orientation for object in objects]
        }
    
    def get_imagehash_from_roi(self, object):
        color = np.frombuffer(object.image.data, dtype=np.uint8).reshape(object.image.height, object.image.width, 3)
        image = Image.fromarray(color)
        return imagehash.phash(image)

    def to_object_interface(self, objects):
        # Clean tracks that are the same person
        black_image = np.zeros((480, 640, 3), dtype=np.uint8)
        retured_tracks = []
        for track in objects:
            # pose = Pose()
            # pose.position.x = round(track.mean[0], 2) if track.kalman_initiated else round(track._pose[0], 2)
            # pose.position.y = round(track.mean[1], 2) if track.kalman_initiated else round(track._pose[1], 2)
            # pose.position.x = person_world_position[0]
            # pose.position.y = person_world_position[1]

            # if any(math.isnan(element) or math.isinf(element) for element in person_world_position): 
            #     continue
            bbox = track.tlbr
            cv2.rectangle(black_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

            print("FILTERED BBOX", int(bbox[0]), int(bbox[1]), int(bbox[2]),int(bbox[3]))
        cv2.imshow("", black_image)
        cv2.waitKey(1)
        #     track_object = Object(
        #         id=int(track.track_id), score=track.score,
        #         left=int(track.bbox[0]), top=int(track.bbox[1]),
        #         right=int(track.bbox[2]),
        #         bot=int(track.bbox[3]), pose = pose, type=track.clase
        #     )
            
        #     if math.isnan(pose.position.x) or math.isinf(pose.position.x) or math.isnan(pose.position.y) or math.isinf(pose.position.y): 
        #         track_object.exist_position = False
        #     else:
        #         track_object.exist_position = True
        #     pose.orientation = track.orientation
        #     speed = Twist()
        #     if track.kalman_initiated:
        #         speed.linear.x = round(track.mean[2]/ track.difference_between_updates, 2)
        #         speed.linear.y = round(track.mean[3]/ track.difference_between_updates, 2)
        #         # speed.angular.z = np.arctan2(speed.linear.x, speed.linear.y)
        #         track_object.speed = speed
        #     retured_tracks.append(track_object)
        #     # speed_module = round(math.sqrt(speed.linear.x ** 2 + speed.linear.y ** 2), 3)
        #     print("SPEED LIST:", track.speed_memory)
        #     print("SPEED LIST:", track.speed)
        #     if track.speed > 1:
        #         print("RÄPIDO")
        #     else:
        #         print("LENTO")
        # tracked_people = ObjectsMSG(objectsmsg = retured_tracks)
        # people_publisher.publish(tracked_people)

if __name__ == '__main__':
    rospy.init_node("people_tracker")
    rospy.loginfo("People tracker node has been started")

    # pub = rospy.Publisher("element_list", Objects, queue_size=10)
    # new_object = Object()
    rate = rospy.Rate(30)
    bytetracker = bytetrack()
    bytetrack_subscriber = rospy.Subscriber("/perceived_people", ObjectsMSG, bytetracker.set_new_people)
    odom_subscriber = rospy.Subscriber("/odom", Odometry, bytetracker.store_robot_pose)
    people_publisher = rospy.Publisher("/tracked_people", ObjectsMSG, queue_size=10)

    while not rospy.is_shutdown():      
        if bytetracker.robot_world_transform_matrix.size != 0:
            bytetracker.execute_bytetrack()
        try:
            rate.sleep()
        except KeyboardInterrupt:
           print("Shutting down")