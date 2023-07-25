import rospy
from yolov8_data.msg import *
from geometry_msgs.msg import *
from nav_msgs.msg import *
from people_msgs.msg import PositionMeasurementArray, People
import sys
import numpy as np
import imagehash
from PIL import Image
import tf
import math
import message_filters
sys.path.append('HashTrack')
from hash import HashTracker

class bytetrack():
    def __init__(self) -> None:
        self.tracker = HashTracker(frame_rate=30)
        # self.objects_data = ObjectsSRVResponse()
        self.tracked_objects_write, self.tracked_objects_read = [], []
        self.yolo_people = []
        self.new_yolo_people = False
        self.leg_detection_people = []
        self.new_leg_detection_people = False
        
    def set_new_yolo_people(self, data):
        print("yolo people received")
        self.yolo_people = data.objectsmsg
        self.new_yolo_people = True

    def set_new_leg_detection_people(self, data):
        print("leg detection people received")
        self.leg_detection_people = data.people
        self.new_leg_detection_people = True

    def set_new_people_data(self, yolo_data, leg_detection_data):
        if yolo_data.objectsmsg:
            print("yolo people received")
            self.yolo_people = yolo_data.objectsmsg
            self.new_yolo_people = True
            return
        if leg_detection_data.people:
            print("leg detection people received")
            self.leg_detection_people = leg_detection_data.people
            self.new_leg_detection_people = True

    def execute_bytetrack(self, event):
        # if self.new_yolo_people and self.new_leg_detection_people:
        yolo_people = self.yolo_people
        leg_detection_people = self.leg_detection_people

        if self.new_yolo_people:
            structured_data = self.read_visual_objects(yolo_people)
            self.to_object_interface(self.tracker.update(np.array(structured_data["scores"]),
                                                                    np.array(structured_data["boxes"]),
                                                                    np.array(structured_data["clases"]),
                                                                    np.array(structured_data["image"]),
                                                                    np.array(structured_data["hash"]),
                                                                    np.array(structured_data["pose"]),
                                                                    np.array(structured_data["orientation"])))
            self.new_yolo_people = False
        
        elif self.new_leg_detection_people:
            self.to_object_interface_legs(leg_detection_people)
            self.new_leg_detection_people = False

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
    
        returned_tracks = []
        for track in objects:
            pose = Pose()
            pose.position.x = round(track.mean[0], 2) if track.kalman_initiated else round(track._pose[0], 2)
            pose.position.y = round(track.mean[1], 2) if track.kalman_initiated else round(track._pose[1], 2)
            
            track_object = Object(
                id=int(track.track_id), score=track.score,
                left=int(track.bbox[0]), top=int(track.bbox[1]),
                right=int(track.bbox[2]),
                bot=int(track.bbox[3]), pose = pose, type=track.clase
            )

            if math.isnan(pose.position.x) or math.isinf(pose.position.x) or math.isnan(pose.position.y) or math.isinf(pose.position.y): 
                track_object.exist_position = False
            else:
                track_object.exist_position = True
            pose.orientation = track.orientation

            speed = Twist()
            if track.kalman_initiated:
                speed.linear.x = round(track.mean[2]/ track.difference_between_updates, 2)
                speed.linear.y = round(track.mean[3]/ track.difference_between_updates, 2)

                track_object.speed = speed
            returned_tracks.append(track_object)
            # print("SPEED:", track.speed)
            # if track.speed > 1.35:
            #     print("RÄPIDO")
            # else:
            #     print("LENTO")
        tracked_people = ObjectsMSG(objectsmsg = returned_tracks)
        people_publisher.publish(tracked_people)

    def to_object_interface_legs(self, leg_detections):
        returned_tracks = []
        for i, track in enumerate(leg_detections):
            track_object = Object()
            track_object.id = int(i)
            track_object.exist_position = True
            pose = Pose()
            pose.position.x = track.position.x
            pose.position.y = track.position.y
            qua_orientation = tf.transformations.quaternion_from_euler(0, 0, np.arctan2(pose.position.y, pose.position.x)) 
            pose.orientation = Quaternion(x=qua_orientation[0], y=qua_orientation[1], z=qua_orientation[2], w=qua_orientation[3])
            track_object.pose = pose

            speed = Twist()
            speed.linear.x = track.velocity.x
            speed.linear.y = track.velocity.y
            track_object.speed = speed
            
            returned_tracks.append(track_object)
        tracked_people = ObjectsMSG(objectsmsg = returned_tracks)
        people_publisher.publish(tracked_people)

if __name__ == '__main__':
    rospy.init_node("people_tracker")
    rospy.loginfo("People tracker node has been started")

    rate = rospy.Rate(30)
    bytetracker = bytetrack()
    # yolo_people_subscriber = rospy.Subscriber("/perceived_people", ObjectsMSG, bytetracker.set_new_yolo_people)

    yolo_people_subscriber = message_filters.Subscriber("/perceived_people", ObjectsMSG)
    leg_detector_people_subscriber = message_filters.Subscriber("/people", People)
    people_sync = message_filters.ApproximateTimeSynchronizer([yolo_people_subscriber, leg_detector_people_subscriber], queue_size=5, slop=0.5)
    people_sync.registerCallback(bytetracker.set_new_people_data)
    
    people_publisher = rospy.Publisher("/tracked_people", ObjectsMSG, queue_size=5)

    rospy.Timer(rospy.Duration(0.033), bytetracker.execute_bytetrack)
    rospy.spin()