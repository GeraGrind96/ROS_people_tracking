#!/usr/bin/env python3
import rospy
from yolov8_data.msg import *
from yolov8_data.srv import *
import sys
import numpy as np

sys.path.append('/home/gerardo/ros_projects/cohan_ws/src/people_tracking/bytetrack/scripts/ByteTrack')

from yolox.tracker.byte_tracker import BYTETracker

class bytetrack():
    def __init__(self) -> None:
        self.tracker = BYTETracker(frame_rate=30)
        # self.objects_data = ObjectsSRVResponse()

    def execute_bytetrack(self, req):
        data = self.read_visual_objects(req)
        objects = self.to_object_interface(self.tracker.update_original(np.array(data["scores"]),
                                                                                       np.array(data["boxes"]),
                                                                                       np.array(data["clases"]),
                                                                                       np.array(data["orientations"])))

        return ObjectsSRVResponse(objects)
    
    def to_object_interface(self, objects):
        returned_people = []
        for track in objects:
            if len(track.last_tlwh) > 0:
                promediate_pose = [0.75 * track.tlwh[i] + 0.25 * track.last_tlwh[i] for i in range(len(track.tlwh))]
            else: 
                promediate_pose = track.tlwh
            returned_people.append(Object(
                id=int(track.track_id), score=track.score,
                left=int(promediate_pose[0]), top=int(promediate_pose[1]),
                right=int(promediate_pose[0] + promediate_pose[2]),
                bot=int(promediate_pose[1] + promediate_pose[3]), type=track.clase, orientation=track.orientation
            ))
        return returned_people

            


    def read_visual_objects(self, objects):
        return {
                "scores": [object.score for object in objects.req],
                "boxes": [[object.left, object.top, object.right, object.bot] for object in objects.req],
                "clases": [object.type for object in objects.req],
                "orientations": [object.orientation for object in objects.req]
            }

if __name__ == '__main__':
    rospy.init_node("bytetrack")
    rospy.loginfo("bytetrack node has been started")

    # pub = rospy.Publisher("element_list", Objects, queue_size=10)
    # new_object = Object()
    rate = rospy.Rate(30)
    bytetracker = bytetrack()
    bytetrack_service = rospy.Service("bytetrack_srv", ObjectsSRV, bytetracker.execute_bytetrack)

    while not rospy.is_shutdown():        
        try:
            rate.sleep()
        except KeyboardInterrupt:
           print("Shutting down")