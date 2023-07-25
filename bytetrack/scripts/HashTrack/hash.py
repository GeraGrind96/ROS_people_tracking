import numpy as np
import matching
from basetrack import BaseTrack, TrackState
from kalman_filter_2d_world import KalmanFilter
from collections import deque
import time
import math
import cv2

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, bbox, pose, score, clase, image, hash, orientation, kalman_enabled=True):

        # wait activate
        self._pose = np.asarray(pose, dtype=np.float)

        self.last_pose = [math.inf, math.inf]

        self.speed_pose = [0, 0]

        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.clase = clase
        self.image = image
        self.hash = hash
        self.bbox = np.asarray(bbox, dtype=np.float)

        self.hash_memory = deque(maxlen=50)
        self.last_hash_stored = time.time()

        self.speed_memory = deque(maxlen=3)
        self.last_speed_stored = time.time()
        # self.store_speed_period = 0.1
        self.speed = 0

        self.store_period = 0.3
        self.score = score
        self.tracklet_len = 0
        self.enable_kalman = kalman_enabled
        self.kalman_initiated = False
        self.last_kalman_update = time.time()
        self.difference_between_updates = 0

        self.orientation = orientation

    def predict(self):
        mean_state = self.mean.copy()
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks if st.kalman_initiated])
            multi_covariance = np.asarray([st.covariance for st in stracks if st.kalman_initiated])

            if len(multi_covariance) > 0 and len(multi_mean) > 0:
                for i, st in enumerate(stracks):
                    # if st.state != TrackState.Tracked:
                    #     multi_mean[i][3] = 0 
                    multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
                for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                    stracks[i].mean = mean
                    stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_class_id(str(self.clase))
        if not any(math.isnan(element) or math.isinf(element) or element == 0 for element in self._pose):
            # print("ACTIVATE FUNCTION POSE:", self._pose)
            if self.enable_kalman:
                self.mean, self.covariance = self.kalman_filter.initiate(self._pose)
                self.last_kalman_update = time.time()
                self.kalman_initiated = True
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        if not any(math.isnan(element) or math.isinf(element) or element == 0 for element in new_track._pose):
            # print("re_activate FUNCTION POSE:", new_track._pose)
            
            if self.enable_kalman:
                if not self.kalman_initiated:
                    self.mean, self.covariance = self.kalman_filter.initiate(new_track._pose)
                    self.last_kalman_update = time.time()
                    self.kalman_initiated = True                    
                else:
                    self.last_pose = [self.mean[0], self.mean[1]]
                    self.mean, self.covariance = self.kalman_filter.update(
                        self.mean, self.covariance, new_track._pose)
                    
                    self.difference_between_updates = time.time() - self.last_kalman_update
                    self.last_kalman_update = time.time()
        self.last_pose = self._pose
        self._pose = new_track._pose
        self.last_pose_update = time.time()
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.bbox = new_track.bbox
        # if time.time() - self.last_orientation_stored > self.store_orientation_period:
        #     self.orientation_memory.append(round(new_track.orientation, 3))
        #     self.last_orientation_stored = time.time()

        self.orientation = new_track.orientation

        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.image = new_track.image
        self.hash = new_track.hash
        self.hash_memory = new_track.hash_memory

    def update(self, new_track, frame_id):
        
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        # TODO: WATCH OUT THIS
        # self.tlwh = new_track.tlwh
        
        if not any(math.isnan(element) or math.isinf(element) or element == 0 for element in new_track._pose):
            # print("update FUNCTION POSE:", new_track._pose)
            if self.enable_kalman:

                # if time.time() - self.last_speed_stored > self.store_speed_period:
                self.difference_between_updates = time.time() - self.last_kalman_update


                if not self.kalman_initiated:
                    self.mean, self.covariance = self.kalman_filter.initiate(new_track._pose)
                    self.last_kalman_update = time.time()
                    self.kalman_initiated = True
                else:
                    self.last_pose = [self.mean[0], self.mean[1]]
                    self.mean, self.covariance = self.kalman_filter.update(
                        self.mean, self.covariance, new_track._pose)
                    self.last_kalman_update = time.time()

                speed_module = round(math.sqrt(self.mean[2] ** 2 + self.mean[3] ** 2), 2)
                print("speed and time difference:", speed_module, self.difference_between_updates)
                self.speed_memory.append(speed_module / self.difference_between_updates)
                self.last_speed_stored = time.time()
                if len(self.speed_memory) > 0:
                    self.speed = sum(self.speed_memory) / len(self.speed_memory)
        self.last_pose = self._pose
        self._pose = new_track._pose
        self.last_pose_update = time.time()
        self.state = TrackState.Tracked
        self.is_activated = True
        self.image = new_track.image
        self.hash = new_track.hash
        self.bbox = new_track.bbox
        self.orientation = new_track.orientation
        # if time.time() - self.last_orientation_stored > self.store_orientation_period:
        #     self.orientation_memory.append(round(new_track.orientation, 3))
        #     self.last_orientation_stored = time.time()
        #     if len(self.orientation_memory) > 0:
        #         self.orientation = sum(self.orientation_memory) / len(self.orientation_memory)

        # if time.time() - self.last_hash_stored > self.store_period:
            # self.hash_memory.append(new_track.hash)
            # self.last_hash_stored = time.time()
        self.score = new_track.score

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

class HashTracker(object):
    def __init__(self, frame_rate=30, buffer_=90, kalman_enabled=True):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.match_thresh = 0.99999999
        self.frame_id = 0
        self.track_thresh = 0.4
        self.det_thresh = self.track_thresh + 0.1
        self.track_buffer = buffer_
        self.buffer_size = int(frame_rate / 30.0 * self.track_buffer)
        self.max_time_lost = self.buffer_size
        self.enable_kalman = kalman_enabled
        self.kalman_filter = KalmanFilter()

        # Metrics ponderation
        self.k_hash = 1
        self.k_iou = 0

        # For specific element mode
        self.tracked_element = None
        self.chosen_track = -1

    def update(self, scores, bboxes, clases, images, hash, poses, orientation):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        bboxes_keep = bboxes[remain_inds]
        bboxes_second = bboxes[inds_second]
        poses_second = poses[inds_second]
        poses_keep = poses[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        clases_keep = clases[remain_inds]
        clases_second = clases[inds_second]
        images_keep = images[remain_inds]
        images_second = images[inds_second]
        hash_keep = hash[remain_inds]
        hash_second = hash[inds_second]
        orientation_keep = orientation[remain_inds]
        orientation_second = orientation[inds_second]

        if len(bboxes_keep) > 0:
            '''Detections'''
            detections = [STrack(bbox, pose, s, clases, image, hash, orientation) for
                          (bbox, pose, s, clases, image, hash, orientation) in zip(bboxes_keep, poses_keep, scores_keep, clases_keep, images_keep, hash_keep, orientation_keep)]
        else:
            detections = []
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        if self.enable_kalman:
            STrack.multi_predict(strack_pool)
        dists_hash = self.k_hash * matching.hash_distance(strack_pool, detections)
        dists_iou = self.k_iou * matching.iou_distance(strack_pool, detections)
        combinated_dists = dists_hash + dists_iou

        dists = matching.fuse_score(combinated_dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(poses_second) > 0:
            '''Detections'''
            detections_second = [STrack(bbox, pose, s, clases, image, hash, orientation) for
                          (bbox, pose, s, clases, image, hash, orientation) in zip(bboxes_second, poses_second, scores_second, clases_second, images_second, hash_second, orientation_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        dists_hash = self.k_hash * matching.hash_distance(r_tracked_stracks, detections_second)
        dists_iou = self.k_iou * matching.iou_distance(r_tracked_stracks, detections_second)
        combinated_dists = dists_hash + dists_iou

        matches, u_track, u_detection_second = matching.linear_assignment(combinated_dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]

        dists_hash = self.k_hash * matching.hash_distance(unconfirmed, detections)
        dists_iou = self.k_iou * matching.iou_distance(unconfirmed, detections)
        combinated_dists = dists_hash + dists_iou

        dists = matching.fuse_score(combinated_dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.tracked_stracks = self.check_distance_between_people(self.tracked_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks.extend(self.lost_stracks)
        return output_stracks

    def joint_stracks(self, tlista, tlistb):
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res


    def sub_stracks(self, tlista, tlistb):
        stracks = {}
        for t in tlista:
            stracks[t.track_id] = t
        for t in tlistb:
            tid = t.track_id
            if stracks.get(tid, 0):
                del stracks[tid]
        return list(stracks.values())


    def remove_duplicate_stracks(self, stracksa, stracksb):
        pdist_hash = 1 - self.k_hash * matching.hash_distance(stracksa, stracksb)
        pdist_iou = 1 - self.k_iou * matching.iou_distance(stracksa, stracksb)
        pairs = np.where((pdist_iou < self.k_iou * 0.15) | (pdist_hash < self.k_hash * 0.20))
        dupa, dupb = list(), list()
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if not i in dupa]
        resb = [t for i, t in enumerate(stracksb) if not i in dupb]
        return resa, resb
    
    def check_distance_between_people(self, objects):
        repeated_objects = []
        for i, track in enumerate(objects):
            track_x = round(track.mean[0], 2) if track.kalman_initiated else round(track._pose[0], 2)
            track_y = round(track.mean[1], 2) if track.kalman_initiated else round(track._pose[1], 2)
            
            for j in range(i+1, len(objects)):
                track_comp = objects[j]
                track_comp_x = round(track_comp.mean[0], 2) if track_comp.kalman_initiated else round(track_comp._pose[0], 2)
                track_comp_y = round(track_comp.mean[1], 2) if track_comp.kalman_initiated else round(track_comp._pose[1], 2)
                
                if abs(track_x - track_comp_x) < 0.2 and abs(track_y - track_comp_y) < 0.2:
                    repeated_objects.append(track_comp.track_id) if track.track_id < track_comp.track_id else repeated_objects.append(track.track_id)
        # print("repeated_objects", repeated_objects)
        filtered_objects = [track for track in objects if track.track_id not in repeated_objects]
        return filtered_objects