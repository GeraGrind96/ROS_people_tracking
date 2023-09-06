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
        self.associated_legs = None

        self.hash_memory = deque(maxlen=100)
        self.last_hash_stored = time.time()

        self.speed_memory = deque(maxlen=3)
        self.last_speed_stored = time.time()
        self.store_speed_period = 0.05
        self.speed = np.array([0,0])

        self.store_period = 0.33
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
        # print("TRACK ID", self.track_id)
        # TODO: WATCH OUT THIS
        # self.tlwh = new_track.tlwh
        
        if not any(math.isnan(element) or math.isinf(element) or element == 0 for element in new_track._pose):
            # print("update FUNCTION POSE:", new_track._pose)
            if self.enable_kalman:
                # if time.time() - self.last_speed_stored > self.store_speed_period:
                

                if not self.kalman_initiated:
                    self.mean, self.covariance = self.kalman_filter.initiate(new_track._pose)
                    self.last_kalman_update = time.time()
                    self.kalman_initiated = True
                else:
                    self.last_pose = [self.mean[0], self.mean[1]]
                    self.difference_between_updates = time.time() - self.last_kalman_update
                    if (time.time() - self.last_kalman_update) > self.store_speed_period:
                        speed_vector = np.array([self.mean[2], self.mean[3]]) / self.difference_between_updates
                        self.speed_memory.append(speed_vector)
                        self.speed = np.mean(self.speed_memory, axis=0)
                        self.last_kalman_update = time.time()
                    self.mean, self.covariance = self.kalman_filter.update(
                        self.mean, self.covariance, new_track._pose)
                    

                if not self.kalman_initiated:
                    self.mean, self.covariance = self.kalman_filter.initiate(new_track._pose)
                    self.last_kalman_update = time.time()
                    self.kalman_initiated = True
                else:
                    self.last_pose = [self.mean[0], self.mean[1]]
                    pose_diff = math.sqrt((self.mean[0] - new_track._pose[0]) ** 2 + (self.mean[1] - new_track._pose[1]) ** 2)
                    if pose_diff < 0.25:
                        self.difference_between_updates = time.time() - self.last_kalman_update
                        if (time.time() - self.last_kalman_update) > self.store_speed_period:
                            speed_vector = np.array([self.mean[2], self.mean[3]]) / self.difference_between_updates
                            self.speed_memory.append(speed_vector)
                            self.speed = np.mean(self.speed_memory, axis=0)
                            self.last_kalman_update = time.time()
                        self.mean, self.covariance = self.kalman_filter.update(
                            self.mean, self.covariance, new_track._pose)
        

        self._pose = new_track._pose
        self.last_pose_update = time.time()
        self.state = TrackState.Tracked
        self.is_activated = True
        self.image = new_track.image
        if time.time() - self.last_hash_stored > self.store_period and cv2.compareHist(new_track.hash, self.hash, cv2.HISTCMP_CORREL) != 1:
            self.hash_memory.append(new_track.hash)
            self.last_hash_stored = time.time()
        self.hash = new_track.hash
        self.bbox = new_track.bbox
        self.orientation = new_track.orientation
        self.score = new_track.score

    @property
    def get_pose(self):
        if self.mean is None:
            return self._pose.copy()
        ret = self.mean[:2].copy()
        return ret

    def set_associated_legs(self, legs_name):
        self.associated_legs = legs_name

    def refresh_memory(self, memory):
        self.hash_memory = memory

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

class HashTracker(object):
    def __init__(self, frame_rate=20, buffer_=60, kalman_enabled=True):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.chosen_strack = None
        self.match_thresh = 0.99
        self.match_thresh_following = 0.99
        self.frame_id = 0
        self.track_thresh = 0.7
        self.det_thresh = self.track_thresh + 0.1
        self.track_buffer = buffer_
        self.buffer_size = int(frame_rate / 30.0 * self.track_buffer)
        self.max_time_lost = self.buffer_size
        self.enable_kalman = kalman_enabled
        self.kalman_filter = KalmanFilter()

        # Metrics weights
        self.k_hash = 0.3
        self.k_iou = 0.7
        self.k_pose = 0.0

        self.k_hash_following = 0.3
        self.k_iou_following = 0.0
        self.k_pose_following = 0.7

        # For specific element mode
        self.chosen_track = -1
        self.time_without_seeing_followed_limit = 4
        self.time_person_is_lost = 0
        self.followed_person_lost = False

    def set_chosen_track(self, track_id):
        self.chosen_track = track_id

    def update(self, scores, bboxes, clases, images, hash, poses, orientations):
        # Cleaning not followed tracks
        if self.chosen_track != -1:
            track_to_remove = None
            for i, track in enumerate(self.tracked_stracks):
                if track.track_id == self.chosen_track:
                    self.chosen_strack = track
                    track_to_remove = i
                    break
            if track_to_remove != None and len(self.chosen_strack.hash_memory) > 0:
                self.tracked_stracks.remove(self.tracked_stracks[track_to_remove])
            elif self.chosen_strack == None:
                return self.update_original(scores, bboxes, clases, images, hash, poses, orientations)
            elif len(self.chosen_strack.hash_memory) == 0:
                return self.update_original(scores, bboxes, clases, images, hash, poses, orientations)
            return self.update_element_following(scores, bboxes, clases, images, hash, poses, orientations)
        else:
            return self.update_original(scores, bboxes, clases, images, hash, poses, orientations)

    def update_element_following(self, scores, bboxes, clases, images, hash, poses, orientation):
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

        if len(bboxes) > 0:
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
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        strack_pool.append(self.chosen_strack)

        if self.enable_kalman:
            STrack.multi_predict(strack_pool)

        ######### ASSOCIATION OF FOLLOWED PERSON #########

        ''' Step 2: First association, with high score detection boxes'''
        # # Predict the current location with KF
        followed_person_found = True
        if detections:
            dists_hash = self.k_hash_following * matching.hash_distance_following([self.chosen_strack], detections)
            dists_iou = self.k_iou_following * matching.iou_distance([self.chosen_strack], detections)
            dists_pose = self.k_pose_following * matching.pose_distance([self.chosen_strack], detections)

            for i in range(len(dists_pose[0])):
                dists_hash[:, i] += dists_iou[0][i] + dists_pose[0][i]
            # print(self.k_hash_following, self.k_iou_following)
            # For associating with detections score
            pos_match, filtered_memory = matching.get_max_similarity_detection(dists_hash, self.chosen_strack.hash_memory)
            # self.tracked_stracks[0].refresh_memory(filtered_memory)
            # print("NEW TRACK MEMORY SIZE:", len(self.tracked_stracks[0].hash_memory))
            if pos_match != -1:
                self.followed_person_lost = False
                self.k_hash_following = 0.5
                self.k_iou_following = 0.0
                self.k_pose_following = 0.5
                self.chosen_strack.update(detections[pos_match], self.frame_id)
                detections.remove(detections[pos_match])

            ##### IF FOLLOWED PERSON IS NOT FOUND, ADJUST METRICs weights for using only visual appearance
            else:
                if not self.followed_person_lost:
                    self.time_person_is_lost = time.time()
                    self.followed_person_lost = True
                else:
                    if time.time() - self.time_person_is_lost > self.time_without_seeing_followed_limit:
                        followed_person_found = False

                # self.chosen_strack.kalman_initiated = False
                self.k_hash_following = 1.0
                self.k_iou_following = 0.0
                self.k_pose_following = 0.0
        else:
            if not self.followed_person_lost:
                # print("PERSON LOST FOR FIRST TIME")
                self.time_person_is_lost = time.time()
                self.followed_person_lost = True
            else:
                if time.time() - self.time_person_is_lost > self.time_without_seeing_followed_limit:
                    # print("PERSON DEFINETLY LOST")
                    followed_person_found = False

        strack_pool.remove(self.chosen_strack)
        dists_hash = self.k_hash * matching.hash_distance(strack_pool, detections)
        dists_iou = self.k_iou * matching.iou_distance(strack_pool, detections)
        dists_pose = self.k_pose * matching.pose_distance(strack_pool, detections)

        # print(dists_hash) 
        # print(dists_iou)
        # print(dists_pose)
        combinated_dists = dists_hash + dists_iou + dists_pose
        # combinated_dists = dists_pose
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
        dists_pose = self.k_pose * matching.pose_distance(r_tracked_stracks, detections_second)
        combinated_dists = dists_hash + dists_iou + dists_pose
        # combinated_dists = dists_pose

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
        dists_pose = self.k_pose * matching.pose_distance(unconfirmed, detections)
        combinated_dists = dists_hash + dists_iou + dists_pose
        # combinated_dists = dists_pose

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
        _, self.lost_stracks = self.remove_duplicate_stracks([self.chosen_strack], self.lost_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        output_stracks = [track for track in self.tracked_stracks]
        output_stracks.extend(self.lost_stracks)
        if followed_person_found:
            output_stracks.append(self.chosen_strack)

        return output_stracks

    def update_original(self, scores, bboxes, clases, images, hash, poses, orientation):
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
        dists_pose = self.k_pose * matching.pose_distance(strack_pool, detections)

        combinated_dists = dists_hash + dists_iou + dists_pose
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
        dists_pose = self.k_pose * matching.pose_distance(r_tracked_stracks, detections_second)
        combinated_dists = dists_hash + dists_iou + dists_pose

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
        dists_pose = self.k_pose * matching.pose_distance(unconfirmed, detections)
        combinated_dists = dists_hash + dists_iou + dists_pose

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
        pdist_hash = self.k_hash * matching.hash_distance(stracksa, stracksb)
        pdist_iou = self.k_iou * matching.iou_distance(stracksa, stracksb)
        pdist_pose = self.k_pose * matching.pose_distance(stracksa, stracksb)
        pairs = np.where((pdist_iou < self.k_iou * 0.15) & (pdist_hash < self.k_hash * 0.4) & (pdist_pose < self.k_pose * 0.3))
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
                
                if abs(track_x - track_comp_x) < 0.3 and abs(track_y - track_comp_y) < 0.3:
                    repeated_objects.append(track_comp.track_id) if track.track_id < track_comp.track_id else repeated_objects.append(track.track_id)
        filtered_objects = [track for track in objects if track.track_id not in repeated_objects]
        return filtered_objects
    
    def associate_leg_detector_with_track(self, association_matrix, detected_legs):
        # First int for visual pose       
        for association in association_matrix:
            strack_index, detected_leg_index = int(association[0]), int(association[1])
            
            strack = self.tracked_stracks[strack_index]
            detected_leg_name = detected_legs[detected_leg_index].name
            
            strack.set_associated_legs(detected_leg_name)

