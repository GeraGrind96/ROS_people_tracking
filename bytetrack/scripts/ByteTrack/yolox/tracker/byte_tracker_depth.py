from inspect import trace
import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
import collections
import more_itertools as m
import math
from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, clase):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        
        # Circular queue data
        self.queue_lenght = 1
        self.occ_queue_lenght = 10
        self.bb_side_margin = 20
        self.clase = clase
        self.last_bb_size = None
        self.last_mean = None
        self.last_covariance = None
        
        self.abs_speed_queue = collections.deque(maxlen=self.queue_lenght)
        self.x_speed_queue = collections.deque(maxlen=self.queue_lenght)
        self.y_speed_queue = collections.deque(maxlen=self.queue_lenght)
        self.same_direction_speed_queue = collections.deque(maxlen=self.queue_lenght)
        
        self.moving_queue = collections.deque(maxlen=self.queue_lenght)
        self.depth_queue = collections.deque(maxlen=self.queue_lenght)
        self.right_depth_queue = collections.deque(maxlen=self.occ_queue_lenght)
        self.left_depth_queue = collections.deque(maxlen=self.occ_queue_lenght)
        
        self.copy_of = None
        self.copied = False
        self.has_copy = False
        
        self.bb_size_difference_list = None
        
        self.occluded = False
        self.moving = False
        
        self.depth = None
        self.abs_speed = None
        self.x_speed = None
        self.y_speed = None
        self.moving_occluded = False
        self.duplicate = False
        self.behind = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                if not stracks[i].occluded:
                    stracks[i].mean = mean
                    stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_class_id(str(self.clase))
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        # possible_occlussion = self.update_circular_queue(self.tlwh_to_tlbr(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        # possible_occlussion = self.update_circular_queue(new_track.tlbr)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id, occluded=False):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        # possible_occlussion = self.update_circular_queue(new_track.tlbr)
        # if occluded:
        #     print("OCCLUDED")
        #     self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self.tlwh))
        # else:
            # if self.behind:
            #     print()
            #     print("UPDATING FROM BEHIND")
            #     self.mean, self.covariance = self.kalman_filter.update(self.last_mean, self.last_covariance, self.tlwh_to_xyah(self.last_bb_size))
            #     self.last_mean, self.last_covariance
            # else:
        self.mean[4] = self.x_speed
        self.mean[5] = self.y_speed
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
            # self.last_bb_size = self.tlwh
             
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    def update_virtual(self, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1

    # For adding x positions of bounding boxes to check if occlussion can exists
    def update_circular_queue(self, bb_size):
        self.bb_size_cq.append(bb_size[2] - bb_size[0])
        self.bb_size_difference_list = [c[0] - c[1] < 0 for c in m.sliding_window(self.bb_size_cq, 2)]
        # print(self.bb_size_difference_list)
        # print(self.bb_size_cq)
        # print(self.bb_size_difference_list)
        if all(self.bb_size_difference_list) and len(self.bb_size_difference_list) == self.queue_lenght - 1:
            return True
        return False

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, frame_rate=40, buffer_=60):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.virtual_stracks = []

        self.frame_id = 0
        #self.args = args
        self.track_thresh = 0.4
        self.mot20 = False
        self.match_thresh = 0.99999999

        self.track_buffer = buffer_
        self.det_thresh = self.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * self.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update_original(self, scores, bboxes, clases):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        clases_keep = clases[remain_inds]
        clases_second = clases[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, clase) for
                          (tlbr, s, clase) in zip(dets, scores_keep, clases_keep)]
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
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.mot20:
            #PEDIR CODIGO PABLO
            dists = matching.fuse_score(dists, detections)
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
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, clase) for
                          (tlbr, s, clase) in zip(dets_second, scores_second, clases_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
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
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
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

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks
    def update2(self, scores, bboxes, depth_image, clases):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        clases_keep = clases[remain_inds]
        clases_second = clases[inds_second]
        # Coge los bb de lo que llega de YOLO
        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, clase) for
                          (tlbr, s, clase) in zip(dets, scores_keep, clases_keep)]
        else:
            detections = []

        for i, det in enumerate(detections):
            if abs(det.tlbr[0] - det.tlbr[2]) < 50:
                detections.pop(i)



        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        # Recorre los bb que han aparecido una vez
        for track in self.tracked_stracks:
            # Si no ha sido trackeado alguna vez...
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

            ''' Check if bounding box moved some frames before'''
            track.x_speed_queue.append(track.mean[4])
            track.y_speed_queue.append(track.mean[5])
            track.x_speed = sum(track.x_speed_queue)/len(track.x_speed_queue)
            track.y_speed = sum(track.y_speed_queue)/len(track.y_speed_queue)
            track.abs_speed_queue.append(math.sqrt(pow(track.mean[4],2) + pow(track.mean[5], 2)))
            track.speed = sum(track.abs_speed_queue)/len(track.abs_speed_queue)
            if track.speed > 0.4:
                track.moving_queue.append(True)
            else:
                track.moving_queue.append(False)
            if len(track.moving_queue) == track.queue_lenght:
                if all(track.moving_queue):
                    track.moving = True
                else:
                    track.moving = False
            x_point = round((track.tlbr[0] + track.tlbr[2])/2)
            y_point = round((track.tlbr[1] + track.tlbr[3])/2)
            if x_point < 480 and y_point < 640 and 0 < x_point and 0 < y_point:
                track.depth_queue.append(depth_image[y_point][x_point])
                track.depth = sum(track.depth_queue)/len(track.depth_queue)

            ''' Get mean depth of rectangles at bounding box side to check possible element occlussions (walls, doors...) '''
            #
            # if all(element < 0 for element in track.speed_queue) or all(element > 0 for element in track.speed_queue):
            #     x_inst_speed = sum(track.speed_queue)/len(track.speed_queue)
            #     print("X INST SPEED", x_inst_speed)
            #     rounded_tlbr = track.tlbr.astype(int)
            #     if rounded_tlbr[1] >= 0 and rounded_tlbr[1] < 640 and rounded_tlbr[3] >= 0 and rounded_tlbr[3] < 640:
            #         if rounded_tlbr[0]-track.bb_side_margin >= 0 and rounded_tlbr[0]-track.bb_side_margin < 480:
            #             bb_left_side = depth_image[rounded_tlbr[1]:rounded_tlbr[3],rounded_tlbr[0]-track.bb_side_margin:rounded_tlbr[0]]
            #             bb_left_side_mean = np.mean(bb_left_side)
            #             print("LEFT DEPTH:", bb_left_side_mean)
            #             if bb_left_side_mean < track.depth:
            #                 track.left_depth_queue.append(True)
            #             else:
            #                 track.left_depth_queue.append(False)
            #             # print(track.left_depth_queue)
            #             if all(track.left_depth_queue) and len(track.left_depth_queue) == track.occ_queue_lenght:
            #                 print("POSSIBLE LEFT OCCLUSSION")
            #         if rounded_tlbr[2]+track.bb_side_margin >= 0 and rounded_tlbr[2]+track.bb_side_margin < 480:
            #             bb_right_side = depth_image[rounded_tlbr[1]:rounded_tlbr[3],rounded_tlbr[2]:rounded_tlbr[2]+track.bb_side_margin]
            #             bb_right_side_mean = np.mean(bb_right_side)
            #             print("RIGHT DEPTH:", bb_right_side_mean)
            #             if bb_right_side_mean < track.depth:
            #                 track.right_depth_queue.append(True)
            #             else:
            #                 track.right_depth_queue.append(False)
            #             # print(track.right_depth_queue)
            #             if all(track.right_depth_queue) and len(track.right_depth_queue) == track.occ_queue_lenght:
            #                 print("POSSIBLE RIGHT OCCLUSSION")
            # print("")
            # print("TRACK:", track)
            # print("Behind:", track.behind)
            # print("LAST BB:", track.last_bb_size)
            # print("BB:", track.tlbr)
            # print("MOVING:", track.moving)
            # print("SPEEDS:", track.x_speed, track.y_speed, track.speed)
            # print("DEPTH:", track.depth)
            # print("STATE:", track.state)
            # print("MOD BB:", track.estimated_pos)

        ''' Check if some intersections between bounding boxes exist and return a list with those that intersect '''
        inter_between_bb = np.triu(matching.iou_distance(tracked_stracks, tracked_stracks))
        inter_list = np.argwhere(inter_between_bb != 0)
        ''' For each pair of bounding boxes that intersect, check distance from camera and speed '''
        for track_a, track_b in inter_list:
            if inter_between_bb[track_a][track_b] != 1:
                # checking depth
                # b farther than a
                if tracked_stracks[track_a].depth < tracked_stracks[track_b].depth:
                    if not tracked_stracks[track_b].moving:
                        tracked_stracks[track_b].occluded = True
                    # Possible occlusion with movement
                    else:
                        if not tracked_stracks[track_b].duplicate:
                            tracked_stracks[track_b].occluded = False
                            linear_track = copy.deepcopy(tracked_stracks[track_b])
                            linear_track.moving_occluded = True
                            linear_track.mean[4] = linear_track.x_speed
                            linear_track.mean[5] = linear_track.y_speed
                            self.virtual_stracks.append(linear_track)
                            tracked_stracks[track_b].duplicate = True
                        
                else:
                    if not tracked_stracks[track_a].moving:
                        tracked_stracks[track_a].occluded = True
                    # Possible occlusion with movement
                    else:
                        if not tracked_stracks[track_a].duplicate:
                            tracked_stracks[track_a].occluded = False
                            linear_track = copy.deepcopy(tracked_stracks[track_a])
                            linear_track.moving_occluded = True 
                            linear_track.mean[4] = linear_track.x_speed
                            linear_track.mean[5] = linear_track.y_speed
                            self.virtual_stracks.append(linear_track)
                            tracked_stracks[track_a].duplicate = True
            else:
                tracked_stracks[track_a].occluded = False
                tracked_stracks[track_b].occluded = False
            
        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)      



        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        if len(self.virtual_stracks) > 0:
            for i, trk in enumerate(self.virtual_stracks):
                if self.frame_id - trk.end_frame > self.max_time_lost:
                    self.virtual_stracks.pop(i)
                    for i, tr_trk in enumerate(self.tracked_stracks):
                        if tr_trk.track_id == trk.track_id:
                            tr_trk.duplicate = False
            STrack.multi_predict(self.virtual_stracks)
        
        # Calcula el solapamiento entre los bbs que ya han sido trackeados y los perdidos; y los que llegan en el frame actual
        dists = matching.iou_distance(strack_pool, detections)
        # print("HUNGARO:", dists)

        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        # Calcula la matriz del húngaro
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)
        # Recorre lo matcheado
        for itracked, idet in matches:
            track = strack_pool[itracked]

            det = detections[idet]


            if track.state == TrackState.Tracked and not track.occluded:
                
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            # En el caso de que se haya reencontrado a la persona, el kalman se reactiva
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, clase) for
                          (tlbr, s, clase) in zip(dets_second, scores_second, clases_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        # print("HUNGARO 2:", dists)
        
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]                   
            det = detections_second[idet]
            # Si ha matcheado pero está behind, mejor darlo por perdido

            if track.state == TrackState.Tracked and not track.occluded:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                if track.duplicate:
                    for i, v_trk in enumerate(self.virtual_stracks):
                        if v_trk.track_id == track.track_id:
                            track = v_trk
                            # print(track.tlbr)
                            self.virtual_stracks.pop(i)
                            break
                    for j, v_trk in enumerate(self.tracked_stracks):
                        if v_trk.track_id == track.track_id:
                            self.tracked_stracks.pop(j)
                            break
                track.mark_lost()
                lost_stracks.append(track)

        ''' Step 4: Deal with unconfirmed tracks, usually tracks with only one beginning frame'''

        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)

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
            # print("PERSON END FRAME:", track.end_frame)                
            if self.frame_id - track.end_frame > self.max_time_lost:
                # print("REMOVED:", track)
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        # print("SELF TRACKED STRACKS", self.tracked_stracks)
        # print("LOST STRACKS", self.lost_stracks)
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks

    # def update2(self, scores, bboxes, img_info, img_size, depth_image):
    #     print("")
    #     print("START")
    #     print("SELF START TRACKED STRACKS", self.tracked_stracks)  
    #     # print("self.tracked_stracks:", self.tracked_stracks)
    #     self.frame_id += 1
    #     # print("FRAME ID:", self.frame_id)
    #     activated_starcks = []
    #     refind_stracks = []
    #     lost_stracks = []
    #     virtual_stracks = []
    #     removed_stracks = []

    #     # img_h, img_w = img_info[0], img_info[1]
    #     # scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
    #     # bboxes /= scale

    #     remain_inds = scores > self.track_thresh
    #     inds_low = scores > 0.1
    #     inds_high = scores < self.track_thresh
    #     # print("SCORES:", scores)

    #     inds_second = np.logical_and(inds_low, inds_high)
    #     dets_second = bboxes[inds_second]
    #     dets = bboxes[remain_inds]
    #     scores_keep = scores[remain_inds]
    #     scores_second = scores[inds_second]
    #     #print("Boxes ", dets, "Scores ", scores_keep)

    #     # Coge los bb de lo que llega de YOLO
    #     if len(dets) > 0:
    #         '''Detections'''
    #         detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
    #                       (tlbr, s) in zip(dets, scores_keep)]
    #     else:
    #         detections = []
            
    #     ''' Add newly detected tracklets to tracked_stracks'''
    #     unconfirmed = []
    #     tracked_stracks = []  # type: list[STrack]
    #     # Recorre los bb que han aparecido una vez
    #     for track in self.tracked_stracks:
    #         # Si no ha sido trackeado alguna vez...
    #         if not track.is_activated:
    #             unconfirmed.append(track)
    #         else:
    #             tracked_stracks.append(track)
            
    #         ''' Check if bounding box moved some frames before''' 
    #         track.x_speed_queue.append(track.mean[4])
    #         track.y_speed_queue.append(track.mean[5])
    #         track.x_speed = sum(track.x_speed_queue)/len(track.x_speed_queue)
    #         track.y_speed = sum(track.y_speed_queue)/len(track.y_speed_queue)
    #         track.abs_speed_queue.append(math.sqrt(pow(track.mean[4],2) + pow(track.mean[5], 2)))
    #         track.speed = sum(track.abs_speed_queue)/len(track.abs_speed_queue)        
    #         if track.speed > 0.4:
    #             track.moving_queue.append(True)
    #         else:
    #             track.moving_queue.append(False)
    #         if len(track.moving_queue) == track.queue_lenght:
    #             if all(track.moving_queue):
    #                 track.moving = True
    #             else:
    #                 track.moving = False
    #         x_point = round((track.tlbr[0] + track.tlbr[2])/2)
    #         y_point = round((track.tlbr[1] + track.tlbr[3])/2)
    #         if x_point < 480 and y_point < 640 and 0 < x_point and 0 < y_point: 
    #             track.depth_queue.append(depth_image[y_point][x_point])
    #             track.depth = sum(track.depth_queue)/len(track.depth_queue)
            
    #         ''' Get mean depth of rectangles at bounding box side to check possible element occlussions (walls, doors...) '''
    #         # 
    #         # if all(element < 0 for element in track.speed_queue) or all(element > 0 for element in track.speed_queue):     
    #         #     x_inst_speed = sum(track.speed_queue)/len(track.speed_queue)
    #         #     print("X INST SPEED", x_inst_speed)
    #         #     rounded_tlbr = track.tlbr.astype(int)
    #         #     if rounded_tlbr[1] >= 0 and rounded_tlbr[1] < 640 and rounded_tlbr[3] >= 0 and rounded_tlbr[3] < 640:
    #         #         if rounded_tlbr[0]-track.bb_side_margin >= 0 and rounded_tlbr[0]-track.bb_side_margin < 480:  
    #         #             bb_left_side = depth_image[rounded_tlbr[1]:rounded_tlbr[3],rounded_tlbr[0]-track.bb_side_margin:rounded_tlbr[0]]
    #         #             bb_left_side_mean = np.mean(bb_left_side)
    #         #             print("LEFT DEPTH:", bb_left_side_mean)
    #         #             if bb_left_side_mean < track.depth:
    #         #                 track.left_depth_queue.append(True)
    #         #             else:
    #         #                 track.left_depth_queue.append(False)
    #         #             # print(track.left_depth_queue)
    #         #             if all(track.left_depth_queue) and len(track.left_depth_queue) == track.occ_queue_lenght:
    #         #                 print("POSSIBLE LEFT OCCLUSSION")
    #         #         if rounded_tlbr[2]+track.bb_side_margin >= 0 and rounded_tlbr[2]+track.bb_side_margin < 480:  
    #         #             bb_right_side = depth_image[rounded_tlbr[1]:rounded_tlbr[3],rounded_tlbr[2]:rounded_tlbr[2]+track.bb_side_margin]
    #         #             bb_right_side_mean = np.mean(bb_right_side)
    #         #             print("RIGHT DEPTH:", bb_right_side_mean)
    #         #             if bb_right_side_mean < track.depth:
    #         #                 track.right_depth_queue.append(True)
    #         #             else:
    #         #                 track.right_depth_queue.append(False)
    #         #             # print(track.right_depth_queue)
    #         #             if all(track.right_depth_queue) and len(track.right_depth_queue) == track.occ_queue_lenght:
    #         #                 print("POSSIBLE RIGHT OCCLUSSION")
    #         # print("")
    #         # print("TRACK:", track)
    #         # print("Behind:", track.behind)
    #         # print("LAST BB:", track.last_bb_size)
    #         # print("BB:", track.tlbr)
    #         # print("MOVING:", track.moving)
    #         # print("SPEEDS:", track.x_speed, track.y_speed, track.speed)
    #         # print("DEPTH:", track.depth)
    #         # print("STATE:", track.state)
    #         # print("MOD BB:", track.estimated_pos)
        
    #     ''' Check if some intersections between bounding boxes exist and return a list with those that intersect '''
    #     inter_between_bb = np.triu(matching.iou_distance(tracked_stracks, tracked_stracks))
    #     inter_list = np.argwhere(inter_between_bb != 0)
    #     for inter in inter_list:
    #         ''' For each pair of bounding boxes that intersect, check distance from camera and speed '''
    #         if inter_between_bb[inter[0]][inter[1]] != 1:
                
    #             if tracked_stracks[inter[0]].depth < tracked_stracks[inter[1]].depth:
    #                     if not tracked_stracks[inter[1]].moving:
    #                         tracked_stracks[inter[1]].occluded = True
    #                         tracked_stracks[inter[1]].behind = False
    #                     else:
    #                         if not tracked_stracks[inter[1]].copied:
    #                             print("COPIA CREADA")
    #                             tracked_stracks[inter[1]].occluded = False
    #                             # tracked_stracks[inter[1]].last_bb_size = [tracked_stracks[inter[1]].last_bb_size[0] + tracked_stracks[inter[1]].x_speed*0.03, tracked_stracks[inter[1]].last_bb_size[1] + tracked_stracks[inter[1]].y_speed*0.03, tracked_stracks[inter[1]].last_bb_size[2] + tracked_stracks[inter[1]].x_speed*0.03, tracked_stracks[inter[1]].last_bb_size[3] + tracked_stracks[inter[1]].y_speed*0.03]
    #                             alt_track = copy.deepcopy(tracked_stracks[inter[1]])
    #                             alt_track.behind = True
    #                             alt_track.copied = True
    #                             # alt_track.track_id = -alt_track.track_id
    #                             alt_track.copy_of = tracked_stracks[inter[1]].track_id
    #                             lost_stracks.append(alt_track)
    #                             tracked_stracks[inter[1]].copied = True
                        
    #             elif tracked_stracks[inter[1]].depth < tracked_stracks[inter[0]].depth:
                    
    #                     if not tracked_stracks[inter[0]].moving:
    #                         tracked_stracks[inter[0]].occluded = True
    #                         tracked_stracks[inter[0]].behind = False
    #                     else:
    #                         if not tracked_stracks[inter[0]].copied:
    #                             print("COPIA CREADA")
    #                             tracked_stracks[inter[0]].occluded = False
    #                             # tracked_stracks[inter[0]].last_bb_size = [tracked_stracks[inter[0]].last_bb_size[0] + tracked_stracks[inter[0]].x_speed*0.03, tracked_stracks[inter[0]].last_bb_size[1] + tracked_stracks[inter[0]].y_speed*0.03, tracked_stracks[inter[0]].last_bb_size[2] + tracked_stracks[inter[0]].x_speed*0.03, tracked_stracks[inter[0]].last_bb_size[3] + tracked_stracks[inter[0]].y_speed*0.03]
    #                             alt_track = copy.deepcopy(tracked_stracks[inter[0]])
    #                             alt_track.behind = True
    #                             alt_track.copied = True
    #                             # alt_track.track_id = -alt_track.track_id
    #                             alt_track.copy_of = tracked_stracks[inter[0]].track_id
    #                             lost_stracks.append(alt_track)
    #                             tracked_stracks[inter[0]].copied = True
                            
    #         else:
    #             tracked_stracks[inter[0]].occluded = False
    #             tracked_stracks[inter[1]].occluded = False      
    #             tracked_stracks[inter[0]].behind = False
    #             tracked_stracks[inter[1]].behind = False
    #             tracked_stracks[inter[0]].copied = False
    #             tracked_stracks[inter[1]].copied = False
    #             tracked_stracks[inter[0]].copy_of == None
    #             tracked_stracks[inter[1]].copy_of == None

    #     # print("tracked_stracks")
    #     # for i in tracked_stracks:
    #     #     print(i)
    #     #     print(i.tlbr)
    #     #     print(i.behind)
    #     #     print(i.mean[4], i.mean[5])
    #     # print("lost_stracks")
    #     # for i in self.lost_stracks:
    #     #     print(i)
    #     #     print(i.tlbr)
    #     #     print(i.behind)
    #     #     print(i.mean[4], i.mean[5])
    #     # print("end")          
            
    #     ''' Step 2: First association, with high score detection boxes'''
    #     strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)      
    #     print(strack_pool)
    #     for i in strack_pool:
    #         print("TRACK:", i, "Behind:", i.behind, "STATE:", i.state, "COPY OF:", i.copy_of, "HAS COPY:", i.copied)
    #     # Predict the current location with KF
    #     STrack.multi_predict(strack_pool)
                
    #     # Calcula el solapamiento entre los bbs que ya han sido trackeados y los perdidos; y los que llegan en el frame actual
    #     dists = matching.iou_distance(strack_pool, detections)
    #     print(dists)
    #     if not self.mot20:
    #         dists = matching.fuse_score(dists, detections)
    #     # Calcula la matriz del húngaro
    #     matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)
    #     # Recorre lo matcheado
    #     for itracked, idet in matches:
    #         track = strack_pool[itracked]
    #         # Si hace match con una copia que no ha desaparecido, salta
    #         if (track.copy_of != None and track.behind) or track.occluded:
    #             print("MATCHED WITH COPY")
    #             track.mark_lost()
    #             lost_stracks.append(track)
    #             continue

    #         # Al coincidir con un dato que entra del frame actual, actualiza Kalman
    #         det = detections[idet]
    #         print(det, "match with", track)

    #         if track.state == TrackState.Tracked:
                
    #             track.update(detections[idet], self.frame_id)
    #             activated_starcks.append(track)
    #         # En el caso de que se haya reencontrado a la persona, el kalman se reactiva
    #         else:
    #             track.re_activate(det, self.frame_id, new_id=False)
    #             refind_stracks.append(track)

    #     ''' Step 3: Second association, with low score detection boxes'''

    #     # association the untrack to the low score detections
    #     if len(dets_second) > 0:
    #         '''Detections'''
    #         detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
    #                       (tlbr, s) in zip(dets_second, scores_second)]
    #     else:
    #         detections_second = []
    #     r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
    #     dists = matching.iou_distance(r_tracked_stracks, detections_second)
        
    #     matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
    #     for itracked, idet in matches:
    #         copied_track = None
    #         track = r_tracked_stracks[itracked]                   
    #         det = detections_second[idet]
    #         # Si ha matcheado pero está behind, mejor darlo por perdido
    #         if (track.copy_of != None and track.behind) or track.occluded:
    #             print("MATCHED WITH COPY")
    #             track.mark_lost()
    #             lost_stracks.append(track)
    #             continue
    #         if track.state == TrackState.Tracked:
    #             track.update(det, self.frame_id)
    #             activated_starcks.append(track)
    #         else:
    #             # print("REIDENTIFIED")
    #             track.re_activate(det, self.frame_id, new_id=False)
    #             refind_stracks.append(track)

    #     for it in u_track:
    #         track = r_tracked_stracks[it]
            
    #         if not track.state == TrackState.Lost:
    #             track.mark_lost()
    #             lost_stracks.append(track)

    #     ''' Step 4: Deal with unconfirmed tracks, usually tracks with only one beginning frame'''

    #     detections = [detections[i] for i in u_detection]
    #     dists = matching.iou_distance(unconfirmed, detections)
    #     if not self.mot20:
    #         dists = matching.fuse_score(dists, detections)

    #     matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
    #     for itracked, idet in matches:
    #         unconfirmed[itracked].update(detections[idet], self.frame_id)
    #         activated_starcks.append(unconfirmed[itracked])
    #     for it in u_unconfirmed:
    #         track = unconfirmed[it]
    #         if not track.copied:
    #             track.mark_removed()
    #             removed_stracks.append(track)

    #     """ Step 4: Init new stracks"""
    #     for inew in u_detection:
    #         track = detections[inew]
    #         if track.score < self.det_thresh:
    #             continue
    #         track.activate(self.kalman_filter, self.frame_id)
    #         activated_starcks.append(track)
    #     """ Step 5: Update state"""
    #     for track in self.lost_stracks:
    #         # print("PERSON END FRAME:", track.end_frame)
    #         if track.behind or track.occluded:
    #             track.behind = False
    #             track.occluded = False
                
    #         if self.frame_id - track.end_frame > self.max_time_lost:
    #             print("REMOVED:", track)
    #             track.mark_removed()
    #             removed_stracks.append(track)

    #     # print('Ramained match {} s'.format(t4-t3))

    #     self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
    #     self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
    #     self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
    #     self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
    #     self.lost_stracks.extend(lost_stracks)
    #     self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
    #     self.removed_stracks.extend(removed_stracks)
    #     self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
    #     # get scores of lost tracks
    #     print("SELF TRACKED STRACKS", self.tracked_stracks)
    #     print("LOST STRACKS", self.lost_stracks)
    #     output_stracks = [track for track in self.tracked_stracks if track.is_activated]
    #     for i in strack_pool:
    #         print("TRACK:", i, "Behind:", i.behind, "STATE:", i.state, "COPY OF:", i.copy_of, "HAS COPY:", i.copied)

    #     return output_stracks

    # def update2_old(self, scores, bboxes, img_info, img_size, depth_image):
    #     # print("self.tracked_stracks:", self.tracked_stracks)
    #     self.frame_id += 1
    #     # print("FRAME ID:", self.frame_id)
    #     activated_starcks = []
    #     refind_stracks = []
    #     lost_stracks = []
    #     removed_stracks = []

    #     # img_h, img_w = img_info[0], img_info[1]
    #     # scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
    #     # bboxes /= scale

    #     remain_inds = scores > self.track_thresh
    #     inds_low = scores > 0.1
    #     inds_high = scores < self.track_thresh
    #     # print("SCORES:", scores)

    #     inds_second = np.logical_and(inds_low, inds_high)
    #     dets_second = bboxes[inds_second]
    #     dets = bboxes[remain_inds]
    #     scores_keep = scores[remain_inds]
    #     scores_second = scores[inds_second]
    #     #print("Boxes ", dets, "Scores ", scores_keep)

    #     if len(dets) > 0:
    #         '''Detections'''
    #         detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
    #                       (tlbr, s) in zip(dets, scores_keep)]
    #     else:
    #         detections = []
            
    #     ''' Add newly detected tracklets to tracked_stracks'''
    #     unconfirmed = []
    #     tracked_stracks = []  # type: list[STrack]
    #     for track in self.tracked_stracks:
    #         if not track.is_activated:
    #             unconfirmed.append(track)
    #         else:
    #             tracked_stracks.append(track)

    #     ''' Step 2: First association, with high score detection boxes'''
    #     strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
    #     for track in strack_pool:            
    #         print(track)
    #         print("BOUNDING:", track.tlbr)
    #     # Predict the current location with KF
    #     STrack.multi_predict(strack_pool)
                
    #     dists = matching.iou_distance(strack_pool, detections)
    #     if not self.mot20:
    #         dists = matching.fuse_score(dists, detections)

    #     matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

    #     for itracked, idet in matches:
    #         track = strack_pool[itracked]
    #         det = detections[idet]
    #         # if not track.occluded:
    #             # print(track, "Kalman enabled")
    #         if track.state == TrackState.Tracked:
    #             track.update(detections[idet], self.frame_id)
    #             activated_starcks.append(track)
    #         else:
    #             track.re_activate(det, self.frame_id, new_id=False)
    #             refind_stracks.append(track)
    #         # else:
    #         #     # print(track, "Kalman disabled")
    #         #     track.update(detections[idet], self.frame_id, True)
    #         #     activated_starcks.append(track)

    #     ''' Step 3: Second association, with low score detection boxes'''

    #     # association the untrack to the low score detections
    #     if len(dets_second) > 0:
    #         '''Detections'''
    #         detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
    #                       (tlbr, s) in zip(dets_second, scores_second)]
    #     else:
    #         detections_second = []
    #     r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
    #     dists = matching.iou_distance(r_tracked_stracks, detections_second)
        
    #     matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
    #     for itracked, idet in matches:
    #         track = r_tracked_stracks[itracked]
    #         det = detections_second[idet]
    #         if track.state == TrackState.Tracked:
    #             track.update(det, self.frame_id)
    #             activated_starcks.append(track)
    #         else:
    #             # print("REIDENTIFIED")
    #             track.re_activate(det, self.frame_id, new_id=False)
    #             refind_stracks.append(track)

    #     for it in u_track:
    #         track = r_tracked_stracks[it]
    #         if not track.state == TrackState.Lost:
    #             track.mark_lost()
    #             lost_stracks.append(track)

    #     ''' Step 4: Deal with unconfirmed tracks, usually tracks with only one beginning frame'''

    #     detections = [detections[i] for i in u_detection]
    #     dists = matching.iou_distance(unconfirmed, detections)
    #     if not self.mot20:
    #         dists = matching.fuse_score(dists, detections)

    #     matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
    #     for itracked, idet in matches:
    #         unconfirmed[itracked].update(detections[idet], self.frame_id)
    #         activated_starcks.append(unconfirmed[itracked])
    #     for it in u_unconfirmed:
    #         track = unconfirmed[it]
    #         track.mark_removed()
    #         removed_stracks.append(track)

    #     """ Step 4: Init new stracks"""
    #     for inew in u_detection:
    #         track = detections[inew]
    #         if track.score < self.det_thresh:
    #             continue
    #         track.activate(self.kalman_filter, self.frame_id)
    #         activated_starcks.append(track)
    #     """ Step 5: Update state"""
    #     for track in self.lost_stracks:
    #         # print("PERSON END FRAME:", track.end_frame)
    #         if self.frame_id - track.end_frame > self.max_time_lost:
    #             track.mark_removed()
    #             removed_stracks.append(track)
    #             print("HAS A COPY:". track.has_copy)
    #             if track.has_copy == True:
                    
    #                 for trk in self.tracked_stracks:
    #                     if trk.track_id < 0 and trk.copy_of == track.track_id:
    #                         trk.track_id = track.track_id

    #     # print('Ramained match {} s'.format(t4-t3))

    #     self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
    #     self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
    #     self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
    #     self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
    #     self.lost_stracks.extend(lost_stracks)
    #     self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
    #     self.removed_stracks.extend(removed_stracks)
    #     self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
    #     # get scores of lost tracks
    #     output_stracks = [track for track in self.tracked_stracks if track.is_activated]

    #     return output_stracks

    # def update2(self, scores, bboxes, img_info, img_size):
    #     print("self.tracked_stracks:", self.tracked_stracks)
    #     self.frame_id += 1
    #     # print("FRAME ID:", self.frame_id)
    #     activated_starcks = []
    #     refind_stracks = []
    #     lost_stracks = []
    #     removed_stracks = []

    #     # img_h, img_w = img_info[0], img_info[1]
    #     # scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
    #     # bboxes /= scale

    #     remain_inds = scores > self.track_thresh
    #     inds_low = scores > 0.1
    #     inds_high = scores < self.track_thresh
    #     # print("SCORES:", scores)

    #     inds_second = np.logical_and(inds_low, inds_high)
    #     dets_second = bboxes[inds_second]
    #     dets = bboxes[remain_inds]
    #     scores_keep = scores[remain_inds]
    #     scores_second = scores[inds_second]
    #     #print("Boxes ", dets, "Scores ", scores_keep)

    #     if len(dets) > 0:
    #         '''Detections'''
    #         detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
    #                         (tlbr, s) in zip(dets, scores_keep)]
    #     else:
    #         detections = []
                        
    #     ''' Add newly detected tracklets to tracked_stracks'''
    #     unconfirmed = []
    #     tracked_stracks = []  # type: list[STrack]
    #     for track in self.tracked_stracks:
    #         if not track.is_activated:
    #             unconfirmed.append(track)
    #         else:
    #             tracked_stracks.append(track)
    #     # print("TRACKED STRACKS LEN", len(tracked_stracks))
    #     # print("LOST STRACKS LEN", len(self.lost_stracks))
    #     ''' Step 2: First association, with high score detection boxes'''
    #     strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
    #     # 

    #     # Predict the current location with KF
    #     STrack.multi_predict(strack_pool)
    #     for strack in strack_pool:   
    #         print("STRACK POOL:", strack)
    #         print("STRACK POOL MEAN:", [round(item) for item in strack.tlbr])
    #     dists = matching.iou_distance(strack_pool, detections)
    #     if not self.mot20:
    #         dists = matching.fuse_score(dists, detections)
    #     print("")
    #     print("STEP 2")
    #     print("")
    #     print("DISTS MATRIX:", dists)
    #     matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

    #     for itracked, idet in matches:
    #         track = strack_pool[itracked]
    #         det = detections[idet]
    #         if track.state == TrackState.Tracked:
    #             track.update(detections[idet], self.frame_id)
    #             activated_starcks.append(track)
    #         else:
    #             track.re_activate(det, self.frame_id, new_id=False)
    #             refind_stracks.append(track)

    #     ''' Step 3: Second association, with low score detection boxes'''
    #     print("")
    #     print("STEP 3")
    #     print("")
    #     # association the untrack to the low score detections
    #     if len(dets_second) > 0:
    #         '''Detections'''
    #         detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
    #                         (tlbr, s) in zip(dets_second, scores_second)]
    #     else:
    #         detections_second = []
    #     r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
    #     dists = matching.iou_distance(r_tracked_stracks, detections_second)
        
    #     matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
    #     for itracked, idet in matches:
    #         track = r_tracked_stracks[itracked]
    #         det = detections_second[idet]
    #         if track.state == TrackState.Tracked:
    #             track.update(det, self.frame_id)
    #             activated_starcks.append(track)
    #         else:
    #             print("REIDENTIFIED")
    #             track.re_activate(det, self.frame_id, new_id=False)
    #             refind_stracks.append(track)

    #     for it in u_track:
    #         track = r_tracked_stracks[it]
    #         if not track.state == TrackState.Lost:
    #             track.mark_lost()
    #             lost_stracks.append(track)

    #     '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
    #     # print("")
    #     # print("STEP 4")
    #     # print("")
    #     detections = [detections[i] for i in u_detection]
    #     dists = matching.iou_distance(unconfirmed, detections)
    #     if not self.mot20:
    #         dists = matching.fuse_score(dists, detections)
    #     print("DISTS MATRIX:", dists)
    #     matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
    #     for itracked, idet in matches:
    #         unconfirmed[itracked].update(detections[idet], self.frame_id)
    #         activated_starcks.append(unconfirmed[itracked])
    #     for it in u_unconfirmed:
    #         track = unconfirmed[it]
    #         track.mark_removed()
    #         removed_stracks.append(track)

    #     """ Step 4: Init new stracks"""
    #     for inew in u_detection:
    #         track = detections[inew]
    #         if track.score < self.det_thresh:
    #             continue
    #         track.activate(self.kalman_filter, self.frame_id)
    #         activated_starcks.append(track)
    #     """ Step 5: Update state"""
    #     for track in self.lost_stracks:
    #         # print("PERSON END FRAME:", track.end_frame)
    #         print("NOT SEEING TIME:", self.frame_id - track.end_frame)
    #         if self.frame_id - track.end_frame > self.max_time_lost:
    #             track.mark_removed()
    #             removed_stracks.append(track)

    #     # print('Ramained match {} s'.format(t4-t3))

    #     self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
    #     self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
    #     self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
    #     self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
    #     self.lost_stracks.extend(lost_stracks)
    #     self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
    #     self.removed_stracks.extend(removed_stracks)
    #     self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
    #     # get scores of lost tracks
    #     output_stracks = [track for track in self.tracked_stracks if track.is_activated]

    #     return output_stracks

    # def update(self, output_results, img_info, img_size):
    #     self.frame_id += 1
    #     activated_starcks = []
    #     refind_stracks = []
    #     lost_stracks = []
    #     removed_stracks = []

    #     if output_results.shape[1] == 5:
    #         scores = output_results[:, 4]
    #         bboxes = output_results[:, :4]
    #     else:
    #         output_results = output_results.cpu().numpy()
    #         scores = output_results[:, 4] * output_results[:, 5]
    #         bboxes = output_results[:, :4]  # x1y1x2y2
    #     img_h, img_w = img_info[0], img_info[1]
    #     scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
    #     bboxes /= scale

    #     remain_inds = scores > self.track_thresh
    #     inds_low = scores > 0.1
    #     inds_high = scores < self.track_thresh

    #     inds_second = np.logical_and(inds_low, inds_high)
    #     dets_second = bboxes[inds_second]
    #     dets = bboxes[remain_inds]
    #     scores_keep = scores[remain_inds]
    #     scores_second = scores[inds_second]
    #     #print("Box ", dets, "Score ", scores_second)
    #     if len(dets) > 0:
    #         '''Detections'''
    #         detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
    #                       (tlbr, s) in zip(dets, scores_keep)]
    #     else:
    #         detections = []

    #     ''' Add newly detected tracklets to tracked_stracks'''
    #     unconfirmed = []
    #     tracked_stracks = []  # type: list[STrack]
    #     for track in self.tracked_stracks:
    #         if not track.is_activated:
    #             unconfirmed.append(track)
    #         else:
    #             tracked_stracks.append(track)

    #     ''' Step 2: First association, with high score detection boxes'''
    #     strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
    #     # Predict the current location with KF
    #     STrack.multi_predict(strack_pool)
    #     dists = matching.iou_distance(strack_pool, detections)
    #     if not self.mot20:
    #         dists = matching.fuse_score(dists, detections)
    #     matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

    #     for itracked, idet in matches:
    #         track = strack_pool[itracked]
    #         det = detections[idet]
    #         if track.state == TrackState.Tracked:
    #             track.update(detections[idet], self.frame_id)
    #             activated_starcks.append(track)
    #         else:
    #             track.re_activate(det, self.frame_id, new_id=False)
    #             refind_stracks.append(track)

    #     ''' Step 3: Second association, with low score detection boxes'''
    #     # association the untrack to the low score detections
    #     if len(dets_second) > 0:
    #         '''Detections'''
    #         detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
    #                       (tlbr, s) in zip(dets_second, scores_second)]
    #     else:
    #         detections_second = []
    #     r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

    #     dists = matching.iou_distance(r_tracked_stracks, detections_second)
    #     matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
    #     for itracked, idet in matches:
    #         track = r_tracked_stracks[itracked]
    #         det = detections_second[idet]
    #         if track.state == TrackState.Tracked:
    #             track.update(det, self.frame_id)
    #             activated_starcks.append(track)
    #         else:
    #             track.re_activate(det, self.frame_id, new_id=False)
    #             refind_stracks.append(track)

    #     for it in u_track:
    #         track = r_tracked_stracks[it]
    #         if not track.state == TrackState.Lost:
    #             track.mark_lost()
    #             lost_stracks.append(track)

    #     '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
    #     detections = [detections[i] for i in u_detection]
    #     dists = matching.iou_distance(unconfirmed, detections)
    #     if not self.mot20:
    #         dists = matching.fuse_score(dists, detections)
    #     matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
    #     for itracked, idet in matches:
    #         unconfirmed[itracked].update(detections[idet], self.frame_id)
    #         activated_starcks.append(unconfirmed[itracked])
    #     for it in u_unconfirmed:
    #         track = unconfirmed[it]
    #         track.mark_removed()
    #         removed_stracks.append(track)

    #     """ Step 4: Init new stracks"""
    #     for inew in u_detection:
    #         track = detections[inew]
    #         if track.score < self.det_thresh:
    #             continue
    #         track.activate(self.kalman_filter, self.frame_id)
    #         activated_starcks.append(track)
    #     """ Step 5: Update state"""
    #     for track in self.lost_stracks:
    #         if self.frame_id - track.end_frame > self.max_time_lost:
    #             track.mark_removed()
    #             removed_stracks.append(track)

    #     # print('Ramained match {} s'.format(t4-t3))

    #     self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
    #     self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
    #     self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
    #     self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
    #     self.lost_stracks.extend(lost_stracks)
    #     self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
    #     self.removed_stracks.extend(removed_stracks)
    #     self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
    #     # get scores of lost tracks
    #     output_stracks = [track for track in self.tracked_stracks if track.is_activated]

    #     return output_stracks


def joint_stracks(tlista, tlistb):
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
        # elif t.has_copy or t.copy_of != None:
            # res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
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
