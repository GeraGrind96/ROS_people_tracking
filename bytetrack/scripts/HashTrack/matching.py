import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
import time

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
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

def hash_distance_following(followed_track, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    if (len(followed_track)>0 and isinstance(followed_track[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        ahashs = followed_track
        bhashs = btracks
    else:
        ahashs = [track.hash_memory for track in followed_track]
        bhashs = [track.hash for track in btracks]
    _hashes = histograms(ahashs[0], bhashs)

    cost_matrix = check_for_classes(followed_track, btracks, _hashes)
    return cost_matrix

def get_max_similarity_detection(distances_matrix, hash_memory):
    print("distances_matrix", distances_matrix)
    
    filtered_array, filtered_memory = remove_arrays_with_values_under_value(distances_matrix, hash_memory, 0.45)
    print("filtered_array", filtered_array)
    min_value_by_memory = np.argmin(filtered_array, axis=-1)
    
    # print("min_value_by_memory", min_value_by_memory)
    counts = np.bincount(min_value_by_memory)
    # print("counts", counts)
    if len(counts) == 0:
        return -1, []
    else:
        print("TIMES MATCHED:", counts[np.argmax(counts)])
        print("PROPORTION RESPECT TOTAL MEMORY:", 0.9 * len(distances_matrix[0]))
        if counts[np.argmax(counts)] > 0.9 * len(distances_matrix[0]):
            return np.argmax(counts), filtered_memory
        else:
            return -1, []

def remove_arrays_with_values_under_value(array, hash_memory, value):
    hash_memory = np.array(hash_memory)
    array_mask = np.any(array <= value, axis=-1)
    result_memory = hash_memory[array_mask[0]]

    return array[array_mask], result_memory.tolist()

def hash_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        ahashs = atracks
        bhashs = btracks
    else:
        ahashs = [track.hash for track in atracks]
        bhashs = [track.hash for track in btracks]
    _hashes = histograms(ahashs, bhashs)
    cost_matrix = check_for_classes(atracks, btracks, _hashes)
    return cost_matrix

def hashes(ahashs, bhashs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    hashes = np.zeros((len(ahashs), len(bhashs)))
    if hashes.size == 0:
        return hashes

    hashes = np.subtract.outer(
        np.ascontiguousarray(ahashs),
        np.ascontiguousarray(bhashs)
    )
    return hashes / 42

def histograms(ahashs, bhashs):
    histograms = np.zeros((len(ahashs), len(bhashs)))
    if histograms.size == 0:
        return histograms

    histograms1 = np.ascontiguousarray(ahashs)
    histograms2 = np.ascontiguousarray(bhashs)
    similarity_matrix = np.zeros((len(histograms1), len(histograms2)))
    for i, hist1 in enumerate(histograms1):
        for j, hist2 in enumerate(histograms2):
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            similarity_matrix[i, j] = similarity
    return 1 - similarity_matrix 

def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.bbox for track in atracks]
        btlbrs = [track.bbox for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    cost_matrix = check_for_classes(atracks, btracks, cost_matrix)
    return cost_matrix

def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious

def check_for_classes(atracks, btracks, cost_matrix):
    """
    Checks that elements i,j belong to the same class, otherwise sets cost to 1
    """
    for i, atrack in enumerate(atracks):
        for j, btrack in enumerate(btracks):
            if atrack.clase != btrack.clase:
                cost_matrix[i, j] = 1
    return cost_matrix


def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix

def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
    