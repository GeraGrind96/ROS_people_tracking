import numpy as np
from collections import OrderedDict


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 1
    _class_count = {}

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    max_count = 50

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        if BaseTrack._count == 50:
            BaseTrack._count = 1
        BaseTrack._count += 1
        
        return BaseTrack._count

    @staticmethod
    def next_class_id(clase):
        if clase in BaseTrack._class_count.keys():
            if BaseTrack._class_count[clase] == BaseTrack.max_count:
                BaseTrack._class_count[clase] = 1
            BaseTrack._class_count[clase] += 1
        else:
            BaseTrack._class_count[clase] = 1
        return BaseTrack._class_count[clase]

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed
