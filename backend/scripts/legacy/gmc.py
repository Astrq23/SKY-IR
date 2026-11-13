import cv2
import numpy as np

class GMC:
    """Global Motion Compensation class."""
    def __init__(self, downscale=2):
        self.downscale = downscale
        self.detector = cv2.FastFeatureDetector_create(threshold=20)
        self.prev_gray = None
        self.prev_kps = None

    def apply(self, raw_frame, tracks):
        height, width = raw_frame.shape[:2]
        frame_gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            frame_gray = cv2.resize(frame_gray, (width // self.downscale, height // self.downscale))
        
        kps = self.detector.detect(frame_gray, None)
        kps = np.float32([kp.pt for kp in kps])

        if self.prev_gray is None or self.prev_kps is None or len(self.prev_kps) == 0 or len(kps) == 0:
            self.prev_gray = frame_gray
            self.prev_kps = kps
            return tracks

        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, self.prev_kps, None)

        if p1 is not None:
            status = st.flatten() == 1
            good_old = self.prev_kps[status]
            good_new = p1[status]
        else:
            good_old = np.array([])
            good_new = np.array([])

        if len(good_old) < 10:
            self.prev_gray = frame_gray
            self.prev_kps = kps
            return tracks

        m, inliers = cv2.estimateAffinePartial2D(good_old, good_new)
        if m is not None:
            if self.downscale > 1:
                m[0, 2] *= self.downscale
                m[1, 2] *= self.downscale
            for track in tracks:
                track.apply_gmc(m)

        self.prev_gray = frame_gray
        self.prev_kps = kps
        return tracks