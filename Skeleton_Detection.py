from Utility import *

class Skeleton:
    def __init__(self):
      self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
    
    def process(self,img:np.asanyarray):
      img.flags.writeable = False
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      return self.pose.process(img)