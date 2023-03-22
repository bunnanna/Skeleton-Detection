from mediapipe.python.solutions.drawing_styles import *

# Pose
_THICKNESS_POSE_LANDMARKS = 1
_POSE_LANDMARKS_LEFT = frozenset([
    PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE,
    PoseLandmark.LEFT_EYE_OUTER, PoseLandmark.LEFT_EAR, PoseLandmark.MOUTH_LEFT,
    PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW,
    PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_PINKY, PoseLandmark.LEFT_INDEX,
    PoseLandmark.LEFT_THUMB, PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE,
    PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL,
    PoseLandmark.LEFT_FOOT_INDEX
])

_POSE_LANDMARKS_RIGHT = frozenset([
    PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE,
    PoseLandmark.RIGHT_EYE_OUTER, PoseLandmark.RIGHT_EAR,
    PoseLandmark.MOUTH_RIGHT, PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST,
    PoseLandmark.RIGHT_PINKY, PoseLandmark.RIGHT_INDEX,
    PoseLandmark.RIGHT_THUMB, PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE,
    PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL,
    PoseLandmark.RIGHT_FOOT_INDEX
])

#custer landmark
_POSE_LANDMARKS_NOSE=frozenset([PoseLandmark.NOSE])

_POSE_LANDMARKS_EYES=frozenset([
    PoseLandmark.LEFT_EYE_INNER,PoseLandmark.LEFT_EYE,
    PoseLandmark.LEFT_EYE_OUTER,PoseLandmark.RIGHT_EYE_INNER,
    PoseLandmark.RIGHT_EYE,PoseLandmark.RIGHT_EYE_OUTER
    ])

_POSE_LANDMARKS_EARS=frozenset([PoseLandmark.LEFT_EAR, PoseLandmark.RIGHT_EAR])

_POSE_LANDMARKS_MOUTH=frozenset([PoseLandmark.MOUTH_LEFT, PoseLandmark.MOUTH_RIGHT])

_POSE_LANDMARKS_BODY=frozenset([
  PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER,
  PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP
  ])

_POSE_LANDMARKS_LEFT_ARM=frozenset([
  PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_PINKY, 
  PoseLandmark.LEFT_INDEX, PoseLandmark.LEFT_THUMB
  ])

_POSE_LANDMARKS_RIGHT_ARM=frozenset([
  PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_PINKY, 
  PoseLandmark.RIGHT_INDEX, PoseLandmark.RIGHT_THUMB
  ])

_POSE_LANDMARKS_LEFT_LEG=frozenset([
  PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE, 
  PoseLandmark.LEFT_HEEL, PoseLandmark.LEFT_FOOT_INDEX
  ])

_POSE_LANDMARKS_RIGHT_LEG=frozenset([
  PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE, 
  PoseLandmark.RIGHT_HEEL, PoseLandmark.RIGHT_FOOT_INDEX
  ])

def get_custom_pose_landmarks_style() -> Mapping[int, DrawingSpec]:
  """Returns the custom pose landmarks drawing style.

  Returns:
      A mapping from each pose landmark to its custom drawing spec.
  """
  pose_landmark_style = {}
  for landmark in _POSE_LANDMARKS_NOSE:
    pose_landmark_style[landmark] = DrawingSpec(color=(239, 63, 82), thickness=_THICKNESS_POSE_LANDMARKS)
  for landmark in _POSE_LANDMARKS_EYES:
    pose_landmark_style[landmark] = DrawingSpec(color= (73, 178, 218), thickness=0)
  for landmark in _POSE_LANDMARKS_EARS:
    pose_landmark_style[landmark] = DrawingSpec(color=(255, 188, 63), thickness=_THICKNESS_POSE_LANDMARKS/2)
  for landmark in _POSE_LANDMARKS_MOUTH:
    pose_landmark_style[landmark] = DrawingSpec(color=(149, 87, 166), thickness=0)
  for landmark in _POSE_LANDMARKS_BODY:
    pose_landmark_style[landmark] = DrawingSpec(color=(102, 204, 153), thickness=_THICKNESS_POSE_LANDMARKS*5)
  for landmark in _POSE_LANDMARKS_LEFT_ARM:
    pose_landmark_style[landmark] = DrawingSpec(color=(255, 102, 102), thickness=_THICKNESS_POSE_LANDMARKS*5)
  for landmark in _POSE_LANDMARKS_RIGHT_ARM:
    pose_landmark_style[landmark] = DrawingSpec(color=(51, 153, 255), thickness=_THICKNESS_POSE_LANDMARKS*5)
  for landmark in _POSE_LANDMARKS_LEFT_LEG:
    pose_landmark_style[landmark] = DrawingSpec(color=(255, 204, 51), thickness=_THICKNESS_POSE_LANDMARKS*5)
  for landmark in _POSE_LANDMARKS_RIGHT_LEG:
    pose_landmark_style[landmark] = DrawingSpec(color=(153, 51, 204), thickness=_THICKNESS_POSE_LANDMARKS*5)
  return pose_landmark_style

# pose connection
# pose_connection = frozenset([(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
#                               (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
#                               (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
#                               (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
#                               (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
#                               (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
#                               (29, 31), (30, 32), (27, 31), (28, 32),
#                               # custom
#                               # (10,12),(9,11)
#                               ])
pose_connection = frozenset([(PoseLandmark.NOSE, PoseLandmark.LEFT_EYE_INNER), 
                             (PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE), 
                             (PoseLandmark.LEFT_EYE, PoseLandmark.LEFT_EYE_OUTER), 
                             (PoseLandmark.LEFT_EYE_OUTER, PoseLandmark.LEFT_EAR), 
                             (PoseLandmark.NOSE, PoseLandmark.RIGHT_EYE_INNER), 
                             (PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE),
                             (PoseLandmark.RIGHT_EYE, PoseLandmark.RIGHT_EYE_OUTER), 
                             (PoseLandmark.RIGHT_EYE_OUTER, PoseLandmark.RIGHT_EAR),
                             (PoseLandmark.MOUTH_LEFT, PoseLandmark.MOUTH_RIGHT), 
                             (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER), 
                             (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
                             (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST), 
                             (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_PINKY), 
                             (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_INDEX), 
                             (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_THUMB), 
                             (PoseLandmark.LEFT_PINKY, PoseLandmark.LEFT_INDEX),
                             (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW), 
                             (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST), 
                             (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_PINKY), 
                             (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_INDEX), 
                             (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_THUMB),
                             (PoseLandmark.RIGHT_PINKY, PoseLandmark.RIGHT_INDEX), 
                             (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP), 
                             (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP), 
                             (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP), 
                             (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
                             (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE), 
                             (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE), 
                             (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE), 
                             (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL), 
                             (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL),
                             (PoseLandmark.LEFT_HEEL, PoseLandmark.LEFT_FOOT_INDEX), 
                             (PoseLandmark.RIGHT_HEEL, PoseLandmark.RIGHT_FOOT_INDEX), 
                             (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_FOOT_INDEX), 
                             (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_FOOT_INDEX),
                             # custom
                             # (10,12),(9,11)
                             ])

#custer connection

_POSE_CONNECTIONS_HEAD=frozenset([
  
  (PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE), 
  (PoseLandmark.LEFT_EYE, PoseLandmark.LEFT_EYE_OUTER), 
  (PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE),
  (PoseLandmark.RIGHT_EYE, PoseLandmark.RIGHT_EYE_OUTER), 
  (PoseLandmark.MOUTH_LEFT, PoseLandmark.MOUTH_RIGHT)
])
_POSE_CONNECTIONS_NO_DRAW=frozenset([
  (PoseLandmark.NOSE, PoseLandmark.LEFT_EYE_INNER), 
  (PoseLandmark.LEFT_EYE_OUTER, PoseLandmark.LEFT_EAR), 
  (PoseLandmark.RIGHT_EYE_OUTER, PoseLandmark.RIGHT_EAR), 
  (PoseLandmark.NOSE, PoseLandmark.RIGHT_EYE_INNER), 
])

_POSE_CONNECTIONS_BODY=frozenset([
  (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
  (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP), 
  (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP), 
  (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP), 
  ])

_POSE_CONNECTIONS_LEFT_ARM=frozenset([
  (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
  (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST), 
  (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_PINKY), 
  (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_INDEX), 
  (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_THUMB), 
  (PoseLandmark.LEFT_PINKY, PoseLandmark.LEFT_INDEX)
  ])

_POSE_CONNECTIONS_RIGHT_ARM=frozenset([
  (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW), 
  (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST), 
  (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_PINKY), 
  (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_INDEX), 
  (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_THUMB),
  (PoseLandmark.RIGHT_PINKY, PoseLandmark.RIGHT_INDEX)
  ])

_POSE_CONNECTIONS_LEFT_LEG=frozenset([
  (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
  (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE), 
  (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL), 
  (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_FOOT_INDEX), 
  (PoseLandmark.LEFT_HEEL, PoseLandmark.LEFT_FOOT_INDEX), 
  
  ])

_POSE_CONNECTIONS_RIGHT_LEG=frozenset([
  (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
  (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE), 
  (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL), 
  (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_FOOT_INDEX), 
  (PoseLandmark.RIGHT_HEEL, PoseLandmark.RIGHT_FOOT_INDEX), 
  ])

_THICKNESS_POSE_CONNECTIONS=2
def get_custom_pose_connections_style() -> Mapping[int, DrawingSpec]:
  """Returns the custom pose connections drawing style.

  Returns:
      A mapping from each pose connection to its custom drawing spec.
  """
  pose_connection_style = {}
  for connection in _POSE_CONNECTIONS_HEAD:
    pose_connection_style[connection] = DrawingSpec(color=(223, 156, 168), thickness=_THICKNESS_POSE_CONNECTIONS)
  for connection in _POSE_CONNECTIONS_NO_DRAW:
    pose_connection_style[connection] = DrawingSpec(color=(223, 156, 168), thickness=0)
  for connection in _POSE_CONNECTIONS_BODY:
    pose_connection_style[connection] = DrawingSpec(color=(178, 225, 196), thickness=_THICKNESS_POSE_CONNECTIONS)
  for connection in _POSE_CONNECTIONS_LEFT_ARM:
    pose_connection_style[connection] = DrawingSpec(color=(255, 170, 170), thickness=_THICKNESS_POSE_CONNECTIONS)
  for connection in _POSE_CONNECTIONS_RIGHT_ARM:
    pose_connection_style[connection] = DrawingSpec(color=(170, 204, 255), thickness=_THICKNESS_POSE_CONNECTIONS)
  for connection in _POSE_CONNECTIONS_LEFT_LEG:
    pose_connection_style[connection] = DrawingSpec(color=(255, 229, 170), thickness=_THICKNESS_POSE_CONNECTIONS)
  for connection in _POSE_CONNECTIONS_RIGHT_LEG:
    pose_connection_style[connection] = DrawingSpec(color=(204, 170, 255), thickness=_THICKNESS_POSE_CONNECTIONS)
  return pose_connection_style

def get_custom_pose_connections_style_3d() -> Mapping[int, DrawingSpec]:
  """Returns the custom pose connections drawing style.

  Returns:
      A mapping from each pose connection to its custom drawing spec.
  """
  pose_connection_style = {}
  for connection in _POSE_CONNECTIONS_HEAD:
    pose_connection_style[connection] = DrawingSpec(color=(223, 156, 168), thickness=_THICKNESS_POSE_CONNECTIONS/2)
  for connection in _POSE_CONNECTIONS_NO_DRAW:
    pose_connection_style[connection] = DrawingSpec(color=(223, 156, 168), thickness=0)
  for connection in _POSE_CONNECTIONS_BODY:
    pose_connection_style[connection] = DrawingSpec(color=(178, 225, 196), thickness=_THICKNESS_POSE_CONNECTIONS*2)
  for connection in _POSE_CONNECTIONS_LEFT_ARM:
    pose_connection_style[connection] = DrawingSpec(color=(255, 170, 170), thickness=_THICKNESS_POSE_CONNECTIONS*2)
  for connection in _POSE_CONNECTIONS_RIGHT_ARM:
    pose_connection_style[connection] = DrawingSpec(color=(170, 204, 255), thickness=_THICKNESS_POSE_CONNECTIONS*2)
  for connection in _POSE_CONNECTIONS_LEFT_LEG:
    pose_connection_style[connection] = DrawingSpec(color=(255, 229, 170), thickness=_THICKNESS_POSE_CONNECTIONS*2)
  for connection in _POSE_CONNECTIONS_RIGHT_LEG:
    pose_connection_style[connection] = DrawingSpec(color=(204, 170, 255), thickness=_THICKNESS_POSE_CONNECTIONS*2)
  return pose_connection_style
