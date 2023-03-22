from Utility import *
from DisplayGraph import DisplayGraph
from Skeleton_Detection import Skeleton
from CameraInput import Camerainput
try:
    camera = Camerainput()
    dgraph = DisplayGraph(camera)
    pose = Skeleton()
    while True:
        dgraph.clear_data()
        image, success = camera.get_image()
        if not success: continue
            
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        dgraph.display(image,results)

        if dgraph.event=="close_event":
            break

except Exception as e:
    print(e)
finally:
    plt.disconnect(dgraph.click)
    plt.disconnect(dgraph.close)
    if mode==0: camera.cap.release()
    cv2.destroyAllWindows()
