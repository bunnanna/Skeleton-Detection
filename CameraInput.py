from Utility import *
class Camerainput:
    def __init__(self) -> None:
        self.depth_frame=None
        if mode==1:
            # Configure depth and color streams
            self.pipeline = rs.pipeline()
            config = rs.config()

            # Get device product line for setting a supporting resolution
            pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()

            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break
            if not found_rgb:
                print("The demo requires Depth camera with Color sensor")
                exit(0)

            config.enable_stream(rs.stream.depth, WIDTH,HEIGHT,  rs.format.z16, 30)
            config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30) 

            # Start streaming
            self.pipeline.start(config)
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            
            self.spat_filter = rs.spatial_filter(smooth_alpha=.25, smooth_delta=45, magnitude=2, hole_fill=1)          # Spatial    - edge-preserving spatial smoothing
            self.temp_filter = rs.temporal_filter(smooth_alpha=.1, smooth_delta= 100, persistence_control=3)

        elif mode==0:
            # # For webcam input:
            self.cap = cv2.VideoCapture(0)
      
    def convert_depth_to_phys_coord_using_realsense(self,x, y):
        #   _intrinsics = depth_intrinsics
        # _intrinsics.model  = rs.distortion.none
        _intrinsics = self.color_intrin
        depth = self.depth_frame.get_distance(x, y)
        result = rs.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)
        #result[0]: right, result[1]: down, result[2]: forward
        return result[2], result[0], -result[1]
    
    def get_image(self):
        if mode == 0:
            success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                return None, False
            return image,True
            
        elif mode == 1:
            frames = self.align.process(self.pipeline.wait_for_frames())
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            depth_frame = self.spat_filter.process(depth_frame)
            depth_frame = self.temp_filter.process(depth_frame)
            
            depth_frame = depth_frame.as_depth_frame()
            
            if not depth_frame or not color_frame:
                return None,False
            
            # Convert images to numpy arrays
            image = np.asanyarray(color_frame.get_data())
            self.depth_frame = depth_frame
            return image, True

if __name__ == "__main__":
    camera = Camerainput()
    print(camera.color_intrin)