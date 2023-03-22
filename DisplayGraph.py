from Utility import *
from custom_style import *
from functools import reduce
import operator
import math
from Geometry3D import *
from typing import List, Mapping, Optional, Tuple, Union

class DisplayGraph():
    def __init__(self,camera):
        self.camera = camera
        self.graph_init()
        self.config_plt_event()
        
    def graph_init(self):
        self.fig = plt.figure(figsize=(15,8))#
        plt.subplots_adjust(left=0.01,
                    bottom=0.01,
                    right=0.99,
                    top=0.99,
                    wspace=0,
                    hspace=0)
        self.ax1 = self.fig.add_subplot(1,2,1)
        self.ax2 = self.fig.add_subplot(1,2,2,projection='3d')
        self.azim=45
        self.elev=20
        self.ax2.view_init(elev=self.elev, azim=self.azim)
        self.rotate = False
        self.boundingbox=[]
        self.boundingbox3D=[]
 
    def config_plt_event(self):
        self.event=None
        def onclick(event):
            if (type(event.inaxes).__name__ in ["AxesSubplot","Axes"]):
                if len(self.boundingbox)>=4:
                    self.boundingbox=[]
                    self.boundingbox3D=[]
                    return
                    
                ix,iy=int(event.xdata),int(event.ydata)
                self.boundingbox.append((ix,iy))
                if self.camera.depth_frame is not None: self.boundingbox3D.append(self.camera.convert_depth_to_phys_coord_using_realsense(ix,iy)) 

            if (type(event.inaxes).__name__ in ["Axes3DSubplot","Axes3D"]) and event.dblclick:
                self.rotate = not self.rotate
            # print(event.dblclick)
            
        def onclose(event):
            self.event=event.name
            
        self.click = plt.connect('button_press_event',onclick)
        self.close = plt.connect('close_event',onclose)   
               
    def clear_data(self):
        self.ax1.cla()
        self.ax2.cla()
        self.config_label()
        
    def config_label(self):
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])

        self.ax2.set_xlim3d(0, 5) # z
        self.ax2.set_ylim3d(-1, 1) # x
        self.ax2.set_zlim3d(-1.5, 1.5) # y
        
        # self.ax2.set_facecolor('black')
        # self.ax2.xaxis.pane.fill = False
        # self.ax2.yaxis.pane.fill = False
        # self.ax2.zaxis.pane.fill = False
        # self.ax2.tick_params(axis='x',colors='white')
        # self.ax2.tick_params(axis='y',colors='white')
        # self.ax2.tick_params(axis='z',colors='white')
        
        self.ax2.w_xaxis.set_pane_color((0,0,0))
        self.ax2.w_yaxis.set_pane_color((0,0,0))
        self.ax2.w_zaxis.set_pane_color((.5,.5,.5))
        self.ax2.invert_xaxis()
        # self.ax2.axes.xaxis.set_ticklabels([])
        # self.ax2.axes.yaxis.set_ticklabels([])
        # self.ax2.axes.zaxis.set_ticklabels([])
        
    
    def display(self,img:np.asanyarray,results,rotation=(4,0)):
        
        self.graph_rotation(*rotation)
        self.draw_mark(img)
        self.draw_boundingbox(img)
        self.plot_mark()
        self.plot_boundingbox()
        if results is not None:
            self.draw_landmarks(img,results.pose_landmarks)
            self.plot_landmarks(results.pose_landmarks)
            self.bone_rect()
        self.ax1.imshow(img,aspect='equal')
        
            
        plt.pause(0.001)
        # plt.show()
        
    def graph_rotation(self,azim:int=0,elev:int=0):
        '''for rotate 3d graph'''
        def rotation_limit(v):
            if v > 360 : v-=360
            if v < 360 : v+=360
        if self.rotate:
            self.azim += azim
            self.elev += elev
            rotation_limit(self.azim)
            rotation_limit(self.elev)
            self.ax2.view_init(elev=self.elev, azim=self.azim)
            
    def draw_mark(self,img):
        for mark in self.boundingbox:
            cv2.circle(img,mark,5,(0,0,0),-1)
            cv2.circle(img,mark,5,(255,255,255),1) 
                
    def plot_mark(self):
        for mark3D in self.boundingbox3D:
            self.ax2.scatter3D(*mark3D,color=(1.,1.,1.),linewidth=5)    
                  
    def draw_boundingbox(self,img:np.asanyarray):
        if len(self.boundingbox)==4:
            bounding_sort_list = self.sort_boundingbox_coor(self.boundingbox)
            for st_coor, ed_coor in zip(bounding_sort_list,[bounding_sort_list[-1]]+bounding_sort_list[:-1]):
                cv2.line(img,st_coor, ed_coor,(0,0,0),5)   
                 
    def plot_boundingbox(self):
        if len(self.boundingbox3D)==4:
            bounding_sort_list = self.sort_boundingbox_coor(self.boundingbox3D)
            for st_coor, ed_coor in zip(bounding_sort_list,[bounding_sort_list[-1]]+bounding_sort_list[:-1]):
                self.ax2.plot3D(
              xs=[st_coor[0], ed_coor[0]],
              ys=[st_coor[1], ed_coor[1]],
              zs=[st_coor[2], ed_coor[2]],
              color=(1.,1.,1.),
              linewidth=1)
                   
    def sort_boundingbox_coor(self,boundingbox_point):
        coords = boundingbox_point
        if not len(coords[0]) in [2,3]:raise Exception("invalid point dimension")
        
        center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
        sorted_boundingbox_point= sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
        return sorted_boundingbox_point
    # @caltime
    def draw_landmarks(self,img,landmark_list,
                       connections=pose_connection,
                       landmark_drawing_spec=get_custom_pose_landmarks_style(),
                    connection_drawing_spec=get_custom_pose_connections_style()):
        if not landmark_list:
            return
        if img.shape[2] != 3:
            raise ValueError('Input image must contain three channel bgr data.')
        image_rows, image_cols, _ = img.shape
        idx_to_coordinates = {}
        for idx, landmark in enumerate(landmark_list.landmark):
            if ((landmark.HasField('visibility') and
                landmark.visibility < VISIBILITY_THRESHOLD)):
                continue
            landmark_px = self._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                        image_cols, image_rows)
            if landmark_px:
                idx_to_coordinates[idx] = landmark_px
        if connections:
            num_landmarks = len(landmark_list.landmark)
            # Draws the connections if the start and end landmarks are both visible.
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                    raise ValueError(f'Landmark index is out of range. Invalid connection '
                                    f'from landmark #{start_idx} to landmark #{end_idx}.')
                if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                    drawing_spec = connection_drawing_spec[connection] if isinstance(
                        connection_drawing_spec, Mapping) else connection_drawing_spec
                    if int(drawing_spec.thickness) <= 0:
                        continue
                    cv2.line(img, idx_to_coordinates[start_idx],
                            idx_to_coordinates[end_idx], drawing_spec.color[::-1],
                            int(drawing_spec.thickness))
            # Draws landmark points after finishing the connection lines, which is
            # aesthetically better.
            if landmark_drawing_spec:
                for idx, landmark_px in idx_to_coordinates.items():
                    drawing_spec = landmark_drawing_spec[idx] if isinstance(
                        landmark_drawing_spec, Mapping) else landmark_drawing_spec
                    if int(drawing_spec.thickness) <= 0:
                        continue
                    # White circle border
                    circle_border_radius = max(drawing_spec.circle_radius + 1,
                                            int(drawing_spec.circle_radius * 1.2))
                    cv2.circle(img, landmark_px, circle_border_radius, WHITE,
                            int(drawing_spec.thickness))
                    # Fill color into the circle
                    cv2.circle(img, landmark_px, drawing_spec.circle_radius,
                            drawing_spec.color[::-1], int(drawing_spec.thickness))
    # @caltime                
    def plot_landmarks(self,landmarks_list,connections=pose_connection,
                                landmark_drawing_spec=get_custom_pose_landmarks_style(),
                                connection_drawing_spec=get_custom_pose_connections_style_3d()):
        self.connection=[]
        if not landmarks_list:
            return

        plotted_landmarks = {}
        for idx, landmark in enumerate(landmarks_list.landmark):
            if (landmark.HasField('visibility') and
                    landmark.visibility < VISIBILITY_THRESHOLD):
                continue

            landmark_px = self._normalized_to_pixel_coordinates(landmark.x, landmark.y,WIDTH, HEIGHT)

            if landmark_px is None: continue
            
            landmark_color = landmark_drawing_spec.color[::-1] if type(
                landmark_drawing_spec) == DrawingSpec else landmark_drawing_spec[idx].color[::-1]
            landmark_thickness = landmark_drawing_spec.thickness if type(
                landmark_drawing_spec) == DrawingSpec else landmark_drawing_spec[idx].thickness

            landmark_thickness = max(int(landmark_thickness),1)

            if self.camera.depth_frame is not None:
                real_world_coor = self.camera.convert_depth_to_phys_coord_using_realsense(int(landmark_px[0]),int(landmark_px[1]))

                self.ax2.scatter3D(
                    *real_world_coor,
                    color=self._normalize_color(landmark_color),
                    linewidth=int(landmark_thickness))
                plotted_landmarks[idx] = real_world_coor
            else:
                self.ax2.scatter3D(
                    xs=[3+landmark.z],
                    ys=[landmark.x],
                    zs=[-landmark.y],
                    color=self._normalize_color(landmark_color),
                    linewidth=int(landmark_thickness))
                plotted_landmarks[idx] = (3+landmark.z, landmark.x, -landmark.y)
    #   draw_head(plotted_landmarks,ax)

        if connections:
            num_landmarks = len(landmarks_list.landmark)
            # Draws the connections if the start and end landmarks are both visible.
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                    raise ValueError(f'Landmark index is out of range. Invalid connection '
                                    f'from landmark #{start_idx} to landmark #{end_idx}.')
                if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                    landmark_pair = [
                        plotted_landmarks[start_idx], plotted_landmarks[end_idx]
                    ]
                    connection_color = connection_drawing_spec.color[::-1] if type(
                        connection_drawing_spec) == DrawingSpec else connection_drawing_spec[(start_idx, end_idx)].color[::-1]
                    connection_thickness = connection_drawing_spec.thickness if type(
                        connection_drawing_spec) == DrawingSpec else connection_drawing_spec[(start_idx, end_idx)].thickness
                    if int(connection_thickness) <= 0:
                        continue
                    try:
                        self.ax2.plot3D(
                            xs=[landmark_pair[0][0], landmark_pair[1][0]],
                            ys=[landmark_pair[0][1], landmark_pair[1][1]],
                            zs=[landmark_pair[0][2], landmark_pair[1][2]],
                            color=self._normalize_color(connection_color),
                            linewidth=int(connection_thickness))
                        self.connection.append(landmark_pair)
                    except:
                        print(connection_thickness)

    def plot_temp_line(self,start=(1,-.2,-.2),end=(4,.1,.2)):
        self.ax2.scatter3D(*start,(255,0,0),5)   
        self.ax2.scatter3D(*end,(255,0,0),5)
        self.ax2.plot3D(
              xs=[start[0], end[0]],
              ys=[start[1], end[1]],
              zs=[start[2], end[2]],
              color=(0,0,0),
              linewidth=1)              
        
        if self.check_position((start,end)):
            self.fig.set_facecolor((1.,0.5,.5))
        else :
            self.fig.set_facecolor((1.,1.,1.))   
    
    def bone_rect(self):
        
        if any(self.check_position(connection) for connection in self.connection):
            self.fig.set_facecolor((1.,0.5,.5))
        else :
            self.fig.set_facecolor((1.,1.,1.))
                       
    def _normalize_color(self,color):
        return tuple(v / 255. for v in color)
    
    def _normalized_to_pixel_coordinates(self,
            normalized_x: float, normalized_y: float, image_width: int,
            image_height: int) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates."""

        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                            math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px,y_px
    
    def check_position(self,connection):
        if len(self.boundingbox3D) <4: return False
        try:
            box=[Point(box_mark) for box_mark in self.sort_boundingbox_coor(self.boundingbox3D)]
            line=[Point(landmark) for landmark in connection]
            geo_line=Segment(*line)
            geo_box1=ConvexPolygon(tuple(box[0:3]))
            geo_box2=ConvexPolygon(tuple(box[1:4]))
            return geo_box1.intersection(geo_line) is not None or geo_box2.intersection(geo_line) is not None
        except:
            return False
        