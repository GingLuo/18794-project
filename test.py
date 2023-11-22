import cv2
import torch
from pathlib import Path
import pyrealsense2 as rs
import math
from collections import defaultdict
import numpy as np

# from yolov5.models.experimental import attempt_load

IMG_H = 480
IMG_W = 640
minimum_detection_distance = 0.5 
maximum_detection_distance = 10

def object_depth_measurement_square(depth_image, label, depth_scale):
    #print("the label is:", label)
    top_left, bottom_right = label 
    #print("at distance function:",top_left, bottom_right, l)
    (minx, miny) = top_left
    minx = math.floor(minx)
    miny = math.floor(miny)
    (maxx, maxy) = bottom_right
    maxx = math.floor(maxx)
    maxy = math.floor(maxy)
    depth_list = defaultdict(int)
    depth_list[0] = 1
    for x in range(miny, maxy):
        for y in range(minx, maxx):
            pixel_depth = depth_image[x][y][0]
            if pixel_depth > minimum_detection_distance/depth_scale and pixel_depth < maximum_detection_distance/depth_scale:
                #round to nearest tenth
                #print(x, y, pixel_depth)
                depth_list[round(pixel_depth,1)] += 1
    #print(depth_list)
    return max(depth_list, key=lambda x: depth_list[x]) * depth_scale



def get_mean_dis(depth_frame,boxs):
    count = 0
    sum_dis = 0
    for i in range(boxs[0],boxs[2]):
        for j in range(boxs[1],boxs[3]):
            if(depth_frame[j,i]!=0):
                sum_dis+=depth_frame[j,i]
                count+=1
    if count!=0:
        return (sum_dis/count)/1000
    else:
        return -1
    

def get_center_dis(depth_frame,boxs,img):
    h_dis =(boxs[3]-boxs[1])//3
    w_dis = (boxs[2]-boxs[0])//3

    count = 0
    sum_dis = 0
    for i in range(boxs[0]+w_dis,boxs[2]-w_dis):
        for j in range(boxs[1]+h_dis,boxs[3]-h_dis):
            if(depth_frame[j,i]!=0):
                sum_dis+=depth_frame[j,i]
                count+=1
    cv2.rectangle(img, (int(boxs[0]+w_dis), int(boxs[1]+h_dis)), (int(boxs[2]-w_dis), int(boxs[3]-h_dis)), (0, 0, 255), 1)
    

    if count!=0:
        return (sum_dis/count)/1000
    else:
        return -1
    

def detection(org_img, boxs,depth_frame,method=0):
    img = org_img.copy()

    
    for box in boxs:
        distance = 0
        if method==0:
            distance =  get_mean_dis(depth_frame,box[:4].astype(int))
        elif method==1:
            distance =  get_center_dis(depth_frame,box[:4].astype(int),img)
        cv2.putText(img,str(box[-1])+", distance:"+"{:.3f}".format(distance), 
                    (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)

    # x, y = 320, 240
    # distance = depth_frame.get_distance(x, y)

    # print(f"Distance at pixel ({x}, {y}): {distance} meters")

    return img
    

def realsense_setup():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, IMG_W, IMG_H, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.bgr8, 30)
    pipeline.start(config)

    return pipeline

def yolo_model_load():
    # Load YOLOv5 model
    model = torch.hub.load('/home/cc/18794_proj/yolov5', 'custom', path='yolo_model/best_m2.pt', source='local')
    # model = attempt_load('/home/cc/yolov5/weights')
    model2 = torch.hub.load('/home/cc/18794_proj/yolov5', 'custom', path='yolo_model/best_traffic.pt', source='local')

    model3 = torch.hub.load('/home/cc/18794_proj/yolov5', 'custom', path='yolo_model/yolov5n.pt', source='local')

    model.eval()
    model2.eval()
    model3.eval()

    return model,model2,model3


# Set up camera
# cap = cv2.VideoCapture(0)  # Use the default camera (change to a different camera index if needed)
# ret, frame = cap.read()
def get_model_img(model,color_image,depth_frame,method):
    results = model(color_image)
    boxs= results.pandas().xyxy[0].values
    return detection(color_image, boxs,depth_frame,method)


def runner_realsense():

    save_num = 6
    pipeline = realsense_setup()
    model1,model2,model3 = yolo_model_load()

    align_to = rs.stream.color
    align = rs.align(align_to)

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        # depth_image = depth_image.reshape((depth_image.shape[0],depth_image.shape[1],1))
        color_image = np.asanyarray(color_frame.get_data())
        depth_map = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,alpha=0.5),cv2.COLORMAP_JET)

        # img0 = frame
        # img = letterbox(img0, new_shape=640)[0]

        # # Convert BGR image to RGB
        # color_image = np.asanyarray(img0)

        
        color_img1= get_model_img(model3,color_image,depth_image,0)
        color_img2= get_model_img(model3,color_image,depth_image,1)
        # color_img2= get_model_img(model2,color_image,depth_frame)
        # color_img3= get_model_img(model3,color_image,depth_frame)
  
        # depth = depth_image[320,240].astype(float)
        # print(depth)
        # cv2.imshow('depth', depth_image)

        height, width, channels = color_image.shape

        combined_image = np.zeros((2*height, 2 * width, channels), dtype=np.uint8)
        combined_image[:height, :width, :] =color_img1
        # Copy the second image to the right half of the combined image
        combined_image[:height, width:, :] = color_img2
        # combined_image[height:, :width, :] =color_img3

        cv2.imshow('img', combined_image)

        # cv2.imshow('img', depth_image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            # do your stuff here
            # pass
            cv2.imwrite("tmp"+str(save_num)+".png", combined_image) 
            save_num+=1
            cv2.imwrite("color"+str(save_num)+".png", color_image) 
            cv2.imwrite("depth"+str(save_num)+".png", depth_map) 

    # Release the camera and close the OpenCV window
    # cap.release()
    cv2.destroyAllWindows()

runner_realsense()
