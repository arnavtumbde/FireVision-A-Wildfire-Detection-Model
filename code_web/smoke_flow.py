import cv2
import numpy as np

def smoke_pixel_segmentation(image):
    hsv_im = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 50, 0], dtype=np.uint8)
    upper_bound = np.array([180, 200, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv_im, lower_bound, upper_bound)
    invert_mask = cv2.bitwise_not(mask)
    
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(invert_mask, kernel, iterations=1)
    smoke_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
    
    return smoke_mask

def smoke_flow(im, prev, smoke_mask, wind_dir, processed_frame, n):
    curr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
#     pyr_scale, levels, winsize, iterations, poly_n, poly_sigma
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 
                                        0.5, 3, 15, 5, 3, 1.5, 0)
    
    # Get average flow direction
    avgx = []
    avgy = []
     
    step = 15
    for y in range(0, processed_frame.shape[0], step):
        for x in range(0, processed_frame.shape[1], step):
            if smoke_mask[y, x] > 0:
                fx, fy = flow[y, x]
                flow_magnitude = np.sqrt(fx**2 + fy**2)
                
                if flow_magnitude > 0.5:
                    avgx.append(fx)
                    avgy.append(fy)

                    end_point = (int(x + fx), int(y + fy))
                    cv2.arrowedLine(processed_frame, (x, y), end_point, (255, 0, 0), 2, tipLength=3)

    avg_direction_angle = np.arctan2(np.mean(avgy), np.mean(avgx))
    angle_in_degrees = np.degrees(avg_direction_angle)
    cv2.putText(processed_frame, f'Smoke Direction: {angle_in_degrees:.2f}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    wind_dir.append(angle_in_degrees)
        
    smoke_dir_nframes = None
    if len(wind_dir) > n:
        smoke_dir_nframes = np.nanmean(wind_dir)
        wind_dir.pop(0)
    
    return curr, processed_frame, smoke_dir_nframes
