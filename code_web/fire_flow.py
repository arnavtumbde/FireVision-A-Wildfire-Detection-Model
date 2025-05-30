import cv2
import numpy as np

# Seven Rules for Segmentation -> https://core.ac.uk/download/pdf/55305301.pdf
def fire_pixel_segmentation(image):
    fire_mask = np.zeros_like(image[:, :, 0])
    
    # YCbCr Color Space and splitting BGR Channel
    YCrCb_im = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(YCrCb_im)
    B, G, R = cv2.split(image)
    
    # Rule 1: R > G > B
    mask1 = (R > G) & (G > B)
    
    # Rule 2: R > 190 & G > 100 & B < 140
    mask2 = (R > 190) & (G > 100) & (B < 140)
    
    # Rule 3 & 4: Y >= Cb & Cr >= Cb
    mask3_4 = (Y >= Cb) & (Cr >= Cb)
    
    # Rule 5: Y >= Ymean & Cb <= Cbmean & Cr >= Crmean
    mask5 = (Y >= np.mean(Y)) & (Cb <= np.mean(Cb)) & (Cr >= np.mean(Cr))
    
    # Rule 6: Cb-Cr >= 70 (Threshold)
    mask6 = abs(Cb-Cr) >= 70
    
    # Rule 7: Cb <= 120 & Cr >= 150
    mask7 = (Cb <= 120) & (Cr >= 150)

    combined_mask = mask1 & mask2 & mask3_4 & mask5 & mask6 & mask7
    fire_mask[combined_mask] = 255

    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(fire_mask, kernel, iterations=1)
    fire_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
    
    return fire_mask

def fire_flow(fire_mask, area_frame, fireX, fireY, processed_frame, m):
    
    centerX = []
    centerY = []
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(processed_frame, contours, -1, (0, 0, 255), 2) 
    
    for contour in contours:
        # Compute the moments of the contour
        area = cv2.contourArea(contour)

        M = cv2.moments(contour)
        if M["m00"] != 0 and area > 100:
            # Calculate the center of the contour
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            centerX.append(cX)
            centerY.append(cY)
            
    fireX.append(np.mean(centerX))
    fireY.append(np.mean(centerY))
    
    endpoint_mframes = None
    if len(fireX) > m:
        x = np.nanmean(fireX)
        y = np.nanmean(fireY)
        
        if np.isfinite(x) and np.isfinite(y):
            endpoint_mframes = (int(x), int(y))
 
        fireX.pop(0)
        fireY.pop(0)


    total_area = 0
    cv2.drawContours(area_frame, contours, -1, (0, 0, 255), -1) 
    binary_image = cv2.cvtColor(area_frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(binary_image, 50, 255, cv2.THRESH_BINARY)
    
    area_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in area_contours:
        total_area += cv2.contourArea(contour)  
    
    return total_area, area_frame, processed_frame, endpoint_mframes
