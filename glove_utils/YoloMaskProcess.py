# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 09:46:58 2023

@author: HP
"""
import cv2
import numpy as np
import copy

def process_img(img, mask, imgsz):
    imgsz = tuple(imgsz)
    orginal_shape = (img.shape[1], img.shape[0])
    img = cv2.resize(img, imgsz)
    mask = cv2.resize(mask, imgsz)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY) 

    slices = mask[:, :, np.newaxis]
    slices = np.all(slices != 0,axis = -1)

    # 1. blur inside the ROI------------------------------------------------
    ksize = (7, 7) # 模糊核大小
    img[slices, :] = cv2.blur(img, ksize)[slices, :]
    #-----------------------------------------------------------------------
    
    # 2. convert to CIELAB and count majority rigon-------------------------
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # ----------------------------------------------------------------------

    # 3. find the most concentrate part------------------------------------
    # 设置直方图参数
    hist_size = [8]
    ranges = [0, 256]

    # 计算直方图
    hist = cv2.calcHist([img], [0], mask, hist_size, ranges)
    
    # 找到最大值所在的坐标
    max_idx = np.argmax(hist)

    # 提取色彩集中的区域
    max_L = (max_idx - 1) * 32
    #-----------------------------------------------------------------------

    # low_bound < L < up_bound
    low_bound = 160
    
    # 140 < a,b < 155
    img = img.astype(np.float64)
    min_val = (np.min(img[slices, 0]), np.min(img[slices, 1]),  np.min(img[slices, 2]))
    max_val = (np.max(img[slices, 0]), np.max(img[slices, 1]),  np.max(img[slices, 2]))
    img[slices, 1] -= min_val[1]
    img[slices, 2] -= min_val[2]
    
    img[slices, 1] /= max_val[1] - min_val[1]
    img[slices, 2] /= max_val[2] - min_val[2]
              
    img[slices, 1] *= 15
    img[slices, 2] *= 15
     
    img[slices, 1] += 140 
    img[slices, 2] += 140 
    img[slices, 0] += low_bound - max_L

    img[slices, 0] = np.minimum(255, img[slices, 0])
    img[slices, 0] = np.maximum(0, img[slices, 0])

    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    img = cv2.resize(img, orginal_shape)

    return img

def compute_IOU(box1, box2):
    # compute bottom left and top right coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    # compute intersection area
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    # compute box area
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    # compute IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def meanshift_postprocess(img, mask, bbox):
    # 1. get mask's bounding box-----------------------------------------
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY) 
    x, y, w, h = cv2.boundingRect(mask)
    mask_bbox = (x, y, x + w, y + h)
    # -------------------------------------------------------------------

    # 2. scale up skeleton's bounding box-----------------------------------------
    # get (x_min, y_min, width, height)
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    # get center
    center_x = x_min + width/2
    center_y = y_min + height/2

    # expand width and height
    new_width = width * 2
    new_height = height * 2

    # new left up 
    new_x_min = int(center_x - new_width/2)
    new_y_min = int(center_y - new_height/2)

    # get new (x_min, y_min, width, height)
    skeleton_bbox = (max(0, new_x_min), max(0, new_y_min), \
                min(img.shape[1], int(new_x_min + new_width)), \
                min(img.shape[0], int(new_y_min + new_height)))
    #-----------------------------------------------------------------------------

    # 3. compute IoU score and filter out low samples--------------------
    IOU_score = compute_IOU(mask_bbox, skeleton_bbox)

    if IOU_score < 0.55:
        return None
    # --------------------------------------------------------------------

    # 4. get slice of skeleton bounding box and run meanshift----------------------
    img_bbox = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_bbox  = img_bbox[skeleton_bbox[1]: skeleton_bbox[3], skeleton_bbox[0]: skeleton_bbox[2], :]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.0001)
    
    slices = mask[skeleton_bbox[1]: skeleton_bbox[3], skeleton_bbox[0]: skeleton_bbox[2], np.newaxis]
    slices = np.all(slices != 0,axis = -1)

    # meanshift filter
    img_bbox[slices, :] = cv2.blur(img_bbox, (7, 7))[slices, :]
    img_bbox[slices, :] = cv2.blur(img_bbox, (7, 7))[slices, :]
    img_bbox = cv2.pyrMeanShiftFiltering(img_bbox, 10, 50, dst=None, maxLevel=None, termcrit=criteria)
    #-------------------------------------------------------------------------------

    # 5. put the ROI back to original position-----------------------
    mask = np.zeros_like(img)
    mask[skeleton_bbox[1]: skeleton_bbox[3], skeleton_bbox[0]: skeleton_bbox[2], :] = img_bbox[...]
    # -----------------------------------------------------------------

    return mask

def generate_mask(meanshift_res, yolo_mask, base_mask, img, sample_mask, num_clasters = 15):
    # 1. convert yolo segmentation result------------------------------------------------
    yolo_mask = cv2.resize(yolo_mask, (meanshift_res.shape[1], meanshift_res.shape[0]))
    yolo_mask = cv2.cvtColor(yolo_mask, cv2.COLOR_BGR2GRAY)
    _, yolo_mask = cv2.threshold(yolo_mask, 1, 255, cv2.THRESH_BINARY) 
    # -----------------------------------------------------------------------------------

    # 2. get mask based on color threshold---------------------------------------------
    # get skeleton based mask's ROI in meanshift's result
    mask_pixels = meanshift_res[np.where(base_mask > 0)]

    # get range of color in skeleton based mask
    lower = np.min(mask_pixels, axis=0)
    upper = np.max(mask_pixels, axis=0)
    # generate new mask with color threshold
    lower = np.array([max(lower[0], 0), max(lower[1], 0), max(lower[2], 0)])
    upper = np.array([min(upper[0], 255), min(upper[1], 255), min(upper[2], 255)])
    mask = np.all((meanshift_res >= lower) & (meanshift_res <= upper), axis=-1)
    mask = np.uint8(mask) * 255
    # bitewise and with yolo's segmentation result
    mask = cv2.bitwise_and(mask, yolo_mask)
    # ----------------------------------------------------------------------------------

    # 3. get the circumscribed circle---------------------------------------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(max_contour)
    center = (int(x), int(y))
    radius = int(1.2 * radius)
    mask = np.zeros_like(mask)
    cv2.circle(mask, center, radius, 255, -1)
    # bitewise and with the yolo's segmentation result
    yolo_mask = cv2.dilate(yolo_mask, (7, 7), iterations=3)
    mask = cv2.bitwise_and(mask, yolo_mask)
    # ----------------------------------------------------------------------------------------

    # 4. get ROI of the circumscribed circle's bounding box and do meanshift fitering in RGB color space---
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    roi = cv2.bitwise_and(img, img, mask=mask)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.0001)
    x_range = (max(center[1]-radius, 0), min(center[1]+radius, roi.shape[0]))
    y_range = (max(center[0]-radius, 0), min(center[0]+radius, roi.shape[1]))
    roi[x_range[0]: x_range[1], y_range[0]: y_range[1], :] = \
    cv2.pyrMeanShiftFiltering(roi[x_range[0]: x_range[1], y_range[0]: y_range[1], :],\
    30, 50, termcrit=criteria)
    # -----------------------------------------------------------------------------------------------------

    # 5. do kmeans in circumscribed circle and get a refined mask-------------------------------------------
    roi_2D = roi.reshape(-1, 3)
    roi_2D = np.float32(roi_2D)
    non_zero_pixels = np.any(roi_2D != [0, 0, 0], axis=-1)
    # non zero index
    roi_2D_non_zero = roi_2D[non_zero_pixels]
    # run kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.01)
    _, labels, _ = cv2.kmeans(roi_2D_non_zero, num_clasters, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    
    # a label image
    label_image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    # reflect label to original position
    label_image.reshape(-1)[non_zero_pixels] = labels.flatten() + 1
    # compute the number of labels in circumscribed circle and a mask generated by landmarks
    sample_mask = cv2.bitwise_and(mask, sample_mask)
    sample_label = cv2.bitwise_and(label_image, sample_mask)
    all_label = cv2.bitwise_and(label_image, mask)
    unique_values_all, counts_all = np.unique(all_label, return_counts=True)
    unique_values_sample, counts_sample = np.unique(sample_label, return_counts=True)
    
    # remove 0(outside mask)
    index = np.where(unique_values_all == 0)
    unique_values_all = np.delete(unique_values_all, index)
    counts_all = np.delete(counts_all, index)

    index = np.where(unique_values_sample == 0)
    unique_values_sample = np.delete(unique_values_sample, index)
    counts_sample = np.delete(counts_sample, index)
    total_sample = np.sum(sample_mask == 255)

    # filter out low percentage labels and get a refined mask
    filtered_value = []
    for i, value in enumerate(unique_values_all):
        if value not in unique_values_sample:
            curr_sample_count = 0
        else:
            index = np.where(unique_values_sample == value)
            index = index[0][0]
            curr_sample_count = counts_sample[index]
        if curr_sample_count / total_sample < 0.01:
            filtered_value.append(value)

    for value in filtered_value:
        label_image[np.where(label_image == value)] = 0
    _, mask = cv2.threshold(label_image, 0, 255, cv2.THRESH_BINARY) 

    # dilate and erode
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=5)
    mask = cv2.erode(mask, kernel, iterations=5)

    # open and close
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    # -----------------------------------------------------------------------------------------------

    # 6. get largest connected domain------------------------------------------------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    max_label = 1
    max_area = stats[1, cv2.CC_STAT_AREA]
    for label in range(2, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > max_area:
            max_label = label
            max_area = area

    max_mask = (labels == max_label).astype('uint8') * 255
    mask = cv2.bitwise_and(max_mask, yolo_mask)
    # -------------------------------------------------------------------------------

    # 7. get largest closed contour--------------------------------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # polygon approximation
    cnt = max_contour
    epsilon = 0.001 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    mask = np.zeros_like(yolo_mask)
    cv2.drawContours(mask, [approx], 0, 255, -1)
    # --------------------------------------------------------------------------------
    return mask, approx
