import os

import cv2
import numpy as np
import torch
from PIL import Image

def gen_config2(video_dir, video_name, image_format):
    if image_format=='RGB' or image_format=='HSI-FalseColor':
        image_type = 'jpg'
    else:
        image_type = 'png'
    # ==============================Getting all the images from the videos =========================================
    img_dir = os.path.join(video_dir, video_name)
    images_in_video = [x for x in os.listdir(img_dir) if x.find(image_type) != -1]
    images_in_video.sort()
    images_in_video = [img_dir + '/' + x for x in images_in_video]
    # images_in_video = [np.array(Image.open(image_file))for image_file in images_in_video]

    # ==============================================================================================================

    # ==============================Getting all the gts for each image in the videos=================================
    gt_path = os.path.join(video_dir, video_name, 'groundtruth_rect.txt')
    f = open(gt_path, 'r')
    lines = f.readlines()
    f.close()
    gts = []

    for line in lines:
        gt_data_per_image = line.split('\t')[:-1]
        if len(gt_data_per_image)!=4:
            gt_data_per_image = line.split('\t')
        gt_data_int = list(map(int, gt_data_per_image))
        gts.append(gt_data_int)
    # ============================================================================================================

    images_in_video = np.asarray(images_in_video)
    gts = np.asarray(gts)

    assert len(images_in_video) == len(gts)

    return images_in_video, gts


def gen_config(video_dir, video_name, image_format):
    if image_format=='RGB' or image_format=='HSI-FalseColor':
        image_type = 'jpg'
    else:
        image_type = 'png'
    # ==============================Getting all the images from the videos =========================================
    img_dir = os.path.join(video_dir, video_name, image_format)
    images_in_video = [x for x in os.listdir(img_dir) if x.find(image_type) != -1]
    images_in_video.sort()
    images_in_video = [img_dir + '/' + x for x in images_in_video]
    # images_in_video = [np.array(Image.open(image_file))for image_file in images_in_video]

    # ==============================================================================================================

    # ==============================Getting all the gts for each image in the videos=================================
    gt_path = os.path.join(video_dir, video_name, image_format, 'groundtruth_rect.txt')
    f = open(gt_path, 'r')
    lines = f.readlines()
    f.close()
    gts = []

    for line in lines:
        gt_data_per_image = line.split('\t')[:-1]
        gt_data_int = list(map(int, gt_data_per_image))
        gts.append(gt_data_int)
    # ============================================================================================================

    images_in_video = np.asarray(images_in_video)
    gts = np.asarray(gts)

    assert len(images_in_video) == len(gts)

    return images_in_video, gts


def gen_config2(video_dir, video_name, image_format):
    if image_format=='RGB' or image_format=='HSI-FalseColor':
        image_type = 'jpg'
    else:
        image_type = 'png'
    # ==============================Getting all the images from the videos =========================================
    img_dir = os.path.join(video_dir, video_name)
    images_in_video = [x for x in os.listdir(img_dir) if x.find(image_type) != -1]
    images_in_video.sort()
    images_in_video = [img_dir + '/' + x for x in images_in_video]
    # images_in_video = [np.array(Image.open(image_file))for image_file in images_in_video]

    # ==============================================================================================================

    # ==============================Getting all the gts for each image in the videos=================================
    gt_path = os.path.join(video_dir, video_name, 'groundtruth_rect.txt')
    f = open(gt_path, 'r')
    lines = f.readlines()
    f.close()
    gts = []

    for line in lines:
        gt_data_per_image = line.split('\t')[:-1]
        if len(gt_data_per_image)!=4:
            gt_data_per_image = line.split('\t')
        gt_data_int = list(map(int, gt_data_per_image))
        gts.append(gt_data_int)
    # ============================================================================================================

    images_in_video = np.asarray(images_in_video)
    gts = np.asarray(gts)

    assert len(images_in_video) == len(gts)

    return images_in_video, gts

def X2Cube(img, B=[4, 4], skip = [4, 4],bandNumber=16):
    # Parameters
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    # Generate Depth indeces
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    # Get all actual indices & index into input array for final output
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    DataCube = out.reshape(M//B[0], N//B[1],bandNumber )
    return DataCube

def X2Cube2(img, B=[5, 5], skip = [5, 5],bandNumber=25):
    # Parameters
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    # Generate Depth indeces
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    # Get all actual indices & index into input array for final output
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    DataCube = out.reshape(M//B[0], N//B[1],bandNumber )
    return DataCube


def crop_image(image, bbox):
    target_height = target_width = 896
    x, y, w, h = bbox
    # Calculate the center of the bounding box
    center_x = x + w // 2
    center_y = y + h // 2

    # Calculate the half-width and half-height of the target size
    half_target_width = target_width // 2
    half_target_height = target_height // 2

    # Calculate the cropping region
    start_x = max(center_x - half_target_width, 0)
    end_x = min(center_x + half_target_width, image.shape[1])
    start_y = max(center_y - half_target_height, 0)
    end_y = min(center_y + half_target_height, image.shape[0])

    # Crop the image
    cropped_image = image[int(start_y):int(end_y), int(start_x):int(end_x)]

    # Pad the cropped image if its size is less than the target size
    if cropped_image.shape[0] < target_height or cropped_image.shape[1] < target_width:
        pad_top = max((target_height - cropped_image.shape[0]) // 2, 0)
        pad_bottom = max(target_height - cropped_image.shape[0] - pad_top, 0)
        pad_left = max((target_width - cropped_image.shape[1]) // 2, 0)
        pad_right = max(target_width - cropped_image.shape[1] - pad_left, 0)
        cropped_image = cv2.copyMakeBorder(cropped_image, pad_top, pad_bottom, pad_left, pad_right,
                                           cv2.BORDER_CONSTANT, value=0)

    return cropped_image



def define_classlist():
    class_dict = {
        'automobile': 0,
        'automobile10': 0,
        'automobile11': 0,
        'automobile12': 0,
        'automobile13': 0,
        'automobile14': 0,
        'automobile2': 0,
        'automobile3': 0,
        'automobile4': 0,
        'automobile5': 0,
        'automobile6': 0,
        'automobile7': 0,
        'automobile8': 0,
        'automobile9': 0,
        'car1': 0,
        'car10': 0,
        'car2': 0,
        'car3': 0,
        'car4': 0,
        'car5': 0,
        'car6': 0,
        'car7': 0,
        'car8': 0,
        'car9': 0,
        'taxi': 0,
        'val_car': 0,
        'val_car2': 0,
        'val_car3': 0,
        'val_truck': 0,
        'basketball': 1,
        'val_basketball': 1,
        'board' : 2,
        'val_board' : 2,
        'val_drive' : 2,
        'val_bus': 3,
        'val_bus2': 3,
        'kangaroo': 4,
        'val_kangaroo': 4,
        'pedestrian': 5,
        'pedestrian4': 5,
        'pedestrain': 5,
        'pedestrian2': 5,
        'pedestrian3': 5,
        'val_campus': 5,
        'val_forest': 5,
        'val_forest2': 5,
        'val_pedestrain': 5,
        'val_pedestrian2': 5,
        'val_player': 5,
        'val_playground': 5,
        'val_student': 5,
        'val_worker': 5,
        'rider1' : 6,
        'rider2': 6,
        'rider3': 6,
        'rider4' : 6,
        'val_rider1': 6,
        'val_rider2' : 6,
        'toy': 7,
        'toy2': 7,
        'val_toy1': 7,
        'val_toy2': 7,
        'val_ball': 8,
        'val_book': 9,
        'val_card': 10,
        'val_coke': 11,
        'val_excavator': 12,
        'val_face': 13,
        'val_face2': 13,
        'val_fruit': 14,
        'val_hand': 15,
        'val_paper': 16,
        'val_rubik': 17,
        'bus': 18,
        'bus2': 18,
        'val_coin': 19

    }

    return class_dict


def cal_iou(box1, box2):
    """

    :param box1: x1,y1,w,h
    :param box2: x1,y1,w,h
    :return: iou
    """
    x11 = box1[0]
    y11 = box1[1]
    x21 = box1[0] + box1[2] - 1
    y21 = box1[1] + box1[3] - 1
    area_1 = (x21 - x11 + 1) * (y21 - y11 + 1)

    x12 = box2[0]
    y12 = box2[1]
    x22 = box2[0] + box2[2] - 1
    y22 = box2[1] + box2[3] - 1
    area_2 = (x22 - x12 + 1) * (y22 - y12 + 1)

    x_left = max(x11, x12)
    x_right = min(x21, x22)
    y_top = max(y11, y12)
    y_down = min(y21, y22)

    inter_area = max(x_right - x_left + 1, 0) * max(y_down - y_top + 1, 0)
    iou = inter_area / (area_1 + area_2 - inter_area)
    return iou

def cal_success(iou):
    success_all = []
    overlap_thresholds = np.arange(0, 1.05, 0.05)
    for overlap_threshold in overlap_thresholds:
        success = sum(np.array(iou) > overlap_threshold) / len(iou)
        success_all.append(success)
    return np.array(success_all)

def calAUC(gtArr,resArr, video_dir):
    # ------------ starting evaluation  -----------
    success_all_video = []
    for idx in range(len(resArr)):
        result_boxes = resArr[idx]
        result_boxes_gt = gtArr[idx]
        result_boxes_gt = [np.array(box) for box in result_boxes_gt]
        iou = list(map(cal_iou, result_boxes, result_boxes_gt))
        success = cal_success(iou)
        auc = np.mean(success)
        success_all_video.append(success)
        # print ('video = ',video_dir[idx],' , auc = ',auc)
    return np.mean(success_all_video)


def define_classlist_2023():
    class_dict = {
        'automobile': 0,
        'automobile10': 0,
        'automobile11': 0,
        'automobile12': 0,
        'automobile13': 0,
        'automobile14': 0,
        'automobile2': 0,
        'automobile3': 0,
        'automobile4': 0,
        'automobile5': 0,
        'automobile6': 0,
        'automobile7': 0,
        'automobile8': 0,
        'automobile9': 0,
        'car1': 0,
        'car10': 0,
        'car2': 0,
        'car3': 0,
        'car4': 0,
        'car5': 0,
        'car6': 0,
        'car7': 0,
        'car8': 0,
        'car9': 0,
        'car13': 0,
        'car14': 0,
        'car15': 0,
        'car16': 0,
        'car17': 0,
        'car18': 0,
        'car19': 0,
        'car20': 0,
        'car21': 0,
        'car22': 0,
        'car23': 0,
        'car24': 0,
        'car25': 0,
        'car26': 0,
        'car27': 0,
        'car28': 0,
        'car29': 0,
        'car30': 0,
        'car31': 0,
        'car32': 0,
        'car33': 0,
        'car34': 0,
        'car35': 0,
        'car36': 0,
        'car37': 0,
        'car38': 0,
        'car39': 0,
        'car40': 0,
        'car41': 0,
        'taxi': 0,
        'val_car': 0,
        'val_car2': 0,
        'val_car3': 0,
        'val_car11': 0,
        'val_car12': 0,
        'val_car49': 0,
        'val_car50': 0,
        'val_car51': 0,
        'val_car52': 0,
        'val_car53': 0,
        'val_car59': 0,
        'val_car60': 0,
        'val_car61': 0,
        'val_car62': 0,
        'val_car63': 0,
        'val_car64': 0,
        'val_car76': 0,
        'val_car77': 0,
        'val_car78': 0,
        'val_car79': 0,
        'val_car80': 0,
        'val_car81': 0,
        'val_car82': 0,
        'val_car83': 0,
        'val_car84': 0,
        'val_car85': 0,

        'basketball': 1,
        'basketball1': 1,
        'val_basketball': 1,
        'val_basketball3': 1,
        'board': 2,
        'val_board': 2,
        'val_drive': 2,
        'val_bus': 3,
        'val_bus2': 3,
        'kangaroo': 4,
        'val_kangaroo': 4,
        'pedestrian': 5,
        'pedestrian2': 5,
        'pedestrian3': 5,
        'pedestrian4': 5,
        'pedestrian5': 5,
        'val_campus': 5,
        'val_forest': 5,
        'val_forest2': 5,
        'val_pedestrian': 5,
        'val_pedestrian2': 5,
        'val_pedestrian7': 5,
        'val_player': 5,
        'val_playground': 5,
        'val_student': 5,
        'val_worker': 5,
        'rainystreet2': 5,
        'rainystreet5': 5,
        'val_rainystreet10': 5,
        'val_rainystreet16': 5,
        'rednir_rainystreet2': 5,
        'rednir_rainystreet5': 5,
        'val_rednir_rainystreet10': 5,
        'val_rednir_rainystreet16': 5,

        'rider1': 6,
        'rider2': 6,
        'rider3': 6,
        'rider4': 6,
        'rider5': 6,
        'rider6': 6,
        'rider7': 6,
        'rider8': 6,
        'rider12': 6,
        'rider13': 6,
        'val_rider1': 6,
        'val_rider2': 6,
        'val_rider11': 6,
        'val_rider16': 6,
        'val_rider17': 6,
        'val_rider18': 6,
        'val_rider19': 6,

        'toy': 7,
        'toy2': 7,
        'val_toy1': 7,
        'val_toy2': 7,
        'val_ball': 8,
        'ball_mirror7': 8,
        'val_ball_mirror9': 8,
        'val_rednir_ball_mirror9': 8,
        'rednir_ball_mirror7': 8,
        'val_book': 9,
        'val_card': 10,

        'val_coke': 11,
        'val_excavator': 12,
        'val_face': 13,
        'val_face2': 13,
        'val_fruit': 14,
        'val_hand': 15,
        'val_paper': 16,
        'val_rubik': 17,
        'bus': 18,
        'bus2': 18,
        'bus3': 18,
        'bus4': 18,
        'val_coin': 19,
        'bytheriver1': 20,
        'rednir_bytheriver1': 20,

        'cloth1': 21,
        'rednir_cloth1': 21,
        'rednir_dice1': 22,

        'dice1': 22,
        'val_dice2': 22,
        'val_rednir_dice2': 22,

        'rednir_duck3': 23,
        'val_rednir_duck5': 23,

        'duck3': 23,
        'val_duck5': 23,
        'glass2': 24,
        'rednir_glass2': 25,
        'rednir_officechair1': 26,
        'officechair1': 26,
        'officefan2': 27,
        'rednir_officefan2': 27,

        'partylights3': 28,
        'val_partylights6': 28,
        'val_rednir_partylights6': 28,
        'rednir_partylights3': 28,

        'pool5': 29,
        'val_pool10': 29,
        'val_pool11': 29,
        'val_rednir_pool10': 29,
        'val_rednir_pool11': 29,
        'rednir_pool5': 29,

        'rednir_receipts3': 30,
        'receipts3': 30,
        'val_truck': 31,
        'truck1': 31,
        'whitecup3': 32,
        'val_whitecup1': 32,
        'val_rednir_whitecup1': 32,

        'rednir_whitecup3': 32,
        'cards11': 33,
        'val_cards16': 33,
        'val_cards19': 33,
        'val_rednir_cards16': 33,
        'val_rednir_cards19': 33,

        'rednir_cards11': 33
    }

    return class_dict


def is_bbox_inside_image(bbox, image_width, image_height):
    x, y, w, h = bbox
    # Calculate the bottom-right coordinates
    x2 = x + w
    y2 = y + h

    # Check if the bounding box is inside the image
    if x >= 0 and y >= 0 and x2 <= image_width and y2 <= image_height:
        return True
    else:
        return False


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    w = x2 - x1  # Calculate width
    h = y2 - y1  # Calculate height
    return [x1, y1, w, h]


def is_bbox_inside_image(bbox, image_width, image_height):
    x, y, w, h = bbox
    # Calculate the bottom-right coordinates
    x2 = x + w
    y2 = y + h

    # Check if the bounding box is inside the image
    if x >= 0 and y >= 0 and x2 <= image_width and y2 <= image_height:
        return True
    else:
        return False




