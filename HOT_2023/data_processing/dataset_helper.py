import os
import random
import shutil
from scipy import spatial
import cv2
import numpy as np
import albumentations as A


def gen_config(video_dir, video_name, image_format):
    if image_format=='RGB' or image_format=='HSI-FalseColor':
        image_type = 'jpg'
    else:
        image_type = 'png'
    # ==============================Getting all the images from the videos =========================================
    img_dir = os.path.join(video_dir, video_name)
    images_in_video = [x for x in os.listdir(img_dir) if x.find(image_type) != -1]
    images_in_video.sort()
    images_in_video = [img_dir + '/' + x for x in images_in_video]

    # ==============================================================================================================

    # ==============================Getting all the gts for each image in the videos=================================
    gt_path = os.path.join(video_dir, video_name, 'groundtruth_rect.txt')
    f = open(gt_path, 'r')
    lines = f.readlines()
    f.close()
    gts = []

    for line in lines:
        gt_data_per_image = line.split('\t')[:-1]
        if(len(gt_data_per_image))==3:
            gt_data_per_image = line.split('\t')
        gt_data_int = list(map(int, gt_data_per_image))
        gts.append(gt_data_int)
    # ============================================================================================================

    images_in_video = np.asarray(images_in_video)
    gts = np.asarray(gts)

    assert len(images_in_video) == len(gts)

    return images_in_video, gts


def make_dataset_directory(dataset_path):
    if os.path.isdir(dataset_path):
        shutil.rmtree(dataset_path)
    os.makedirs(dataset_path)


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



def random_flip(image, bounding_box):
    image = np.asarray(image)
    flip_code = random.choice([1, 0])  # -1 for horizontal flip, 0 for vertical flip, 1 for both flips disabled
    flipped_image = cv2.flip(image, flip_code)

    width = image.shape[1]
    height = image.shape[0]

    if flip_code == 1:  # Horizontal flip
        # Adjust the bounding box coordinates
        flipped_bounding_box = bounding_box.copy()
        flipped_bounding_box[0] = image.shape[1] - bounding_box[0] - bounding_box[2]

    elif flip_code == 0:  # Vertical flip
        flipped_bounding_box = bounding_box.copy()
        flipped_bounding_box[1] = image.shape[0] - bounding_box[1] - bounding_box[3]

    return flipped_image, flipped_bounding_box


def rotate_image_with_bounding_box(image, bounding_box):
    image = np.asarray(image)
    # Get the image dimensions
    height, width = image.shape[:2]

    # Randomly select the rotation angle between -45 and 45 degrees
    angle = random.uniform(-45, 45)

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # Perform the rotation on the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # Adjust the bounding box coordinates
    x, y, w, h = bounding_box
    box_center_x = x + w / 2
    box_center_y = y + h / 2

    # Rotate the bounding box center point
    rotated_center = np.dot(rotation_matrix, np.array([[box_center_x], [box_center_y], [1]]))
    rotated_center_x = rotated_center[0, 0]
    rotated_center_y = rotated_center[1, 0]

    # Calculate the rotated bounding box coordinates
    rotated_x = rotated_center_x - w / 2
    rotated_y = rotated_center_y - h / 2

    rotated_bounding_box = [int(rotated_x), int(rotated_y), int(w), int(h)]

    return rotated_image, rotated_bounding_box


def augmented_image(img):
    img = np.array(img)
    t_number = np.random.randint(1, 15)
    brightness = np.random.random()
    contrast = np.random.random()
    saturation = np.random.random()
    hue = np.random.random()
    transform = A.Compose([
        A.ChannelShuffle(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),
        A.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue,
                      always_apply=False, p=0.2),
        A.AdvancedBlur(blur_limit=(3, 7), sigmaX_limit=(0.2, 1.0), sigmaY_limit=(0.2, 1.0),
                       rotate_limit=90, beta_limit=(0.5, 8.0),
                       noise_limit=(0.9, 1.1), always_apply=False, p=0.2),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.2),
        A.Equalize(mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.2),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.2),
        A.ZoomBlur (max_factor=1.31, step_factor=(0.01, 0.03), always_apply=False, p=0.2),
        A.UnsharpMask (blur_limit=(3, 7), sigma_limit=0.0, alpha=(0.2, 0.5), threshold=10, always_apply=False, p=0.2),
        A.Superpixels (p_replace=0.1, n_segments=100, max_size=128, interpolation=1, always_apply=False, p=0.2),
        A.ToGray(p=0.2),
        A.ToSepia(p=0.2),

        ])

    transformed = transform(image=img)
    transformed_image = transformed['image']
    return transformed_image


def convert_to_yolo(bbox, image_width, image_height):
    x, y, w, h = bbox

    center_x = (x + (w / 2)) / image_width
    center_y = (y + (h / 2)) / image_height
    width = w / image_width
    height = h / image_height

    return center_x, center_y, width, height


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
    #DataCube = DataCube.transpose(1, 0, 2)
    #DataCube = DataCube / DataCube.max() * 255
    #DataCube.astype('uint8')
    return DataCube

def crop_and_paste(image, bbox):
    # Extract coordinates of the bounding box
    x_min, y_min, w, h = bbox
    x_max = x_min + w
    y_max = y_min + h
    # x_min, y_min, x_max, y_max = bbox

    # Crop the bounding box from the image
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Generate random positions for pasting
    max_x = image.shape[1] - (x_max - x_min)
    max_y = image.shape[0] - (y_max - y_min)
    new_x = random.randint(0, max_x)
    new_y = random.randint(0, max_y)

    # Extract the region where the cropped image will be pasted
    paste_region = image[new_y:new_y + (y_max - y_min), new_x:new_x + (x_max - x_min)]

    # Calculate the average color of the surrounding area
    surrounding_area = np.copy(image)
    surrounding_area[new_y:new_y + (y_max - y_min), new_x:new_x + (x_max - x_min)] = 0
    avg_color = np.mean(surrounding_area, axis=(0, 1))

    # Paste the cropped image randomly in another position
    image[new_y:new_y + (y_max - y_min), new_x:new_x + (x_max - x_min)] = cropped_image

    # Fill the previous position with the average color
    image[y_min:y_max, x_min:x_max] = avg_color

    return image, [new_x, new_y, w, h]


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
        'val_trucker': 31,
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