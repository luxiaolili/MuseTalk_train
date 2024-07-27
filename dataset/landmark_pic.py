import os
import sys 
import numpy as np
import cv2
import time
from tqdm import tqdm
import multiprocessing
import glob

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from face_landmark import FaceLandmarker
from draw_util import FaceMeshVisualizer


def crop_face_mask(img, mask, lmk, expand=1.1):
    
    H, W, _ = img.shape
    lmks = lmk
    lmks[:, 0] *= W
    lmks[:, 1] *= H

    x_min = np.min(lmks[:, 0])
    x_max = np.max(lmks[:, 0])
    y_min = np.min(lmks[:, 1])
    y_max = np.max(lmks[:, 1])

    width = x_max - x_min
    height = y_max - y_min
    
    if width*height >= W*H*0.15:
        if W == H:
            return img
        size = min(H, W)
        offset = int((max(H, W) - size)/2)
        if size == H:
            return img[:, offset:-offset]
        else:
            return img[offset:-offset, :]
    else:
        center_x = x_min + width / 2
        center_y = y_min + height / 2

        width *= expand
        height *= expand

        size = max(width, height)

        x_min = int(center_x - size / 2)
        x_max = int(center_x + size / 2)
        y_min = int(center_y - size / 2)
        y_max = int(center_y + size / 2)

        top = max(0, -y_min)
        bottom = max(0, y_max - img.shape[0])
        left = max(0, -x_min)
        right = max(0, x_max - img.shape[1])
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        cropped_img = img[y_min + top:y_max + top, x_min + left:x_max + left]
        cropped_mask = mask[y_min + top:y_max + top, x_min + left:x_max + left]

    return cropped_img, cropped_mask

def crop_face(img, lmk, expand=1.1):
    H, W, _ = img.shape
    lmks = lmk
    lmks[:, 0] *= W
    lmks[:, 1] *= H

    x_min = np.min(lmks[:, 0])
    x_max = np.max(lmks[:, 0])
    y_min = np.min(lmks[:, 1])
    y_max = np.max(lmks[:, 1])

    width = x_max - x_min
    height = y_max - y_min
    
    if width*height >= W*H*0.15:
        if W == H:
            return img
        size = min(H, W)
        offset = int((max(H, W) - size)/2)
        if size == H:
            return img[:, offset:-offset]
        else:
            return img[offset:-offset, :]
    else:
        center_x = x_min + width / 2
        center_y = y_min + height / 2

        width *= expand
        height *= expand

        size = max(width, height)

        x_min = int(center_x - size / 2)
        x_max = int(center_x + size / 2)
        y_min = int(center_y - size / 2)
        y_max = int(center_y + size / 2)

        top = max(0, -y_min)
        bottom = max(0, y_max - img.shape[0])
        left = max(0, -x_min)
        right = max(0, x_max - img.shape[1])
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        cropped_img = img[y_min + top:y_max + top, x_min + left:x_max + left]

    return cropped_img



def get_landmark(detector, image):
    detection_result, mesh3d = detector.detect(image)
    bs_list = detection_result.face_blendshapes
    if len(bs_list) == 1:
        bs = bs_list[0]
        bs_values = []
        for index in range(len(bs)):
            bs_values.append(bs[index].score)
        bs_values = bs_values[1:] # remove neutral
        trans_mat = detection_result.facial_transformation_matrixes[0]
        face_landmarks_list = detection_result.face_landmarks
        face_landmarks = face_landmarks_list[0]
        lmks = []
        for index in range(len(face_landmarks)):
            x = face_landmarks[index].x
            y = face_landmarks[index].y
            z = face_landmarks[index].z
            lmks.append([x, y, z])
        lmks = np.array(lmks)
        lmks3d = np.array(mesh3d.vertex_buffer)
        lmks3d = lmks3d.reshape(-1, 5)[:, :3]
        mp_tris = np.array(mesh3d.index_buffer).reshape(-1, 3) + 1

    return {
                "lmks": lmks,
                'lmks3d': lmks3d,
                "trans_mat": trans_mat,
                'faces': mp_tris,
                "bs": bs_values
            }


mode = mp.tasks.vision.FaceDetectorOptions.running_mode.IMAGE
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
base_options.delegate = mp.tasks.BaseOptions.Delegate.CPU
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            running_mode=mode,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)

     
detector = FaceLandmarker.create_from_options(options)


def scanner_video(rootDir):
    video_list = []
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            file_name = os.path.join(root,file)
            if file_name.endswith('.png'):
                video_list.append(file_name)
    return video_list


if __name__ =='__main__':
    img_path = sys.argv[1]
    imgs = scanner_video(img_path)
    print(len(imgs))
    num = 0
    for im_p in imgs:
        print(num)
        num += 1
        img = cv2.imread(im_p)
        image_org = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_org)
        lmk_result = get_landmark(detector, image)
        vis = FaceMeshVisualizer(forehead_edge=False)
        lmks = lmk_result['lmks'].astype(np.float32)
        draw_result = vis.draw_landmarks((image_org.shape[1], image_org.shape[0]), lmks, normed=True)
        #cv2.imwrite('draw_result.png', draw_result)
        new_path = im_p.replace('images', 'landmarks')
        new_dir = os.path.dirname(new_path)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        cv2.imwrite(new_path, draw_result)
        '''
        crop_img, crop_mask = crop_face_mask(img, draw_result, lmks)
        cv2.imwrite('crop_face.png', crop_img)
        cv2.imwrite('crop_mask.png', crop_mask)'''