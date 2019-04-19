import cv2
import time
import math
import os
import numpy as np
import lanms
import shutil
import torch
from dataset.data_utils import restore_rectangle, polygon_area
from torchvision import transforms
from tqdm import tqdm
from cal_recall import cal_recall_precison_f1


def rotate(box_List, image):
    # xuan zhuan tu pian

    n = len(box_List)
    c = 0
    angle = 0
    for i in range(n):
        box = box_List[i]
        y1 = min(box[0][1], box[1][1], box[2][1], box[3][1])
        y2 = max(box[0][1], box[1][1], box[2][1], box[3][1])
        x1 = min(box[0][0], box[1][0], box[2][0], box[3][0])
        x2 = max(box[0][0], box[1][0], box[2][0], box[3][0])
        for j in range(4):
            if (box[j][1] == y2):
                k1 = j
        for j in range(4):
            if (box[j][0] == x2 and j != k1):
                k2 = j
        c = (box[k1][0] - box[k2][0]) * 1.0 / (box[k1][1] - box[k2][1])
        if (c < 0):
            c = -c
        if (c > 1):
            c = 1.0 / c
        angle = math.atan(c) + angle
    angle = angle / n
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    scale = 1
    M = cv2.getRotationMatrix2D(center, angle, scale)
    image_new = cv2.warpAffine(image, M, (w, h))
    return image_new


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    """
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.

    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)
    """

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    # resize_h, resize_w = 512, 512
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=1e-5, box_thresh=1e-8, nms_thres=0.1):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
    # print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start
    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]
    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def change_box(box_List):
    n = len(box_List)
    for i in range(n):
        box = box_List[i]
        y1 = min(box[0][1], box[1][1], box[2][1], box[3][1])
        y2 = max(box[0][1], box[1][1], box[2][1], box[3][1])
        x1 = min(box[0][0], box[1][0], box[2][0], box[3][0])
        x2 = max(box[0][0], box[1][0], box[2][0], box[3][0])
        box[0][1] = y1
        box[0][0] = x1
        box[1][1] = y1
        box[1][0] = x2
        box[3][1] = y2
        box[3][0] = x1
        box[2][1] = y2
        box[2][0] = x2
        box_List[i] = box
    return box_List


def save_box(box_List, image, img_path):
    n = len(box_List)
    box_final = []
    for i in range(n):
        box = box_List[i]
        y1_0 = int(min(box[0][1], box[1][1], box[2][1], box[3][1]))
        y2_0 = int(max(box[0][1], box[1][1], box[2][1], box[3][1]))
        x1_0 = int(min(box[0][0], box[1][0], box[2][0], box[3][0]))
        x2_0 = int(max(box[0][0], box[1][0], box[2][0], box[3][0]))
        y1 = max(int(y1_0 - 0.1 * (y2_0 - y1_0)), 0)
        y2 = min(int(y2_0 + 0.1 * (y2_0 - y1_0)), image.shape[0] - 1)
        x1 = max(int(x1_0 - 0.25 * (x2_0 - x1_0)), 0)
        x2 = min(int(x2_0 + 0.25 * (x2_0 - x1_0)), image.shape[1] - 1)
        image_new = image[y1:y2, x1:x2]

        # # 图像处理
        gray_2 = image_new[:, :, 0]
        gradX = cv2.Sobel(gray_2, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray_2, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        blurred = cv2.blur(gradX, (2, 2))
        (_, thresh) = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)
        # closed = cv2.erode(thresh, None, iterations = 1)
        # closed = cv2.dilate(closed, None, iterations = 1)
        closed = thresh
        x_plus = []
        x_left = 1
        x_right = closed.shape[1]
        for jj in range(0, closed.shape[1]):
            plus = 0
            for ii in range(0, closed.shape[0]):
                plus = plus + closed[ii][jj]
            x_plus.append(plus)

        for jj in range(0, int(closed.shape[1] * 0.5 - 1)):
            if (x_plus[jj] > 0.4 * max(x_plus)):
                x_left = max(jj - 5, 0)
                break
        for ii in range(closed.shape[1] - 1, int(closed.shape[1] * 0.5 + 1), -1):
            if (x_plus[ii] > 0.4 * max(x_plus)):
                x_right = min(ii + 5, closed.shape[1] - 1)
                break

        image_new = image_new[:, x_left:x_right]
        cv2.imwrite("." + img_path.split(".")[1] + '_' + str(i) + ".jpg", image_new)
        box[0][1] = y1
        box[0][0] = x1 + x_left
        box[1][1] = y1
        box[1][0] = x1 + x_right
        box[3][1] = y2
        box[3][0] = x1 + x_left
        box[2][1] = y2
        box[2][0] = x1 + x_right
        box_List[i] = box
    return box_List


def transform_for_test():
    """
    CV2 => PI => tensor
    """
    # image = Image.fromarray(np.uint8(img))

    transform_list = []

    transform_list.append(transforms.ToTensor())

    transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    transform = transforms.Compose(transform_list)

    return transform


def predict_one(model, img_path, device, draw_img=False):
    im = cv2.imread(img_path)[:, :, ::-1]

    im_resized, (ratio_h, ratio_w) = resize_image(im)
    im_resized = im_resized.astype(np.float32)

    im_resized = im_resized.transpose(2, 0, 1)
    im_resized = torch.from_numpy(im_resized)
    im_resized = im_resized.to(device)
    im_resized = im_resized.unsqueeze(0)

    timer = {'net': 0, 'restore': 0, 'nms': 0}
    start = time.time()
    with torch.no_grad():
        score, geometry = model(im_resized)
    timer['net'] = time.time() - start
    score = score.permute(0, 2, 3, 1)
    geometry = geometry.permute(0, 2, 3, 1)
    score = score.data.cpu().numpy()
    geometry = geometry.data.cpu().numpy()
    # 从网络输出中获取bbox
    boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
    return_bboxs = []
    if boxes is not None:
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h
        for box in boxes:
            box = sort_poly(box.astype(np.int32))

            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                # print('wrong direction')
                continue

            if box[0, 0] < 0 or box[0, 1] < 0 or box[1, 0] < 0 or box[1, 1] < 0 or box[2, 0] < 0 or box[2, 1] < 0 or \
                    box[3, 0] < 0 or box[3, 1] < 0:
                continue

            poly = np.array(
                [[box[0, 0], box[0, 1]], [box[1, 0], box[1, 1]], [box[2, 0], box[2, 1]], [box[3, 0], box[3, 1]]])

            p_area = polygon_area(poly)
            if p_area > 0:
                poly = poly[(0, 3, 2, 1), :]
            return_bboxs.append(poly)
            if draw_img:
                cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0),
                              thickness=1)

    return np.array(return_bboxs), im

def eval(model, save_path, test_path, device):
    model.eval()
    img_path = os.path.join(test_path, 'img')
    gt_path = os.path.join(test_path, 'gt')
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_paths = [os.path.join(img_path, x) for x in os.listdir(img_path)]
    for img_path in tqdm(img_paths, desc='test model'):
        img_name = os.path.basename(img_path).split('.')[0]
        save_name = os.path.join(save_path, 'res_' + img_name + '.txt')
        bboxs, im = predict_one(model,img_path,device,draw_img=False)
        np.savetxt(save_name, bboxs.reshape(-1, 8), delimiter=',', fmt='%d')
    result_dict = cal_recall_precison_f1(gt_path, save_path)
    return result_dict['recall'], result_dict['precision'], result_dict['hmean']