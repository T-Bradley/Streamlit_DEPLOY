import openvino as ov

import ipywidgets as widgets
import yaml

import cv2 
import numpy as np
from ultralytics.utils.plotting import colors
from typing import Tuple

core = ov.Core()
model = core.read_model(model = "models/best.xml")
compiled_model = core.compile_model(model = model, device_name = "AUTO")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

with open('models/metadata.yaml') as info:
    info_dict = yaml.load(info, Loader=yaml.Loader)

labels = info_dict['names']

def letterbox(img: np.ndarray, new_shape:Tuple[int, int] = (640, 640), 
              color:Tuple[int, int, int] = (114, 114, 114), 
              auto:bool = False, scale_fill:bool = False, 
              scaleup:bool = False, stride:int = 32):
    """
    Resize image and padding for detection. Takes image as input, 
    resizes image to fit into new shape with saving original aspect ratio and pads it to meet stride-multiple constraints
    
    Parameters:
      img (np.ndarray): image for preprocessing
      new_shape (Tuple(int, int)): image size after preprocessing in format [height, width]
      color (Tuple(int, int, int)): color for filling padded area
      auto (bool): use dynamic input size, only padding for stride constrins applied
      scale_fill (bool): scale image to fill new_shape
      scaleup (bool): allow scale image if it is lower then desired input size, can affect model accuracy
      stride (int): input padding stride
    Returns:
      img (np.ndarray): image after preprocessing
      ratio (Tuple(float, float)): hight and width scaling ratio
      padding_size (Tuple(int, int)): height and width padding size
    
    
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def prepare_data(image, input_layer):
    input_w, input_h = input_layer.shape[2], input_layer.shape[3]
    input_image = letterbox(np.array(image))[0]  
    input_image = cv2.resize(input_image, (input_w, input_h))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image/255
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, 0)
 
    return input_image

def evaluate(output, conf_threshold):
    
    boxes = []
    scores = []
    label_key = []
    label_index = 0
    
    for class_ in output[0][4:]:
  
        
        for index in range (len(class_)):
            confidence = class_[index]

            if  confidence > conf_threshold:

                xcen = output[0][0][index]
                ycen = output[0][1][index]
                w = output[0][2][index]
                h = output[0][3][index]

                xmin = int(xcen - (w/2))
                xmax = int(xcen + (w/2))
                ymin = int(ycen - (h/2))
                ymax = int(ycen + (h/2))

                box = (xmin, ymin, xmax, ymax)
                boxes.append(box)
                scores.append(confidence)

                label_key.append(label_index)
  
        label_index += 1 
        
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    return boxes, scores, label_key


def non_max_suppression(boxes, scores, threshold):	
    assert boxes.shape[0] == scores.shape[0]
    # bottom-left origin
    ys1 = boxes[:, 0]
    xs1 = boxes[:, 1]
    # top-right target
    ys2 = boxes[:, 2]
    xs2 = boxes[:, 3]
    # box coordinate ranges are inclusive-inclusive
    areas = (ys2 - ys1) * (xs2 - xs1)
    scores_indexes = scores.argsort().tolist()
    boxes_keep_index = []
    while len(scores_indexes):
        index = scores_indexes.pop()
        boxes_keep_index.append(index)
        if not len(scores_indexes):
            break
        ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index],
                           areas[scores_indexes])
        filtered_indexes = set((ious > threshold).nonzero()[0])
        # if there are no more scores_index
        # then we should pop it
        scores_indexes = [
            v for (i, v) in enumerate(scores_indexes)
            if i not in filtered_indexes
        ]
    return np.array(boxes_keep_index)


def compute_iou(box, boxes, box_area, boxes_area):
    # this is the iou of the box against all other boxes
    assert boxes.shape[0] == boxes_area.shape[0]
    # get all the origin-ys
    # push up all the lower origin-xs, while keeping the higher origin-xs
    ys1 = np.maximum(box[0], boxes[:, 0])
    # get all the origin-xs
    # push right all the lower origin-xs, while keeping higher origin-xs
    xs1 = np.maximum(box[1], boxes[:, 1])
    # get all the target-ys
    # pull down all the higher target-ys, while keeping lower origin-ys
    ys2 = np.minimum(box[2], boxes[:, 2])
    # get all the target-xs
    # pull left all the higher target-xs, while keeping lower target-xs
    xs2 = np.minimum(box[3], boxes[:, 3])
    # each intersection area is calculated by the
    # pulled target-x minus the pushed origin-x
    # multiplying
    # pulled target-y minus the pushed origin-y
    # we ignore areas where the intersection side would be negative
    # this is done by using maxing the side length by 0
    intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
    # each union is then the box area
    # added to each other box area minusing their intersection calculated above
    unions = box_area + boxes_area - intersections
    # element wise division
    # if the intersection is 0, then their ratio is 0
    ious = intersections / unions
    return ious
    
def visualize(image, nms_output, boxes, label_key,scores, conf_threshold):
    image_h, image_w, c = image.shape
    input_w, input_h = input_layer.shape[2], input_layer.shape[3]

    for i in nms_output:
        xmin, ymin, xmax, ymax = boxes[i]
 
        xmin = int(xmin*image_w/input_w)
        xmax = int(xmax*image_w/input_w)
        ymin = int(ymin*image_h/input_h)
        ymax = int(ymax*image_h/input_h)
 
        label = label_key[i]
        color = colors(label)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(int(scores[i]*100)) + "%" + labels[label]
        font_scale= (image_w/1000)
        label_width, label_height = cv2.getTextSize(text, font,font_scale, 1)[0]
        cv2.rectangle(image, (xmin, ymin-label_height), (xmin + label_width, ymin), color, -1)
        
        cv2.putText(image, text, (xmin+2, ymin), font, font_scale, (255,255,255), 1, cv2.LINE_AA)    
    return image

def predict_image(image, conf_threshold):
    input_image = prepare_data(image, input_layer)
    output = compiled_model([input_image])[output_layer]
    boxes, scores, label_key = evaluate(output, conf_threshold)
 

    if len(boxes):
        nms_output = non_max_suppression(boxes, scores, conf_threshold)
  
        visualized_image = visualize(image, nms_output, boxes, label_key,scores, conf_threshold)

        return visualized_image
    else:
        return image































































