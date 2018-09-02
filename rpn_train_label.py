import xml.etree.ElementTree as ET
import os
import shutil
import re
import numpy as np
import cv2
import sys
from math import ceil
from data_generators import calc_rpn
from config import Config
# Following variables needed setting by hand
IMAGENET_ROOT = '/opt/pyc/imagenet'
NETWORK_SCALE = 16
RESIZED_HEIGHT, RESIZED_WIDTH = (160, 256)
# SYSNETS = ['n01055165', 'n01581434', 'n01629819', 'n01665541', 'n02691156']
SYSNETS = ['n02691156']
# Following paths are generated relatively
TRAIN_ROOT = os.path.join(IMAGENET_ROOT, '')
ANNOTATION_ROOT = os.path.join(IMAGENET_ROOT, 'Annotation')


def get_wnid_name_dict():
    with open(os.path.join(IMAGENET_ROOT, 'words.txt'), 'rt', errors='ignore') as f:
        lines = f.readlines()
    wnid_name_dict = dict(list(map(lambda x: (x.split('\t')[0], x.split('\t')[1].rstrip('\n')), lines)))
    return wnid_name_dict


def parse_boundbox_xml(xml_path, wnid_name_dict=None):
    '''
    parse the annotation xml file indicated by xml_path (optional using wnid_name_dict
    to translate wnid to string)
    :param xml_path:
    :param wnid_name_dict: a dict like { wnid: name }
    :return: { 'shape': [, , ,],
               'objects': [{'name':...,
                            'xmin':...,
                            'ymin':...,
                            'xmax':...,
                            'ymax':...} ... ]}
    '''
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # for i in root:
    #     print(i.tag, i.attrib)
    img_shape = ()
    for i in root.findall('size'):
        width = int(i.find('width').text)
        height = int(i.find('height').text)
        depth = int(i.find('depth').text)
#         follow the shape format of matplotlib interface
        img_shape = (height, width, depth)
#         print('shape:', img_shape)
    objs = []
    for obj in root.findall('object'):
        obj_dict = {'name':obj.find('name').text}
        if wnid_name_dict is not None:
            try:
                obj_dict['name'] = wnid_name_dict[obj_dict['name']]
            except:
                pass
        obj_dict.update({'xmin':int(obj.find('bndbox').find('xmin').text),
                        'ymin':int(obj.find('bndbox').find('ymin').text),
                        'xmax':int(obj.find('bndbox').find('xmax').text),
                        'ymax':int(obj.find('bndbox').find('ymax').text)})
#         print(obj_dict)
        objs.append(obj_dict)
    return {"shape": img_shape, "objects": objs}


def tranform_bbox(input_dict, original_height, original_width):
    '''
    Transform xml parsed dict to dict used by calc_rpn
    :param input_dict: dict return by parse_boundbox_xml
    :param original_width: img width from real file
    :param original_height: img height from real file
    :return: dict containing 'bboxes'
    '''
    ratio_h =  original_height / input_dict['shape'][0]
    ratio_w = original_width / input_dict['shape'][1]
    bboxes = []
    for obj in input_dict['objects']:
        box = {'class': obj['name'],
               'x1':obj['xmin'] * ratio_w,
               'x2':obj['xmax'] * ratio_w,
               'y1':obj['ymin'] * ratio_h,
               'y2':obj['ymax'] * ratio_h}
        bboxes.append(box)
    return {'bboxes':bboxes,
            'shape':[original_height, original_width]}


def calc_network_outshape(input_width, input_height):
    return (ceil(input_width / NETWORK_SCALE), ceil(input_height / NETWORK_SCALE))


def draw_img_bbox_cv(img_path, bbox_dict):
    img = cv2.imread(img_path)
    new_shape = img.shape
    for box in bbox_dict['bboxes']:
        coords = [int(box['x1'] ),
                  int(box['y1'] ),
                  int(box['x2'] ),
                  int(box['y2'] )
                 ]
        color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
        cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), color)
        cv2.putText(img, box['class'], (coords[0], coords[1] - 15), cv2.FONT_HERSHEY_DUPLEX, 1, color, thickness=2)
    cv2.imshow('img', img)
    return cv2.waitKey(10)


def copy_file(from_path, to_path):
    if not os.path.isfile(from_path):
        print("%s not exists!" % from_path)
    else:
        fpath, fname = os.path.split(to_path)
        if not os.path.exists(fpath):
            os.mkdir(fpath)
        shutil.copyfile(from_path, to_path)

'''[
 ('n01055165', 'camping, encampment, bivouacking, tenting'),
 ('n01581434', 'Rocky Mountain jay, Perisoreus canadensis capitalis'),
 ('n01629819', 'European fire salamander, Salamandra salamandra'),
 ('n01665541', 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea'),
 ('n02691156', 'airplane, aeroplane, plane')
]
'''
wnid_name_dict = get_wnid_name_dict()
C = Config()
y_rpn_cls_list = None
y_rpn_regr_list = None
label_name_list = []
for sysnet in SYSNETS:
    for img_name in os.listdir(os.path.join(TRAIN_ROOT, sysnet)):
        img_path = os.path.join(TRAIN_ROOT, sysnet, img_name)
        annots_path = os.path.join(ANNOTATION_ROOT, sysnet, img_name.rstrip('.jpg') + '.xml')
        if os.path.exists(img_path) and os.path.exists(annots_path):
            label_name_list.append(img_name)
            xml_dict = parse_boundbox_xml(annots_path, wnid_name_dict)
            img = cv2.imread(img_path)
            bbox_dict = tranform_bbox(xml_dict, img.shape[0], img.shape[1])
            # ret_key = draw_img_bbox_cv(img_path, bbox_dict)
            y_rpn_cls, y_rpn_regr = calc_rpn(C, bbox_dict, img.shape[1], img.shape[0],
                                             RESIZED_WIDTH, RESIZED_HEIGHT,
                                             calc_network_outshape)
            if y_rpn_cls_list is None:
                y_rpn_cls_list = y_rpn_cls
            else:
                y_rpn_cls_list = np.concatenate([y_rpn_cls_list, y_rpn_cls], axis=0)
            if y_rpn_regr_list is None:
                y_rpn_regr_list = y_rpn_regr
            else:
                y_rpn_regr_list = np.concatenate([y_rpn_regr_list, y_rpn_regr], axis=0)
np.save('y_rpn_cls', np.array(y_rpn_cls_list))
np.save('y_rpn_regr', np.array(y_rpn_regr_list))
with open('label_name.txt', 'wt') as f:
    f.writelines([i + '\n' for i in label_name_list])




