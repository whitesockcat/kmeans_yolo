import json

json_path = 'pascal_train2007.json'
# json_path = 'instances_train2017.json'
# json_path = 'instances_val2017.json'

with open(json_path, 'r') as jsonfile:
    voc = json.load(jsonfile)

# {
#     "info": info,
#     "licenses": [license],
#     "images": [image],
#     "annotations": [annotation],
#     "categories": [category]
# }
# image{
#     "id": int, <--->annotation{'image_id'}
#     "width": int,
#     "height": int,
#     "file_name": str,
#     "license": int,
#     "flickr_url": str,
#     "coco_url": str,
#     "date_captured": datetime,
# }
# annotation{
#     "id": int,    
#     "image_id": int,
#     "category_id": int,
#     "segmentation": RLE or [polygon],
#     "area": float,
#     "bbox": [x,y,width,height],
#     "iscrowd": 0 or 1,
# }
# categoriesd{
#     "id": int,
#     "name": str,
#     "supercategory": str,
# }

def get_area_rate(h, w, bboxh, bboxw, min_size=800, max_size=1333):
    min_original_size = float(min((w, h)))
    max_original_size = float(max((w, h)))
    if max_original_size / min_original_size * min_size > max_size:
        min_size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == min_size) or (h <= w and h == min_size):
        size_rate = 1

    if w < h:
        size_rate = min_size / w
    else:
        size_rate = min_size / h

    area_rate = size_rate ** 2
    return area_rate
    
catid2label = ['bg', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',\
    'cahair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',\
    'sheep', 'sofa', 'train', 'tvmonitor']
catid = 4
images = voc['images']
anns = voc['annotations']
# print(len(images))
# cnt_box = 0
# for ann in anns:
#     if ann['category_id'] == 1:
#         cnt_box += 1
# print(cnt_box)
h_w_s = []
b_r_w_s = []
b_r_h_s = []
size_32, size_64, size_128, size_256, size_512, size_1024 = \
    {'0.5':0, '1':0, '2':0}, \
    {'0.5':0, '1':0, '2':0}, \
    {'0.5':0, '1':0, '2':0}, \
    {'0.5':0, '1':0, '2':0}, \
    {'0.5':0, '1':0, '2':0}, \
    {'0.5':0, '1':0, '2':0}
maxh, maxw = 0, 0
thre0 = 0.7
thre1 = 1.5
for img in images:
    id = img['id']
    for ann in anns:
        if id == ann['image_id']:
            height = img['height']
            width = img['width']
            # maxh = height if height > maxh else maxh
            # maxw = width if width > maxw else maxw
            bbox = ann['bbox']
            area = ann['area']
            bboxw = bbox[2]
            bboxh = bbox[3]
            if ann['category_id'] != catid:
                continue
            if bboxh <= 0 or bboxw <= 0:
                continue
            # print(height, width, bboxh, bboxw)
            area_rate = get_area_rate(height, width, bboxh, bboxw)
            resize_area = area * area_rate
            h_w  = round(bboxh / bboxw, 2)
# print(maxh, maxw)
            if resize_area < 32 ** 2 * 2:
                if h_w < thre0:
                    size_32['0.5'] += 1
                elif h_w < thre1:
                    size_32['1'] += 1
                else:
                    size_32['2'] += 1
                # size_32 += 1
            elif resize_area < 64 ** 2 * 2:
                if h_w < thre0:
                    size_64['0.5'] += 1
                elif h_w < thre1:
                    size_64['1'] += 1
                else:
                    size_64['2'] += 1
            elif resize_area < 128 ** 2 * 2:
                if h_w < thre0:
                    size_128['0.5'] += 1
                elif h_w < thre1:
                    size_128['1'] += 1
                else:
                    size_128['2'] += 1
            elif resize_area < 256 ** 2 * 2:
                if h_w < thre0:
                    size_256['0.5'] += 1
                elif h_w < thre1:
                    size_256['1'] += 1
                else:
                    size_256['2'] += 1
#             elif resize_area < 512 ** 2 * 2:
#                 if h_w < thre0:
#                     size_512['0.5'] += 1
#                 elif h_w < thre1:
#                     size_512['1'] += 1
#                 else:
#                     size_512['2'] += 1
            else:
                if h_w < thre0:
                    size_512['0.5'] += 1
                elif h_w < thre1:
                    size_512['1'] += 1
                else:
                    size_512['2'] += 1
print(catid2label[catid])
print(size_32)
print(size_64)
print(size_128)
print(size_256)
print(size_512)
# print(size_1024)
            # h_w  = round(bboxh / bboxw, 2)
            # bbox_relative_w = round(bboxw / width, 2)
            # bbox_relative_h = round(bboxh / height, 2)
            # h_w_s.append(h_w)
            # b_r_w_s.append(bbox_relative_w)
            # b_r_h_s.append(bbox_relative_h)
            
# print(h_w_s)
# a, b, c, d, e = 0, 0, 0, 0, 0
# f, g, h, i, j = 0, 0, 0, 0, 0
# for h_w in b_r_h_s:
#     if h_w > 0.9:
#         a += 1
#     elif h_w > 0.8:
#         b += 1
#     elif h_w > 0.7:
#         c += 1
#     elif h_w > 0.6:
#         d += 1
#     elif h_w > 0.5:
#         e += 1
#     elif h_w > 0.4:
#         f += 1
#     elif h_w > 0.3:
#         g += 1
#     elif h_w > 0.2:
#         h += 1
#     elif h_w > 0.1:
#         i += 1
#     else:
#         j += 1


# bbox = {'h_w_rate': h_w_s, 'w': b_r_w_s, 'h': b_r_h_s}
# with open('bbox_coco_train.json', 'w') as jsonfile:
#     json.dump(bbox, jsonfile)

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline
# 0.05 ~ 0.95 for h or w NOT for h/w
scale = [i/10 + 0.05 for i in range(10)]

# for h_w in h_w_s:
#     if h_w > 2.5:
#         a += 1
#     elif h_w > 2.25:
#         b += 1
#     elif h_w > 2:
#         c += 1
#     elif h_w > 1.75:
#         d += 1
#     elif h_w > 1.5:
#         e += 1
#     elif h_w > 1.25:
#         f += 1
#     elif h_w > 1:
#         g += 1
#     elif h_w > 0.5:
#         h += 1
#     elif h_w > 0.25:
#         i += 1
#     else:
#         j += 1
# # print(a)
# # print(b)
# # print(c)
# # print(d)
# # print(e)
# # print(f)
# # print(g)
# # print(h)
# # print(i)
# # print(j)
# scale = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]

# x = np.array(scale)
# frequnce = [a, b, c, d, e, f, g, h, i, j]
# y = np.array(frequnce[::-1])


# xnew = np.linspace(x.min(),x.max(),300)
# y_smooth = spline(x, y, xnew)

# plt.plot(xnew, y_smooth, label = r'$h$')

# a, b, c, d, e = 0, 0, 0, 0, 0
# f, g, h, i, j = 0, 0, 0, 0, 0
# for h_w in b_r_w_s:
#     if h_w > 0.9:
#         a += 1
#     elif h_w > 0.8:
#         b += 1
#     elif h_w > 0.7:
#         c += 1
#     elif h_w > 0.6:
#         d += 1
#     elif h_w > 0.5:
#         e += 1
#     elif h_w > 0.4:
#         f += 1
#     elif h_w > 0.3:
#         g += 1
#     elif h_w > 0.2:
#         h += 1
#     elif h_w > 0.1:
#         i += 1
#     else:
#         j += 1

# frequnce = [a, b, c, d, e, f, g, h, i, j]
# y = np.array(frequnce[::-1])

# xnew = np.linspace(x.min(),x.max(),300)
# y_smooth = spline(x, y, xnew)

# plt.plot(xnew, y_smooth, label = r'$w$')
# plt.legend(loc = 9)
# plt.show()