from flask import Flask, request, jsonify
import torch
from PIL import Image
import base64
import io
import numpy as np
import cv2
from ultralytics import YOLO
import pickle
import hashlib
import time
import os
from scipy.ndimage import gaussian_filter
import asyncio

app = Flask(__name__)
import json
# 模型加载

model = "models/best_new.pt"
model = YOLO(model)

#unique signal
def generate_unique_id():
    timestamp = str(int(time.time() * 1000)) # 时间戳
    uid = hashlib.md5(timestamp.encode()).hexdigest() # 使用MD5哈希函数生成唯一标识符
    return uid

# 保存推理结果

def polygons_to_mask2(img_shape, polygons):
    '''
    边界点生成mask
    :param img_shape: [h,w]
    :param polygons: labelme JSON中的边界点格式 [[x1,y1],[x2,y2],[x3,y3],...[xn,yn]]
    :return:
    '''
    mask = np.zeros(img_shape, dtype=np.uint8)
    polygons = np.asarray(polygons, np.int32) # 这里必须是int32，其他类型使用fillPoly会报错
    # cv2.fillPoly(mask, polygons, 1) # 非int32 会报错
    cv2.fillConvexPoly(mask, polygons, 1)  # 非int32 会报错
    return mask

# 预测
def predict(image,parament):
    # 预处理图片
    img = preprocess(image)
    img = np.asarray(img)
    print(parament)
    with torch.no_grad():
        out = model.predict(img, conf=float(parament['conf']), save_txt=False, save_crop=False, boxes=False, retina_masks=True, device='0')
        for result in out:
            masks = result.masks  # Masks object for segmentation masks outputs
        coordinates = masks.xy
        list1 = []
        for i in coordinates:
            # mask位置
            h, w = result.orig_shape
            mask = polygons_to_mask2([h, w], i)
            mask = mask.astype(np.uint8)
            # mask = gaussian_filter(mask, sigma=2)

            # mask所在坐标矩形框
            x = i[:, 0]
            y = i[:, 1]
            y1 = int(min(y))
            y2 = int(max(y))
            x1 = int(min(x))
            x2 = int(max(x))
            # 创建与原图大小全黑图，用于提取.

            res = np.zeros_like(img)
            # 提取>0部分到新图并进行裁剪
            res[mask > 0] = img[mask > 0]

            # 裁剪后的图
            masked = res[y1:y2, x1:x2]
            list1.append(masked)

        #推理照片保存
        # save_results(list1)
        return list1


# 预处理
def preprocess(image):
    return Image.open(io.BytesIO(image))

# 返回base64
def transform(outputs):
    image_list = []
    for img in outputs:
        # 转换为Image格式
        pil_img = Image.fromarray(img)
        # 编码为base64
        buff = io.BytesIO()
        pil_img.save(buff, format="PNG")
        img_str = base64.b64encode(buff.getvalue()).decode('utf-8')
        # 添加到列表
        image_list.append(img_str)
    return image_list

def save_results(result):
    unique_id = generate_unique_id() # 生成唯一标识符
    # 创建子目录
    subdir = os.path.join('app_results', unique_id)
    print(subdir)
    os.makedirs(subdir, exist_ok=True)
    for i in result:
        timestamp = str(int(time.time() * 1000))
        crop_path = os.path.join(subdir, f'{timestamp}.jpg')
        print(crop_path)
        cv2.imwrite(crop_path, i)
    # 文件名为推理结果的索引加上时间戳后缀

#接口
@app.route('/predict', methods=['POST'])
def get_prediction():
    file = request.files['file']
    parament = request.files['parament'].read()
    parament = pickle.loads(parament)
    img_bytes = file.read()
    result = predict(img_bytes, parament)
    result = transform(result)

    return jsonify({'content': result})


if __name__ == '__main__':
    app.run(
        host='172.16.1.152',
        port=5000,
        debug=False,
    )
