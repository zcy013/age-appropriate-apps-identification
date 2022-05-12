#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# author: zcy

# TODO hsv 用字典存储，改好后相应的修改learner_image，已改好，未debug

import os
import shutil
import subprocess
import time
import dhash
import json
import cv2
import numpy as np
from PIL import Image
from google.cloud import vision
from skimage.io import imread
from skimage.measure import shannon_entropy
import pytesseract
from multiprocessing import Lock, Pool

from configs import dir_apps, dir_unzips, dir_images_meta, dir_images_resc, dir_images_temp, dir_features_img, dir_features_all, market
from configs import image_file_types
from utils import read_apk_features


size_threshold = 144 * 144
diff_hash_bit_threshold = 5
img_limit = 20

# If google cloud cannot be accessed directly, set up proxy here
os.environ['http_proxy'] = 'http://10.20.48.184:7890'
os.environ['https_proxy'] = 'http://10.20.48.184:7890'


def get_dhash(path):
    image = Image.open(path)
    return dhash.dhash_int(image)


def get_similarity_from_hash(hash0, hash1):
    # sim = 1 - dhash.get_num_bits_different(hash0, hash1) / 128
    # return sim
    return dhash.get_num_bits_different(hash0, hash1)


def load_meta_imgs_dhash(pkg):  # TODO
    img_list = []
    for img_path in os.path.join(dir_images_meta % market, pkg):
        if img_path.rsplit('.', 1)[-1].lower() not in image_file_types:
            continue
        img_list.append((os.path.join(dir_images_meta % market, pkg, img_path), None,
                         get_dhash(os.path.join(dir_images_meta % market, pkg, img_path))))
    return img_list


def get_unsimilar_imgs(img_list, limit, img_list_fixed):
    # 获取指定数量上限的不相似图片
    img_list_new = []
    for tuple in img_list:
        if len(img_list_new) >= limit:
            break
        try:
            tuple_new = (tuple[0], tuple[1], get_dhash(tuple[0]))  # tuple中增加dhash信息
        # 受损图片
        except:
            continue

        # 与已记录的每个图片和截图图标对比哈希值
        is_similar = False
        for tuple_existed in img_list_fixed + img_list_new:
            if get_similarity_from_hash(tuple_new[2], tuple_existed[2]) <= diff_hash_bit_threshold:
                is_similar = True
                break

        if not is_similar:
            img_list_new.append(tuple_new)
    return [(t[0], t[1]) for t in img_list_new]


def filter_one_img(path):
    # 格式过滤
    ext = path.split('.')[-1].lower()
    if ext not in image_file_types:
        return None, None

    try:
        img = Image.open(path)
        img.verify()
        img = Image.open(path)  # 调用verify后需要重新打开

        # 尺寸过滤
        if img.width * img.height < size_threshold:
            return None, None

        # 对webp格式图片做格式转换
        if ext == 'webp' or img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert('RGB')
            os.remove(path)
            # 更改后缀名，以新格式保存
            path = path.rsplit('.', 1)[0] + '.jpg'
            img.save(path, 'jpeg')
            ext = 'jpg'
    except Exception as e:
        print(e)
        return None, None
    return path, ext


def get_img_weight(path, mode='entropy'):
    if mode == 'entropy':
        try:
            # 计算香农熵
            # TODO 常见bug，打不开图像，导致熵为0
            img = imread(path, as_gray=True)
            # img_gray = rgb2gray(img)
            entropy = shannon_entropy(img)
            return entropy
        except Exception as e:
            print(path, e)
            return 0
    else:
        return os.path.getsize(path)


def image_filter_one_app(pkg):
    # 单独处理meta图片（只做格式转换不做过滤？）
    for img_name in os.listdir(os.path.join(dir_images_meta % market, pkg)):
        filter_one_img(os.path.join(dir_images_meta % market, pkg, img_name))

    img_list = []
    # 加载meta图片
    # if os.path.exists(os.path.join(dir_images_resc, pkg)):
    #     for file in os.listdir(os.path.join(dir_images_resc, pkg)):
    #
    # else:
    #     os.mkdir(os.path.join(dir_images_resc, pkg))

    # 为解决分卷压缩包及乱码问题，改用jar解压，区别是jar只能在运行命令的路径存放解压出的文件，不需要更改后缀名
    # try:
    #     # shutil.move('%s/%s.apk' % (dir_apps, pkg), '%s/%s.zip' % (dir_apps, pkg))
    #     shutil.copy('%s/%s.apk' % (dir_apps, pkg), '%s/%s/%s.apk' % (dir_unzips, pkg, pkg))
    # except:
    #     pass

    # 1. 遍历apk目录下所有文件，若格式为图像且尺寸足够大，拷贝到tmp文件夹（重命名以防有同名）
    try:
        # mkdir(dir_unzips/pkg), jar -xf, os.walk(dir_unzips/pkg), rm(dir_unzips/pkg)
        # 解压apk文件
        # subprocess.run(['unzip -o -q -O UTF8 -d {0}/{2} {1}/{2}.zip'.format(dir_unzips, dir_apps, pkg)], shell=True)
        if not os.path.exists(os.path.join(dir_unzips % market, pkg)):
            os.mkdir(os.path.join(dir_unzips % market, pkg))
        try:
            subprocess.run(['jar -xf ../../apps/%s.apk' % pkg], shell=True, cwd=os.path.join(dir_unzips % market, pkg))
        except subprocess.CalledProcessError as e:
            print('unzip error at app =', pkg, e)
        for root, dirs, files in os.walk('%s/%s' % (dir_unzips % market, pkg)):
            for name in files:
                # 此处可能经过格式转换，图片位置未改变
                new_path, new_ext = filter_one_img(os.path.join(root, name))
                if not new_path:
                    continue

                # 选择图片的依据，entropy（默认）或size
                weight = get_img_weight(new_path)
                if weight > 0:
                    if not os.path.exists(os.path.join(dir_images_temp % market, pkg)):
                        os.mkdir(os.path.join(dir_images_temp % market, pkg))
                    dest = '%s/%s/%s%d.%s' % (dir_images_temp % market, pkg, pkg, len(img_list) + 1, new_ext)
                    # 移动到图片临时文件夹
                    shutil.move(new_path, dest)
                    img_list.append((dest, weight))

    except Exception as e:
        print('error occur at app filter %s: %s' % (pkg, str(e)))
    finally:
        # 删除解压缩文件
        if os.path.exists(os.path.join(dir_unzips % market, pkg)):
            shutil.rmtree(os.path.join(dir_unzips % market, pkg))
        # shutil.move('%s/%s.zip' % (dir_apps, pkg), '%s/%s.apk' % (dir_apps, pkg))
        # os.remove('%s/%s.zip' % (dir_unzips, pkg))

    # 2. 对所有解包出的图像，按同一依据从大到小排序，根据dhash去重，记录前N个最大的图像
    if img_list:
        img_list = sorted(img_list, key=lambda t: t[1], reverse=True)  # (file, weight)
        img_list_meta = load_meta_imgs_dhash(pkg)
        img_list_filtered = get_unsimilar_imgs(img_list, img_limit, img_list_meta)  # (file, weight)
        # print([(img[0].split('/')[-1], img[1]) for img in img_list_filtered])

        # 把选中的图像移动到对应的resc图像文件夹下
        if img_list_filtered:
            if not os.path.exists(os.path.join(dir_images_resc % market, pkg)):
                os.mkdir(os.path.join(dir_images_resc % market, pkg))
            for img in img_list_filtered:
                shutil.move(img[0], img[0].replace(dir_images_temp % market, dir_images_resc % market))
        # 删除临时文件夹
        if os.path.exists(os.path.join(dir_images_temp % market, pkg)):
            shutil.rmtree(os.path.join(dir_images_temp % market, pkg))


def image_filter_one_process(pkg):
    print('[%s] %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), pkg))
    try:
        image_filter_one_app(pkg)
    except Exception as e:
        print('error occur at app process %s' % pkg, str(e))
        if os.path.exists(os.path.join(dir_images_resc % market, pkg)):
            shutil.rmtree(os.path.join(dir_images_resc % market, pkg))


def image_filter(process_num):
    pool = Pool(process_num)
    print('[%s] filtering images ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    # 对每个apk文件，解包并拷贝出尺寸足够大的图像类型文件，放于tmp文件夹
    for file in os.listdir(dir_apps % market):
        if not file.endswith('.apk'):
            continue
        pkg = file[:-4]
        # 检查是否已提取过feature
        if os.path.exists(os.path.join(dir_features_img % market, pkg) + '.txt'):
            continue
        # 检查是否已提取过图片
        if os.path.exists(os.path.join(dir_images_resc % market, pkg)):
            continue
        pool.apply_async(image_filter_one_process, (pkg, ))
    pool.close()
    pool.join()


# 处理图片文件夹，删除重复图片（程序多次运行会产生的问题，应该已经被修复，不需要再用此方法）
# 存在问题，如果有重名不同格式的图（程序多次运行产生的），会被覆盖，图像数量可能减少
def remove_duplicate_images():
    # 对每个应用中所有图片做过滤
    for pkg, _, img_files in os.walk(dir_images_resc % market):
        # 先记录所有图片
        imgs_all = []
        for img_file in img_files:
            new_path, new_ext = filter_one_img(os.path.join(pkg, img_file))
            if new_path:
                weight = get_img_weight(new_path, 'size')
                if weight > 0:
                    imgs_all.append((new_path, weight))
            # 删除无效文件
            else:
                os.remove(os.path.join(pkg, img_file))
        # 选出weight最大且不重复的N个图像，注：因为不同格式图片重名问题，imgs_all中可能有重复元素
        if imgs_all:
            imgs_all = sorted(imgs_all, key=lambda t: t[1], reverse=True)
            imgs_filtered = get_unsimilar_imgs(imgs_all, img_limit)
            # 删除未选中的图像的文件
            for img in imgs_all:
                if img not in imgs_filtered and os.path.exists(img[0]):
                    os.remove(img[0])
        print('[%s] %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), pkg))


# 处理所有图片feature，检查是否有同名不同格式的图片的feature（提取feature后更改图片格式后又提取feature产生的）
def remove_duplicate_features():
    for file in os.listdir(dir_features_img % market):
        features_all = read_apk_features(os.path.join(dir_features_img % market, file))
        pkg = file.rsplit('.', 1)[0]
        features_to_remove = []
        for img_name, img_features in features_all.items():
            path = os.path.join(dir_images_meta % market, pkg, img_name)
            if not os.path.exists(path) and not os.path.exists(path.replace(dir_images_meta % market, dir_images_resc % market)):
                if os.path.exists(path.rsplit('.', 1)[0] + '.jpg'):
                    print('extension changed: %s' % path)
                    features_to_remove.append(img_name)
                else:
                    print('else 1: %s' % path)

        if features_to_remove:
            for feature in features_to_remove:
                del features_all[feature]
            with open(os.path.join(dir_features_img % market, file), "w", encoding='utf-8') as handle:
                json.dump(features_all, handle, ensure_ascii=False)


# 处理图片feature文件夹
# 检查对应的图片文件夹，删除已删除图像的feature
# 对已抽取google api feature的图像，在其feature文件中增加更多feature
# 注意：之后要添加到feature extractor中去，此方法不再调用
def update_features():
    files = os.listdir(dir_features_img % market)
    files.sort()
    for file in files[:]:
        features_all = read_apk_features(os.path.join(dir_features_img % market, file))
        if not features_all:
            continue
        pkg = file.rsplit('.', 1)[0]
        # if not os.path.exists(os.path.join(dir_images_resc, pkg)):
        #     continue
        print('\n[%s] %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), pkg))
        for img_name, img_features in features_all.items():
            if 'ocr_raw' in features_all[img_name]:
                continue
            path = os.path.join(dir_images_resc % market, pkg, img_name)
            # 若图片本体不存在
            # if not os.path.exists(path):
            #     # 检查是否是格式变动
            #     if os.path.exists(path.rsplit('.', 1)[0] + '.jpg'):
            #         img_name_ = img_name.rsplit('.', 1)[0] + '.jpg'
            #         path = os.path.join(dir_images_resc, pkg, img_name_)
            #         features_update[img_name_] = features_all[img_name]
            #         print(img_name, '->', img_name_)
            #         img_name = img_name_
            #     # 则删除对应的feature
            #     else:
            #         print(img_name, 'removed')
            #         continue
            # else:
            #     features_update[img_name] = features_all[img_name]
            # print(img_name, 'updating')
            # 若图片本体存在，增加feature
            try:
                img = cv2.imread(path)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # 一维hsv
                hsv_flat = hsv.reshape(-1, hsv.shape[-1])
                # 图片中所有出现过的颜色值
                colors = np.unique(hsv_flat, axis=0)
                # 颜色数量
                features_all[img_name]['color_count'] = len(colors)
                # hsv量化（减少颜色数量）
                hsv_reduc = np.floor(np.divide(hsv_flat, [60.1, 85.1, 51.1]))  # 180/3, 255/3, 255/5)，.1用来减少一个端点值
                # 量化的颜色值及数量
                colors2, counts2 = np.unique(hsv_reduc, axis=0, return_counts=True)
                size = hsv.shape[0] * hsv.shape[1]
                color_props = np.zeros(45)  # 3*3*5
                for color, count in zip(colors2, counts2):
                    # color[0] * 15 + color[1] * 5 + color[2] 将45个三维颜色值映射到0-44
                    color_props[int(color[0] * 15 + color[1] * 5 + color[2])] = count / size
                features_all[img_name]['color_props'] = color_props.tolist()
                features_all[img_name]['ocr_raw'] = pytesseract.image_to_string(img, lang='chi_sim+eng')
            except Exception as e:
                print('error occur at img %s: %s' % (img_name, str(e)))
                # del features_all[img_name]
        if features_all.keys():
            with open(os.path.join(dir_features_img % market, file), "w", encoding='utf-8') as handle:
                json.dump(features_all, handle, ensure_ascii=False)
        else:
            os.remove(os.path.join(dir_features_img % market, file))


def detect_image(path, is_uri=False):
    client = vision.ImageAnnotatorClient()
    features_to_detect = [{'type_': vision.Feature.Type.LABEL_DETECTION}
                          # {'type_': vision.Feature.Type.SAFE_SEARCH_DETECTION}
                          # {'type_': vision.Feature.Type.OBJECT_LOCALIZATION}
                          ]
    # local
    if not is_uri:
        with open(path, 'rb') as image_file:
            content = image_file.read()
        response = client.annotate_image({
            'image': {'content': content},
            'features': features_to_detect})
    # remote
    else:
        response = client.annotate_image({
            'image': {'source': {'image_uri': path}},
            'features': features_to_detect})

    # detect labels
    labels = response.label_annotations
    # print('Labels:')
    dict_labels = {}
    for label in labels:
        dict_labels[label.description] = label.score
        # print('{} (confidence: {})'.format(label.description, label.score))

    # safe search
    safe = response.safe_search_annotation
    dict_safe = {}
    # 值为vision.Likelihood，1-5离散值，除以5归一化
    dict_safe['adult'] = safe.adult / 5
    dict_safe['medical'] = safe.medical / 5
    dict_safe['spoofed'] = safe.spoof / 5
    dict_safe['violence'] = safe.violence / 5
    dict_safe['racy'] = safe.racy / 5
    # likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY')
    # print('\nSafe search:')
    # print('adult: {}'.format(likelihood_name[safe.adult]))
    # print('medical: {}'.format(likelihood_name[safe.medical]))
    # print('spoofed: {}'.format(likelihood_name[safe.spoof]))
    # print('violence: {}'.format(likelihood_name[safe.violence]))
    # print('racy: {}'.format(likelihood_name[safe.racy]))

    # # localize objects
    # objects = response.localized_object_annotations
    # dict_objects = {}
    # # print('\nNumber of objects found: {}'.format(len(objects)))
    # for object_ in objects:
    #     dict_objects[object_.name] = object_.score
    #     # print('\n{} (confidence: {})'.format(object_.name, object_.score))
    #     # print('Normalized bounding polygon vertices: ')
    #     # for vertex in object_.bounding_poly.normalized_vertices:
    #     #     print(' - ({}, {})'.format(vertex.x, vertex.y))

    # if response.error.message:
    #     raise Exception('{}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'
    #                     .format(response.error.message))

    return {'labels': dict_labels, 'safe': dict_safe}  # , 'objects': dict_objects}


# path指向一个应用的经过滤的图片的目录
def feature_extractor_of_imgs_in(pkg):
    img_features = {}
    # 先读取已获得的图片的feature（若有）
    if os.path.exists(os.path.join(dir_features_img % market, pkg) + '.txt'):
        img_features = read_apk_features(os.path.join(dir_features_img % market, pkg) + '.txt')
    # 处理目录下每个图片文件，没有提取feature（或之前提取失败）的图片加入list
    path_list = []
    # 更改为只处理前6张截图，不处理资源图像
    for file in os.listdir(os.path.join(dir_images_meta % market, pkg)):
        if file not in img_features.keys():
            name = file.split('.')[0]
            if name == 'icon' or int(name.replace('screenshot', '')) <= 6:
                path_list.append((os.path.join(dir_images_meta % market, pkg, file), file))
    # for file in os.listdir(os.path.join(dir_images_resc % market, pkg)):
    #     if file not in img_features.keys():
    #         path_list.append((os.path.join(dir_images_resc % market, pkg, file), file))
    # 若没有需要处理的图片，返回空字典，以防feature文件被完全同样的内容覆盖
    if not path_list:
        return {}

    print('[%s] %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), pkg))
    for img_path, file in path_list:
        if img_path.split('.')[-1] in image_file_types:

            # google vision api features
            result = None
            try_times = 0
            # 可重试两次
            while not result and try_times < 3:
                try:
                    result = detect_image(img_path)
                except Exception as e:
                    print('error! while calling api: %s' % str(e))
                    try_times += 1
                    time.sleep(20)
            if not result:
                print('error! failed to detect %s/%s for 3 times' % (pkg, file))
                continue
            img_features[file] = result

            # 图片本身features
            try:
                img_features[file]['dhash'] = get_dhash(img_path)
                img_features[file]['size'] = get_img_weight(img_path, 'size')
                img_features[file]['entropy'] = get_img_weight(img_path, 'entropy')
                img = cv2.imread(img_path)
                img_features[file]['width'] = img.shape[1]
                img_features[file]['height'] = img.shape[0]

                # Hue[0, 179], Saturation[0, 255], Vue[0, 255]
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # print(hsv.mean(axis=0).mean(axis=0))
                avg_hsv = hsv.mean(axis=0).mean(axis=0).tolist()
                img_features[file]['avg_hsv'] = {'avg_h': avg_hsv[0], 'avg_s': avg_hsv[1], 'avg_v': avg_hsv[2]}

                # 一维hsv
                hsv_flat = hsv.reshape(-1, hsv.shape[-1])
                # 图片中所有出现过的颜色值
                colors = np.unique(hsv_flat, axis=0)
                # 颜色数量
                img_features[file]['color_count'] = len(colors)

                # hsv量化（减少颜色数量）
                hsv_reduc = np.floor(np.divide(hsv_flat, [60.1, 85.1, 51.1]))  # 180/3, 255/3, 255/5)，.1用来减少一个端点值
                # 量化的颜色值及数量
                colors2, counts2 = np.unique(hsv_reduc, axis=0, return_counts=True)
                size = hsv.shape[0] * hsv.shape[1]
                color_props = np.zeros(45)  # 3*3*5
                for color, count in zip(colors2, counts2):
                    # color[0] * 15 + color[1] * 5 + color[2] 将45个三维颜色值映射到0-44
                    color_props[int(color[0] * 15 + color[1] * 5 + color[2])] = count / size
                img_features[file]['color_props'] = color_props.tolist()

                try:
                    img_features[file]['ocr_raw'] = pytesseract.image_to_string(img, lang='chi_sim+eng')
                except Exception as ee:
                    print('error! while getting ocr %s/%s: %s' % (pkg, file, str(ee)))
                    img_features[file]['ocr_raw'] = ''
            except Exception as e:
                print('error! occur at img %s/%s: %s' % (pkg, file, str(e)))
                print(file, img_features[file])
                del img_features[file]

    return img_features


def feature_extractor_single_process(pkg):
    features = feature_extractor_of_imgs_in(pkg)
    # print(dict_img_features)
    if features.keys():
        with open(os.path.join(dir_features_img % market, pkg) + '.txt', "w", encoding='utf-8') as handle:
            json.dump(features, handle, ensure_ascii=False)


def feature_extractor(process_num):
    pool = Pool(process_num)
    # with open(pkg_record_file, 'r') as f:
    #     pkg_list = f.readlines()
    # process_tag = 1
    # step = 1000  # 100 apps per hour per thread
    # pkg_list = pkg_list[step*(process_tag-1): step*process_tag]

    print('[%s] extracting features ...' % time.strftime('%Y-%m-%d %H:%M:%S'))
    # 待处理的应用list
    for pkg in os.listdir(dir_images_meta % market):
    # 指定包名list
    # with open('../sampler/google_sampled_pkgs.txt', 'r') as f:
    #     pkgs = f.readlines()
    # for pkg in pkgs:
        pkg = pkg.strip()
        # 已提取过不能跳过，应在之后检查有无遗漏的图片
        # if os.path.exists(os.path.join(dir_features_img, pkg) + '.txt'):
        #     continue
        # 处理每个应用
        # if os.path.isdir(os.path.join(dir_images_resc % market, pkg)) and \
        # if os.path.exists('%s/%s.txt' % (dir_features_all % market, pkg)):
        pool.apply_async(feature_extractor_single_process, (pkg, ))
            # feature_extractor_single_process(path, pkg)
    pool.close()
    pool.join()


if __name__ == '__main__':
    # img = imread('../data/google/icon.png')
    # img = imread('../data/huawei/icon.png')
    # remove_duplicate_features()
    # update_features()
    # image_filter(60)
    feature_extractor(16)
