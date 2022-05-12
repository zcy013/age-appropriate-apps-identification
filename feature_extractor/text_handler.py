 #!/usr/bin/python3
# -*- coding: UTF-8 -*-
# author: zcy

import os
import time
import re
import json

from configs import dir_features_img, market
from utils import read_apk_features

# variables


# TODO 代码整合进image_handler
def ocr_preprocess():
    for file in os.listdir(dir_features_img % market):
        features_all = read_apk_features(os.path.join(dir_features_img % market, file))
        if not features_all:
            continue
        pkg = file.rsplit('.', 1)[0]
        print('\n[%s] %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), pkg))
        for img_name, img_features in features_all.items():
            # if 'string' in features_all[img_name] or 'ocr_raw' not in features_all[img_name]:
            #     continue
            try:
                ocr_raw = features_all[img_name]['ocr_raw'].strip()

                # method 1 : regex
                # strs_cn = re.findall('[\u4e00-\u9fa5]+', ocr_raw)
                # strs_en = re.findall('[a-zA-Z\d]+', ocr_raw)
                # result = ''.join(strs_cn) + ' ' + ' '.join(strs_en)

                # method 2 : manually traverse
                ocr_clean = []
                keep_space = False
                for char in ocr_raw:
                    if re.match('[\u4e00-\u9fa5]', char):
                        ocr_clean.append(char)
                        keep_space = False
                    elif re.match('[a-zA-Z\d]', char):
                        ocr_clean.append(char)
                        keep_space = True
                    # 保留除空格以外的常见符号
                    elif char in "\n,.;:'?!()-/\\%，。；：？！（）":
                        if (len(ocr_clean) > 0 and char != ocr_clean[-1]) or len(ocr_clean) < 1:
                            ocr_clean.append(char)
                        keep_space = False
                    else:
                        if keep_space:
                            ocr_clean.append(' ')
                            keep_space = False
                result = ''.join(ocr_clean)

                features_all[img_name]['string'] = result
            except Exception as e:
                print('error occur at img %s/%s: %s' % (pkg, img_name, str(e)))
                # del features_all[img_name]

        if features_all.keys():
            with open(os.path.join(dir_features_img % market, file), "w", encoding='utf-8') as handle:
                json.dump(features_all, handle, ensure_ascii=False)
        else:
            os.remove(os.path.join(dir_features_img % market, file))


if __name__ == '__main__':
    ocr_preprocess()
