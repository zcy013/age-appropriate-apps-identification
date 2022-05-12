#!/usr/bin/python3
import os
from androguard.core.analysis import analysis
from androguard.core.bytecodes.dvm import DalvikVMFormat
from androguard.core.analysis.analysis import Analysis
from androguard.decompiler.decompiler import DecompilerDAD
from androguard.core.bytecodes.apk import APK
from androguard.misc import AnalyzeAPK
import re
import hashlib
import json
import mysql.connector
import subprocess
import time
from multiprocessing import Lock, Pool

from configs import table, dir_features_all, dir_apps, market


class AndroidAPK:
    def __init__(self, path, pkg, cursor):
        self.path = path
        self.pkg = pkg
        # self.application = APK(self.path)
        # 不全面啊，应该传入get_all_de，不如直接用AnalyzeAPK
        # self.application_dex = DalvikVMFormat(self.application.get_dex())
        # self.application_x = Analysis(self.application_dex)
        # self.application_dex.set_decompiler(DecompilerDAD(self.application_dex, self.application_x))
        self.app, _, self.ana = AnalyzeAPK(path)
        self.features = {}
        self.cursor = cursor

    def get_identical_features(self):
        # 用于比较同一应用
        self.features['pkg'] = self.pkg
        self.features['android_version_code'] = self.app.get_androidversion_code()
        self.features['android_version_name'] = self.app.get_androidversion_name()
        self.features['max_sdk'] = self.app.get_max_sdk_version()
        self.features['min_sdk'] = self.app.get_min_sdk_version()
        self.features['target_sdk'] = self.app.get_target_sdk_version()
        self.features['md5'] = hashlib.md5(self.app.get_raw()).hexdigest()
        self.features['sha256'] = hashlib.sha256(self.app.get_raw()).hexdigest()
        # self.features['android_version_code'] = self.application.get_androidversion_code()
        # self.features['android_version_name'] = self.application.get_androidversion_name()
        # self.features['max_sdk'] = self.application.get_max_sdk_version()
        # self.features['min_sdk'] = self.application.get_min_sdk_version()
        # self.features['target_sdk'] = self.application.get_target_sdk_version()
        # self.features['md5'] = hashlib.md5(self.application.get_raw()).hexdigest()
        # self.features['sha256'] = hashlib.sha256(self.application.get_raw()).hexdigest()

    def get_metadata_features(self):
        # metadata
        if 'google' in table or 'huawei' in table:
            sql = 'select minAge, name, category, adLabel, developer, score, desLong, releaseDate from `%s` where pkg="%s"' % (table, self.pkg)
        else:
            sql = 'select pkg, name, category, adLabel, developer, score, desLong, releaseDate from `%s` where pkg="%s"' % (table, self.pkg)

        self.cursor.execute(sql)
        meta = self.cursor.fetchone()
        if 'google' in table or 'huawei' in table:
            self.features['minAge'] = meta[0]
        self.features['name'] = meta[1]
        self.features['category'] = meta[2]
        self.features['ad_label'] = bool(meta[3])
        self.features['developer'] = meta[4]
        self.features['score'] = float(meta[5])
        self.features['description'] = meta[6]
        self.features['release_date'] = str(meta[7])

    def get_manifest_features(self):
        # 硬件
        self.features['hardware'] = self.app.get_features()
        # 库
        self.features['libraries'] = self.app.get_libraries()
        # 权限
        self.features['permissions'] = self.app.get_details_permissions()  # name: [level, descShort, descLong]

        # 硬件
        # self.features['hardware'] = self.application.get_features()
        # # 库
        # self.features['libraries'] = self.application.get_libraries()
        # # 权限
        # self.features['permissions'] = self.application.get_details_permissions()  # name: [level, descShort, descLong]

    def get_code_features(self):
        # code
        self.features['urls'] = self.get_urls()
        self.features['apis'] = self.get_refine_apis()
        self.features['classes_length'] = len(self.ana.get_classes())
        self.features['methods_length'] = len(self.ana.get_methods())
        self.features['fields_length'] = len(self.ana.get_fields())

        # 四大组件
        self.features['activities'] = self.app.get_activities()
        self.features['providers'] = self.app.get_providers()
        self.features['receivers'] = self.app.get_receivers()
        self.features['services'] = self.app.get_services()

        # self.features['is_obfuscation'] = True if analysis.is_ascii_obfuscation(self.application_dex) else False

        # self.features['loc'] = self.get_line_of_code()

        s = self.app.get_all_attribute_value("action", "name")
        self.features['intents'] = [i for i in s]

        # code
        # self.features['urls'] = self.get_urls()
        # self.features['apis'] = self.get_refine_apis()
        # self.features['classes_length'] = len(self.application_dex.get_classes())
        # self.features['methods_length'] = len(self.application_dex.get_methods())
        # self.features['fields_length'] = len(self.application_dex.get_fields())
        #
        # # 四大组件
        # self.features['activities'] = self.application.get_activities()
        # self.features['providers'] = self.application.get_providers()
        # self.features['receivers'] = self.application.get_receivers()
        # self.features['services'] = self.application.get_services()
        #
        # self.features['is_obfuscation'] = True if analysis.is_ascii_obfuscation(self.application_dex) else False
        #
        # # self.features['loc'] = self.get_line_of_code()
        #
        # s = self.application.get_all_attribute_value("action", "name")
        # self.features['intents'] = [i for i in s]

    # def get_resource_features(self):
    #     # resource
    #     # 后续需要处理，从element中提取string
    #     self.features['strings'] = bytes.decode(self.application.get_android_resources().get_strings_resources())
    #     # 解压缩（统一放到另一个文件夹），遍历所有文件，若格式为图像，拷贝出来（重命名以防有同名）
    #     imgs = []
    #     subprocess.run(['unzip -o -q -d {0}/{2} {1}/{2}.zip'.format(dir_unzips, dir_apps, self.pkg)], shell=True)
    #     for root, dirs, files in os.walk('%s/%s' % (dir_unzips, self.pkg)):
    #         for name in files:
    #             ext = name.split('.')[-1]
    #             if ext in image_file_types:
    #                 if not os.path.exists(os.path.join(dir_images_meta, self.pkg)):
    #                     subprocess.run(['mkdir', self.pkg], cwd=dir_images_meta)
    #                 dest = '%s/%s/%s%d.%s' % (dir_images_meta, self.pkg, self.pkg, len(imgs)+1, ext)
    #                 subprocess.run(['mv -f "%s/%s" %s' % (root, name, dest)], shell=True)
    #                 imgs.append(dest)
    #     # 删除解压缩文件
    #     subprocess.run(['rm -rf {0}/{1}'.format(dir_unzips, self.pkg)], shell=True)
    #     # 压缩图片资源
    #     output = subprocess.check_output(['7za a -sdel %s.7z ./*' % pkg], shell=True, cwd=os.path.join(dir_images_meta, self.pkg))
    #     # 似乎有点多余
    #     # self.features['images'] = '\n'.join(imgs)

    def get_urls(self):
        strs = self.application_dex.get_strings()
        # pattern = 'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        # pattern = re.compile(r'([hH][tT]{2}[pP]:\/\/|[hH][tT]{2}[pP][sS]:\/\/|[wW]{3}.|[wW][aA][pP].|[fF][tT][pP].|[fF][iI][lL][eE].)[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]')
        pattern = re.compile(r'((http|ftp|https)://)(([a-zA-Z0-9\._-]+\.[a-zA-Z]{2,6})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(:[0-9]{1,4})*(/[a-zA-Z0-9\&%_\./-~-]*)?')
        urls = []
        for str in strs:
            it = re.finditer(pattern, str)
            for i in it:
                if i not in urls:
                    urls.append(i.group())
        return urls

    # def get_apis(self):
    #     methods = set()
    #     cs = [cc.get_name() for cc in self.application_dex.get_classes()]  # 在该应用中定义的所有类
    #     for method in self.application_dex.get_methods():  # 程序中所有方法名
    #         g = self.application_x.get_method(method)
    #         if method.get_code() is None:
    #             continue
    #         for i in g.get_basic_blocks().get():
    #             for ins in i.get_instructions():
    #                 output = ins.get_output()
    #                 match = re.search(r'(L[^;]*;)->[^\(]*\([^\)]*\).*', output)
    #                 if match and match.group(1) not in cs:  # 不在应用内定义的类中定义的方法调用就是api
    #                     methods.add(match.group())
    #     methods = list(methods)
    #     return methods

    #The ExternalClass is used for all classes that are not defined in the
        # DEX file, thus are external classes.
    def get_apis(self):
        methods = set()
        for m in self.ana.get_methods():
            # Method must be external to be an API
            if m.is_external():
            # if m.is_android_api():
                methods.add(m.get_method())
        return list(methods)

    def get_refine_apis(self):
        api_list = set()
        for api in self.get_apis():
            start_index = api.find('num_layers')
            end_index = api.find('(')
            start_index += 1
            api = api[start_index:end_index] + '()'
            api_list.add(api)
        return list(api_list)

    # TODO 这样不行，没有统计java源码
    # def get_line_of_code(self):
    #     # 统计代码行数
    #     cloc_output = subprocess.check_output(['/home/zcy/cloc-1.84/cloc --csv --quiet {0}.zip'.format(self.pkg)],
    #                                           shell=True, cwd=dir_apps)
    #     # cloc_output = subprocess.check_output(['cloc --csv --quiet {0}.zip'.format(self.pkg)],
    #     #                                       shell=True, cwd=dir_apps)
    #     cloc_output = bytes.decode(cloc_output).split('\n')
    #     for i in range(-1, -len(cloc_output) - 1, -1):
    #         line = cloc_output[i].split(',')
    #         if len(line) == 5:
    #             if line[1] == 'SUM':
    #                 return int(line[-1])
    #     # 之前没有加这个return，导致有些loc为null
    #     return 0

    def get_all_features(self):
        # try:
        #     subprocess.run(['mv {0}.apk {0}.zip'.format(self.pkg)], shell=True, cwd=dir_apps)
        # except:
        #     pass

        self.get_identical_features()
        self.get_metadata_features()
        self.get_manifest_features()
        self.get_code_features()
        # self.get_resource_features()
        # finally:
        #     subprocess.run(['mv {0}.zip {0}.apk'.format(self.pkg)], shell=True, cwd=dir_apps)

    def save_refine_apk(self, saved_folder):
        self.get_all_features()
        if self.features == {}:
            print('fail to get features')
            return

        path = saved_folder + "/" + self.pkg + '.txt'
        with open(path, "w", encoding='utf-8') as handle:
            json.dump(self.features, handle, ensure_ascii=False)
            # pickle.dump(self.features, handle)


def search_app_in_db(cursor, pkg):
    sql = 'SELECT * FROM `%s` WHERE pkg="%s"' % (table, pkg)
    try:
        cursor.execute(sql)
        result = cursor.fetchone()
    except Exception as e:
        print('[%s] SQL error on searching for app %s: %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), pkg, str(e)))
        return False
    if result:
        return result
    return False


def single_process(apk_path, pkg):
    if os.path.exists(dir_features_all % market + '/' + pkg + '.txt'):
        return

    mydb = mysql.connector.connect(
        host='10.20.48.184',
        port=7777,
        user='root',
        passwd='123456',
        database='app_metadata',
        auth_plugin='mysql_native_password',
        charset='utf8mb4'
    )
    mycursor = mydb.cursor()

    # 先查询在本地是否有安装包，在数据库中是否有数据
    if os.path.exists(apk_path) and search_app_in_db(mycursor, pkg):
        # 若有，取出数据库中的特征，提取安装包中的特征，存入文件
        # print('[%s] %s initializing ...' % (time.strftime('%Y-%m-%d %H:%M:%S'), pkg))
        try:
            result = AndroidAPK(apk_path, pkg, mycursor)
        except Exception as e:
            print('error initializing apk %s: %s' % (pkg, str(e)))
        # print('[%s] %s getting features ...' % (time.strftime('%Y-%m-%d %H:%M:%S'), pkg))
        result.save_refine_apk(dir_features_all % market)
        # print('[%s] %s done' % (time.strftime('%Y-%m-%d %H:%M:%S'), pkg))
    mydb.close()


if __name__ == '__main__':
    pool = Pool(20)
    for file in os.listdir(dir_apps % market):
        if not file.endswith('.apk'):
            continue
        try:
            pkg = file.rsplit('.apk', 1)[0]
            # apk_path = os.path.join(dir_apps, pkg + '.apk')
            # print('\n[%s] handling %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), pkg))
            # single_process(os.path.join(dir_apps, file), pkg)
            pool.apply_async(single_process, (os.path.join(dir_apps % market, file), pkg))

        except Exception as e:
            print('[%s] error occur: %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), str(e)))
    pool.close()
    pool.join()

# for test
# if __name__ == '__main__':
#     apk_path = '../data/huawei/apps/com.yongyou.apk'
#     result = AndroidAPK(apk_path, 'com.yongyou.apk', None)
#     result.save_refine_apk(dir_features_all)

    # features = read_apk_features(dir_features_all + '/com.eg.android.AlipayGphone.txt')
    # print(features)


# if __name__ == '__main__':
#     counter = []
#     for app in os.listdir(dir_apps % market):
#         path = os.path.join(dir_apps % market, app)
#         try:
#             counter.append(len(bytes.decode(APK(path).get_android_resources().get_strings_resources())))
#             if len(counter) == 100:
#                 break
#         except:
#             continue
#     print(counter)
#     print(sum(counter))
