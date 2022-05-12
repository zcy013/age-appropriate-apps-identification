#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# author: zcy

import os
import requests
from urllib import request
import ssl
import time
import mysql.connector
from multiprocessing import Lock, Pool, cpu_count
import google_play_metadata_parser as googleplay_parser

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# url = "https://play.google.com/store/apps/details?id=%s&gl=cn&hl=zh-CN"  # 多为机翻中文或无中文
url = "https://play.google.com/store/apps/details?id=%s&gl=cn&hl=en"
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'


def crawler(pkg_name, to_retry=1):  # 每个包名爬取失败后可以重试一次
    try:
        req = request.Request(url % pkg_name)
        req.add_header('User-Agent', user_agent)
        web = request.urlopen(req, timeout=30, context=ssl._create_unverified_context())
        if web.code != 200:
            return None
        data = web.read().decode("utf-8")
    except Exception as e:
        if 'code' in dir(e) and e.code == 429:
            # TODO 待确认retry-after在哪里给出
            print(req.headers)
            time.sleep(int(e.headers["Retry-After"]))
        if 'code' not in dir(e) or e.code != 404:
            if to_retry:
                time.sleep(30)
                return crawler(pkg_name, to_retry - 1)
            else:
                print('[%s] error occur on crawling app %s: %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), pkg_name, str(e)))
                return None
        else:
            return None

    app_meta = googleplay_parser.get_app_basic_info(data)
    app_meta.update(googleplay_parser.get_app_image_urls(data))
    app_meta['desLong'] = googleplay_parser.get_app_description(data)
    return app_meta


def search_app_in_db(cursor, pkg):
    sql = 'SELECT * FROM googleplay WHERE pkg="%s"' % pkg
    try:
        cursor.execute(sql)
        result = cursor.fetchone()
    except Exception as e:
        print('[%s] SQL error on searching for app %s: %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), pkg, str(e)))
        return False
    if result:
        return result
    return False


def insert_app_into_db(cursor, db, dict):
    cols = ', '.join(dict.keys())
    vals = str(list(dict.values()))[1:-1].replace(', None, ', ', NULL, ').replace(', None, ', ', NULL, ')
    sql = 'INSERT INTO googleplay (%s) VALUE (%s)' % (cols, vals)
    # print(sql)
    try:
        cursor.execute(sql)
        db.commit()
        # print(time.strftime('%Y-%m-%d %H:%M:%S'), 'inserted', dict['pkg'])
    except Exception as e:
        print('[%s] SQL error on inserting app %s: %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), dict['pkg'], str(e)))
        print(sql)


def single_app_handler(pkg, metadata):
    mydb = mysql.connector.connect(
        host='10.20.48.184',
        port=7777,
        # host='localhost',
        user='root',
        passwd='123456',
        database='app_metadata',
        auth_plugin='mysql_native_password',
        charset='utf8mb4'
    )
    mycursor = mydb.cursor()

    # print(pkg)
    # print('searching for', pkg)
    if not search_app_in_db(mycursor, pkg):
    # if True:
        # print('crawling for', pkg)
        data = crawler(pkg)
        if not data or not data['name']:
            return
        data['sha256'] = metadata[0]
        data['sha1'] = metadata[1]
        data['md5'] = metadata[2]
        data['pkg'] = pkg
        data['verCode'] = int(metadata[6])
        try:
            data['vt_detection'] = int(metadata[7])
            data['vt_scan_date'] = metadata[8].split(' ')[0]
        except:
            data['vt_detection'] = None
            data['vt_scan_date'] = None
        data['markets'] = metadata[10]
        # print('inserting for', pkg)
        insert_app_into_db(mycursor, mydb, data)

    mycursor.close()


if __name__ == '__main__':

    # 按下载量选择部分应用下载？

    # 1. 从androzoo list中拿到不重复的最新应用列表
    # get_latest_record_for('./latest2106.csv.gz', 'play.google.com')

    # 2. 在应用市场中爬取所有应用meta信息，存入数据库（多进程）
    pool = Pool(processes=100)
    with open('../../data/apps_google.csv', 'r') as f:
        for line in f.readlines()[1:]:
            tmp = line.split(',')
            pkg = tmp[5].replace('"', '')
            pool.apply_async(single_app_handler, args=(pkg, tmp))
            # single_app_handler(pkg, tmp)
    pool.close()
    pool.join()

    # 3. 从数据库中按下载量？选取部分应用用sha256通过androzoo下载到本地
    #sha256,sha1,md5,dex_date,apk_size,pkg_name,vercode, vt_detection,vt_scan_date,dex_size,markets
