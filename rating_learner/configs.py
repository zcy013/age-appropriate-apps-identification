#!/usr/bin/python3
# -*- coding: UTF-8 -*-
# author: zcy

# import

random_state = 98765
# dir_root =
market = 'google'
table = market #if market not in ['google'] else market + '_subset'
dir_apps = '../data/%s/nkid/apps'
dir_unzips = '../data/%s/nkid/unzips'
dir_images_meta = '../data/%s/nkid/images_meta'
dir_images_resc = '../data/%s/nkid/images_resc'
dir_images_temp = '../data/%s/nkid/images_temp'

dir_features_all = '../data/%s/%s/features_all'
dir_features_img = '../data/%s/%s/features_img'

category_mappings = {
    # 'huawei':
    #     {'game-Action': ['动作射击'], 'game-Strategy': ['经营策略'], 'Tools': ['实用工具'], 'game-Casino': [],
    #      'game-Role Playing': ['角色扮演'], 'News & Magazines': ['新闻阅读'], 'game-Simulation': [],
    #      'game-Casual': ['休闲益智'], 'Books & Reference': [], 'Dating': [], 'Sports': [], 'Business': ['商务'],
    #      'Music & Audio': [], 'Personalization': [], 'game-Adventure': [], 'Social': ['社交通讯'],
    #      'Maps & Navigation': ['出行导航'], 'Art & Design': [], 'Auto & Vehicles': ['汽车'], 'Entertainment': ['影音娱乐'],
    #      'Food & Drink': ['美食'], 'game-Educational': [], 'game-Arcade': [], 'Lifestyle': ['便捷生活'],
    #      'Photography': ['拍摄美化'], 'Education': ['教育'], 'game-Puzzle': [], 'Libraries & Demo': [],
    #      'Finance': ['金融理财'], 'Communication': [], 'Shopping': ['购物比价'], 'Productivity': [],
    #      'Health & Fitness': ['运动健康'], 'Weather': [], 'game-Card': [], 'Medical': [], 'Travel & Local': ['旅游住宿'],
    #      'game-Racing': [], 'Events': [], 'Comics': [], 'game-Trivia': [], 'House & Home': [],
    #      'Video Players & Editors': [], 'game-Sports': ['体育竞速'], 'game-Board': ['棋牌桌游'], 'Parenting': ['儿童'],
    #      'game-Music': [], 'Beauty': [], 'game-Word': []},
    'google':
        {'game-Action': ['game-Action'], 'game-Strategy': ['game-Strategy'], 'Tools': ['Tools'],
         'game-Casino': ['game-Casino'], 'game-Role Playing': ['game-Role Playing'],
         'News & Magazines': ['News & Magazines'], 'game-Simulation': ['game-Simulation'],
         'game-Casual': ['game-Casual'], 'Books & Reference': ['Books & Reference'], 'Dating': ['Dating'],
         'Sports': ['Sports'], 'Business': ['Business'], 'Music & Audio': ['Music & Audio'],
         'Personalization': ['Personalization'], 'game-Adventure': ['game-Adventure'], 'Social': ['Social'],
         'Maps & Navigation': ['Maps & Navigation'], 'Art & Design': ['Art & Design'],
         'Auto & Vehicles': ['Auto & Vehicles'], 'Entertainment': ['Entertainment'], 'Food & Drink': ['Food & Drink'],
         'game-Educational': ['game-Educational'], 'game-Arcade': ['game-Arcade'], 'Lifestyle': ['Lifestyle'],
         'Photography': ['Photography'], 'Education': ['Education'], 'game-Puzzle': ['game-Puzzle'],
         'Libraries & Demo': ['Libraries & Demo'], 'Finance': ['Finance'], 'Communication': ['Communication'],
         'Shopping': ['Shopping'], 'Productivity': ['Productivity'], 'Health & Fitness': ['Health & Fitness'],
         'Weather': ['Weather'], 'game-Card': ['game-Card'], 'Medical': ['Medical'], 'Travel & Local': ['Travel & Local'],
         'game-Racing': ['game-Racing'], 'Events': ['Events'], 'Comics': ['Comics'], 'game-Trivia': ['game-Trivia'],
         'House & Home': ['House & Home'], 'Video Players & Editors': ['Video Players & Editors'],
         'game-Sports': ['game-Sports'], 'game-Board': ['game-Board'], 'Parenting': ['Parenting'],
         'game-Music': ['game-Music'], 'Beauty': ['Beauty'], 'game-Word': ['game-Word']},
    'huawei':
        {'教育': ['教育'], '金融理财': ['金融理财'], '拍摄美化': ['拍摄美化'], '购物比价': ['购物比价'],
         '儿童': ['儿童'], '出行导航': ['出行导航'], '社交通讯': ['社交通讯'], '运动健康': ['运动健康'],
         '新闻阅读': ['新闻阅读'], '经营策略': ['经营策略'], '旅游住宿': ['旅游住宿'], '实用工具': ['实用工具'],
         '影音娱乐': ['影音娱乐'], '商务': ['商务'], '便捷生活': ['便捷生活'], '汽车': ['汽车'],
         '棋牌桌游': ['棋牌桌游'], '美食': ['美食'], '角色扮演': ['角色扮演'], '动作射击': ['动作射击'],
         '休闲益智': ['休闲益智'],  # '体育竞速': ['体育竞速'], 'other': [],
         },
    # 'google':
    #     {'教育': ['Education'], '金融理财': ['Finance'], '拍摄美化': ['Photography', 'Video Players & Editors'],
    #      '购物比价': ['Shopping'], '儿童': ['Parenting'], '出行导航': ['Maps & Navigation'],
    #      '社交通讯': ['Social', 'Dating', 'Communication'], '运动健康': ['Health & Fitness', 'Medical', 'Sports'],
    #      '新闻阅读': ['Books & Reference', 'News & Magazines', 'Comics'], '经营策略': ['game-Simulation', 'game-Strategy'],
    #      '旅游住宿': ['Travel & Local'], '实用工具': ['Tools'], '影音娱乐': ['Entertainment', 'Music & Audio'],
    #      '商务': ['Business', 'Productivity'], '便捷生活': ['House & Home', 'Weather', 'Lifestyle', 'Beauty', 'Events'],
    #      '汽车': ['Auto & Vehicles'], '棋牌桌游': ['game-Board', 'game-Casino'], '美食': ['Food & Drink'],
    #      '角色扮演': ['game-Role Playing', 'game-Card', 'game-Adventure'],  # '动作射击': ['game-Action', 'game-Arcade'],
    #      '休闲益智': ['game-Puzzle', 'game-Casual', 'game-Trivia', 'game-Music', 'game-Word', 'game-Educational'],
    #      '体育竞速': ['game-Racing', 'game-Sports']},
    #     # 'other': ['Personalization'··, 'Art & Design', 'Libraries & Demo'],
    '360':
        {'tools': ['系统安全'], 'social communication': ['通讯社交'], 'entertainment': ['影音视听'],
         'photography': ['摄影摄像'], 'reading': ['新闻阅读'], 'location': [], 'travelling': ['地图旅游'],
         'shopping': ['购物优惠'], 'finance': ['金融理财'], 'education': ['教育学习'], 'health': ['健康医疗'],
         'lifestyle': ['生活休闲'], 'traffic': [], 'business': ['办公商务'], 'dinning': [], 'kids': ['儿童'],
         'other': ['主题壁纸'],
         'role playing': ['游戏-角色扮演', '游戏-网络游戏'], 'casual puzzle': ['游戏-休闲益智', '游戏-儿童游戏'],
         'strategy': ['游戏-经营策略'], 'sports racing': ['游戏-体育竞速'], 'action': ['游戏-动作冒险', '游戏-飞行射击'],
         'board': ['游戏-棋牌天地']},
    'tencent':
        {'tools': ['工具', '系统', '安全'], 'social communication': ['社交', '通讯'], 'entertainment': ['视频', '音乐', '娱乐'],
         'photography': ['摄影', '美化'], 'reading': ['阅读', '新闻'], 'location': ['出行'], 'travelling': ['旅游'],
         'shopping': ['购物'], 'finance': ['理财'], 'education': ['教育'], 'health': ['健康'],
         'lifestyle': ['生活'], 'traffic': [], 'business': ['办公'], 'dinning': [], 'kids': ['儿童'],
         'other': ['主题壁纸'],
         'role playing': ['游戏-角色扮演', '游戏-网络游戏'], 'casual puzzle': ['游戏-休闲益智'],
         'strategy': ['游戏-经营策略'], 'sports racing': ['游戏-体育竞速'], 'action': ['游戏-动作冒险', '游戏-飞行射击'],
         'board': ['游戏-棋牌中心']},
}

sensitive_words_dict = {
    'huawei':
        ['暴力', '不良', '惊吓', '理财', '直播', '恋爱', '交友', '成人'],
    'google':
        ['violence', 'bad', 'frightening', 'nudity', 'gambling', 'sexual', 'expletive', 'tobacco', 'drug', 'criminal', 'discriminatory', 'illegal']
        # ['暴力', '粗俗', '裸露', '赌博', '淫秽', '脏话', '色情', '刺激', '抽烟', '吸毒', '犯罪', '杀戮', '歧视', '违禁', '药品']
}
sensitive_words = sensitive_words_dict[market]

# 参考Android官网https://developer.android.com/guide/topics/media/media-formats
image_file_types = ['bmp', 'jpg', 'jpeg', 'png', 'webp', 'heic', 'heif']

features_to_count = ['hardware', 'libraries', 'permissions', 'urls', 'apis', 'minAge', #'category',
                     'activities', 'providers', 'receivers', 'services', 'intents']  # 需要进行初筛（子特征过多，先统计一次频率，之后只考虑频率高的子特征）
# 训练模型时保留的feature
# features_in_number = ['classes_length', 'methods_length', 'fields_length']  # , 'score', 'loc']
features_manifest = ['permissions']  # 'hardware', 'libraries']  # to count
features_code = ['apis']  # 'urls', 'apis', 'activities', 'providers', 'receivers', 'services', 'intents']  # to count
features_cate = ['category']
features_ads = ['ad_label']
features_name = ['name']
features_text = ['description']
features_imgs = ['labels']
features_ocrs = ['ocrs']
features_rate = ['minAge']
features_colo = ['avg_hsv']

MODEL_MANI = 'manifest'
MODEL_CODE = 'code'
MODEL_CATE = 'category'
MODEL_ADS = 'ads'
MODEL_NAME = 'name'
MODEL_TEXT = 'text'
MODEL_IMGS = 'image'
MODEL_OCRS = 'ocrs'
MODEL_RATE = 'rating'
MODEL_HSV = 'hsv'

# 选择模型时只需修改此变量
model2features = {
    # MODEL_MANI: features_manifest,
    MODEL_CODE: features_code,
    MODEL_CATE: features_cate,
    MODEL_ADS: features_ads,
    # MODEL_NAME: features_name,
    # MODEL_TEXT: features_text,
    MODEL_IMGS: features_imgs,
    # MODEL_OCRS: features_ocrs,
    # MODEL_RATE: features_rate,
    # MODEL_HSV: features_colo
}
# 集成时的权重，值为单个模型在所有数据上的AUC
model_weights_market = {
    'google': {
        MODEL_MANI: 94.16,
        MODEL_CODE: 96.53,
        MODEL_CATE: 93.12,
        MODEL_ADS:  58.02,
        MODEL_RATE: 64.47,
        MODEL_NAME: 84.13,
        MODEL_TEXT: 94.38,
        MODEL_IMGS: 92.93,
        MODEL_OCRS: 90.78,
        MODEL_HSV:  72.55
    }, 'huawei': {
        MODEL_MANI: 57.88,
        MODEL_CODE: 56.36,
        # MODEL_CATE: 86.01,
        MODEL_ADS:  50.00,
        MODEL_TEXT: 100.0,
        MODEL_IMGS: 100.0,
        MODEL_OCRS: 96.47
    }
}
model_weights = model_weights_market[market]

feature_size_market = {
    'google': {
        MODEL_MANI: 240,
        MODEL_CODE: 2750,
        MODEL_NAME: 100,
        MODEL_TEXT: 120,
        MODEL_IMGS: 130,
        MODEL_OCRS: 50
    },
    'huawei': {

    }
}
feature_size_assigned = feature_size_market[market]
