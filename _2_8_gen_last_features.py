
# coding: utf-8

# 生成用户上一次访问相关的特征

# In[1]:

import os
import pickle
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import (load_pickle, dump_pickle, get_nominal_dfal, feats_root,
                   mem_usage, reduce_mem_usage, nominal_cate_cols,
                   ordinal_cate_cols, identity_cols, continual_cols, top, freq,
                   unique, vrange, percentile)

pd.set_option('display.max_columns', 1000)


# In[5]:

def gen_user_time_delta_feature(updata=False):
    dump_file = './feats/user_time_delta_feature.pkl'

    if os.path.exists(dump_file) and not updata:
        print('Found ' + dump_file)
    else:
        print('gen_user_time_delta_feature ing...')
        dfal = get_nominal_dfal()
        dfal = dfal[['user_id', 'dt']].sort_values('dt', ascending=True)
        ugp = dfal.groupby('user_id')
        
        threshold_3day = 60 * 60 * 24 * 3
        threshold_2day = 60 * 60 * 24 * 2
        threshold_1day = 60 * 60 * 24 * 1
        
        dfal['agg_user_ts_delta1'] = ugp['dt'].transform(lambda x: x.diff(1))
        dfal['agg_user_new'] = dfal.agg_user_ts_delta1.isnull().apply(lambda x: 1 if x else 0)
        dfal.agg_user_ts_delta1 = dfal.agg_user_ts_delta1.apply(lambda x: x.total_seconds())
        dfal.agg_user_ts_delta1 = dfal.agg_user_ts_delta1.apply(lambda x: x if x < threshold_3day else threshold_3day)
        dfal.agg_user_ts_delta1 = dfal.agg_user_ts_delta1.fillna(threshold_3day)
        dfal['agg_user_da_delta1'] = dfal.agg_user_ts_delta1.apply(lambda x:x//86400)
        dfal['agg_user_ho_delta1'] = dfal.agg_user_ts_delta1.apply(lambda x:x//3600)
        dfal['agg_user_mi_delta1'] = dfal.agg_user_ts_delta1.apply(lambda x:x//60)
        
        #dfal['agg_user_noa_05mi'] = dfal.agg_user_mi_delta.apply(lambda x: 1 if x >=  5 else 0)
        #dfal['agg_user_noa_10mi'] = dfal.agg_user_mi_delta.apply(lambda x: 1 if x >= 10 else 0)
        #dfal['agg_user_noa_30mi'] = dfal.agg_user_mi_delta.apply(lambda x: 1 if x >= 30 else 0)
        
        #dfal['agg_user_noa_2ho'] = dfal.agg_user_ho_delta.apply(lambda x: 1 if x >= 2 else 0)
        #dfal['agg_user_noa_5ho'] = dfal.agg_user_ho_delta.apply(lambda x: 1 if x >= 5 else 0)
        #dfal['agg_user_noa_9ho'] = dfal.agg_user_ho_delta.apply(lambda x: 1 if x >= 9 else 0)
        
        #dfal['agg_user_noa_1da'] = dfal.agg_user_ts_delta.apply(lambda x: 1 if x >= threshold_1day else 0)
        #dfal['agg_user_noa_2da'] = dfal.agg_user_ts_delta.apply(lambda x: 1 if x >= threshold_2day else 0)
        #dfal['agg_user_noa_3da'] = dfal.agg_user_ts_delta.apply(lambda x: 1 if x >= threshold_3day else 0)
        dfal = dfal.drop(['dt', 'user_id'], axis=1)
        dfal, _ = reduce_mem_usage(dfal)
        dump_pickle(dfal, dump_file)
        print('gen_user_time_delta_feature completed')
        del dfal


# In[6]:

def gen_user_last_attrs_feature(updata=False):
    dump_file = './feats/user_last_attrs_feature.pkl'
    if os.path.exists(dump_file) and not updata:
        print('Found ' + dump_file)
    else:
        print('gen_user_last_attrs_feature ing...')
        attrs_cols = [
            'item_category_list', 'item_brand_id', 'item_city_id', 'item_price_level',
            'item_sales_level', 'item_collected_level', 'item_pv_level',
            'context_page_id', 'shop_review_num_level', 'shop_star_level',
            'shop_review_positive_rate', 'shop_score_delivery',
            'shop_score_description', 'shop_score_service'
        ]

        level_cols = [
            'item_price_level', 'item_sales_level', 'item_collected_level',
            'item_pv_level', 'shop_review_num_level', 'shop_star_level',
            'shop_review_positive_rate', 'shop_score_delivery',
            'shop_score_description', 'shop_score_service'
        ]
        dfal = get_nominal_dfal()
        dfal = dfal[['user_id', 'dt'] + attrs_cols].sort_values('dt', ascending=True)
        ugp = dfal.groupby('user_id')

        dfal[['agg_last1_' + c for c in attrs_cols]] = ugp[attrs_cols].transform(lambda x: x.shift(1))
        dfal[['agg_last2_' + c for c in attrs_cols]] = ugp[attrs_cols].transform(lambda x: x.shift(2))
        dfal[['agg_diff1_' + c for c in level_cols]] = ugp[level_cols].transform(lambda x: x.diff(1))
        dfal[['agg_diff2_' + c for c in level_cols]] = ugp[level_cols].transform(lambda x: x.diff(2))
        dfal = dfal.drop(attrs_cols + ['dt', 'user_id'], axis=1)
        dump_pickle(dfal, dump_file)
        print('gen_user_last_attrs_feature completed')
        del dfal


# In[7]:

if __name__ == '__main__':
    gen_user_time_delta_feature(True)
    gen_user_last_attrs_feature(True)


# In[ ]:



