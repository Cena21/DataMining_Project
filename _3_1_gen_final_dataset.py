
# coding: utf-8

# In[ ]:

#get_ipython().system('jupyter nbconvert --to script *.ipynb')


# In[1]:

import os
import pickle
import gc
import pandas as pd
pd.set_option('display.max_columns', 1000)
import numpy as np
from tqdm import tqdm_notebook, tnrange, tqdm
from utils import (reduce_mem_usage, load_pickle, dump_pickle, get_nominal_dfal, feats_root,
                   ordinal_cate_cols, nominal_cate_cols, identity_cols,
                   fit_cat, fit_lgb, verbose_feature_importance_cat)

import warnings
warnings.filterwarnings(action='ignore')

from sklearn.preprocessing import Imputer
import h5py


# In[2]:

from _2_1_gen_user_features import (
    add_user_click_stats, add_user_da_feature_click, add_user_ho_feature_click,
    add_user_total_da_click)
from _2_2_gen_item_features import (
    add_item_click_stats, add_item_da_feature_click, add_item_ho_feature_click,
    add_item_total_da_click)
from _2_3_gen_shop_features import (
    add_shop_click_stats, add_shop_da_feature_click, add_shop_ho_feature_click,
    add_shop_total_da_click)
from _2_4_gen_acc_sum_counts import add_global_count_sum
from _2_5_gen_smooth_cvr import add_hist_cvr_smooth
from _2_6_gen_bagging_features import add_target_features
from _2_7_gen_level_features import add_level_features


# In[5]:

def gen_dataset(data, dump_file, scope='tr', last_da=24, updata=False):
    if os.path.exists(dump_file) and not updata:
        print('Found ' + dump_file)
        store = pd.HDFStore(dump_file, mode='r',complevel=9,)
        data = store['dataset']
        store.close()
    else:
        print('Generating {} Dataset...'.format(scope))
        ##################################################################
        # add user click
        data = add_user_click_stats(data)
        data = add_user_total_da_click(data)
        data = add_user_da_feature_click(data)
        data = add_user_ho_feature_click(data)
        # add item click
        data = add_item_click_stats(data)
        data = add_item_total_da_click(data)
        data = add_item_da_feature_click(data)
        data = add_item_ho_feature_click(data)
        # add shop click
        data = add_shop_click_stats(data)
        data = add_shop_total_da_click(data)
        data = add_shop_da_feature_click(data)
        data = add_shop_ho_feature_click(data)
        # add global count sum
        data = add_global_count_sum(data, last_da)

        # add smooth cvr
        for c in tqdm(identity_cols, desc='add_hist_cvr_smooth'):
            data = add_hist_cvr_smooth(data, c)
        print('add_hist_cvr_smooth completed')
        
        for c in tqdm(['item_id', 'shop_id', 'user_id', 'item_brand_id', 'item_city_id'], desc='add_target_features'):
            data = add_target_features(data, c)
        print('add_target_features completed')
        
        for c in tqdm(['item_id', 'shop_id', 'user_id'], desc='add_level_features'):
            data = add_level_features(data, c)
        
        drop_cols = []
        for c in data.columns:
            if c.startswith('agg_level_item_id_shop_'):
                drop_cols.append(c)
        data = data.drop(drop_cols, axis=1)
        print('add_level_features completed')
        
        if scope == 'te':
            data.drop('is_trade', axis=1, inplace=True)
            
        nan_cols = []
        for c in data.columns:
            nan_count = data[data[c].isnull()].shape[0]
            if nan_count>0:
                print(c, nan_count)
                nan_cols.append(c)
        data[nan_cols] = Imputer(strategy='median').fit_transform(data[nan_cols])
        print(data.shape)
        
        
        store = pd.HDFStore(dump_file, mode='w',complevel=9)
        store['dataset'] = data
        store.close()
        print('Generated {} Dataset'.format(scope))
    
    
    return data


# In[6]:

def gen_final_dataset(tr_start_da, tr_end_da, te_da=24, updata=False):
    dfal = get_nominal_dfal()
    dfal = dfal.sort_values('dt')
    user_time_delta_feature = load_pickle('./feats/user_time_delta_feature.pkl')
    user_last_attrs_feature = load_pickle('./feats/user_last_attrs_feature.pkl')
    dfal = pd.concat([dfal, user_time_delta_feature, user_last_attrs_feature], axis=1)
    print(dfal.shape)
    
    dftr = dfal.loc[(dfal.da >= tr_start_da) & (dfal.da <= tr_end_da)]
    tr_dump_file = './cache/final_dataset_tr_{}_{}.h5'.format(tr_start_da, tr_end_da)
    dftr = gen_dataset(dftr,tr_dump_file, 'tr', tr_end_da, updata)
    
    dfte = dfal.loc[dfal.da == te_da]
    te_dump_file = './cache/final_dataset_te_{}.h5'.format(te_da)
    dfte = gen_dataset(dfte,te_dump_file, 'te', te_da, updata)
    
    del dfal
    gc.collect()
    return dftr, dfte


# In[7]:

ignore_cols = ['instance_id', 'dt', 'da', 'user_id', 'item_id', 'shop_id']

def get_dataset(updata=False):

    dftr, dfte = gen_final_dataset(19, 23, 24, updata)
    trset = dftr.loc[(dftr.da > 18) & (dftr.da <= 22), :].drop(ignore_cols, axis=1)
    vaset = dftr.loc[dftr.da == 23, :].drop(ignore_cols, axis=1)
    teset = dfte.loc[dfte.da == 24, :].drop(ignore_cols, axis=1)

    del dftr
    del dfte
    gc.collect()

    X_tr = trset.drop('is_trade', axis=1)
    X_va = vaset.drop('is_trade', axis=1)
    X_te = teset
    y_tr = trset.is_trade
    y_va = vaset.is_trade

    del trset
    del vaset
    del teset
    gc.collect()
    return X_tr, y_tr, X_va, y_va, X_te


# In[8]:

X_tr, y_tr, X_va, y_va, X_te =  get_dataset(True)


# In[9]:

X_tr.shape,y_tr.shape, X_va.shape, y_va.shape, X_te.shape


# In[10]:

cates_cols = [
    'item_category_list', 'item_city_id', 'user_gender_id',
    'user_occupation_id', 'item_brand_id'
]

cates_cols = cates_cols +  list(filter(lambda x : 'user' not in x, ['agg_last1_'+ c for c in cates_cols]))

cates_cols


# In[11]:

lgb1 = fit_lgb(X_tr, y_tr, X_va, y_va, cates_cols)


# In[12]:

lgb1.best_score


# In[13]:

y_hat_lgb1 = lgb1.predict(X_te, num_iteration=lgb1.best_iteration)


# In[14]:

unimportant_features = []
for x in sorted(zip(lgb1.feature_name(), lgb1.feature_importance("gain")), key=lambda x: x[1]):
    if x[1] < 1000:
        unimportant_features.append(x[0])

X_tr.drop(unimportant_features, axis=1, inplace=True)
X_va.drop(unimportant_features, axis=1, inplace=True)
X_te.drop(unimportant_features, axis=1, inplace=True)
cates_cols = list(filter(lambda x : x in X_tr.columns.values.tolist(), cates_cols))
print(cates_cols)
lgb2 = fit_lgb(X_tr, y_tr, X_va, y_va, cates_cols)


# In[15]:

lgb2.best_score


# In[16]:

y_hat_lgb2 = lgb2.predict(X_te, num_iteration=lgb2.best_iteration)


# ## catboost

# In[17]:

X_tr, y_tr, X_va, y_va, X_te =  get_dataset(False)


# In[18]:

cates_cols = [
    'item_category_list', 'item_city_id', 'user_gender_id',
    'user_occupation_id', 'item_brand_id'
]

cates_cols = cates_cols +  list(filter(lambda x : 'user' not in x, ['agg_last1_'+ c for c in cates_cols]))

cates_cols


# In[19]:

X_tr[cates_cols] = X_tr[cates_cols].astype(int)
X_va[cates_cols] = X_va[cates_cols].astype(int)
X_te[cates_cols] = X_te[cates_cols].astype(int)


# In[20]:

cates_idx = [X_tr.columns.values.tolist().index(c) for c in cates_cols]


# In[21]:

cat1 = fit_cat(X_tr, y_tr, X_va, y_va, cates_idx)


# In[22]:

y_hat_cat1 = cat1.predict_proba(X_te)[:,1]


# In[23]:

for score, name in sorted(zip(cat1.feature_importances_ , X_tr.columns), reverse=True):
    if score <= 0.1:
        del X_tr[name]
        del X_va[name]
        del X_te[name]
        print('{}: {}'.format(name, score))


# In[24]:

X_tr.shape,X_va.shape,X_te.shape


# In[25]:

best_cat_params = cat1.get_params().copy()
best_cat_params.update({'use_best_model': True})


# In[26]:

import catboost as cb


# In[27]:

cat2 = cb.CatBoostClassifier(**best_cat_params)
cat2 = cat2.fit(X_tr, y_tr, eval_set=(X_va, y_va))


# In[28]:

verbose_feature_importance_cat(cat2, X_tr)


# In[29]:

y_hat_cat2 = cat2.predict_proba(X_te)[:,1]


# In[30]:

for score, name in sorted(zip(cat2.feature_importances_, X_tr.columns), reverse=True):
    if score <= 0.1:
        del X_tr[name]
        del X_va[name]
        del X_te[name]
        print('{}: {}'.format(name, score))
print(X_tr.shape, X_va.shape, X_te.shape)


# In[31]:

cat3 = cb.CatBoostClassifier(**best_cat_params)
cat3 = cat3.fit(X_tr, y_tr, eval_set=(X_va, y_va))


# In[32]:

y_hat_cat3 = cat3.predict_proba(X_te)[:,1]


# In[33]:

verbose_feature_importance_cat(cat3, X_tr)


# In[35]:

dfal = get_nominal_dfal()
hat = dfal.loc[dfal.da==24, ['instance_id']]
del dfal
gc.collect()


# In[36]:

hat.shape, y_hat_cat1.shape


# In[37]:

hat['lgb1'] = y_hat_lgb1
hat['lgb2'] = y_hat_lgb2
hat['cat1'] = y_hat_cat1
hat['cat2'] = y_hat_cat2
hat['cat3'] = y_hat_cat3


# In[38]:

hat = hat.set_index('instance_id')
hat.head()


# In[39]:

hat['predicted_score'] = hat['cat1'] #hat.mean(axis=1)


# In[40]:

hat.head()


# In[41]:

from time import time


# In[42]:

hat.to_csv(
    './rests/20180422-lgb-cat-{}.txt'.format(int(time() // 1)),
    index=True,
    header=True,
    sep=' ',
    columns=['predicted_score'])


# In[ ]:

get_ipython().system('head ./rests/20180422-lgb-cat.txt')


# In[ ]:



