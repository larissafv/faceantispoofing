import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
seed = 123
rng = np.random.default_rng(seed)

def create_csv(df, list_id, filename):
    """
    Arguments:
        df: DataFrame, shape ``[n_images, 3]``
        list_id: List
        filename: String
    """
    
    data_dict = {
            'id': [],
            'path': [],
            'live': []
        }
    
    aux_live = []
    aux_spoof = []
    
    for i in list_id:
        live = df[(df['id'] == i) & (df['live'] == 1)]
        spoof = df[(df['id'] == i) & (df['live'] == 0)]
        #to ensure the ratio of 1:1 between live and spoof
        if live.shape[0] > 0 and spoof.shape[0] > 0:
            m = min(live.shape[0], spoof.shape[0])

            idx = rng.choice(range(live.shape[0]), m, False)
            for j in idx:
                data_dict['id'].append(i)
                data_dict['path'].append(live.iloc[j]['path'])
                data_dict['live'].append(1)

            idx = rng.choice(range(spoof.shape[0]), m, False)
            for j in idx:
                data_dict['id'].append(i)
                data_dict['path'].append(spoof.iloc[j]['path'])
                data_dict['live'].append(0)
        elif live.shape[0] > 0:
            aux_live.append(i)
        elif spoof.shape[0] > 0:
            aux_spoof.append(i)
    
    live = df[df['id'].isin(aux_live)]
    spoof = df[df['id'].isin(aux_spoof)]
    m = min(live.shape[0], spoof.shape[0])
    
    idx = rng.choice(range(live.shape[0]), m, False)
    for j in idx:
        data_dict['id'].append(live.iloc[j]['id'])
        data_dict['path'].append(live.iloc[j]['path'])
        data_dict['live'].append(1)

    idx = rng.choice(range(spoof.shape[0]), m, False)
    for j in idx:
        data_dict['id'].append(spoof.iloc[j]['id'])
        data_dict['path'].append(spoof.iloc[j]['path'])
        data_dict['live'].append(0)

    pd.DataFrame.from_dict(data_dict).to_csv('./img_data/'+filename, index = False)

    
data = pd.read_csv('./img_data/data.csv')
total = data['id'].unique()

#train -> 80%, val -> 10%, test -> 10%
train_id, aux_id = train_test_split(total, test_size = 0.2, random_state = seed)
val_id, test_id = train_test_split(aux_id, test_size = 0.5, random_state = seed)

create_csv(data, train_id, 'train.csv')
create_csv(data, val_id, 'val.csv')
create_csv(data, test_id, 'test.csv')
