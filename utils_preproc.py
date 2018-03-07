import pandas as pd
from pandas import HDFStore
import os
from IPython.display import Image, display

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler

def load_descriptors_layer_1(desc_file, ev_class=False):
    
    desc_db = HDFStore(desc_file)
    df = desc_db.select('event_desc')
    desc_db.close()

    if isinstance(ev_class,pd.DataFrame):
        df = df.merge(ev_class)
        id_cols = ['event_id', 'year', 'valid_event']
    else:
        id_cols = ['event_id', 'year']

    feat = df[df.columns.difference( id_cols )]
    ids  = df[id_cols].copy()
#    ids['year_eid'] = ids['year'].astype(str) + '-' + ids['event_id'].astype(str)
    print('Dataset size', feat.shape)
    
    return feat, ids

def split_train_test(x, y):
    # Split in train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12345)
    X_train.reset_index(inplace=True,drop=True)
    X_test.reset_index(inplace=True,drop=True)
    y_train.reset_index(inplace=True,drop=True)
    y_test.reset_index(inplace=True,drop=True)
    print('Train size:\t', X_train.shape)
    print('Test size:\t', X_test.shape)
    return X_train, X_test, y_train, y_test

def scale(x):
    # Scale features
    cols = x.columns
    x_sc = x.copy()
    scaler  = StandardScaler().fit(x)
    x_sc[cols] = scaler.transform(x)
    return x_sc, scaler

def pca_transform(x, n_comp):
    # Perform PCA
    pca = PCA().fit(x)
    print (pca.explained_variance_ratio_.cumsum())

    pca = PCA(n_components=n_comp).fit(x)

    x = pca.transform(x)
    x = pd.DataFrame(x, columns=['pc'+str(i) for i in range(1,n_comp+1)])
    
    return x, pca

def class_balance_ros(x,y, label_col = 'valid_event'):
    
    # Prepare dataframes to keep identifiers
    extra_cols = y.columns.difference([label_col])
    x[extra_cols] = y[extra_cols]
    
    # Balance classes
    ros = RandomOverSampler()
    x_ros, y_ros = ros.fit_sample(x, y[label_col])
    
    # Return datasets to initial columns structure
    x_ros = pd.DataFrame(x_ros, columns=x.columns)
    y_ros = pd.DataFrame(y_ros, columns=[label_col])
    y_ros[extra_cols] = x_ros[extra_cols]
    x_ros.drop(extra_cols, axis=1, inplace=True)
    
    return x_ros, y_ros
    
def pipeline(x,y):
    
    X_train, X_test, y_train, y_test = split_train_test(x, y)
    
    X_train, y_train = class_balance_ros(x,y)

    X_train, scaler = scale(X_train)
    X_train, pca = pca_transform(X_train, 10)
    
    X_test = scaler.transform(X_test)
    X_test = pca.transform(X_test)
    
    X_test = pd.DataFrame(X_test, columns=['pc'+str(i) for i in range(1,pca.n_components_+1)])
    
    print('Final datasets size:', )
    print('Train size:\t', X_train.shape)
    print('Test size:\t', X_test.shape)
    
    return X_train, X_test, y_train, y_test

def display_event( plots_dir, ev_id, year ):
    
    route = os.path.join( plots_dir[year], str(ev_id)+".png" )  # Display stored picture
    if os.path.exists(route):
        display(Image(filename=route))
    else:
        print ("File not found", route )
    return
    