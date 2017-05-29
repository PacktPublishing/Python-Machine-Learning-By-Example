import numpy as np


import csv

def read_ad_click_data(n, offset=0):
    X_dict, y = [], []
    with open('train.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for i in range(offset):
            next(reader)
        i = 0
        for row in reader:
            i += 1
            y.append(int(row['click']))
            del row['click'], row['id'], row['hour'], row['device_id'], row['device_ip']
            X_dict.append(row)
            if i >= n:
                break
    return X_dict, y

n = 10000
X_dict_train, y_train = read_ad_click_data(n)

from sklearn.feature_extraction import DictVectorizer
dict_one_hot_encoder = DictVectorizer(sparse=False)
X_train = dict_one_hot_encoder.fit_transform(X_dict_train)

X_dict_test, y_test = read_ad_click_data(n, n)
X_test = dict_one_hot_encoder.transform(X_dict_test)

X_train_10k = X_train
y_train_10k = np.array(y_train)


# Feature selection with random forest

from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=30, n_jobs=-1)
random_forest.fit(X_train_10k, y_train_10k)


# bottom 10 weights and the corresponding 10 least important features
print(np.sort(random_forest.feature_importances_)[:10])
print(np.argsort(random_forest.feature_importances_)[:10])
# top 10 weights and the corresponding 10 most important features
print(np.sort(random_forest.feature_importances_)[-10:])
print(np.argsort(random_forest.feature_importances_)[-10:])

print(dict_one_hot_encoder.feature_names_[393])

top500_feature = np.argsort(random_forest.feature_importances_)[-500:]
X_train_10k_selected = X_train_10k[:, top500_feature]
print(X_train_10k_selected.shape)
