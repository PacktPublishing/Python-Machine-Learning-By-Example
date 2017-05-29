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
            X_dict.append(dict(row))
            if i >= n:
                break
    return X_dict, y

n = 100000
X_dict_train, y_train = read_ad_click_data(n)
print(X_dict_train[0])
print(X_dict_train[1])


from sklearn.feature_extraction import DictVectorizer
dict_one_hot_encoder = DictVectorizer(sparse=False)
X_train = dict_one_hot_encoder.fit_transform(X_dict_train)
print(len(X_train[0]))

X_dict_test, y_test = read_ad_click_data(n, n)
X_test = dict_one_hot_encoder.transform(X_dict_test)
print(len(X_test[0]))


from sklearn.tree import DecisionTreeClassifier
parameters = {'max_depth': [3, 10, None]}
decision_tree = DecisionTreeClassifier(criterion='gini', min_samples_split=30)

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(decision_tree, parameters, n_jobs=-1, cv=3, scoring='roc_auc')

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

decision_tree_best = grid_search.best_estimator_
pos_prob = decision_tree_best.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score
print('The ROC AUC on testing set is: {0:.3f}'.format(roc_auc_score(y_test, pos_prob)))



from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=30, n_jobs=-1)
grid_search = GridSearchCV(random_forest, parameters, n_jobs=-1, cv=3, scoring='roc_auc')
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)

random_forest_best = grid_search.best_estimator_
pos_prob = random_forest_best.predict_proba(X_test)[:, 1]
print('The ROC AUC on testing set is: {0:.3f}'.format(roc_auc_score(y_test, pos_prob)))


