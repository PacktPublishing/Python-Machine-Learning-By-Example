import numpy as np
import csv
from sklearn.metrics import roc_auc_score


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

n = 100000
X_dict_train, y_train = read_ad_click_data(n)
dict_one_hot_encoder = DictVectorizer(sparse=False)
X_train = dict_one_hot_encoder.fit_transform(X_dict_train)

X_train_100k = X_train
y_train_100k = np.array(y_train)

X_dict_test, y_test_next10k = read_ad_click_data(10000, 100000)
X_test_next10k = dict_one_hot_encoder.transform(X_dict_test)

# Use scikit-learn package
from sklearn.linear_model import SGDClassifier
sgd_lr = SGDClassifier(loss='log', penalty=None, fit_intercept=True, n_iter=5, learning_rate='constant', eta0=0.01)
sgd_lr.fit(X_train_100k, y_train_100k)

predictions = sgd_lr.predict_proba(X_test_next10k)[:, 1]
print('The ROC AUC on testing set is: {0:.3f}'.format(roc_auc_score(y_test_next10k, predictions)))



# Feature selection with L1 regularization

l1_feature_selector = SGDClassifier(loss='log', penalty='l1', alpha=0.0001, fit_intercept=True, n_iter=5, learning_rate='constant', eta0=0.01)
l1_feature_selector.fit(X_train_10k, y_train_10k)
X_train_10k_selected = l1_feature_selector.transform(X_train_10k)
print(X_train_10k_selected.shape)
print(X_train_10k.shape)

# bottom 10 weights and the corresponding 10 least important features
print(np.sort(l1_feature_selector.coef_)[0][:10])
print(np.argsort(l1_feature_selector.coef_)[0][:10])
# top 10 weights and the corresponding 10 most important features
print(np.sort(l1_feature_selector.coef_)[0][-10:])
print(np.argsort(l1_feature_selector.coef_)[0][-10:])



# Online learning

# The number of iterations is set to 1 if using partial_fit.
sgd_lr = SGDClassifier(loss='log', penalty=None, fit_intercept=True, n_iter=1, learning_rate='constant', eta0=0.01)

import timeit
start_time = timeit.default_timer()

# there are 40428968 labelled samples, use the first ten 100k samples for training, and the next 100k for testing
for i in range(20):
    X_dict_train, y_train_every_100k = read_ad_click_data(100000, i * 100000)
    X_train_every_100k = dict_one_hot_encoder.transform(X_dict_train)
    sgd_lr.partial_fit(X_train_every_100k, y_train_every_100k, classes=[0, 1])


print("--- %0.3fs seconds ---" % (timeit.default_timer() - start_time))

X_dict_test, y_test_next10k = read_ad_click_data(10000, (i + 1) * 200000)
X_test_next10k = dict_one_hot_encoder.transform(X_dict_test)


predictions = sgd_lr.predict_proba(X_test_next10k)[:, 1]
print('The ROC AUC on testing set is: {0:.3f}'.format(roc_auc_score(y_test_next10k, predictions)))


# Multiclass classification with logistic regression

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer

all_names = set(names.words())
lemmatizer = WordNetLemmatizer()

def letters_only(astr):
    for c in astr:
        if not c.isalpha():
            return False
    return True

def clean_text(docs):
    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(' '.join([lemmatizer.lemmatize(word.lower())
                                        for word in doc.split()
                                        if letters_only(word)
                                        and word not in all_names]))
    return cleaned_docs

data_train = fetch_20newsgroups(subset='train', categories=None, random_state=42)
data_test = fetch_20newsgroups(subset='test', categories=None, random_state=42)

cleaned_train = clean_text(data_train.data)
label_train = data_train.target
cleaned_test = clean_text(data_test.data)
label_test = data_test.target

tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english', max_features=40000)
term_docs_train = tfidf_vectorizer.fit_transform(cleaned_train)
term_docs_test = tfidf_vectorizer.transform(cleaned_test)

# combined with grid search
from sklearn.model_selection import GridSearchCV
parameters = {'penalty': ['l2', None],
              'alpha': [1e-07, 1e-06, 1e-05, 1e-04],
              'eta0': [0.01, 0.1, 1, 10]}

sgd_lr = SGDClassifier(loss='log', learning_rate='constant', eta0=0.01, fit_intercept=True, n_iter=10)

grid_search = GridSearchCV(sgd_lr, parameters, n_jobs=-1, cv=3)

grid_search.fit(term_docs_train, label_train)
print(grid_search.best_params_)

sgd_lr_best = grid_search.best_estimator_
accuracy = sgd_lr_best.score(term_docs_test, label_test)
print('The accuracy on testing set is: {0:.1f}%'.format(accuracy*100))
