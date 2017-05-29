from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
import glob
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer


emails, labels = [], []

file_path = '../enron1/spam/'
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with open(filename, 'r', encoding = "ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(1)

file_path = '../enron1/ham/'
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with open(filename, 'r', encoding = "ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(0)

def letters_only(astr):
    for c in astr:
        if not c.isalpha():
            return False
    return True

all_names = set(names.words())
lemmatizer = WordNetLemmatizer()

def clean_text(docs):
    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(' '.join([lemmatizer.lemmatize(word.lower())
                                        for word in doc.split()
                                        if letters_only(word)
                                        and word not in all_names]))
    return cleaned_docs

cleaned_emails = clean_text(emails)

from sklearn.model_selection import StratifiedKFold
k = 10
k_fold = StratifiedKFold(n_splits=k)
# convert to numpy array for more efficient slicing
cleaned_emails_np = np.array(cleaned_emails)
labels_np = np.array(labels)

smoothing_factor_option = [1.0, 2.0, 3.0, 4.0, 5.0]
from collections import defaultdict
auc_record = defaultdict(float)

for train_indices, test_indices in k_fold.split(cleaned_emails, labels):
    X_train, X_test = cleaned_emails_np[train_indices], cleaned_emails_np[test_indices]
    Y_train, Y_test = labels_np[train_indices], labels_np[test_indices]
    tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english', max_features=8000)
    term_docs_train = tfidf_vectorizer.fit_transform(X_train)
    term_docs_test = tfidf_vectorizer.transform(X_test)
    for smoothing_factor in smoothing_factor_option:
        clf = MultinomialNB(alpha=smoothing_factor, fit_prior=True)
        clf.fit(term_docs_train, Y_train)
        prediction_prob = clf.predict_proba(term_docs_test)
        pos_prob = prediction_prob[:, 1]
        auc = roc_auc_score(Y_test, pos_prob)
        auc_record[smoothing_factor] += auc

print(auc_record)

print('max features  smoothing  fit prior  auc')
for smoothing, smoothing_record in auc_record.items():
        print('       8000      {0}      true    {1:.4f}'.format(smoothing, smoothing_record/k))


