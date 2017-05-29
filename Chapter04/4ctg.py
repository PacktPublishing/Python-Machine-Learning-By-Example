import pandas as pd
df = pd.read_excel('CTG.xls', "Raw Data")

X = df.ix[1:2126, 3:-2].values
Y = df.ix[1:2126, -1].values   # 3 class classification
# Y = df.ix[2:2126, -2].values

from collections import Counter
Counter(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

from sklearn.svm import SVC
svc = SVC(kernel='rbf')

parameters = {'C': (100, 1e3, 1e4, 1e5),
              'gamma': (1e-08, 1e-7, 1e-6, 1e-5)
              }
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(svc, parameters, n_jobs=-1, cv=3)


import timeit
start_time = timeit.default_timer()
grid_search.fit(X_train, Y_train)
print("--- %0.3fs seconds ---" % (timeit.default_timer() - start_time))

print(grid_search.best_params_)
print(grid_search.best_score_)

svc_best = grid_search.best_estimator_

accuracy = svc_best.score(X_test, Y_test)
print('The accuracy on testing set is: {0:.1f}%'.format(accuracy*100))

prediction = svc_best.predict(X_test)
from sklearn.metrics import classification_report
report = classification_report(Y_test, prediction)
print(report)
