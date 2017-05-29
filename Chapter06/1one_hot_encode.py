from sklearn.feature_extraction import DictVectorizer


X_dict = [{'interest': 'tech', 'occupation': 'professional'},
          {'interest': 'fashion', 'occupation': 'student'},
          {'interest': 'fashion', 'occupation': 'professional'},
          {'interest': 'sports', 'occupation': 'student'},
          {'interest': 'tech', 'occupation': 'student'},
          {'interest': 'tech', 'occupation': 'retired'},
          {'interest': 'sports', 'occupation': 'professional'}]

dict_one_hot_encoder = DictVectorizer(sparse=False)
X_encoded = dict_one_hot_encoder.fit_transform(X_dict)
print(X_encoded)

print(dict_one_hot_encoder.vocabulary_)

new_dict = [{'interest': 'sports', 'occupation': 'retired'}]
new_encoded = dict_one_hot_encoder.transform(new_dict)
print(new_encoded)

print(dict_one_hot_encoder.inverse_transform(new_encoded))


# new category not encountered before
new_dict = [{'interest': 'unknown_interest', 'occupation': 'retired'},
            {'interest': 'tech', 'occupation': 'unseen_occupation'}]
new_encoded = dict_one_hot_encoder.transform(new_dict)
print(new_encoded)



import numpy as np
X_str = np.array([['tech', 'professional'],
                  ['fashion', 'student'],
                  ['fashion', 'professional'],
                  ['sports', 'student'],
                  ['tech', 'student'],
                  ['tech', 'retired'],
                  ['sports', 'professional']])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
X_int = label_encoder.fit_transform(X_str.ravel()).reshape(*X_str.shape)
print(X_int)

one_hot_encoder = OneHotEncoder()
X_encoded = one_hot_encoder.fit_transform(X_int).toarray()
print(X_encoded)



# new category not encountered before
new_str = np.array([['unknown_interest', 'retired'],
                  ['tech', 'unseen_occupation'],
                  ['unknown_interest', 'unseen_occupation']])

def string_to_dict(columns, data_str):
    data_dict = []
    for sample_str in data_str:
        data_dict.append({column: value for column, value in zip(columns, sample_str)})
    return data_dict

columns = ['interest', 'occupation']
new_encoded = dict_one_hot_encoder.transform(string_to_dict(columns, new_str))
print(new_encoded)