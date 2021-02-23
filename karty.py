#!/usr/bin/python3

import pandas as pn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trainFile = "./cards_available.csv"
predictFile = "./cards_available_test.csv"

train = pn.read_csv(trainFile)
assert isinstance(train, pn.DataFrame)
# print(train)
# print(train.dtypes)
data = train.loc[:, "playerClass": "damage"]
# print(data)
target = train["owner"]
# print(target)
column_transformer = ColumnTransformer(
    [
        ('test', OneHotEncoder(), [0, 7])
    ]
)
pipe = make_pipeline(
    column_transformer,
    KNeighborsClassifier()
)
# clf = KNeighborsClassifier()
# clf.fit(data, target)
clf = pipe.fit(data, target)
predict = pn.read_csv(predictFile)
ow = clf.predict(predict)
for r in ow:
    print(r)
