import DocumentClassifier
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
TF_IDF_classifier = DocumentClassifier.TFIDFDocumentClassifier()
dataset_path = '../Data/701505'
# create a dictionary to map classes to integer labels
classes = os.listdir(dataset_path)
int_labels = [i for i in range(len(classes))]
mapping = {}
for i in range(len(classes)):
    mapping[classes[i]] = int_labels[i]

# load dataset class by class
strs = []
labels = []
for class_ in classes:
    class_path = os.path.join(dataset_path, class_)
    files = os.listdir(class_path)
    class_text = []
    for file in files:
        with open(os.path.join(class_path, file), encoding='utf8') as f:
            class_text += [f.read()]
    class_labels = [mapping[class_]] * len(class_text)

    strs += class_text
    labels += class_labels
data_array = np.array([strs, labels]).transpose()

dataframe = pd.DataFrame(data=data_array, columns=['text', 'label'])
dataframe = dataframe.sample(frac=1.0)
strs = dataframe['text'].to_list()
labels = dataframe['label'].to_list()

strs_train, strs_test, labels_train, labels_test = train_test_split(strs, labels, test_size=0.1)

TF_IDF_classifier.fit(strs_train, labels_train)
print(TF_IDF_classifier.eval(strs_train, labels_train))
print(TF_IDF_classifier.eval(strs_test, labels_test))
