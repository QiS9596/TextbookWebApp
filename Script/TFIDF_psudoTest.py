import DocumentClassifier
import os
import pandas as pd
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

TF_IDF_classifier.fit(strs, labels)
result = TF_IDF_classifier.predict(strs)
hit = 0
for i in range(len(result)):
    if result[i] == labels[i]:
        hit += 1

print(float(hit)/float(len(result)))
