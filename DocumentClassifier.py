"""
DocumentClassifier
"""
from abc import abstractmethod, ABC
import TBAppExceptions
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from joblib import dump, load
import streamlit as st

class BaseDocumentClassifier(ABC):
    def __init__(self, classifier=None):
        self.classifier = classifier

    @abstractmethod
    def predict(self, text):
        """
        Predict the type of the document based on the input text
        The length of returned list will have the same length as the input list text
        :raise TBAppExceptions.NotInitializedException if self.classifier is not initialized
        :param text: list of string; a list of text of the document to be classified
        :return: list of int; the predicted class
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, text, target):
        """
        Train the prediction model with data
        Assertions: len(text) == len(target)
        :param text: list of strings; a list of text of the document
        :param target: list of ints; list of the prediction target for the document in the text list
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path, **kwargs):
        """
        Save the current model to target location at path
        :raise TBAppExceptions.NotInitializedException if self.classifier is not initialized
        :param path: string; the place and file name for saving the file
        :return: None
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path, **kwargs):
        """
        Load an existing trained model in
        :raise FileNotFoundError: raise FileNotFoundError when the target file in the path does not exist
        :param path: path to target file
        :return: a BaseDocumentClassifier object
        """

    def eval(self, test_text, test_target, metrics=['acc']):
        """
        Perform evaluation on test data
        Assertions: len(test_text) == len(test_target)
        Supported metrics:
            acc: prediction accuracy
        :raise TBAppExceptions.NotInitializedException if self.classifier is not initialized
        :param test_text: list of strings; each string is a text of the document to be classified
        :param test_target: list of integers; each integer is a classification label for a document
        :param metrics: list of strings; each string represent a evaluation metric
        :return: a dictionary object, dictionary.keys() is same as metrics,
        """
        result = {}
        predicted = self.predict(test_text)
        if 'acc' in metrics:
            correct = 0
            for i in range(len(predicted)):
                if predicted[i] == test_target[i]:
                    correct += 1
            result['acc'] = float(correct) / float(len(predicted))
        return result


class SKLearnBasedDocumentClassifier(BaseDocumentClassifier, ABC):
    def save(self, path, **kwargs):
        if self.classifier is None:
            raise TBAppExceptions.NotInitializedException(
                type(self).__name__ + ' when trying to save model: classifier not initialized')
        dump(self.classifier, path)

    @classmethod
    def load(cls, path, **kwargs):
        return cls(classifier=load(path))


class TFIDFDocumentClassifier(SKLearnBasedDocumentClassifier):
    def __init__(self, classifier=None, count_vect=None):
        super().__init__(classifier)
        self.TFIDF_matrix = None
        if count_vect is None:
            self.count_vect = CountVectorizer()
        else:
            self.count_vect = count_vect
        self.tfidf_transformer = TfidfTransformer()

    @st.cache
    def predict(self, text):
        if self.classifier is None:
            raise TBAppExceptions.NotInitializedException(
                type(self).__name__ + 'when trying to call predict: classifier not initialized.')
        return self.classifier.predict(self.count_vect.transform(text))

    def fit(self, text, target):
        word_counts = self.count_vect.fit_transform(text)
        self.TFIDF_matrix = self.tfidf_transformer.fit_transform(word_counts)
        self.classifier = MultinomialNB().fit(self.TFIDF_matrix, target)

    def save(self, path, **kwargs):
        if self.classifier is None:
            raise TBAppExceptions.NotInitializedException(
                type(self).__name__ + ' when trying to save model: classifier not initialized')
        dump(self.classifier, path)

    def save(self, path, **kwargs):
        """
        Save method specifically for TFIDFDocumentClassifier
        :param path: path to save the Multinominal Naive-Bayes Model
        :param kwargs: should contain vpath, which is the place for saving the count vectorizer
        :return: None
        """
        dump(self.classifier, path)
        dump(self.count_vect, kwargs['vpath'])

    @classmethod
    def load(cls, path, **kwargs):
        """
        Load model method sepcifically for TF-IDF document classifier
        :param path: path to the Multinominal Naive-Bayes Model
        :param kwargs: should contain vpath, that lead to count vectorizer
        :return: a TFIDFDocumentClassifier instance
        """

        return cls(classifier=load(path), count_vect=load(kwargs['vpath']))
