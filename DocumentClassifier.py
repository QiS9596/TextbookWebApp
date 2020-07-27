"""
DocumentClassifier
"""
from abc import abstractmethod, ABC
import TBAppExceptions
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


class BaseDocumentClassifier(ABC):
    def __init__(self):
        self.classifier = None

    @abstractmethod
    def predict(self, text):
        """
        Predict the type of the document based on the input text
        The length of returned list will have the same length as the input list text
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
    def save(self, path):
        """
        Save the current model to target location at path
        :param path: string; the place and file name for saving the file
        :return: None
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path):
        """
        Load an existing trained model in
        :raise FileNotFoundError: raise FileNotFoundError when the target file in the path does not exist
        :param path: path to target file
        :return: a BaseDocumentClassifier object
        """

class TFIDFDocumentClassifier(BaseDocumentClassifier):
    def __init__(self):
        super().__init__()
        self.TFIDF_matrix = None
        self.count_vect = CountVectorizer()
        self.tfidf_transformer = TfidfTransformer()

    def predict(self, text):
        if self.classifier is None:
            raise TBAppExceptions.NotInitializedException
        return self.classifier.predict(self.count_vect.transform(text))

    def fit(self, text, target):
        word_counts = self.count_vect.fit_transform(text)
        self.TFIDF_matrix = self.tfidf_transformer.fit_transform(word_counts)
        self.classifier = MultinomialNB().fit(self.TFIDF_matrix, target)

    def save(self, path):
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        raise NotImplementedError