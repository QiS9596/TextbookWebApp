"""
Exceptions used in this project
"""


class DocumentClassifierException(Exception):
    """
    Base class for Exceptions in DocumentClassifier
    """
    def __init__(self, message):
        super().__init__()
        self.message = message


class NotInitializedException(DocumentClassifierException):
    """
    When calling a unfitted module, or uninitialized child object in any kind of DocumentClassifier object
    will raise this Exception
    """
    def __init__(self, message):
        super(NotInitializedException, self).__init__(message)
