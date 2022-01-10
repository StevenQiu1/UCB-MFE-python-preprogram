import unittest
import inspect

def logPoint(context):
    'utility function used for module functions and class methods'
    callingFunction = inspect.stack()[1][3]
    print 'in %s - %s()' % (context, callingFunction)

def setUpModule():
    'called once, before anything else in this module'
    logPoint('module %s' % __name__)

def tearDownModule():
    'called once, after everything else in this module'
    logPoint('module %s' % __name__)

class TestFixtures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        'called once, before any tests'
        logPoint('class %s' % cls.__name__)

    @classmethod
    def tearDownClass(cls):
        'called once, after all tests, if setUpClass successful'
        logPoint('class %s' % cls.__name__)

    def setUp(self):
        'called multiple times, before every test method'
        self.logPoint()

    def tearDown(self):
        'called multiple times, after every test method'
        self.logPoint()

    def test_1(self):
        'a test'
        self.logPoint()

    def test_2(self):
        'another test'
        self.logPoint()

    def logPoint(self):
        'utility method to trace control flow'
        callingFunction = inspect.stack()[1][3]
        currentTest = self.id().split('.')[-1]
        print 'in %s - %s()' % (currentTest, callingFunction)