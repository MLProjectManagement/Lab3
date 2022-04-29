import os

PREFIX = ".." if os.path.dirname(os.getcwd()) == "src" else ""

DATA_PATH = os.path.join(PREFIX, 'data/')
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw/')
TRANSFORMED_DATA_PATH = os.path.join(DATA_PATH, 'transformed/')

X_TRAIN_PATH = os.path.join(TRANSFORMED_DATA_PATH, 'X_train.csv')
X_TEST_PATH = os.path.join(TRANSFORMED_DATA_PATH, 'X_test.csv')
Y_TRAIN_PATH = os.path.join(TRANSFORMED_DATA_PATH, 'y_train.csv')
Y_TEST_PATH = os.path.join(TRANSFORMED_DATA_PATH, 'y_test.csv')

LOG_PATH = os.path.join(PREFIX, 'logs/')

