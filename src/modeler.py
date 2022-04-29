import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import dagshub

from config import *


print('Initialize Modeling')
print('    Loading training and test datasets')
X_train = np.loadtxt(X_TRAIN_PATH, delimiter = ",")
X_test = np.loadtxt(X_TEST_PATH, delimiter = ",")
y_train = np.loadtxt(Y_TRAIN_PATH, delimiter = ",")
y_test = np.loadtxt(Y_TEST_PATH, delimiter = ",")


if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)

print('    Running logistic regression')
with dagshub.dagshub_logger(metrics_path="logs/test_metrics.csv", hparams_path="logs/pred_pipe_params.yml") as logger: 
    log_reg_classifier = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    
    # log experiment definition
    logger.log_hyperparams(model_class=type(log_reg_classifier).__name__)
    logger.log_hyperparams({'model': log_reg_classifier.get_params()})

    
    log_reg_classifier.fit(X_train, y_train)
    y_pred = log_reg_classifier.predict(X_test)

    # log reported performance
    score = accuracy_score(y_test, y_pred)
    logger.log_metrics({f'accuracy_score':score})
    
    print('    Finished modeling with accuracy score', score)

