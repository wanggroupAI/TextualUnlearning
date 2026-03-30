import numpy as np
import scipy
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def InOutTest(keep, scores, check_keep, check_scores, in_size=100000, out_size=100000,
                  fix_variance=False):
    """
    Fit a two predictive models using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    dat_in = []
    dat_out = []

    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:,j],j])
        dat_out.append(scores[~keep[:,j],j])

    in_size = min(min(map(len,dat_in)), in_size)
    out_size = min(min(map(len,dat_out)), out_size)

    dat_in = np.array([x[:in_size] for x in dat_in])
    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_in = np.median(dat_in, 1)
    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_in = np.std(dat_in)
        std_out = np.std(dat_in)
    else:
        std_in = np.std(dat_in, 1)
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in+1e-30)
        pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out+1e-30)
        score = pr_in-pr_out
        #print(score)
        #print(ans)
        #exit()
        prediction.extend(score)
        answers.extend(ans)

    return prediction, answers
def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc


def method(keep, lgts_o,lgts_u, test_keep, test_lgt_o, test_lgt_u):


    X_train = np.concatenate((lgts_o,lgts_u, lgts_o-lgts_u), axis=-1)  # shape: [85, 64, 6]

    y_train = np.where(keep, 1, -1)  # True -> 1, False -> -1

    X_test = np.concatenate((test_lgt_o, test_lgt_u, test_lgt_o- test_lgt_u), axis=-1)  # shape: [15, 64, 6]

    y_test = np.where(test_keep, 1, -1)  # True -> 1, False -> -1

    all_predictions = []
    all_labels=[]


    for i in range(keep.shape[1]):

        X_train_i = X_train[:, i, :]
        y_train_i = y_train[:, i]
        X_test_i = X_test[:, i, :]
        y_test_i = y_test[:, i]

        train_data = lgb.Dataset(X_train_i, label=y_train_i)

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 2,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbosity': -1 
        }


        model = lgb.train(params, train_data)

        predictions = model.predict(X_test_i)
        all_predictions.append(predictions)
        all_labels.append(y_test_i)

    all_predictions = np.array(all_predictions)  # shape: [64, 15]

    fpr, tpr, _ = roc_curve((np.array(all_labels)==1).flatten(), all_predictions.flatten())
    acc = np.max(1-(fpr+(1-tpr))/2)
    auc_v=auc(fpr, tpr)
    low01 = tpr[np.where(fpr<.001)[0][-1]]
    low05 = tpr[np.where(fpr<.005)[0][-1]]
    low1 = tpr[np.where(fpr<.01)[0][-1]]
    low5 = tpr[np.where(fpr<.05)[0][-1]]
    print('TULA %s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR: %.4f, TPR@0.5%%FPR: %.4f, TPR@1%%FPR: %.4f, TPR@5%%FPR: %.4f' %("", auc_v,acc, low01, low05, low1, low5))

