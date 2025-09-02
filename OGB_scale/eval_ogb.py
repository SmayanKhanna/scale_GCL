# custom eval code -> using log reg as probe. We note that PyGCL's paper doesn't have an implementation for evaluating graph-level GCL on OGB level datasets.

#sklearn
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#OGB
from ogb.graphproppred import Evaluator

def linear_probe(Z_train, y_train, Z_valid, y_valid, Z_test, y_test, dataset = 'ogbg-molhiv', random_state=0):
    evaluator = Evaluator(dataset)
    
    # tune C on valid
    best_auc, best_C = -1, None
    for C in [0.01,0.1,1,10]:
        clf = make_pipeline(StandardScaler(),
            LogisticRegression(max_iter=1000, C=C, class_weight='balanced', random_state=random_state))
        clf.fit(Z_train, y_train)
        prob = clf.predict_proba(Z_valid)[:,1]
        auc = evaluator.eval({'y_true': y_valid.reshape(-1,1),'y_pred': prob.reshape(-1,1)})['rocauc']
        if auc>best_auc: best_auc, best_C = auc, C
            
    # test using standard OGB evaluator
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=best_C, class_weight='balanced', random_state=random_state))
    clf.fit(Z_train, y_train)
    prob = clf.predict_proba(Z_test)[:,1]
    test_auc = evaluator.eval({'y_true': y_test.reshape(-1,1),'y_pred': prob.reshape(-1,1)})['rocauc']
    
    return best_auc, test_auc

def simple_linear_probe(Z_train, y_train, Z_valid, y_valid, Z_test, y_test, dataset = 'ogbg-molhiv', C = 1, random_state=0):
    # linear probe with no grid search.
    
    evaluator = Evaluator(dataset)
    
    # tune C on valid
    clf = make_pipeline(StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight='balanced', C = C, random_state=random_state))
    clf.fit(Z_train, y_train)
    prob = clf.predict_proba(Z_valid)[:,1]
    val_auc = evaluator.eval({'y_true': y_valid.reshape(-1,1),'y_pred': prob.reshape(-1,1)})['rocauc']
            
    # test using standard OGB evaluator
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=C, class_weight='balanced', random_state=random_state))
    clf.fit(Z_train, y_train)
    prob = clf.predict_proba(Z_test)[:,1]
    test_auc = evaluator.eval({'y_true': y_test.reshape(-1,1),'y_pred': prob.reshape(-1,1)})['rocauc']
    
    return val_auc, test_auc

