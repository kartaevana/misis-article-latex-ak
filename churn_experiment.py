import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import BaseEstimator, ClassifierMixin, clone

SEED = 42

class WeightedClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, balanced=False):
        self.estimator = estimator
        self.balanced = balanced

    def fit(self, X, y):
        weights = compute_sample_weight('balanced', y) if self.balanced else None
        self.estimator_ = clone(self.estimator).fit(X, y, sample_weight=weights)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

def evaluate(y, p):
    order = np.argsort(-p, kind='stable')
    total = int(y.sum())
    top = {}
    for q in (.05, .10, .20):
        k = int(np.ceil(q * len(y)))
        hits = int(y[order[:k]].sum())
        top[str(q)] = dict(k=k, hits=hits, precision=hits/k, recall=hits/total,
                           lift=(hits/k)/(total/len(y)))
    return dict(ap=average_precision_score(y,p), auc=roc_auc_score(y,p),
                brier=brier_score_loss(y,p), loss=log_loss(y,p), top=top)

def main(csv, output):
    df = pd.read_csv(csv)
    assert df.CLIENTNUM.is_unique
    leaked = [c for c in df if c.startswith('Naive_Bayes_Classifier')]
    assert len(leaked) == 2
    y = df.Attrition_Flag.map({'Existing Customer':0, 'Attrited Customer':1})
    assert y.notna().all()
    y = y.to_numpy()
    X = df.drop(columns=['CLIENTNUM','Attrition_Flag',*leaked])
    train, holdout = train_test_split(np.arange(len(y)), test_size=.4, stratify=y, random_state=SEED)
    cal, test = train_test_split(holdout, test_size=.5, stratify=y[holdout], random_state=SEED)
    test = np.random.default_rng(SEED).permutation(test)
    models = {
        'logistic': (LogisticRegression(max_iter=2000), {'C':[.05,.5,5]}),
        'forest': (RandomForestClassifier(random_state=SEED,n_jobs=1), {
            'n_estimators':[100,200], 'min_samples_leaf':[1,5], 'max_features':['sqrt',.6]}),
        'boosting': (GradientBoostingClassifier(random_state=SEED), {
            'n_estimators':[100,200], 'max_depth':[2,3], 'learning_rate':[.05,.1]}),
    }
    result = {'version':sklearn.__version__, 'sha256':hashlib.sha256(csv.read_bytes()).hexdigest(),
              'splits':{name:[len(ids),int(y[ids].sum())] for name,ids in
                        [('train',train),('cal',cal),('test',test)]},
              'features':X.shape[1], 'models':{},
              'constant':evaluate(y[test], np.full(len(test),y[train].mean()))}
    for name,(estimator,grid) in models.items():
        numeric = [SimpleImputer(strategy='median')]
        if name == 'logistic':
            numeric.append(StandardScaler())
        preprocess = ColumnTransformer([
            ('numeric',make_pipeline(*numeric),make_column_selector(dtype_include=np.number)),
            ('category',make_pipeline(SimpleImputer(strategy='constant',fill_value='(missing)'),
                         OneHotEncoder(handle_unknown='ignore',sparse_output=False)),
             make_column_selector(dtype_exclude=np.number))])
        pipeline = Pipeline([('preprocess',preprocess),('model',WeightedClassifier(estimator))])
        params = {'model__balanced':[False,True],
                  **{'model__estimator__'+key:values for key,values in grid.items()}}
        cv = GridSearchCV(pipeline,params,scoring='average_precision',
                          cv=StratifiedKFold(5,shuffle=True,random_state=SEED),
                          n_jobs=2,error_score='raise')
        cv.fit(X.iloc[train],y[train])
        variants = {'raw':evaluate(y[test],cv.predict_proba(X.iloc[test])[:,1])}
        for method in ('sigmoid','isotonic'):
            model = CalibratedClassifierCV(FrozenEstimator(cv.best_estimator_),method=method)
            model.fit(X.iloc[cal],y[cal])
            variants[method] = evaluate(y[test],model.predict_proba(X.iloc[test])[:,1])
        results = cv.cv_results_
        cv_by_weight = {
            str(weight): max(score for score, candidate in zip(results['mean_test_score'],results['params'])
                             if candidate['model__balanced'] == weight)
            for weight in (False,True)}
        result['models'][name] = {'best':cv.best_params_,'cv_ap':cv.best_score_,
                                  'cv_by_weight':cv_by_weight,'variants':variants}
        output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(name,cv.best_params_,flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    main(args.csv,args.output)
