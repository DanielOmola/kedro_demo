from sklearn.compose import ColumnTransformer

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.base import BaseEstimator

def build_pipeline_model(model:BaseEstimator, name:str): 
    """
        This method is defining a sklearn pipeline model for a machine
        learning task. 
        The pipeline model is a combination of preprocessing steps 
        and a machine learning model. 
        The preprocessing steps include text encoding using a combination
        of CountVectorizer and TfidfTransformer. 
        
        The machine learning model is specified by the model parameter,
        which can be any instance of a scikit-learn estimator class.
    """

    # build preprocessing transformer
    text_encoding_pipeline = Pipeline([
        # ('imputer', SimpleImputer(strategy="constant", fill_value='', add_indicator=False)),
        ('vect', CountVectorizer(lowercase=False, ngram_range=(1,2))),
        ('tfidf', TfidfTransformer(norm='l2', sublinear_tf=True))
    ])
    preprocessing = ColumnTransformer(
        [("text_encoding", text_encoding_pipeline, 'text')]
    )

    # build a model using preprocessing transformer
    model = Pipeline([
            ('preprocessing', preprocessing),
            (name, model),
            ])
    
    return model


def build_model_dict(model:BaseEstimator, name:str, hpar:dict={}) -> dict:
    model_dict = {
        'model': build_pipeline_model(model=model, name=name),
        'name': name,
        'hpar': hpar
    }
    return model_dict


def generate_model_list()-> list:
    
    random_state = 42
    
    nb = build_model_dict(model=MultinomialNB(), 
                          name='MultinomialNB', 
                          hpar={'alpha':[1, 0.1, 0.01,
                                          0.001, 0.0001, 0.00001],
                                'fit_prior':[True, False],
                                'class_prior':[None], }
                                )
    
    sgd = build_model_dict(model=SGDClassifier(random_state=random_state),
                           name='SGDClassifier', 
                           hpar={'alpha':[1e-03, 1e-04, 1e-05],
                                 'max_iter':[40, 50, 60],
                                  'fit_intercept': [True],
                                  'penalty':['l2']}
                                  )
    

    lr = build_model_dict(model=LogisticRegression(multi_class='multinomial', solver='lbfgs'), 
                          name='LogisticRegression', 
                          hpar={
                                'C': [0.1, 1.0, 10.0],
                                'penalty': ['l2']
                                }
                          )
    

    rf = build_model_dict(model=RandomForestClassifier(random_state=random_state),
                          name='RandomForestClassifier',
                          hpar={'n_estimators': [100, 200, 300],
                                'max_depth': [3, 5, 7],
                                'min_samples_split': [2, 5, 10]}
                          )
    
    knn = build_model_dict(model=KNeighborsClassifier(),
                          name='KNN Classifier',
                          hpar= {'n_neighbors': [3, 5, 7], 
                                 'weights': ['uniform', 'distance']}
                          )

    xgb  = build_model_dict(model=XGBClassifier(random_state=random_state),
                          name='XGBoost Classifier',
                          hpar= {
                                'n_estimators': [50, 100, 200],
                                'learning_rate': [0.1, 0.01, 0.001],
                                'max_depth': [3, 5, 7]
                                }
                          )

    adaboost   = build_model_dict(model=AdaBoostClassifier(random_state=random_state),
                          name='AdaBoost Classifier',
                          hpar= {'n_estimators': [50, 100, 200],
                                'learning_rate': [0.1, 1.0, 10.0]}
                          )
    

    
    
    # model_list = [nb, sgd, lr, rf, knn, xgb, adaboost]
    model_list = [nb, sgd, lr, rf, xgb, adaboost]
    return model_list