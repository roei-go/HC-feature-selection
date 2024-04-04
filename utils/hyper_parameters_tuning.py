import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from hyperopt import hp
from hyperopt import space_eval
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, space_eval


class model_hypopt:
    def __init__(self, model, param_space, X, y, features=None, target_func='accuracy', iterations=150, num_folds=5):
        self.model = model
        self.param_space = param_space
        self.X = X
        self.y = y
        self.target_func = target_func
        self.iterations = iterations
        self.num_folds = num_folds

    def obj_fnc(self, params):
        loss = -cross_val_score(self.model(**params), self.X, self.y, cv=self.num_folds, scoring=self.target_func).mean()
        return {'loss': loss, 'status': STATUS_OK}

    def run(self):
        hypopt_trials = Trials()
        best_params = fmin(self.obj_fnc, self.param_space, algo=tpe.suggest, max_evals=self.iterations, trials=hypopt_trials)
        best_params = space_eval(self.param_space, best_params)
        return best_params
