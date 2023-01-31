import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from hyperopt import hp
from hyperopt import space_eval
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, space_eval

class model_hypopt:
    
    def __init__(self, model, param_space, X_train, y_train, features=None, target_func='accuracy', iterations=150, num_folds=5):
        self.model = model
        self.param_space = param_space
        self.x_train = X_train
        self.y_train = y_train
        self.target_func = target_func
        self.iterations = iterations
        self.num_folds = num_folds
        
            
    def obj_fnc(self, params):
        loss = -cross_val_score(self.model(**params), self.x_train, self.y_train, cv=self.num_folds, scoring=self.target_func).mean()
        return {'loss': loss, 'status': STATUS_OK}
            
    def run(self):
        hypopt_trials = Trials()
        best_params = fmin(self.obj_fnc, self.param_space, algo=tpe.suggest,max_evals=self.iterations, trials= hypopt_trials)
        best_params = space_eval(self.param_space, best_params)
        return best_params
    
    
def hyper_opt_train_eval(model,search_space,train_df,test_df,features,target_metric,hyper_opt_iterations=100,target_col_name='target'):
    y_train = train_df[target_col_name]
    hyper_opt = model_hypopt(model=model,param_space=search_space,
                             data=train_df, features=features, y_train=y_train, target_func=target_metric,iterations=hyper_opt_iterations)
    best_params = hyper_opt.run()
    print(f'Best hyper-parameters found are: {best_params}')
    
    # train model with the best parameters on the entire train set
    trained_model = model(**best_params)
    model_params_dict = trained_model.get_params()
    trained_model.fit(train_df[features], train_df[target_col_name])
    
    y_pred = trained_model.predict(test_df[features])
    metrics_dict = get_classification_metrics(test_df[target_col_name], y_pred)
    
    disp = ConfusionMatrixDisplay.from_predictions(test_df[target_col_name], y_pred,normalize='true')
    conf_mat_fname = 'results/conf_mat.png'
    disp.figure_.set_size_inches(9,9)
    disp.figure_.savefig(conf_mat_fname)
    plt.show()
    
    return trained_model,model_params_dict, metrics_dict

    