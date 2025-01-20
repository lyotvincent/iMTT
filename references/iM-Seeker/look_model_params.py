import pickle

brf = pickle.load(open("./pickle_model_classification.pkl", 'rb'))
print(brf.get_params())
'''
{'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': 32, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 200, 'n_jobs': None, 'oob_score': False, 'random_state': 0, 'replacement': False, 'sampling_strategy': 'all', 'verbose': 0, 'warm_start': False}
'''