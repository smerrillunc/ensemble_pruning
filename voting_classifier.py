import numpy as np
import pandas as pd
from utils import generate_binary_vector
from sklearn.tree import DecisionTreeClassifier

class voting_classifier():
    def __init__(self, ensemble, num_features):
        self.ensemble = np.array(ensemble)
        self.ensemble_size = len(ensemble)
        self.num_features = num_features
        self.feature_encodings = self.get_all_feature_encodings()
        self.train_info, self.splitters, self.min_samples_leafs, self.max_depths = self.get_all_train_info()
    
    def predict_model(self, x_test, model_idx):
        return self.ensemble[model_idx].predict(x_test[:, np.where(self.feature_encodings[model_idx] == 1)[0]])
    
    def predict_proba_model(self, x_test, model_idx):
        return self.ensemble[model_idx].predict_proba(x_test[:, np.where(self.feature_encodings[model_idx] == 1)[0]])
        
    def get_model_preds(self, x_test):
        preds_shape = (x_test.shape[0], 2)
        test_preds = []

        for model_idx in range(self.ensemble_size):    
            preds = self.predict_model(x_test, model_idx)
            test_preds.append(preds)
        return np.array(test_preds)
    
    def get_model_pred_probs(self, x_test):
        test_probs = []

        for model_idx in range(self.ensemble_size):    
            probs = self.predict_proba_model(x_test, model_idx)
            test_probs.append(probs)
        return np.array(test_probs)

    def get_model_preds(self, x_test):
        preds_shape = (x_test.shape[0], 2)
        test_preds = []

        for model_idx in range(self.ensemble_size):    
            preds = self.predict_model(x_test, model_idx)
            test_preds.append(preds)
        return np.array(test_preds)

    def get_ensemble_preds(self, x_test):
        test_preds = np.zeros(shape=x_test.shape[0], dtype=np.float32)
        for model_idx in range(self.ensemble_size):   
            test_preds += self.predict_model(x_test, model_idx)
        return np.array(test_preds)
            
    def predict(self, x_test):
        ensemble_preds = self.get_ensemble_preds(x_test)
        return np.where(ensemble_preds/self.ensemble_size > 0.5, 1, 0)
    
    def predict_proba(self, x_test):
        c1_probs = self.get_ensemble_preds(x_test)/self.ensemble_size
        c0_probs = 1 - c1_probs
        return np.vstack([c0_probs, c1_probs]).T
    
    def get_all_feature_encodings(self):
        feature_encodings = []
        for model in self.ensemble:
            try:
                feature_encoding = generate_binary_vector(self.num_features, model.feature_ones, model.feature_seed)
            except:
                feature_encoding = generate_binary_vector(self.num_features, model.feature_ones, model.random_state)

            feature_encodings.append(feature_encoding)
            
        return np.array(feature_encodings)

    def get_all_train_info(self):
        train_infos = []
        max_depths = []
        splitters = []
        min_samples_leafs = []
        
        for model in self.ensemble:
            try:
                train_info = model.training_frac
            except:
                train_info = model.num_clusters
            train_infos.append(train_info)
            
            splitters.append(model.splitter)
            min_samples_leafs.append(model.min_samples_leaf)
            max_depths.append(model.max_depth)
            
        return np.array(train_infos), np.array(splitters), np.array(min_samples_leafs), np.array(max_depths)