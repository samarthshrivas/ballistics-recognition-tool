import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle

class BallisticsModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def prepare_data(self, df):
        """
        Prepare data for training
        """
        # Select features for training
        features = ['Weight', 'V0', 'V100', 'E0', 'E100']
        target = 'BC'
        
        # Drop rows with missing values
        df = df.dropna(subset=features + [target])
        
        X = df[features]
        y = df[target]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
        
    def train(self, X, y):
        """
        Train the model
        """
        self.model.fit(X, y)
        return self.model
    
    def predict(self, input_data):
        """
        Make prediction on new data
        """
        # Scale input data
        input_scaled = self.scaler.transform(input_data)
        
        # Make prediction
        prediction = self.model.predict(input_scaled)
        
        return prediction
    
    def save_model(self, filepath):
        """
        Save model to file
        """
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load model from file
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls()
        instance.model = data['model']
        instance.scaler = data['scaler']
        return instance 