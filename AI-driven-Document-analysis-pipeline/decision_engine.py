# src/decision_engine.py

import torch
import pickle
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

class DecisionNNAdvanced(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=4):
        super(DecisionNNAdvanced, self).__init__()

        # CNN Layers for feature extraction
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Attention mechanism for dynamic focus (multi-head)
        self.attn_layer = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        # Fully connected layers with residual connection
        self.fc1 = nn.Linear(128, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Softmax for classification
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # CNN Layers with batch normalization
        x = x.unsqueeze(1)  # Add channel dimension for CNN
        x = self.pool(self.bn1(self.conv1(x)))
        x = self.pool(self.bn2(self.conv2(x)))

        # Attention mechanism
        x, _ = self.attn_layer(x, x, x)

        # Flattening for FC layers
        x = torch.flatten(x, 1)

        # Fully connected layers with dropout and batch normalization
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return self.softmax(x)


class DecisionEngineAdvanced:
    def __init__(self, model_path="model/decision_model_advanced.pth", scaler_path="model/scaler.pkl"):
        self.model = torch.load(model_path)  # Load pre-trained model
        self.model.eval()  # Set model to evaluation mode
        
        # Load scaler from pickle
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

    def extract_features(self, document_data):
        """
        Extract features from the document data for prediction.
        :param document_data: A dictionary containing extracted information from the document.
        :return: A normalized feature vector for the model.
        """
        features = [
            document_data.get('amount', 0),
            len(document_data.get('vendor', '')),
            self.extract_vendor_type_feature(document_data.get('vendor', '')),
            self.date_feature(document_data.get('date', ''))
        ]
        
        features = np.array(features).reshape(1, -1)  # Reshape for the model input
        scaled_features = self.scaler.transform(features)  # Apply scaling using the pre-trained scaler
        return torch.tensor(scaled_features, dtype=torch.float32)

    def extract_vendor_type_feature(self, vendor_name):
        """
        Extract a feature based on the vendor's name or category.
        :param vendor_name: Vendor name extracted from the document.
        :return: A feature representing the vendor's type.
        """
        vendor_categories = ['Amazon', 'Microsoft', 'Google', 'Apple']
        return 1 if vendor_name in vendor_categories else 0

    def date_feature(self, date_string):
        """
        Convert the date string to a numerical feature (days since epoch).
        :param date_string: Date extracted from the document.
        :return: A numeric representation of the date.
        """
        if not date_string:
            return 0
        date = datetime.strptime(date_string, "%Y-%m-%d")
        return (date - datetime(1970, 1, 1)).days  # Return days since the epoch

    def make_decision(self, document_data):
        """
        Make a decision based on the extracted features using the advanced deep learning model.
        :param document_data: Extracted document data (amount, vendor, date, etc.).
        :return: A decision label.
        """
        features = self.extract_features(document_data)
        
        with torch.no_grad():  # Disable gradient computation for inference
            output = self.model(features)
        
        decision_idx = torch.argmax(output, dim=1).item()
        decision_map = {0: 'Approved', 1: 'Under Review', 2: 'Flagged for Fraud', 3: 'Rejected'}
        return decision_map[decision_idx]
