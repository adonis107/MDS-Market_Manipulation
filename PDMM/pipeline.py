import os
import json
import joblib
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

import PDMM.preprocessing as prep
import PDMM.machine_learning as ml


class LazySequenceDataset(Dataset):
    def __init__(self, data_array, seq_length, targets=None, indices=None):
        self.data = torch.tensor(data_array, dtype=torch.float32)
        self.seq_length = seq_length
        if indices is None: self.indices = np.arange(len(self.data) - self.seq_length)
        else: self.indices = indices

        self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None
        self._shape = (len(self.indices), self.seq_length, self.data.shape[1])

    def __len__(self):
        return self._shape[0]

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_length
        
        # Create sequence view
        x = self.data[start_idx : end_idx]
        
        if self.targets is not None:
            # Target corresponds to the value at the end of the sequence
            y = self.targets[start_idx]
            return x, y
        
        # Autoencoder: target is input
        return x, x
            
    @property
    def shape(self):
        return self._shape


class AnomalyDetectionPipeline:
    def __init__(self, seq_length=25, batch_size=64, device=None, random_state=0):
        
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # State variables
        self.raw_df = None
        self.processed_df = None
        self.feature_names = []
        self.scaler = None
        self.model = None
        self.detector = None # e.g., OC-SVM
        self.model_type = None
        self.target_col = 'log_return' # For PNN

        # Data containers
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None

        self.data_scaled = None
        self.targets_scaled = None
        
        print(f"Pipeline initialized on device: {self.device}")

    def load_data(self, filepath, nrows=None):
        """
        Loads data from CSV/Parquet.
        
        Args:
            filepath (str): Path to the data file.
            nrows (int, optional): Number of rows to read. Defaults to None (read all).
        """
        print(f"Loading data from {filepath}...")

        if filepath.endswith('.csv') or filepath.endswith('.csv.gz'):
            self.raw_df = pd.read_csv(filepath, nrows=nrows)

        elif filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
            if nrows: self.raw_df = df.head(nrows)
            else: self.raw_df = df

        else:
            raise ValueError("Unsupported file format")
        
        print(f"Successfully loaded {len(self.raw_df)} rows.")
        return self

    def engineer_features(self, feature_sets=['base', 'tao', 'poutre', 'hawkes', 'ofi']):
        """
        Applies feature engineering based on selected sets.
        Options: 'base' (Basic LOB), 'tao' (Weighted Imbalance), 
                 'poutre' (Rapidity), 'hawkes' (Memory), 'ofi' (Elasticity)
        """
        print(f"Engineering features: {feature_sets}...")

        df = self.raw_df.copy()
        
        # Base extraction
        features = prep.extract_features(df, window=50)
        
        if 'tao' in feature_sets:
            features["Weighted_Imbalance_decreasing"] = prep.compute_weighted_imbalance(df, weights=[0.1, 0.1, 0.2, 0.2, 0.4], levels=5)
            features["Weighted_Imbalance_increasing"] = prep.compute_weighted_imbalance(df, weights=[0.4, 0.2, 0.2, 0.1, 0.1], levels=5)
            features["Weighted_Imbalance_constant"] = prep.compute_weighted_imbalance(df, weights=[0.2, 0.2, 0.2, 0.2, 0.2], levels=5)
        
        if 'poutre' in feature_sets:
            features = prep.compute_rapidity_event_flow_features(df, features)
            
        if 'hawkes' in feature_sets:
            features = prep.compute_hawkes_and_weighted_flow(df, data=features)

        if 'ofi' in feature_sets:
            features = prep.compute_order_flow_imbalance(df, data=features)

        # Cleanup
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features = features.fillna(0)
        
        # Clip extreme outliers
        lower = features.quantile(0.001)
        upper = features.quantile(0.999)
        features = features.clip(lower=lower, upper=upper, axis=1)
        
        self.processed_df = features
        self.feature_names = features.columns.tolist()
        print(f"Feature Engineering complete. Total features: {len(self.feature_names)}")
        return self

    def scale_and_sequence(self, method='minmax', train_ratio=0.7, val_ratio=0.15):
        """
        Scales data and creates sequences.
        method: 'minmax' (default), 'standard', 'box-cox'
        val_ratio: proportion of data to use for validation (default 0.15)
        """
        print(f"Preprocessing with method: {method}...")

        # Drop constant columns
        constant_cols = [col for col in self.processed_df.columns if self.processed_df[col].nunique() <= 1]

        # Check for zero variance (numerical constants)
        std_devs = self.processed_df.std()
        zero_var_cols = std_devs[std_devs < 1e-9].index.tolist()

        # Combine and drop columns
        cols_to_drop = list(set(constant_cols + zero_var_cols))

        if cols_to_drop:
            print(f"Dropping {len(cols_to_drop)} constant/zero-variance features: {cols_to_drop}")
            self.processed_df = self.processed_df.drop(columns=cols_to_drop)
            self.feature_names = self.processed_df.columns.tolist()

        if self.target_col not in self.feature_names:
            raise ValueError(f"Target column '{self.target_col}' was dropped because it is constant.")

        data_values = self.processed_df.values
        
        if method == 'minmax':
            self.scaler = MinMaxScaler()
            data_scaled = self.scaler.fit_transform(data_values)

        elif method == 'standard':
            self.scaler = StandardScaler()
            data_scaled = self.scaler.fit_transform(data_values)

        elif method == 'box-cox':
            self.scaler = prep.data_preprocessor()
            data_scaled = self.scaler.fit_transform(data_values)

        else:
            raise ValueError(f"Unknown scaler method: {method}")

        # Convert to float32 for efficiency
        self.data_scaled = data_scaled.astype(np.float32)

        # Prepare targets
        target_idx = self.feature_names.index(self.target_col)
        full_targets = self.data_scaled[:, target_idx]
        self.targets_scaled = np.roll(full_targets, -self.seq_length)
        self.targets_scaled[-self.seq_length:] = 0 # Invalid

        # Split into Train/Val/Test
        n_total = len(data_scaled)
        train_end = int(n_total * train_ratio)
        val_end = int(n_total * (train_ratio + val_ratio))

        # Train set
        self.X_train = LazySequenceDataset(self.data_scaled, self.seq_length, indices=np.arange(0, train_end))
        self.y_train = LazySequenceDataset(self.data_scaled, self.seq_length, targets=self.targets_scaled, indices=np.arange(0, train_end))

        # Val set
        self.X_val = LazySequenceDataset(self.data_scaled, self.seq_length, indices=np.arange(train_end, val_end))
        self.y_val = LazySequenceDataset(self.data_scaled, self.seq_length, targets=self.targets_scaled, indices=np.arange(train_end, val_end))
        
        # Test set
        self.X_test = LazySequenceDataset(self.data_scaled, self.seq_length, indices=np.arange(val_end, n_total))
        self.y_test = LazySequenceDataset(self.data_scaled, self.seq_length, targets=self.targets_scaled, indices=np.arange(val_end, n_total))

        print(f"Data split: Train {len(self.X_train)}, Val {len(self.X_val)}, Test {len(self.X_test)}")
        return self

    def _get_dataloader(self, dataset, shuffle=True, return_indices=False):
        """
        Creates DataLoader.
        """
        if isinstance(dataset, Dataset):
             if return_indices:
                 class IndexWrapper(Dataset):
                     def __init__(self, ds): self.ds = ds
                     def __len__(self): return len(self.ds)
                     def __getitem__(self, i): 
                         data = self.ds[i]
                         return data[0], i
                 return DataLoader(IndexWrapper(dataset), batch_size=self.batch_size, shuffle=shuffle)
             else:
                 return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

        tensor_x = torch.tensor(dataset, dtype=torch.float32)
        indices = torch.arange(len(dataset))

        if return_indices: ds = TensorDataset(tensor_x, indices)
        else: ds = TensorDataset(tensor_x, tensor_x) 

        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def train_model(self, model_type='transformer_ocsvm', epochs=5, lr=1e-3, nu=0.01, hidden_dim=64, lambda_reg=None, patience=5):
        """
        Trains the selected model architecture.
        model_type: 'transformer_ocsvm' (default), 'pnn', 'prae'
        """
        self.model_type = model_type
        num_feat = self.X_train.shape[2]
        
        # Validation set for early stopping
        if self.X_val is None:
            raise ValueError("Validation set is not defined. Please run scale_and_sequence with a validation split.")
        
        early_stopping = ml.EarlyStopping(patience=patience, verbose=True, path='best_model_checkpoint.pth')

        # Validation DataLoader
        val_dataset = self.y_val if model_type == 'pnn' else self.X_val
        val_loader = self._get_dataloader(val_dataset, shuffle=False)

        if model_type == 'transformer_ocsvm':
            print("Initializing Transformer Autoencoder...")
            self.model = ml.TransformerAutoencoder(
                num_features=num_feat,
                model_dim=64,
                num_heads=4,
                num_layers=2,
                representation_dim=128,
                sequence_length=self.seq_length
            ).to(self.device)
            
            # Train Autoencoder
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            train_loader = self._get_dataloader(self.X_train)
            
            self.model.train()
            print(f"Training Autoencoder (Max Epochs={epochs})...")
            for epoch in range(epochs):
                self.model.train()
                train_loss = 0
                for batch_data, _ in train_loader:
                    batch_data = batch_data.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(batch_data)
                    loss = criterion(output, batch_data)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_data, _ in val_loader:
                        batch_data = batch_data.to(self.device)
                        output = self.model(batch_data)
                        loss = criterion(output, batch_data)
                        val_loss += loss.item()

                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

                early_stopping(val_loss, self.model)
                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break

            self.model.load_state_dict(torch.load('best_model_checkpoint.pth'))
            
            # Extract Latent Representations
            print("Extracting Latent Representations for OC-SVM...")
            z_train = self._get_latent(self.X_train)
            
            # Scale Latent Space (Important for SVM convergence)
            self.latent_scaler = StandardScaler()
            z_train_scaled = self.latent_scaler.fit_transform(z_train)
            
            # Train OC-SVM
            print(f"Training One-Class SVM (nu={nu})...")
            self.detector = OneClassSVM(kernel='rbf', gamma='auto', nu=nu)
            self.detector.fit(z_train_scaled)
            
        elif model_type == 'pnn':
            print("Initializing Probabilistic Neural Network (PNN)...")

            # Input dimension
            input_dim = self.seq_length * num_feat
            self.model = ml.ProbabilisticNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim
            ).to(self.device)

            criterion = ml.SkewedGaussianNLL()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

            train_loader = self._get_dataloader(self.y_train)

            self.model.train()
            print(f"Training PNN (Max Epochs={epochs})...")
            for epoch in range(epochs):
                self.model.train()
                train_loss = 0
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    batch_x_flat = batch_x.view(batch_x.size(0), -1)
                    
                    optimizer.zero_grad()
                    mu, sigma, alpha = self.model(batch_x_flat)
                    loss = criterion(batch_y, mu, sigma, alpha)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)

                        batch_x_flat = batch_x.view(batch_x.size(0), -1)

                        mu, sigma, alpha = self.model(batch_x_flat)
                        loss = criterion(batch_y, mu, sigma, alpha)
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f} | Val: {val_loss:.6f}")

                early_stopping(val_loss, self.model)
                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break

            self.model.load_state_dict(torch.load('best_model_checkpoint.pth'))

        elif model_type == 'prae':
            print("Initializing Probabilistic Robust Autoencoder (PRAE)...")

            # Base Autoencoder
            base_ae = ml.TransformerAutoencoder(num_features=num_feat, model_dim=64, num_heads=4, num_layers=2, representation_dim=128, sequence_length=self.seq_length)

            # PRAE Wrapper
            self.model = ml.ProbabilisticRobustAutoencoder(base_autoencoder=base_ae, num_train_samples=len(self.X_train)).to(self.device)

            # Regularization parameter: lambda
            # Uses mean energy of samples
            if lambda_reg is None:
                loader_sample = self._get_dataloader(self.X_train, shuffle=False)
                sq_sum = 0
                count = 0
                for i, (bx, _) in enumerate(loader_sample):
                    if i > 50: break
                    sq_sum += torch.sum(bx**2).item()
                    count += bx.numel()
                mean_energy = sq_sum / count * (num_feat * self.seq_length)
                lambda_reg = mean_energy / (self.seq_length * num_feat) # normalized by input dim
                print(f"Auto-tuned lambda (Mean Energy Heuristic): {lambda_reg:.6f}")

            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            train_loader = self._get_dataloader(self.X_train, return_indices=True)

            self.model.train()
            print(f"Training PRAE (lambda={lambda_reg:.6f}, Max Epochs={epochs})...")

            for epoch in range(epochs):
                self.model.train()
                total_loss_epoch = 0

                for batch_x, batch_idx in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_idx = batch_idx.to(self.device)

                    optimizer.zero_grad()
                    
                    # Forward pass
                    reconstruced, z = self.model(batch_x, indices=batch_idx, training=True)

                    # Per-sample reconstruction loss (MSE), shape: (batch_size,)
                    error_per_sample = torch.mean((reconstruced - batch_x)**2, dim=[1, 2])

                    # Loss
                    # Using mean instead of sum for stability
                    loss_reconstruction = torch.mean(z * error_per_sample)
                    loss_regularization = - lambda_reg * torch.mean(z)
                    loss = loss_reconstruction + loss_regularization

                    loss.backward()
                    optimizer.step()

                    total_loss_epoch += loss.item()

                # Validation
                self.model.eval()
                val_rec_error = 0
                with torch.no_grad():
                    for batch_x, _ in val_loader:
                        batch_x = batch_x.to(self.device)
                        reconstructed, _ = self.model(batch_x, training=False)
                        err = torch.mean((reconstructed - batch_x)**2, dim=[1, 2])
                        val_rec_error += torch.mean(err).item()
                
                val_metric = val_rec_error / len(val_loader)
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss_epoch/len(train_loader):.6f} | Val MSE: {val_metric:.6f}")
                
                early_stopping(val_metric, self.model)
                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break

            self.model.load_state_dict(torch.load('best_model_checkpoint.pth'))

        if os.path.exists('best_model_checkpoint.pth'):
            os.remove('best_model_checkpoint.pth')

        return self

    def _get_latent(self, dataset):
        """Helper to get embeddings in batches (for Transformer)."""

        loader = self._get_dataloader(dataset, shuffle=False)
        self.model.eval()
        reps = []

        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(self.device)
                if self.model_type == 'prae':
                    r = self.model.ae.get_representation(inputs)
                else:
                    r = self.model.get_representation(inputs)
                reps.append(r.cpu().numpy())

        return np.concatenate(reps, axis=0)

    def evaluate_transformer_ocsvm(self, y_true=None):
        """
        Evaluate Transformer + OC-SVM model.

        Args:
            y_true (array-like, optional): True labels for evaluation. If None, synthetic anomalies are created. Defaults to None.

        Returns:
            tuple: A tuple containing true labels, anomaly scores, and predictions.
        """

        # Normal Test Data Representations
        z_test_normal = self._get_latent(self.X_test)
        z_test_normal = self.latent_scaler.transform(z_test_normal)

        # Synthetic Anomalies
        z_test_anom = z_test_normal * 5.0

        if y_true is None:
            # Create synthetic test set
            X_eval = np.concatenate([z_test_normal, z_test_anom], axis=0)
            y_eval = np.concatenate([np.zeros(len(z_test_normal)), np.ones(len(z_test_anom))])
        else:
            X_eval = z_test_normal
            y_eval = y_true
        
        # Score
        scores = -self.detector.score_samples(X_eval) # Higher = more anomalous
        preds = self.detector.predict(X_eval)
        preds = np.where(preds == -1, 1, 0)

        return y_eval, scores, preds

    def _compute_nll(self, dataset):
        """Computes Negative Log-Likelihood for PNN."""
        loader = self._get_dataloader(dataset, shuffle=False)
        self.model.eval()
        nlls = []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                batch_x_flat = batch_x.view(batch_x.size(0), -1)

                mu, sigma, alpha = self.model(batch_x_flat)

                # Compute NLL per sample
                y_true = batch_y.view_as(mu)
                z = (y_true - mu) / sigma

                # PDF
                phi = (1.0 / np.sqrt(2 * np.pi)) * torch.exp(-0.5 * z**2)
                Phi = 0.5 * (1 + torch.erf(alpha * z / np.sqrt(2)))
                pdf = (2.0 / sigma) * phi * Phi

                log_pdf = -torch.log(pdf + 1e-9)
                nlls.append(log_pdf.cpu().numpy().flatten())

        return np.concatenate(nlls)

    def evaluate_pnn(self, y_true=None):
        """_summary_

        Args:
            y_true (_type_, optional): _description_. Defaults to None.
        """
        nll_normal = self._compute_nll(self.y_test)

        loader = self._get_dataloader(self.y_test, shuffle=False)
        nlls_anom = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device) * 5.0 # Scale up for anomalies
                batch_y = batch_y.to(self.device)
                batch_x_flat = batch_x.view(batch_x.size(0), -1)

                mu, sigma, alpha = self.model(batch_x_flat)

                y_true_batch = batch_y.view_as(mu)
                z = (y_true_batch - mu) / sigma

                phi = (1.0 / np.sqrt(2 * np.pi)) * torch.exp(-0.5 * z**2)
                Phi = 0.5 * (1 + torch.erf(alpha * z / np.sqrt(2)))
                pdf = (2.0 / sigma) * phi * Phi
                log_pdf = -torch.log(pdf + 1e-9)
                nlls_anom.append(log_pdf.cpu().numpy().flatten())
        nll_anom = np.concatenate(nlls_anom)
        
        if y_true is None:
            scores = np.concatenate([nll_normal, nll_anom])
            y_eval = np.concatenate([np.zeros(len(nll_normal)), np.ones(len(nll_anom))])
        else:
            scores = nll_normal
            y_eval = y_true
        
        threshold = np.mean(nll_normal) + 2 * np.std(nll_normal)
        preds = (scores > threshold).astype(int)

        return y_eval, scores, preds

    def evaluate_prae(self, y_true=None):
        """_summary_

        Args:
            y_true (_type_, optional): _description_. Defaults to None.
        """
        # Evaluate on test set (reconstruction error)
        loader = self._get_dataloader(self.X_test, shuffle=False)
        self.model.eval()
        rec_errors = []

        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(self.device)
                
                # Disable gates using training=False
                reconstructed, _ = self.model(inputs, training=False)
                
                # MSE per sample
                err = torch.mean((reconstructed - inputs)**2, dim=[1, 2])
                rec_errors.append(err.cpu().numpy())

        scores_test = np.concatenate(rec_errors)

        # Synthetic Anomalies if no true labels
        rec_errors_anom = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(self.device) * 5.0 # Scale up for anomalies
                
                reconstructed, _ = self.model(inputs, training=False)
                
                err = torch.mean((reconstructed - inputs)**2, dim=[1, 2])
                rec_errors_anom.append(err.cpu().numpy())

        scores_anom = np.concatenate(rec_errors_anom)

        if y_true is None:
            scores = np.concatenate([scores_test, scores_anom])
            y_eval = np.concatenate([np.zeros(len(scores_test)), np.ones(len(scores_anom))])
        else:
            scores = scores_test
            y_eval = y_true

        # Threshold
        threshold = np.mean(scores_test) + 3 * np.std(scores_test)
        preds = (scores > threshold).astype(int)

        return y_eval, scores, preds

    def evaluate(self, y_true=None):
        """
        Evaluates the model.
        For unsupervised Autoencoder: create synthetic anomalies (scale latent).
        For PNN: create synthetic anomalies (scale input) and check NLL.
        """
        print("Evaluating model...")
        
        # Transformer + OC-SVM Evaluation
        if self.model_type == 'transformer_ocsvm':
            y_eval, scores, preds = self.evaluate_transformer_ocsvm(y_true)

        elif self.model_type == 'pnn':
            y_eval, scores, preds = self.evaluate_pnn(y_true)

        elif self.model_type == 'prae':
            y_eval, scores, preds = self.evaluate_prae(y_true)

        else:
            raise ValueError("Model not implemented or unknown model type.")

        # Metrics
        results = {
            "AUROC": roc_auc_score(y_eval, scores),
            "AUPRC": average_precision_score(y_eval, scores),
            "F4_Score": fbeta_score(y_eval, preds, beta=4)
        }
        cm = confusion_matrix(y_eval, preds)
        
        return results, cm

    def get_feature_importance(self, n_repeats=3):
        """
        Calculates permutation importance on the Test set (Normal data).
        """
        print("Calculating Feature Importance (Permutation)...")

        self.model.eval()
        
        target_dataset = self.y_test if self.model_type == 'pnn' else self.X_test

        # Subset random indices
        all_indices = np.arange(len(target_dataset))
        subset_idx = np.random.choice(all_indices, size=min(1000, len(all_indices)), replace=False)

        X_subset_list = []
        y_subset_list = []

        for i in subset_idx:
            data_tuple = target_dataset[i]
            x = data_tuple[0]
            X_subset_list.append(x.numpy())
            if len(data_tuple) > 1:
                y_subset_list.append(data_tuple[1].item())

        X_subset = np.stack(X_subset_list)
        y_subset = np.stack(y_subset_list) if y_subset_list else None
        
        # Baseline score (mean anomaly score of normal data)
        if self.model_type == 'transformer_ocsvm':
            X_tensor = torch.tensor(X_subset, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                z_base = self.model.get_representation(X_tensor).cpu().numpy()
            z_base = self.latent_scaler.transform(z_base)
            base_scores = -self.detector.score_samples(z_base)

        elif self.model_type == 'pnn':
            ds_base = TensorDataset(torch.tensor(X_subset, dtype=torch.float32), torch.tensor(y_subset, dtype=torch.float32))
            base_scores = self._compute_nll(ds_base)
        
        elif self.model_type == 'prae':
            X_tensor = torch.tensor(X_subset, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                reconstructed, _ = self.model(X_tensor, training=False)
                err = torch.mean((reconstructed - X_tensor)**2, dim=[1, 2])
                base_scores = err.cpu().numpy()
            
        base_mean = np.mean(base_scores)
        importances = []
        tensor_subset = torch.tensor(X_subset, dtype=torch.float32).to(self.device)

        for i, name in enumerate(self.feature_names):
            diffs = []
            for _ in range(n_repeats):
                # Permute feature i
                permuted = tensor_subset.clone()
                idx = torch.randperm(permuted.size(0))
                permuted[:, :, i] = permuted[idx, :, i]
                
                if self.model_type == 'transformer_ocsvm':
                    with torch.no_grad():
                        z_perm = self.model.get_representation(permuted).cpu().numpy()
                    z_perm = self.latent_scaler.transform(z_perm)
                    perm_scores = -self.detector.score_samples(z_perm)

                elif self.model_type == 'pnn':
                    ds_perm = TensorDataset(permuted, torch.tensor(y_subset))
                    perm_scores = self._compute_nll(ds_perm)

                elif self.model_type == 'prae':
                    with torch.no_grad():
                        reconstructed, _ = self.model(permuted, training=False)
                        err = torch.mean((reconstructed - permuted)**2, dim=[1, 2])
                        perm_scores = err.cpu().numpy()

                # Impact: How much did the anomaly score deviate from baseline?
                # We expect shuffling a key feature to make normal data look anomalous
                diff = np.mean(np.abs(perm_scores - base_scores))
                diffs.append(diff)
            
            importances.append(np.mean(diffs))
            
        imp_df = pd.DataFrame({'Feature': self.feature_names, 'Importance': importances})
        imp_df = imp_df.sort_values(by='Importance', ascending=False)
        
        # Normalize
        imp_df['Importance'] = (imp_df['Importance'] / imp_df['Importance'].max()) * 100
        
        return imp_df
    
    def detect_spoofing(self, Q_spoof=50_000, delta_ticks=5, maker_fee=0.0, taker_fee=0.0005):
        """
        Scans the test set for spoofing opportunities using the trained PNN.
        Computes Delta C (Expected Gain) for a hypothetical spoof order.

        Args:
            Q_spoof (float, optional): Size of the hypothetical spoof order. Defaults to 50_000.
            delta_ticks (int, optional): Distance from best quote to place spoof order. Defaults to 5.
            maker_fee (float, optional): Maker fee rate. Defaults to 0.0.
            taker_fee (float, optional): Taker fee rate. Defaults to 0.0005.
        """
        print(f"Scanning for spoofing (Q={Q_spoof}, dist={delta_ticks} ticks)...")

        # Setup
        self.model.eval()
        fees = {'maker': maker_fee, 'taker': taker_fee}

        gains = []
        indices = []

        hawkes_indices = [i for i, c in enumerate(self.feature_names) if 'Hawkes_L' in c]
        spread_idx = self.feature_names.index('spread') if 'spread' in self.feature_names else 0

        sample_indices = range(0, len(self.X_test), 10)

        for i in sample_indices:
            x_orig_seq, _ = self.X_test[i]
            x_orig_seq = x_orig_seq.to(self.device)

            # Create spoofed sequence
            x_spoof_seq = x_orig_seq.clone()
            x_spoof_seq[-1, hawkes_indices] += 1.0

            # Flatten for model input
            x_orig_flat = x_orig_seq.view(1, -1)
            x_spoof_flat = x_spoof_seq.view(1, -1)

            # Raw spread for cost calculation
            # Force float64 (double) precision to avoid overflow during inverse Box-Cox
            vector_f64 = x_orig_seq[-1].cpu().double().numpy().reshape(1, -1)
            raw_spread = self.scaler.inverse_transform(vector_f64)[0, spread_idx]

            # Distance in price units
            tick_size = 0.01
            delta_price = delta_ticks * tick_size

            # Compute Gain
            q_genuine = 100

            gain = ml.compute_spoofing_gain(
                self.model,
                x_orig_flat,
                x_spoof_flat,
                spread=raw_spread,
                delta_a=0, # Genuine order at best ask
                delta_b=delta_price, # Spoof order deep in the book
                Q=Q_spoof,
                q=q_genuine,
                fees=fees,
                side='ask' # Assuming we want to sell
            )

            if gain > 0:
                gains.append(gain)
                indices.append(i)

        print(f"Found {len(indices)} potential spoofing opportunities.")
        return pd.DataFrame({'Index': indices, 'Expected_Gain': gains})

    def predict(self, dataset):
        """
        Generates anomaly scores for the input dataset.
        Higher score = More anomalous.
        """
        self.model.eval()

        if isinstance(dataset, torch.Tensor):
            loader = DataLoader(TensorDataset(dataset), batch_size=self.batch_size, shuffle=False)
        elif isinstance(dataset, DataLoader):
            loader = dataset
        else:
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            
        scores = []

        with torch.no_grad():
            for batch in loader:
                # Unpack batch (TensorDataset usually returns a tuple)
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch

                x = x.to(self.device)
                
                # Transformer + OCSVM
                if self.model_type == 'transformer_ocsvm':
                    z = self.model.get_representation(x)
                    z_np = z.cpu().numpy()

                    # OCSVM decision_function: Positive = Inlier, Negative = Outlier
                    # We negate it so Higher = More Anomalous
                    if self.detector is not None:
                        batch_scores = -self.detector.decision_function(z_np)
                        scores.append(batch_scores)
                    else:
                        raise ValueError("OC-SVM detector not found in the pipeline.")

                # Probabilistic Robust Autoencoder (PRAE)
                elif self.model_type == 'prae':
                    # PRAE Score: Reconstruction Error (MSE)
                    reconstructed, _ = self.model(x, training=False)
                    
                    # Calculate MSE per sequence
                    # Shape: (Batch, Seq, Feat) -> (Batch, )
                    batch_mse = torch.mean((x - reconstructed) ** 2, dim=[1, 2])
                    scores.append(batch_mse.cpu().numpy())

                # Probabilistic Neural Network (PNN)
                elif self.model_type == 'pnn':
                    # PNN Score: Negative Log Likelihood (NLL)
                    mu, sigma, alpha = self.model(x)
                    
                    y_true = x.view_as(mu) 
                    
                    z_score = (y_true - mu) / sigma

                    # Phi (CDF) and phi (PDF) functions
                    phi = (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * z_score**2)
                    Phi = 0.5 * (1 + torch.erf(alpha * z_score / math.sqrt(2)))
                    pdf = (2.0 / sigma) * phi * Phi

                    # NLL = -log(pdf)
                    nll_per_feature = -torch.log(pdf + 1e-10)
                    
                    # Mean NLL per sample
                    batch_nll = torch.mean(nll_per_feature, dim=1) 
                    if len(batch_nll.shape) > 1:
                        batch_nll = torch.mean(batch_nll, dim=1)
                        
                    scores.append(batch_nll.cpu().numpy())

        return np.concatenate(scores)
             

def load_model(config_path, test_df_features, feature_names):
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    base_path = config_path.replace('_config.json', '')
    model_type = config['model_type']
    seq_length = config['seq_length']

    # Initialize pipeline
    pipeline = AnomalyDetectionPipeline(seq_length=seq_length)
    pipeline.feature_names = feature_names
    pipeline.model_type = model_type

    # Load scaler and transform data
    pipeline.scaler = joblib.load(f"{base_path}_scaler.pkl")
    try:
        X_test_values = test_df_features[config['feature_names']].values
    except KeyError:
        common_feats = [c for c in config['feature_names'] if c in test_df_features.columns]
        X_test_values = test_df_features[common_feats].values
    
    X_test_scaled = pipeline.scaler.transform(X_test_values)

    # Prepare sequences
    X_seqs = prep.create_sequences(X_test_scaled, seq_length)
    X_tensor = torch.tensor(X_seqs, dtype=torch.float32)
    pipeline.X_test = TensorDataset(X_tensor, X_tensor)

    target_col_idx = config['feature_names'].index('log_return') if 'log_return' in config['feature_names'] else -1
    if target_col_idx != -1:
        y_targets = X_test_scaled[seq_length:, target_col_idx]
        y_targets = y_targets[:len(X_seqs)]
        y_tensor = torch.tensor(y_targets, dtype=torch.float32)

        pipeline.y_test = TensorDataset(X_tensor, y_tensor)
    else: 
        pipeline.y_test = None

    # Initialize model architecture
    input_dim = config['input_dim']

    if model_type == 'transformer_ocsvm':
        pipeline.model = ml.TransformerAutoencoder(num_features=input_dim, model_dim=64, num_heads=4, num_layers=2, representation_dim=128, sequence_length=seq_length)
        pipeline.detector = joblib.load(f"{base_path}_ocsvm_detector.pkl")
        pipeline.latent_scaler = joblib.load(f"{base_path}_latent_scaler.pkl")
    
    elif model_type == 'prae':
        base_ae = ml.TransformerAutoencoder(num_features=input_dim, model_dim=64, num_heads=4, num_layers=2, representation_dim=128, sequence_length=seq_length)
        pipeline.model = ml.ProbabilisticRobustAutoencoder(base_ae, num_train_samples=1) # mu shape = (1,), not needed for inference

    elif model_type == 'pnn':
        pipeline.model = ml.ProbabilisticNN(input_dim=seq_length * input_dim, hidden_dim=64)

    # Load model weights
    pipeline.model.to(pipeline.device)
    state_dict = torch.load(f"{base_path}_weights.pth", map_location=pipeline.device)

    if model_type == 'prae':
        if 'mu' in state_dict:
            del state_dict['mu']

    pipeline.model.load_state_dict(state_dict, strict=False)

    return pipeline, config


def evaluate_model(pipeline, config):
    model_type = config['model_type']

    # Evaluate model
    results, cm = pipeline.evaluate()

    if model_type == 'transformer_ocsvm':
        y_eval, scores, _ = pipeline.evaluate_transformer_ocsvm()
    elif model_type == 'prae':
        y_eval, scores, _ = pipeline.evaluate_prae()
    elif model_type == 'pnn':
        y_eval, scores, _ = pipeline.evaluate_pnn()
    
    return results, y_eval, scores, cm


def sequential_training_pipeline(data_dir, num_days=24, first_hour_minutes=60, train_block_minutes=5, 
                                 val_block_minutes=5, model_type='transformer_ocsvm', 
                                 feature_sets=['base', 'tao', 'poutre', 'hawkes', 'ofi'],
                                 epochs=5, lr=1e-3, nu=0.01, hidden_dim=64, lambda_reg=None, 
                                 patience=3, seq_length=25, batch_size=64):
    """
    Sequential training over multiple days with time-based splits.
    
    Args:
        data_dir: Directory containing daily CSV/Parquet files (named chronologically).
        num_days: Number of days to train on (default 24).
        first_hour_minutes: Minutes in first hour to use for train/val (default 60).
        train_block_minutes: Minutes per training block (default 5).
        val_block_minutes: Minutes per validation block (default 5).
        model_type: 'transformer_ocsvm', 'pnn', or 'prae'.
        feature_sets: List of feature engineering sets to apply.
        epochs: Max epochs per training block.
        lr: Learning rate.
        nu: OC-SVM parameter (for transformer_ocsvm).
        hidden_dim: Hidden dimension (for PNN).
        lambda_reg: Regularization parameter (for PRAE).
        patience: Early stopping patience.
        seq_length: Sequence length for models.
        batch_size: Batch size.
    
    Returns:
        dict: Results containing metrics for each day and final evaluation.
    """
    
    # Get sorted list of files
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(('.csv', '.csv.gz', '.parquet'))])
    if len(files) < num_days + 1:
        raise ValueError(f"Need at least {num_days + 1} files, found {len(files)}")
    
    results = {
        'daily_metrics': [],
        'final_day_morning_metrics': None,
        'final_day_rest_metrics': None
    }
    
    # Initialize pipeline
    pipeline = AnomalyDetectionPipeline(seq_length=seq_length, batch_size=batch_size)
    
    print(f"Starting sequential training on {num_days} days...")
    print(f"Train block: {train_block_minutes} min, Val block: {val_block_minutes} min")
    print("="*80)
    
    # Train on days 1 to num_days
    for day_idx in range(num_days):
        day_file = os.path.join(data_dir, files[day_idx])
        print(f"Day {day_idx + 1}/{num_days}: {files[day_idx]}")
        print("-"*80)
        
        # Load data
        pipeline.load_data(day_file)
        df = pipeline.raw_df.copy()
        
        # Ensure time column exists (assumes 'xltime' or 'timestamp')
        time_col = 'xltime' if 'xltime' in df.columns else 'timestamp'
        if time_col not in df.columns:
            raise ValueError(f"Time column not found. Available columns: {df.columns.tolist()}")
        
        df = df.sort_values(time_col).reset_index(drop=True)

        time_factor = 1.0 / (24.0 * 60.0)
        open_fraction = 8.0 / 24.0

        base_date = np.floor(df[time_col].iloc[0])
        official_open = base_date + open_fraction

        pre_market_count = len(df[df[time_col] < official_open])
        
        if pre_market_count > 0:
            print(f"Filtered {pre_market_count} pre-market samples.")
            df = df[df[time_col] >= official_open].reset_index(drop=True)
        
        start_time = official_open

        # Extract first hour
        start_time = df[time_col].iloc[0]
        first_hour_mask = df[time_col] < start_time + (first_hour_minutes * time_factor)
        df_first_hour = df[first_hour_mask].copy()
        df_rest_of_day = df[~first_hour_mask].copy()
        
        print(f"First hour samples: {len(df_first_hour)}, Rest of day: {len(df_rest_of_day)}")
        
        # Split first hour into time blocks
        block_duration = (train_block_minutes + val_block_minutes) * time_factor
        num_blocks = int(first_hour_minutes / (train_block_minutes + val_block_minutes))
        
        print(f"Training on {num_blocks} time blocks...")
        
        for block_idx in range(num_blocks):
            block_start_time = start_time + (block_idx * block_duration)
            train_end_time = block_start_time + (train_block_minutes * time_factor)
            val_end_time = train_end_time + (val_block_minutes * time_factor)
            
            # Create masks
            train_mask = (df_first_hour[time_col] >= block_start_time) & \
                        (df_first_hour[time_col] < train_end_time)
            val_mask = (df_first_hour[time_col] >= train_end_time) & \
                      (df_first_hour[time_col] < val_end_time)
            
            df_train_block = df_first_hour[train_mask].copy()
            df_val_block = df_first_hour[val_mask].copy()
            
            if len(df_train_block) < seq_length * 2 or len(df_val_block) < seq_length:
                print(f"Block {block_idx + 1}: Insufficient data, skipping...")
                continue
            
            print(f"Block {block_idx + 1}/{num_blocks}: Train={len(df_train_block)}, Val={len(df_val_block)}")
            
            # Feature engineering on train block
            pipeline.raw_df = df_train_block
            pipeline.engineer_features(feature_sets)

            if time_col in pipeline.processed_df.columns:
                pipeline.processed_df = pipeline.processed_df.drop(columns=[time_col])
                pipeline.feature_names = pipeline.processed_df.columns.tolist()
        
            train_features = pipeline.processed_df.copy()
            
            # Feature engineering on val block
            pipeline.raw_df = df_val_block
            pipeline.engineer_features(feature_sets)

            if time_col in pipeline.processed_df.columns:
                pipeline.processed_df = pipeline.processed_df.drop(columns=[time_col])
                pipeline.feature_names = pipeline.processed_df.columns.tolist()

            val_features = pipeline.processed_df.copy()
            
            # Scale data
            if block_idx == 0 and day_idx == 0:
                # First block: fit scaler
                scaler = StandardScaler()
                train_scaled = scaler.fit_transform(train_features.values)
                pipeline.scaler = scaler
            else:
                # Subsequent blocks: use existing scaler
                train_scaled = pipeline.scaler.transform(train_features.values)
            
            val_scaled = pipeline.scaler.transform(val_features.values)
            
            # Create sequences
            pipeline.data_scaled = train_scaled.astype(np.float32)
            pipeline.targets_scaled = np.roll(train_scaled[:, pipeline.feature_names.index('log_return')], -seq_length)
            pipeline.targets_scaled[-seq_length:] = 0
            
            pipeline.X_train = LazySequenceDataset(train_scaled, seq_length)
            pipeline.y_train = LazySequenceDataset(train_scaled, seq_length, targets=pipeline.targets_scaled)
            
            pipeline.X_val = LazySequenceDataset(val_scaled, seq_length)
            val_targets = np.roll(val_scaled[:, pipeline.feature_names.index('log_return')], -seq_length)
            val_targets[-seq_length:] = 0
            pipeline.y_val = LazySequenceDataset(val_scaled, seq_length, targets=val_targets)
            
            # Train model (accumulate training on first day, then incrementally)
            if block_idx == 0 and day_idx == 0:
                print(f"Initializing {model_type} model...")
                pipeline.train_model(model_type=model_type, epochs=epochs, lr=lr, nu=nu, 
                                   hidden_dim=hidden_dim, lambda_reg=lambda_reg, patience=patience)
            else:
                print(f"Incremental training...")
                # Incremental training (fewer epochs)
                pipeline.train_model(model_type=model_type, epochs=max(2, epochs // 2), lr=lr * 0.5, 
                                   nu=nu, hidden_dim=hidden_dim, lambda_reg=lambda_reg, patience=patience)
        
        # Evaluate on rest of day N
        print(f"\nEvaluating on rest of Day {day_idx + 1}...")
        
        if len(df_rest_of_day) > seq_length:
            pipeline.raw_df = df_rest_of_day
            pipeline.engineer_features(feature_sets)

            if time_col in pipeline.processed_df.columns:
                pipeline.processed_df = pipeline.processed_df.drop(columns=[time_col])
                pipeline.feature_names = pipeline.processed_df.columns.tolist()

            test_features = pipeline.processed_df.copy()
            test_scaled = pipeline.scaler.transform(test_features.values)
            
            pipeline.X_test = LazySequenceDataset(test_scaled, seq_length)
            test_targets = np.roll(test_scaled[:, pipeline.feature_names.index('log_return')], -seq_length)
            test_targets[-seq_length:] = 0
            pipeline.y_test = LazySequenceDataset(test_scaled, seq_length, targets=test_targets)
            
            # Evaluate
            day_results, cm = pipeline.evaluate()
            results['daily_metrics'].append({
                'day': day_idx + 1,
                'file': files[day_idx],
                'test_samples': len(pipeline.X_test),
                **day_results
            })
            
            print(f"Day {day_idx + 1} Results: AUROC={day_results['AUROC']:.4f}, "
                  f"AUPRC={day_results['AUPRC']:.4f}, F4={day_results['F4_Score']:.4f}")
        else:
            print(f"Not enough data in rest of day for evaluation.")
    
    # Evaluate on Final Day (split into morning and rest)
    print(f"\n{'='*80}")
    print(f"Final Day (Test Day): {files[num_days]}")
    print("-"*80)
    
    final_day_file = os.path.join(data_dir, files[num_days])
    pipeline.load_data(final_day_file)
    df_final_day = pipeline.raw_df.copy()
    df_final_day = df_final_day.sort_values(time_col).reset_index(drop=True)
    
    base_date = np.floor(df_final_day[time_col].iloc[0])
    open_fraction = 8.0 / 24.0
    official_open = base_date + open_fraction

    pre_market_count = len(df_final_day[df_final_day[time_col] < official_open])
    if pre_market_count > 0:
        print(f"Filtered {pre_market_count} pre-market samples.")
        df_final_day = df_final_day[df_final_day[time_col] >= official_open].reset_index(drop=True)

    start_time_final_day = official_open
    morning_mask = df_final_day[time_col] < start_time_final_day + (first_hour_minutes * time_factor)
    
    df_final_day_morning = df_final_day[morning_mask].copy()
    df_final_day_rest = df_final_day[~morning_mask].copy()
    
    print(f"Morning samples: {len(df_final_day_morning)}, Rest of day: {len(df_final_day_rest)}")
    
    # Evaluate on morning of Final Day
    if len(df_final_day_morning) > seq_length:
        print("\nEvaluating on Final Day Morning...")
        pipeline.raw_df = df_final_day_morning
        pipeline.engineer_features(feature_sets)

        if time_col in pipeline.processed_df.columns:
            pipeline.processed_df = pipeline.processed_df.drop(columns=[time_col])
            pipeline.feature_names = pipeline.processed_df.columns.tolist()

        test_features = pipeline.processed_df.copy()
        test_scaled = pipeline.scaler.transform(test_features.values)
        
        pipeline.X_test = LazySequenceDataset(test_scaled, seq_length)
        test_targets = np.roll(test_scaled[:, pipeline.feature_names.index('log_return')], -seq_length)
        test_targets[-seq_length:] = 0
        pipeline.y_test = LazySequenceDataset(test_scaled, seq_length, targets=test_targets)
        
        morning_results, cm_morning = pipeline.evaluate()
        results['final_day_morning_metrics'] = {
            'test_samples': len(pipeline.X_test),
            **morning_results
        }
        
        print(f"Final Day Morning: AUROC={morning_results['AUROC']:.4f}, "
              f"AUPRC={morning_results['AUPRC']:.4f}, F4={morning_results['F4_Score']:.4f}")
    
    # Evaluate on rest of Final Day
    if len(df_final_day_rest) > seq_length:
        print("\nEvaluating on Final Day Rest...")
        pipeline.raw_df = df_final_day_rest
        pipeline.engineer_features(feature_sets)

        if time_col in pipeline.processed_df.columns:
            pipeline.processed_df = pipeline.processed_df.drop(columns=[time_col])
            pipeline.feature_names = pipeline.processed_df.columns.tolist()
            
        test_features = pipeline.processed_df.copy()
        test_scaled = pipeline.scaler.transform(test_features.values)
        
        pipeline.X_test = LazySequenceDataset(test_scaled, seq_length)
        test_targets = np.roll(test_scaled[:, pipeline.feature_names.index('log_return')], -seq_length)
        test_targets[-seq_length:] = 0
        pipeline.y_test = LazySequenceDataset(test_scaled, seq_length, targets=test_targets)
        
        rest_results, cm_rest = pipeline.evaluate()
        results['final_day_rest_metrics'] = {
            'test_samples': len(pipeline.X_test),
            **rest_results
        }
        
        print(f"Final Day Rest: AUROC={rest_results['AUROC']:.4f}, "
              f"AUPRC={rest_results['AUPRC']:.4f}, F4={rest_results['F4_Score']:.4f}")
    
    print(f"\n{'='*80}")
    print("Sequential Training Complete!")
    
    return results, pipeline


class FlattenWrapper(torch.nn.Module):
    """
    Wraps PNN to accept (Batch, Seq, Feat) input by flattening it internally.
    Necessary for Integrated Gradients to work on the sequence structure.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # x shape: (Batch, Seq, Feat)
        # Flatten to: (Batch, Seq * Feat)
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)
        return self.model(x_flat)
    

