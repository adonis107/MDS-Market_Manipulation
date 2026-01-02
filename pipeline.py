import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import preprocessing as prep
import machine_learning as ml

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

        target_idx = self.feature_names.index(self.target_col)

        # Create sequences
        all_sequences = prep.create_sequences(data_scaled, self.seq_length)
        self.y_targets = data_scaled[self.seq_length:, target_idx]
        
        min_len = min(len(all_sequences), len(self.y_targets))
        self.X_seqs = all_sequences[:min_len]
        self.y_targets = self.y_targets[:min_len]

        # Split into Train/Val/Test
        n_total = len(self.X_seqs)
        train_end = int(n_total * train_ratio)
        val_end = int(n_total * (train_ratio + val_ratio))

        self.X_train = self.X_seqs[:train_end]
        self.X_val = self.X_seqs[train_end:val_end]
        self.X_test = self.X_seqs[val_end:]

        self.y_train = self.y_targets[:train_end]
        self.y_val = self.y_targets[train_end:val_end]
        self.y_test = self.y_targets[val_end:]
        
        print(f"Data split: Train {self.X_train.shape}, Val {self.X_val.shape}, Test {self.X_test.shape}")
        return self

    def _get_dataloader(self, X, y=None, shuffle=True, return_indices=False):
        """
        Creates DataLoader.

        Args:
            X (_type_): input data.
            y (_type_, optional): target data. Defaults to None.
            shuffle (bool, optional): If True, shuffles the data. Defaults to True.
            return_indices (bool, optional): If True, returns (data, index) for PRAE gate updates. Defaults to False.
        """
        tensor_x = torch.tensor(X, dtype=torch.float32)
        indices = torch.arange(len(X))

        if y is not None:
            tensor_y = torch.tensor(y, dtype=torch.float32)
            dataset = TensorDataset(tensor_x, tensor_y)

        elif return_indices:
            dataset = TensorDataset(tensor_x, indices)

        else:
            dataset = TensorDataset(tensor_x, tensor_x) # Autoencoder target is input

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

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

        if model_type == 'pnn':
            val_loader = self._get_dataloader(self.X_val, self.y_val, shuffle=False)
        else:
            val_loader = self._get_dataloader(self.X_val, shuffle=False)

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

            train_loader = self._get_dataloader(self.X_train, self.y_train)

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
                train_tensor = torch.tensor(self.X_train, dtype=torch.float32).view(len(self.X_train), -1)
                mean_energy = torch.mean(torch.sum(train_tensor**2, dim=1)).item()

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

    def _get_latent(self, X):
        """Helper to get embeddings in batches (for Transformer)."""

        loader = self._get_dataloader(X, shuffle=False)
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

    def _compute_nll(self, X, y):
        """Computes Negative Log-Likelihood for PNN."""
        loader = self._get_dataloader(X, y, shuffle=False)
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
        nll_normal = self._compute_nll(self.X_test, self.y_test)

        # Synthetic Anomalies
        X_test_anom = self.X_test * 5.0
        nll_anom = self._compute_nll(X_test_anom, self.y_test)

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
        loader_anom = self._get_dataloader(self.X_test * 5.0, shuffle=False)
        rec_errors_anom = []
        with torch.no_grad():
            for batch in loader_anom:
                inputs = batch[0].to(self.device)
                
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
        
        # Use a subset for speed
        subset_idx = np.random.choice(len(self.X_test), size=min(1000, len(self.X_test)), replace=False)
        X_subset = self.X_test[subset_idx]

        if self.model_type == 'pnn':
            y_subset = self.y_test[subset_idx]
        
        # Baseline score (mean anomaly score of normal data)
        if self.model_type == 'transformer_ocsvm':
            z_base = self._get_latent(X_subset)
            z_base = self.latent_scaler.transform(z_base)
            base_scores = -self.detector.score_samples(z_base)
        elif self.model_type == 'pnn':
            base_scores = self._compute_nll(X_subset, y_subset)
            
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
                    perm_scores = self._compute_nll(permuted.cpu().numpy(), y_subset)

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
        spread_idx = self.feature_names.index('spread')

        X_tensor = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)

        for i in range(0, len(X_tensor), 10):
            x_orig_seq = X_tensor[i]

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