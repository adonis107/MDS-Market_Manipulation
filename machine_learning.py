import numpy as np
import pandas as pd
import math
from scipy.stats import skewnorm
from scipy.special import erf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class BottleneckTransformerEncoder(nn.Module):
    """
    Transformer Encoder with Bottleneck Representation
    Takes (Batch, Sequence_length, Number_features) as input and outputs (Batch, Representation_dimension)
    """
    def __init__(self, num_features, model_dim, num_heads, num_layers, representation_dim, sequence_length):
        super(BottleneckTransformerEncoder, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(num_features, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)

        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.flatten = nn.Flatten()
        self.bottleneck = nn.Linear(sequence_length * model_dim, representation_dim)

    def forward(self, x):
        x = self.embedding(x) * np.sqrt(self.model_dim)

        x_pos = x.permute(1, 0, 2) # (Seq_len, Batch, Features)
        x_pos = self.pos_encoder(x_pos)
        x = x_pos.permute(1, 0, 2) # (Batch, Seq_len, Features)

        encoded_seq = self.transformer_encoder(x)

        flattened = self.flatten(encoded_seq)
        representation = self.bottleneck(flattened)

        return representation
    
    
class BottleneckTransformerDecoder(nn.Module):
    """
    Transformer Decoder with Bottleneck Representation
    Takes (Batch, Representation_dimension) as input and outputs (Batch, Sequence_length, Number_features)
    """
    def __init__(self, num_features, model_dim, num_heads, num_layers, representation_dim, sequence_length):
        super(BottleneckTransformerDecoder, self).__init__()
        self.model_dim = model_dim
        self.sequence_length = sequence_length

        self.expand = nn.Linear(representation_dim, sequence_length * model_dim)

        decoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layers, num_layers=num_layers)

        self.output_layer = nn.Linear(model_dim, num_features)

    def forward(self, x):
        expanded = self.expand(x)
        expanded = expanded.view(-1, self.sequence_length, self.model_dim)

        decoded_seq = self.transformer_decoder(expanded)

        output = self.output_layer(decoded_seq)

        return output
    

class TransformerAutoencoder(nn.Module):
    def __init__(self, num_features, model_dim, num_heads, num_layers, representation_dim, sequence_length):
        super(TransformerAutoencoder, self).__init__()
        self.encoder = BottleneckTransformerEncoder(num_features, model_dim, num_heads, num_layers, representation_dim, sequence_length)
        self.decoder = BottleneckTransformerDecoder(num_features, model_dim, num_heads, num_layers, representation_dim, sequence_length)

    def forward(self, x):
        representation = self.encoder(x)
        reconstructed = self.decoder(representation)
        return reconstructed
    
    def get_representation(self, x):
        with torch.no_grad():
            return self.encoder(x)
        

class ProbabilisticRobustAutoencoder(nn.Module):
    """
    PRAE: Probabilistic Robust Autoencoder
    Based on: "Probabilistic Robust Autoencoders for Outlier Detection" by Lindenbaum et al.

    Wrapper to add stochastic gates (mu) to a base autoencoder model.
    """
    def __init__(self, base_autoencoder, num_train_samples, sigma=0.5):
        super(ProbabilisticRobustAutoencoder, self).__init__()
        self.ae = base_autoencoder
        self.num_samples = num_train_samples
        self.sigma = sigma

        self.mu = nn.Parameter(torch.full((num_train_samples,), 0.9))

    def forward(self, x, indices=None, training=True):
        """
        Args:
            x (torch.Tensor): Input data batch
            indices (torch.Tensor, optional): Indices of the batch samples in the original training set. Defaults to None.
            training (bool, optional): Flag indicating training mode. If True, uses stochastic gates. If False, acts as a standard autoencoder. Defaults to True.

        Returns:
            reconstructed (torch.Tensor): Reconstructed output from the autoencoder.
            z (torch.Tensor or None): Stochastic gates applied during training, None if not training or indices not provided.
        """
        reconstructed = self.ae(x)

        z = None
        if training and indices is not None:
            batch_mu = self.mu[indices]

            # Stochastic Gate: z[i] = max(0, min(1, mu[i] + epsilon))
            noise = torch.randn_like(batch_mu) * self.sigma
            z = torch.clamp(batch_mu + noise, 0.0, 1.0)

        return reconstructed, z
    
    def get_outlier_scores(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        with torch.no_grad():
            return self.mu.cpu().numpy()


class ProbabilisticNN(nn.Module):
    """
    PNN: Fabre & Challet
    Architecture: 1 hidden layer, 64 neurons
    Prediction: 3 parameters of the skewed gaussian distribution
    """
    def __init__(self, input_dim, hidden_dim):
        super(ProbabilisticNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 3)  # Output: mu, sigma, alpha

        # Activation for sigma to ensure positivity
        self.softplus = nn.Softplus()

    def forward(self, x):
        """
        Forward pass
        Returns: mu (location), sigma (scale > 0), alpha (skewness)
        """
        h = self.relu(self.fc1(x))
        output = self.fc2(h)

        mu = output[:, 0:1]
        sigma_raw = output[:, 1:2]
        alpha = output[:, 2:3]
        sigma = self.softplus(sigma_raw) + 1e-6  # Ensure sigma is positive
        
        return mu, sigma, alpha
    

class SkewedGaussianNLL(nn.Module):
    """
    Negative Log-Likelihood Loss for Skewed Gaussian Distribution
    Based on Equation (20) from Fabre & Challet:
    f(x) = (2/sigma) * phi((x-mu)/sigma) * Phi(alpha * (x-mu)/sigma)
    """
    def __init__(self):
        super(SkewedGaussianNLL, self).__init__()
    
    def _phi(self, z):
        """Standard normal PDF"""
        return (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * z**2)
    
    def _Phi(self, z):
        """Standard normal CDF"""
        return 0.5 * (1 + torch.erf(z / math.sqrt(2)))

    def forward(self, y_true, mu, sigma, alpha):
        """
        Compute NLL for skewed Gaussian

        Args:
            y_true: Target values (price moves)
            mu: Location parameter
            sigma: Scale parameter (must be > 0)
            alpha: Skewness parameter
        """
        y_true = y_true.view_as(mu)
        z = (y_true - mu) / sigma

        # Skewed Gaussian PDF
        pdf = (2.0 / sigma) * self._phi(z) * self._Phi(alpha * z)
        
        # Negative Log-Likelihood
        log_pdf = -torch.log(pdf + 1e-10)  # Add small constant for numerical stability
        
        return torch.mean(log_pdf)


def skewed_gaussian_cdf(x, mu, sigma, alpha):
    """
    Skewed Gaussian PDF
    Based on Fabre & Challet : Equation 35
    F_alpha((x-mu)/sigma)
    """
    # z = (x - mu) / sigma

    # phi = lambda t: (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * t**2)
    # Phi = lambda t: 0.5 * (1 + erf(t / np.sqrt(2)))

    cdf = skewnorm.cdf(x, a=alpha, loc=mu, scale=sigma)
    return cdf


def conditional_expectation_skewed_gaussian(x_thresh, mu, sigma, alpha, upper=True):
    """
    Conditional expectation E[X | X > x_thresh] or E[X | X <= x_thresh]
    Based on Fabre & Challet : Equation 36 and 37

    Args:
        x_thresh: Threshold value
        mu, sigma, alpha: Skewed Gaussian parameters
        upper: If True, compute E[X | X > x_thresh], else E[X | X <= x_thresh]
    """
    z = (x_thresh - mu) / sigma
    beta = alpha / np.sqrt(1 + alpha**2)

    phi = lambda t: (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * t**2)
    Phi = lambda t: 0.5 * (1 + erf(t / np.sqrt(2.0)))

    # CDF value
    F_alpha_z = skewed_gaussian_cdf(x_thresh, mu, sigma, alpha)

    if upper:
        term1 = np.sqrt(2/np.pi) * beta * (1 - Phi(np.sqrt(1 + alpha**2) * z))
        term2 = np.exp(-0.5 * z**2) * Phi(alpha * z)

        E_cond = mu + sigma * (term1 + term2) / (1 - F_alpha_z + 1e-10)
    else:
        term1 = np.sqrt(2/np.pi) * beta * Phi(np.sqrt(1 + alpha**2) * z)
        term2 = np.exp(-0.5 * z**2) * Phi(alpha * z)

        E_cond = mu + sigma * (term1 - term2) / (F_alpha_z + 1e-10)

    return E_cond
    

def calculate_expected_cost(mu, sigma, alpha, spread, delta_a, delta_b, Q, q, epsilon_plus=0.0, epsilon_minus=0.05, p_bid=None, p_ask=None, side='ask'):
    """
    Calculate Expected Cost for spoofing detection
    Based on Fabre & Challet: Equations 27 and 28

    Args:
        mu, sigma, alpha: Skewed Gaussian parameters
        spread: Current bid-ask spread
        delta_a: Distance of ask order from best ask
        delta_b: Distance of bid order from best bid
        Q: Spoof order size
        q: Genuine order size
        epsilon_plus: Maker fee (default 0%)
        epsilon_minus: Taker fee (default 5%)
        p_bid, p_ask: Best bid and ask prices
        side: 'ask' for selling spoofer, 'bid' for buying spoofer

    Returns:
        expected_cost: Expected cost of spoofing strategy
    """
    if torch.is_tensor(mu):
        mu = mu.detach().cpu().numpy()
        sigma = sigma.detach().cpu().numpy()
        alpha = alpha.detach().cpu().numpy()
    
    thresh_a = delta_a + 0.5 * spread # Ask side
    thresh_b = -(delta_b + 0.5 * spread) # Bid side

    if p_bid is None:
        p_bid = 1.0
    if p_ask is None:
        p_ask = 1.0 + spread

    if side == 'ask': # Spoofer wants to sell
        P_ask_filled = 1.0 - skewed_gaussian_cdf(thresh_a, mu, sigma, alpha)
        P_bid_filled = skewed_gaussian_cdf(thresh_b, mu, sigma, alpha)

        E_dp_ask_not_filled = conditional_expectation_skewed_gaussian(thresh_a, mu, sigma, alpha, upper=False)
        E_dp_bid_filled = conditional_expectation_skewed_gaussian(thresh_b, mu, sigma, alpha, upper=False)

        # Cost components
        # Revenue from executed bona fide sell order
        cost_1 = - P_ask_filled * (1 - epsilon_plus) * q * (p_ask + delta_a)

        # Loss from executed non-bona fide buy order
        cost_2 = P_bid_filled * (1 + epsilon_plus) * Q * (p_bid - delta_b)

        # Cost of liquidating unfilled bona fide order (market sell)
        cost_3 = -(1 - P_ask_filled) * (1 - epsilon_minus) * q *(p_bid + E_dp_ask_not_filled)

        # Cost of liquidating filled non-bona fide order (market sell)
        cost_4 = -P_bid_filled * (1 - epsilon_minus) * Q * (p_bid + E_dp_bid_filled)

        expected_cost = cost_1 + cost_2 + cost_3 + cost_4
    
    else: # side == 'bid', Spoofer wants to buy
        P_bid_filled = skewed_gaussian_cdf(thresh_b, mu, sigma, alpha)
        P_ask_filled = 1.0 - skewed_gaussian_cdf(thresh_a, mu, sigma, alpha)

        E_dp_bid_not_filled = conditional_expectation_skewed_gaussian(thresh_b, mu, sigma, alpha, upper=True)
        E_dp_ask_filled = conditional_expectation_skewed_gaussian(thresh_a, mu, sigma, alpha, upper=True)
        
        # Cost components
        cost_1 = P_bid_filled * (1 + epsilon_plus) * q * (p_bid - delta_b)
        cost_2 = -P_ask_filled * (1 - epsilon_plus) * Q * (p_ask + delta_a)
        cost_3 = (1 - P_bid_filled) * (1 + epsilon_minus) * q * (p_ask + E_dp_bid_not_filled)
        cost_4 = P_ask_filled * (1 + epsilon_minus) * Q * (p_ask + E_dp_ask_filled)
        
        expected_cost = cost_1 + cost_2 + cost_3 + cost_4

    return expected_cost


def compute_spoofing_gain(model, x_original, x_spoofed, spread, delta_a, delta_b, Q, q, fees, side='ask'):
    """
    Compute expected gain from spoofing strategy
    Based on Fabre & Challet: Equation 31
    Delta_C(Q, delta) = E[C_spoof(0, delta_a, 0, q) | x0] - E[C_spoof(delta, delta_a, Q, q) | x]

    Args:
        model: Trained PNN model
        x_original: Features without spoof order
        x_spoofed: Features with spoof order
        spread: Current bid-ask spread
        delta_a: Distance of ask order from best ask
        delta_b: Distance of bid order from best bid
        Q: Spoof order size
        q: Genuine order size
        fees: Tuple of (epsilon_plus, epsilon_minus)
        side: 'ask' for selling spoofer, 'bid' for buying spoofer
    Returns:
        spoofing_gain: Positive if spoofing is profitable
    """
    model.eval()
    with torch.no_grad():
        # Without spoofing
        mu0, sigma0, alpha0 = model(x_original)
        cost_no_spoof = calculate_expected_cost(mu0, sigma0, alpha0, spread, delta_a, 0.0, 0.0, q, fees['maker'], fees['taker'], side=side)

        # With spoofing
        mu1, sigma1, alpha1 = model(x_spoofed)
        cost_with_spoof = calculate_expected_cost(mu1, sigma1, alpha1, spread, delta_a, delta_b, Q, q, fees['maker'], fees['taker'], side=side)

        spoofing_gain = cost_no_spoof - cost_with_spoof
        
    return spoofing_gain


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss does not improve.
    """
    def __init__(self, patience=5, verbose=False, delta=0.0, path='checkpoint.pth'):
        """
        Args:
            patience (int, optional): How long to wait after last time validation loss improved. Defaults to 5.
            verbose (bool, optional): If True, prints a message for each validation loss improvement. Defaults to False.
            delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement. Defaults to 0.0.
            path (str, optional): Path for the checkpoint to be saved to. Defaults to 'checkpoint.pth'.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def compute_permutation_importance(pipeline, X_test, baseline_mse, n_repeats=1):
    """
    Compute permutation feature importance.
    """
    importances = []
    X_np = X_test.cpu().numpy()

    for i, feature_name in enumerate(pipeline.feature_names):
        mse_scores = []

        for _ in range(n_repeats):
            X_permuted = X_np.copy()
            np.random.shuffle(X_permuted[:, :, i])

            X_permuted_tensor = torch.tensor(X_permuted, dtype=torch.float32).to(pipeline.device)
            dataset = TensorDataset(X_permuted_tensor)
            loader = DataLoader(dataset, batch_size=pipeline.batch_size, shuffle=False)

            mse_values = []
            with torch.no_grad():
                for (x_batch,) in loader:
                    reconstructed, _ = pipeline.model(x_batch)
                    mse = torch.mean((reconstructed - x_batch)**2, dim=[1, 2])
                    mse_values.append(mse)

            mse = torch.cat(mse_values).mean().item()
            mse_scores.append(mse)

        mean_mse = np.mean(mse_scores)
        importance = mean_mse - baseline_mse
        importances.append({'Feature': feature_name, 'Importance': importance})
    
    return pd.DataFrame(importances).sort_values(by='Importance', ascending=False)


def analyze_root_causes(pipeline, anomaly_indices, feature_names):
    """
    Identifies features with the highest reconstruction error for anomalies.
    """
    print(f"Analyzing {len(anomaly_indices)} anomalies for root causes...")
    
    max_idx = len(pipeline.X_test) - 1
    valid_indices = anomaly_indices[anomaly_indices <= max_idx]

    n_dropped = len(anomaly_indices) - len(valid_indices)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} anomaly indices that were out of bounds for the test set.")

    if len(valid_indices) == 0:
        print("No valid anomaly indices to analyze.")
        return pd.DataFrame(columns=['Feature', 'Contribution'])

    pipeline.model.eval()
    
    anom_seqs = []
    for idx in valid_indices:
        seq, _ = pipeline.X_test[idx]
        anom_seqs.append(seq)
        
    X_anom = torch.stack(anom_seqs).to(pipeline.device)
    
    # Feature Contribution = Squared Reconstruction Error
    if pipeline.model_type in ['prae', 'transformer_ocsvm']:
        with torch.no_grad():
            if pipeline.model_type == 'prae':
                reconstructed, _ = pipeline.model(X_anom, training=False)
            else:
                reconstructed = pipeline.model(X_anom)
                
        # Error per feature
        feature_errors = torch.mean((X_anom - reconstructed)**2, dim=1).cpu().numpy()
        
        mean_feature_contribution = np.mean(feature_errors, axis=0)

    else:
        # PNN: how far the anomaly is from the global mean
        X_anom_np = X_anom.cpu().numpy()
        X_anom_mean_time = np.mean(X_anom_np, axis=1)
        
        # Global mean of the test set
        loader = pipeline._get_dataloader(pipeline.X_test, shuffle=True)
        batch, _ = next(iter(loader))
        global_mean = torch.mean(batch, dim=[0, 1]).cpu().numpy()
        
        # Contribution = Absolute Deviation
        mean_feature_contribution = np.mean(np.abs(X_anom_mean_time - global_mean), axis=0)

    contribution_df = pd.DataFrame({
        'Feature': feature_names,
        'Contribution': mean_feature_contribution
    }).sort_values(by='Contribution', ascending=False)
    
    return contribution_df


class IntegratedGradients:
    """
    Implements Integrated Gradients.
    Designed to explain anomaly scores in HFT time series.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def attribute(self, inputs, baseline=None, target_func=None, n_steps=50):
        """
        Computes feature attributions using Integrated Gradients.
        
        Args:
            inputs (torch.Tensor): Input data (Batch, Seq_Len, Features)
            baseline (torch.Tensor, optional): Baseline reference. Defaults to zeros.
            target_func (callable): Function that takes the model output and returns a scalar 
                                    (the anomaly score) to differentiate.
            n_steps (int): Number of interpolation steps.
            
        Returns:
            attributions (torch.Tensor): Importance scores same shape as inputs.
        """
        if inputs.shape[0] != 1:
            raise ValueError("Current IG implementation supports single-sample explanation (Batch=1).")

        if baseline is None:
            baseline = torch.zeros_like(inputs)

        # Generate interpolated inputs
        # Shape: (n_steps, Batch, Seq, Feat)
        alphas = torch.linspace(0, 1, n_steps + 1).to(inputs.device)
        
        # Create scaled inputs
        scaled_inputs = baseline + alphas[:, None, None, None] * (inputs - baseline)
        scaled_inputs = scaled_inputs.squeeze(1) 
        scaled_inputs.requires_grad_(True)

        # Forward pass
        model_output = self.model(scaled_inputs)
        
        # Apply the target function to get the scalar anomaly score
        score = target_func(model_output, scaled_inputs)
        
        # Compute Gradients
        grads = torch.autograd.grad(torch.sum(score), scaled_inputs)[0]
        
        # Integral Approximation
        # Average gradients across steps
        avg_grads = torch.mean(grads[:-1] + grads[1:], dim=0) / 2.0
        
        # Scale by (Input - Baseline)
        attributions = (inputs - baseline) * avg_grads.unsqueeze(0)
        
        return attributions


def prae_anomaly_score_func(model_output, inputs):
    """
    Target function for Probabilistic Robust Autoencoder.
    Returns: Mean Squared Error (Reconstruction Loss) per sample.
    """
    reconstructed, _ = model_output
    
    # Gradients of the squared error sum
    loss = torch.sum((reconstructed - inputs) ** 2, dim=[1, 2])
    return loss


def pnn_uncertainty_func(model_output, inputs):
    """
    Target function for Probabilistic Neural Network.
    Returns: Predicted Sigma (Uncertainty) sum.
    
    Explains why the model is uncertain/predicting high volatility.
    """
    mu, sigma, alpha = model_output

    return sigma.sum()


def pnn_nll_func(model_output, inputs, target_y=None):
    """
    Target function for PNN Negative Log Likelihood.
    Requires target_y to explain high loss.
    """
    mu, sigma, alpha = model_output
    
    criterion = SkewedGaussianNLL()
    
    # If target_y is not provided, we can't calculate NLL.
    if target_y is None:
        raise ValueError("Target Y required for NLL attribution")
        
    return criterion(target_y, mu, sigma, alpha)


def explain_occlusion(pipeline, x_seq, feature_names, baseline_mode='mean'):
    """
    Performs Feature Ablation (Occlusion) to explain Transformer+OCSVM anomalies.
    
    Args:
        pipeline: The trained AnomalyDetectionPipeline.
        x_seq (torch.Tensor): The target sequence of shape (1, Seq_Len, Num_Features).
        feature_names (list): List of feature names.
        
    Returns:
        pd.DataFrame: Feature importance sorted by contribution to the anomaly.
    """
    pipeline.model.eval()
    
    # Setup Data
    num_features = len(feature_names)

    if x_seq.dim() == 2:
        x_seq = x_seq.unsqueeze(0)

    # Determine mask value
    if baseline_mode == 'mean':
        baseline_values = x_seq.mean(dim=1, keepdim=True)
    else:
        baseline_values = torch.zeros_like(x_seq[:, 0:1, :])

    batch_tensor = x_seq.repeat(num_features + 1, 1, 1).clone()

    # Occlusion
    for i in range(num_features):
        # Set the i-th feature to 0 across the entire time sequence
        batch_tensor[i + 1, :, i] = 0.0

    # Batch Inference
    with torch.no_grad():
        scores = pipeline.predict(batch_tensor)

    # Calculate Importance
    original_score = scores[0]
    occluded_scores = scores[1:]

    importances = original_score - occluded_scores

    # Format Results
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Original_Score': original_score,
        'New_Score': occluded_scores
    }).sort_values(by='Importance', ascending=False)

    return importance_df