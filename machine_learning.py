import numpy as np
import math
from scipy.stats import skewnorm
from scipy.special import erf
import torch
import torch.nn as nn


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
