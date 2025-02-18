import numpy as np
from scipy.fft import fft, fftfreq
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt

def apply_sine_modulation(X: np.ndarray, frequencies: np.ndarray, num_cycles: int = 10, sample_rate: int = 100):
    """
    Applies sine wave modulation to each feature in X with a unique frequency.
    
    :param X: (n_samples, n_features) input feature matrix
    :param frequencies: (n_features,) array of unique frequencies per feature
    :param num_cycles: Number of cycles per feature modulation
    :param sample_rate: Number of time steps per cycle
    :return: (time_series_samples, n_features) modulated feature matrix
    """
    n_samples, n_features = X.shape
    time_steps = num_cycles * sample_rate
    t = np.linspace(0, num_cycles, time_steps)
    X_modulated = np.zeros((time_steps, n_features))
    
    for i in range(n_features):
        X_modulated[:, i] = X.mean(axis=0)[i] + np.sin(2 * np.pi * frequencies[i] * t)
    
    return t, X_modulated


def analyze_spectral_importance(y_series: np.ndarray, frequencies: np.ndarray, sample_rate: int):
    """
    Extracts feature importance by analyzing the power spectrum of the model's prediction.
    
    :param y_series: Time-series predictions from the model
    :param frequencies: Frequencies associated with each feature
    :param sample_rate: Sample rate used in modulation
    :return: Feature importance dictionary
    """
    N = len(y_series)
    yf = fft(y_series)
    freq_axis = fftfreq(N, d=1/sample_rate)
    
    importance = {}
    for f in frequencies:
        idx = np.argmin(np.abs(freq_axis - f))
        importance[f] = np.abs(yf[idx])
    
    return importance

def feature_importance_analysis(model: BaseEstimator, X: np.ndarray, frequencies: np.ndarray, sample_rate: int = 100):
    """
    Computes feature importance using frequency-based analysis.
    
    :param model: Trained machine learning model
    :param X: (n_samples, n_features) feature matrix
    :param frequencies: (n_features,) unique frequencies for each feature
    :param sample_rate: Sample rate for time discretization
    :return: Dictionary of feature importances
    """
    t, X_modulated = apply_sine_modulation(X, frequencies, sample_rate=sample_rate)
    y_series = model.predict(X_modulated)
    importance = analyze_spectral_importance(y_series, frequencies, sample_rate)
    
    # Normalize importance values
    max_val = max(importance.values())
    importance = {f: v / max_val for f, v in importance.items()}
    
    return importance

# Example usage with a trained model:
# model = SomeTrainedModel()
# X_sample = np.random.randn(100, 5)  # Example input data
# freqs = np.array([1, 2, 3, 5, 7])  # Unique frequencies for each feature
# importance = feature_importance_analysis(model, X_sample, freqs)
# print(importance)
