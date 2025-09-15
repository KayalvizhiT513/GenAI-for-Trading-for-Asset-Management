"""Translate Matlab efficient frontier code to Python."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize


@dataclass
class EfficientFrontierResult:
    target_returns: np.ndarray
    volatilities: np.ndarray
    weights: List[np.ndarray]
    tangency_index: int
    min_variance_index: int


def calculate_returns(prices: np.ndarray, period: int = 1) -> np.ndarray:
    """Compute net returns over ``period`` intervals."""
    if period <= 0:
        raise ValueError("period must be positive")
    return prices[period:] / prices[:-period] - 1.0


def solve_efficient_portfolio(
    cov_matrix: np.ndarray,
    mean_returns: np.ndarray,
    target_return: float,
) -> Tuple[np.ndarray, float]:
    """Solve the long-only minimum variance portfolio for a target return."""
    num_assets = mean_returns.size

    def objective(weights: np.ndarray) -> float:
        return float(weights @ cov_matrix @ weights)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return 2.0 * (cov_matrix @ weights)

    constraints = (
        {
            "type": "eq",
            "fun": lambda w, target=target_return: float(w @ mean_returns - target),
            "jac": lambda w, target=target_return: mean_returns,
        },
        {
            "type": "eq",
            "fun": lambda w: float(np.sum(w) - 1.0),
            "jac": lambda w: np.ones_like(w),
        },
    )
    bounds = [(0.0, None)] * num_assets
    x0 = np.full(num_assets, 1.0 / num_assets)
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    if not result.success:
        message = result.message if result.message else "optimization failed"
        raise RuntimeError(f"Optimization failed for target return {target_return}: {message}")

    return np.asarray(result.x), float(result.fun)


def build_efficient_frontier(mean_returns: np.ndarray, cov_matrix: np.ndarray) -> EfficientFrontierResult:
    target_returns = np.linspace(np.min(mean_returns), np.max(mean_returns), 21)
    variances = np.empty_like(target_returns)
    weights: List[np.ndarray] = []

    for idx, target in enumerate(target_returns):
        w, var = solve_efficient_portfolio(cov_matrix, mean_returns, float(target))
        variances[idx] = var
        weights.append(w)

    volatilities = np.sqrt(variances)
    sharpe_ratios = np.divide(target_returns, volatilities, out=np.zeros_like(target_returns), where=volatilities > 0)
    tangency_index = int(np.argmax(sharpe_ratios))
    min_variance_index = int(np.argmin(volatilities))

    return EfficientFrontierResult(target_returns, volatilities, weights, tangency_index, min_variance_index)


def plot_frontier(result: EfficientFrontierResult) -> None:
    plt.scatter(result.volatilities, result.target_returns)
    plt.scatter(
        result.volatilities[result.tangency_index],
        result.target_returns[result.tangency_index],
        color="red",
        label="Tangency portfolio",
    )
    plt.scatter(
        result.volatilities[result.min_variance_index],
        result.target_returns[result.min_variance_index],
        color="green",
        label="Minimum variance portfolio",
    )
    plt.xlabel("Standard Deviation")
    plt.ylabel("Mean Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def load_data(mat_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = loadmat(mat_file, squeeze_me=True)
    stocks = np.array([str(s) for s in data["stocks"].tolist()])
    cl = np.array(data["cl"], dtype=float)

    mask = (stocks != "EWZ") & (stocks != "FXI")
    stocks = stocks[mask]
    cl = cl[:, mask]

    returns = calculate_returns(cl, 1)
    mean_returns = np.mean(returns, axis=0)
    covariance = np.cov(returns, rowvar=False)
    return stocks, mean_returns, covariance


def main() -> None:
    parser = argparse.ArgumentParser(description="Efficient frontier for ETF universe")
    parser.add_argument(
        "mat_file",
        type=Path,
        default=Path("inputDataOHLCDaily_ETF_20150417.mat"),
        nargs="?",
        help="Path to the MATLAB data file",
    )
    args = parser.parse_args()

    _, mean_returns, covariance = load_data(args.mat_file)
    frontier = build_efficient_frontier(mean_returns, covariance)

    tangency_weights = frontier.weights[frontier.tangency_index]
    min_variance_weights = frontier.weights[frontier.min_variance_index]

    np.set_printoptions(precision=12, suppress=True)
    print("Tangency portfolio weights:")
    print(tangency_weights)
    print("Minimum variance portfolio weights:")
    print(min_variance_weights)

    plot_frontier(frontier)


if __name__ == "__main__":
    main()
