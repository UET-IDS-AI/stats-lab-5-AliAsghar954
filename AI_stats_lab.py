import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.

    f(x) = lam * exp(-lam*x) for x >= 0
    """
    if x < 0:
        return 0
    return lam * np.exp(-lam * x)


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using analytical formula.
    """
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a, b, n=100000):
    """
    Simulate exponential samples and estimate
    P(a < X < b).
    """
    samples = np.random.exponential(scale=1, size=n)
    prob = np.mean((samples > a) & (samples < b))
    return prob


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    """
    coeff = 1 / (np.sqrt(2 * np.pi) * sigma)
    exponent = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return coeff * exponent


def posterior_probability(time):
    """
    Compute P(B | X = time) using Bayes rule.

    Priors:
    P(A)=0.3
    P(B)=0.7

    Distributions:
    A ~ N(40,4)
    B ~ N(45,4)
    """

    pA = 0.3
    pB = 0.7

    muA, sigmaA = 40, 2
    muB, sigmaB = 45, 2

    fA = gaussian_pdf(time, muA, sigmaA)
    fB = gaussian_pdf(time, muB, sigmaB)

    numerator = pB * fB
    denominator = pA * fA + pB * fB

    return numerator / denominator


def simulate_posterior_probability(time, n=100000):
    """
    Estimate P(B | X=time) using simulation.
    """

    pA = 0.3
    pB = 0.7

    # choose groups
    groups = np.random.choice(["A", "B"], size=n, p=[pA, pB])

    times = np.zeros(n)

    # generate times
    times[groups == "A"] = np.random.normal(40, 2, np.sum(groups == "A"))
    times[groups == "B"] = np.random.normal(45, 2, np.sum(groups == "B"))

    # select swimmers near the observed time
    tolerance = 0.5
    mask = (times > time - tolerance) & (times < time + tolerance)

    if np.sum(mask) == 0:
        return 0

    selected_groups = groups[mask]

    return np.mean(selected_groups == "B")
