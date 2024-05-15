import numpy as np
from scipy.stats import t

def welch_t_test(sample1, sample2):
    """
    Perform Welch's t-test (Two-Sample t-test) to compare the means of two independent samples.

    This function assumes that the samples have unequal variances and possibly unequal sample sizes.
    It calculates the t-statistic and p-value for the hypothesis test.

    Args:
        sample1 (array-like): First sample data.
        sample2 (array-like): Second sample data.

    Returns:
        tuple: A tuple containing the t-statistic and p-value.
            - t_statistic (float): The calculated t-statistic.
            - p_value (float): The corresponding p-value for the hypothesis test.

    Raises:
        ValueError: If the sample sizes are less than 2 for either sample.
    """
    # Convert input samples to numpy arrays
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)

    # Check if sample sizes are at least 2
    if len(sample1) < 2 or len(sample2) < 2:
        raise ValueError("Sample sizes must be at least 2 for both samples.")

    # Calculate sample means
    mean1 = sample1.mean()
    mean2 = sample2.mean()

    # Calculate sample variances
    var1 = sample1.var(ddof=1)  # ddof=1 for unbiased variance estimate
    var2 = sample2.var(ddof=1)

    # Get sample sizes
    n1 = sample1.size
    n2 = sample2.size

    # Calculate the t-statistic using Welch's formula
    t_statistic = (mean1 - mean2) / np.sqrt(var1/n1 + var2/n2)

    # Calculate the degrees of freedom using the Welch-Satterthwaite equation
    dof = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))

    # Calculate the p-value using the t-distribution
    p_value = 2 * (1 - t.cdf(np.abs(t_statistic), dof))

    print("Welch's t-test: t = %g, p = %g" % (t_statistic, p_value))

    return t_statistic, p_value