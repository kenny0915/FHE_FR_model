"""Example user-designed quadratic for --activation-file."""

from stability_analysis.activations import PolynomialActivation


def make_activation(name, original_module):
    # Target: GELU/PReLU proxy on [-4, 4]. Coefficients are [C, B, A]
    # for A*x^2 + B*x + C. Refit these values for the measured distribution.
    return PolynomialActivation(
        coefficients=[0.15, 0.5, 0.125],
        interval=(-4.0, 4.0),
        target="user GELU/PReLU proxy",
    )
