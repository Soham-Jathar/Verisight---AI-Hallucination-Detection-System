from app.services.math_notation import format_math_notation


def test_math_notation_removes_latex_delimiters_and_commands() -> None:
    answer = r"The integral is $\int \cos(x) dx = \sin(x) + C$."
    assert format_math_notation(answer) == "The integral is ∫ cos(x) dx = sin(x) + C."


def test_math_notation_formats_common_fraction_and_exponent() -> None:
    answer = r"\frac{x^{n+1}}{n+1} + C"
    assert format_math_notation(answer) == "xⁿ⁺¹/(n+1) + C"
