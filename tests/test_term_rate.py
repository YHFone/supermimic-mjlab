"""Termination-rate EMA: first sample initializes, later samples smooth."""

from supermimic.algo.trainer import _ema_term_rate


def test_term_rate_ema_initializes_on_first_nonzero_sample():
  assert _ema_term_rate(0.0, 0.026) == 0.026


def test_term_rate_ema_smooths_later_samples():
  got = _ema_term_rate(0.026, 0.003)
  assert abs(got - (0.99 * 0.026 + 0.01 * 0.003)) < 1e-12


def test_term_rate_ema_handles_zero_terminations():
  # A zero batch must not reset a previously established rate.
  assert abs(_ema_term_rate(0.026, 0.0) - 0.99 * 0.026) < 1e-12
