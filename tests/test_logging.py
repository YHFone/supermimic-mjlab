import pytest

from supermimic.algo.trainer import window_timings


def test_timing_parts_sum_to_iteration_time():
  row = window_timings(2.757, 1.157, 1.246)
  assert row["time/collect"] == pytest.approx(0.354)
  total = row["time/collect"] + row["time/world_model"] + row["time/policy"]
  assert total == pytest.approx(2.757)


def test_timing_never_negative_when_clock_jitters():
  row = window_timings(1.0, 0.9, 0.95)
  assert row["time/collect"] >= 0.0
  assert sum(row.values()) == pytest.approx(1.0)
