from __future__ import annotations

import os

from recall.services.system_state import (
    EmbedPreconditionResult,
    LoadThreshold,
    check_load,
    check_power,
    resolve_load_threshold,
)


class TestCheckLoad:
    def test_returns_ok_when_load_below_threshold(self, monkeypatch):
        monkeypatch.setattr(os, "getloadavg", lambda: (1.0, 1.0, 1.0))
        monkeypatch.setattr(os, "cpu_count", lambda: 8)
        result = check_load(LoadThreshold(fraction=0.7, on_ac=True))
        assert result.ok

    def test_returns_skip_when_load_above_threshold(self, monkeypatch):
        monkeypatch.setattr(os, "getloadavg", lambda: (7.0, 5.0, 3.0))
        monkeypatch.setattr(os, "cpu_count", lambda: 8)
        result = check_load(LoadThreshold(fraction=0.7, on_ac=True))
        assert not result.ok
        assert "load" in result.reason
        assert "threshold" in result.reason

    def test_returns_ok_when_getloadavg_unavailable(self, monkeypatch):
        monkeypatch.delattr(os, "getloadavg", raising=False)
        result = check_load(LoadThreshold(fraction=0.7, on_ac=True))
        assert result.ok

    def test_reason_names_the_battery_threshold_and_the_power_state(self, monkeypatch):
        """A reason naming a bare "threshold" cannot be checked against `pmset`.

        The battery fraction and the AC fraction produce the same sentence, so an
        operator seeing a ceiling far below the configured one cannot tell whether
        the daemon read the power state differently or the reason is stale
        (REQ-ADAPT-006).
        """
        monkeypatch.setattr(os, "getloadavg", lambda: (41.0, 5.0, 3.0))
        monkeypatch.setattr(os, "cpu_count", lambda: 18)

        result = check_load(
            LoadThreshold(fraction=0.3, on_ac=False, power_detail="on battery: Battery Power")
        )

        assert not result.ok
        assert result.reason == "load 41.0 > battery threshold 5.4 (on battery: Battery Power)"

    def test_reason_names_the_ac_threshold_and_the_power_state(self, monkeypatch):
        monkeypatch.setattr(os, "getloadavg", lambda: (41.0, 5.0, 3.0))
        monkeypatch.setattr(os, "cpu_count", lambda: 18)

        result = check_load(LoadThreshold(fraction=0.7, on_ac=True))

        assert not result.ok
        assert result.reason == "load 41.0 > load threshold 12.6 (AC power)"


class TestResolveLoadThreshold:
    def test_ac_power_puts_the_load_threshold_in_force(self, monkeypatch):
        monkeypatch.setattr(
            "recall.services.system_state.check_power",
            lambda: EmbedPreconditionResult(ok=True),
        )

        threshold = resolve_load_threshold(load_threshold=0.7, battery_threshold=0.3)

        assert threshold.fraction == 0.7
        assert threshold.name == "load threshold"
        assert threshold.power == "AC power"

    def test_battery_puts_the_battery_threshold_in_force(self, monkeypatch):
        monkeypatch.setattr(
            "recall.services.system_state.check_power",
            lambda: EmbedPreconditionResult(ok=False, reason="on battery: Battery Power"),
        )

        threshold = resolve_load_threshold(load_threshold=0.7, battery_threshold=0.3)

        assert threshold.fraction == 0.3
        assert threshold.name == "battery threshold"
        assert threshold.power == "on battery: Battery Power"


class TestCheckPower:
    def test_returns_ok_on_non_macos(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "linux")
        result = check_power()
        assert result.ok
