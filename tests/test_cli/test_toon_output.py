"""Unit tests for TOON output format support (REQ-CLI-013, REQ-CLI-014)."""

from __future__ import annotations

import json
from io import StringIO
from unittest.mock import patch

import pytest
from recall.cli.contract import (
    CliError,
    ErrorCode,
    OutputFormat,
    emit_data,
    emit_error,
    parse_output_format,
    render_dry_run,
    resolve_output_format,
)


@pytest.mark.parametrize("name", ["toon", "TOON", "Toon"])
def test_parse_output_format_toon_case_insensitive(name: str) -> None:
    assert parse_output_format(name) == OutputFormat.TOON


def test_resolve_non_tty_defaults_to_toon() -> None:
    """REQ-CLI-013: non-TTY defaults to TOON."""
    with patch("sys.stdout") as mock_stdout:
        mock_stdout.isatty.return_value = False
        result = resolve_output_format(json_output=False, format_name=None)
    assert result == OutputFormat.TOON


def test_resolve_tty_defaults_to_text() -> None:
    with patch("sys.stdout") as mock_stdout:
        mock_stdout.isatty.return_value = True
        result = resolve_output_format(json_output=False, format_name=None)
    assert result == OutputFormat.TEXT


def test_json_flag_overrides_toon_default() -> None:
    """REQ-CLI-013: --json forces JSON."""
    result = resolve_output_format(json_output=True, format_name=None)
    assert result == OutputFormat.JSON


def test_format_json_overrides_toon_default() -> None:
    """REQ-CLI-013: --format json forces JSON."""
    result = resolve_output_format(json_output=False, format_name="json")
    assert result == OutputFormat.JSON


def test_emit_data_toon_produces_valid_toon() -> None:
    """Verify emit_data with TOON format produces decodable TOON output."""
    from toon_format import decode as toon_decode

    captured = StringIO()
    with patch("typer.echo", side_effect=lambda s, **_: captured.write(str(s) + "\n")):
        emit_data(
            [{"id": "abc", "source": "claude_code"}],
            output_format=OutputFormat.TOON,
        )

    decoded = toon_decode(captured.getvalue())
    assert isinstance(decoded, list)
    assert decoded[0]["id"] == "abc"
    assert decoded[0]["source"] == "claude_code"


def test_emit_data_toon_round_trip_object() -> None:
    """Verify TOON round-trip for object data."""
    from toon_format import decode as toon_decode

    data = {"total": 5, "indexed": 3, "skipped": 2}
    captured = StringIO()
    with patch("typer.echo", side_effect=lambda s, **_: captured.write(str(s) + "\n")):
        emit_data(data, output_format=OutputFormat.TOON)

    decoded = toon_decode(captured.getvalue())
    assert isinstance(decoded, dict)
    assert decoded["total"] == 5
    assert decoded["indexed"] == 3


def test_emit_error_toon() -> None:
    """Verify errors are TOON-encoded in TOON mode."""
    from toon_format import decode as toon_decode

    error = CliError(code=ErrorCode.VALIDATION, message="bad input")
    captured = StringIO()
    with patch("typer.echo", side_effect=lambda s, **_: captured.write(str(s) + "\n")):
        emit_error(error, output_format=OutputFormat.TOON)

    decoded = toon_decode(captured.getvalue())
    assert isinstance(decoded, dict)
    assert decoded["error"]["code"] == "VALIDATION"
    assert decoded["error"]["message"] == "bad input"


def test_render_dry_run_toon() -> None:
    """Verify dry-run output is TOON-encoded in TOON mode."""
    from toon_format import decode as toon_decode

    captured = StringIO()
    with patch("typer.echo", side_effect=lambda s, **_: captured.write(str(s) + "\n")):
        render_dry_run(
            command="index",
            request={"full": True},
            safety={"mutates": True, "destructive": False, "idempotent": True},
            output_format=OutputFormat.TOON,
        )

    decoded = toon_decode(captured.getvalue())
    assert isinstance(decoded, dict)
    assert decoded["dry_run"] is True
    assert decoded["command"] == "index"


def test_toon_fallback_when_import_missing() -> None:
    """REQ-CLI-014: falls back to JSON with stderr warning when toon-format unavailable."""
    import recall.cli.contract as contract

    # Reset the warning flag
    original = contract._toon_warning_emitted
    contract._toon_warning_emitted = False

    stdout_captured = StringIO()
    stderr_captured = StringIO()

    def mock_echo(s, err=False, **_kwargs):
        if err:
            stderr_captured.write(str(s) + "\n")
        else:
            stdout_captured.write(str(s) + "\n")

    try:
        # Mock toon_format import to fail
        import builtins

        real_import = builtins.__import__

        def fail_toon_import(name, *args, **kwargs):
            if name == "toon_format":
                raise ImportError("mocked", name="toon_format")
            return real_import(name, *args, **kwargs)

        with (
            patch("typer.echo", side_effect=mock_echo),
            patch("builtins.__import__", side_effect=fail_toon_import),
        ):
            emit_data({"key": "value"}, output_format=OutputFormat.TOON)

        # Should have emitted JSON fallback on stdout
        payload = json.loads(stdout_captured.getvalue().strip())
        assert payload == {"key": "value"}

        # Should have emitted warning on stderr
        assert "toon-format not installed" in stderr_captured.getvalue()
    finally:
        contract._toon_warning_emitted = original


def test_fields_projection_works_with_toon() -> None:
    """Verify --fields projection works with TOON output."""
    from toon_format import decode as toon_decode

    captured = StringIO()
    with patch("typer.echo", side_effect=lambda s, **_: captured.write(str(s) + "\n")):
        emit_data(
            [{"id": "abc", "source": "claude_code", "extra": "ignored"}],
            output_format=OutputFormat.TOON,
            fields=("id", "source"),
            allowed_fields={"id", "source", "extra"},
        )

    decoded = toon_decode(captured.getvalue())
    assert isinstance(decoded, list)
    assert sorted(decoded[0].keys()) == ["id", "source"]
