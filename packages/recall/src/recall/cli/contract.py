from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from enum import Enum, StrEnum
from typing import Any

import typer
from pydantic import BaseModel


class OutputFormat(StrEnum):
    AUTO = "auto"
    TEXT = "text"
    JSON = "json"
    JSONL = "jsonl"
    TOON = "toon"


class ErrorCode(StrEnum):
    VALIDATION = "VALIDATION"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    LOCKED = "LOCKED"
    NOT_FOUND = "NOT_FOUND"
    RUNTIME = "RUNTIME"


@dataclass(frozen=True)
class CliError(Exception):
    code: ErrorCode
    message: str
    details: dict[str, Any] | None = None
    exit_code: int = 1


# typer >= 0.22 vendors click as typer._click, so ctx.get_parameter_source
# returns members of a ParameterSource enum class that is NOT the one
# importable from standalone click. Identity-based set membership silently
# fails across that boundary — every explicit CLI flag reads as defaulted and
# --params payload values override explicit flags — so explicitness is
# compared by enum member name instead of enum identity.
_EXPLICIT_SOURCE_NAMES = {"COMMANDLINE", "ENVIRONMENT", "PROMPT"}


def is_explicit_source(ctx: typer.Context, name: str) -> bool:
    """True when the parameter was supplied explicitly (CLI, env, or prompt)."""
    source = ctx.get_parameter_source(name)
    return source is not None and getattr(source, "name", None) in _EXPLICIT_SOURCE_NAMES


def parse_output_format(value: str | None) -> OutputFormat:
    if value is None:
        return OutputFormat.AUTO
    normalized = value.strip().lower()
    match normalized:
        case "auto":
            return OutputFormat.AUTO
        case "text":
            return OutputFormat.TEXT
        case "json":
            return OutputFormat.JSON
        case "jsonl" | "ndjson":
            return OutputFormat.JSONL
        case "toon":
            return OutputFormat.TOON
        case _:
            raise ValueError(f"unsupported output format: {value}")


def resolve_output_format(*, json_output: bool, format_name: str | None) -> OutputFormat:
    if json_output:
        return OutputFormat.JSON
    requested = parse_output_format(format_name)
    if requested != OutputFormat.AUTO:
        return requested
    # REQ-CLI-013: non-TTY defaults to TOON for token-efficient agent output
    return OutputFormat.TEXT if sys.stdout.isatty() else OutputFormat.TOON


def resolve_bool_param(
    ctx: typer.Context,
    payload: dict[str, Any],
    name: str,
    current: bool,
) -> bool:
    """Resolve a boolean param from payload with strict type validation.

    Rejects truthy strings like "false" that bool() would coerce to True.
    """
    if is_explicit_source(ctx, name):
        return current
    value = payload.get(name, current)
    if not isinstance(value, bool):
        raise ValueError(
            f"--params field '{name}' must be a boolean (true/false), got {type(value).__name__}"
        )
    return value


def resolve_int_param(
    ctx: typer.Context,
    payload: dict[str, Any],
    name: str,
    current: int,
) -> int:
    """Resolve an integer param from payload with strict type validation."""
    if is_explicit_source(ctx, name):
        return current
    value = payload.get(name, current)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"--params field '{name}' must be an integer, got {type(value).__name__}")
    return value


def resolve_optional_int_param(
    ctx: typer.Context,
    payload: dict[str, Any],
    name: str,
    current: int | None,
) -> int | None:
    """Resolve an optional integer param from payload with strict type validation."""
    if is_explicit_source(ctx, name):
        return current
    value = payload.get(name, current)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            f"--params field '{name}' must be an integer or null, got {type(value).__name__}"
        )
    return value


def resolve_optional_float_param(
    ctx: typer.Context,
    payload: dict[str, Any],
    name: str,
    current: float | None,
) -> float | None:
    """Resolve an optional number param from payload with strict type validation.

    Accepts an integer as a number -- JSON has one numeric type, and a deadline
    written `5` means five seconds, not a type error.
    """
    if is_explicit_source(ctx, name):
        return current
    value = payload.get(name, current)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"--params field '{name}' must be a number or null, got {type(value).__name__}"
        )
    return float(value)


def resolve_optional_bool_param(
    ctx: typer.Context,
    payload: dict[str, Any],
    name: str,
    current: bool | None,
) -> bool | None:
    """Resolve an optional boolean param (true/false/null) with strict type validation."""
    if is_explicit_source(ctx, name):
        return current
    value = payload.get(name, current)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(
            f"--params field '{name}' must be a boolean or null, got {type(value).__name__}"
        )
    return value


def _require_bool(payload: dict[str, Any], key: str) -> bool:
    """Validate that a --params value is a true boolean, not a truthy string."""
    value = payload[key]
    if not isinstance(value, bool):
        raise ValueError(
            f"--params field '{key}' must be a boolean (true/false), got {type(value).__name__}"
        )
    return value


def _require_str_or_none(payload: dict[str, Any], key: str) -> str | None:
    """Validate that a --params value is a string or null."""
    value = payload[key]
    if value is not None and not isinstance(value, str):
        raise ValueError(f"--params field '{key}' must be a string, got {type(value).__name__}")
    return value


def resolve_output_format_from_params(
    ctx: typer.Context,
    payload: dict[str, Any],
    *,
    json_output: bool,
    format_name: str | None,
) -> OutputFormat:
    """Resolve output format, honoring --params overrides per REQ-CLI-009.

    Explicit CLI flags (--json, --format) take precedence over --params values.
    Payload keys use the public names "format" and "json".
    """
    if "json" in payload and not is_explicit_source(ctx, "json_output"):
        json_output = _require_bool(payload, "json")
    if "format" in payload and not is_explicit_source(ctx, "format_name"):
        format_name = _require_str_or_none(payload, "format")
    return resolve_output_format(json_output=json_output, format_name=format_name)


def resolve_output_format_early(
    ctx: typer.Context,
    *,
    json_output: bool,
    format_name: str | None,
    raw_params: str | None,
) -> OutputFormat:
    """Resolve output format considering --params before full payload validation.

    This ensures that error responses from parse_params use the correct output
    format when --params contains format/json overrides (REQ-CLI-009 + REQ-CLI-005).
    """
    if raw_params is not None:
        try:
            peek = json.loads(raw_params)
        except (json.JSONDecodeError, TypeError):
            peek = {}
        if isinstance(peek, dict):
            if (
                "json" in peek
                and isinstance(peek["json"], bool)
                and not is_explicit_source(ctx, "json_output")
            ):
                json_output = peek["json"]
            if (
                "format" in peek
                and isinstance(peek["format"], str)
                and not is_explicit_source(ctx, "format_name")
            ):
                format_name = peek["format"]
    return resolve_output_format(json_output=json_output, format_name=format_name)


def parse_fields(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"fields must be a comma-separated string, got {type(value).__name__}")
    fields = tuple(part.strip() for part in value.split(",") if part.strip())
    if not fields:
        raise ValueError("fields must not be empty")
    return fields


# Valid source values accepted by the CLI (both kebab-case and underscore forms)
VALID_SOURCES = {
    "claude-code",
    "claude_code",
    "codex",
    "pi-agent",
    "pi_agent",
    "pi",
    "grok",
    "grok-build",
    "grok_build",
    "kimi",
    "kimi-code",
    "kimi_code",
}


def validate_source(value: str | None) -> None:
    """Validate source against accepted values, if provided."""
    if value is not None and value not in VALID_SOURCES:
        raise ValueError(f"unsupported source: {value}")


# Maps manifest type names to allowed Python types for --params validation.
# "string" allows str or None; "boolean" allows bool only (not int); etc.
_PARAM_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "boolean": (bool,),
    "integer": (int,),
    "number": (int, float),
    "json": (str,),
    "path": (str,),
    "string[]": (list,),
}


def parse_params(
    raw: str | None,
    *,
    allowed: set[str],
    types: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Parse and validate a --params JSON payload.

    Args:
        raw: The raw JSON string from --params, or None.
        allowed: Set of allowed field names.
        types: Optional mapping of field name to manifest type name
               (e.g. {"limit": "integer", "source": "string"}).
               When provided, payload values are validated against these types.
    """
    if raw is None:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as err:
        raise ValueError(f"invalid JSON for --params: {err.msg}") from err
    if not isinstance(payload, dict):
        raise ValueError("--params must decode to a JSON object")
    # JSON object keys are always strings; make that explicit so the sorted()
    # argument is a comparable set[str] rather than set[object].
    unknown = sorted({str(key) for key in payload} - allowed)
    if unknown:
        raise ValueError(f"unsupported params fields: {', '.join(unknown)}")
    if types:
        _validate_param_types(payload, types)
    return payload


def _validate_param_types(payload: dict[str, Any], types: dict[str, str]) -> None:
    """Validate payload values against declared manifest types."""
    for key, value in payload.items():
        if value is None:
            continue  # null is acceptable for optional fields
        expected_type_name = types.get(key)
        if expected_type_name is None:
            continue  # no type constraint declared (e.g. output-control fields)
        allowed_types = _PARAM_TYPE_CHECKS.get(expected_type_name)
        if allowed_types is None:
            continue  # unknown type name, skip validation
        # bool is a subclass of int in Python; reject bools for integer fields
        if expected_type_name == "integer" and isinstance(value, bool):
            raise ValueError(f"--params field '{key}' must be an integer, got boolean")
        if expected_type_name == "number" and isinstance(value, bool):
            raise ValueError(f"--params field '{key}' must be a number, got boolean")
        if not isinstance(value, allowed_types):
            raise ValueError(
                f"--params field '{key}' must be {expected_type_name}, got {type(value).__name__}"
            )
        if expected_type_name == "string[]" and any(not isinstance(item, str) for item in value):
            raise ValueError(f"--params field '{key}' must contain only strings")


def resolve_param(
    ctx: typer.Context,
    payload: dict[str, Any],
    name: str,
    current: Any,
) -> Any:
    if is_explicit_source(ctx, name):
        return current
    return payload.get(name, current)


def emit_data(
    data: Any,
    *,
    output_format: OutputFormat,
    fields: tuple[str, ...] | None = None,
    allowed_fields: set[str] | None = None,
) -> None:
    serialized = serialize_data(data)
    if fields is not None:
        serialized = project_fields(serialized, fields=fields, allowed_fields=allowed_fields)

    if output_format == OutputFormat.JSON:
        typer.echo(json.dumps(serialized, separators=(",", ":")))
        return

    if output_format == OutputFormat.JSONL:
        if isinstance(serialized, list):
            for item in serialized:
                typer.echo(json.dumps(item, separators=(",", ":")))
            return
        typer.echo(json.dumps(serialized, separators=(",", ":")))
        return

    if output_format == OutputFormat.TOON:
        _emit_toon(serialized)
        return

    raise RuntimeError("text output must be rendered by the command")


def emit_error(error: CliError, *, output_format: OutputFormat) -> None:
    if output_format == OutputFormat.TEXT:
        typer.echo(f"error: {error.message}", err=True)
        return
    error_payload: dict[str, Any] = {
        "code": error.code.value,
        "message": error.message,
    }
    if error.details:
        error_payload["details"] = error.details
    payload: dict[str, Any] = {"error": error_payload}
    if output_format == OutputFormat.TOON:
        _emit_toon(payload)
        return
    typer.echo(json.dumps(payload, separators=(",", ":")))


DRY_RUN_FIELDS = {"dry_run", "command", "request", "safety"}


def render_dry_run(
    *,
    command: str,
    request: dict[str, Any],
    safety: dict[str, Any],
    output_format: OutputFormat,
    fields: tuple[str, ...] | None = None,
    ctas: list[Cta] | None = None,
    include_cta: bool = False,
) -> None:
    payload = {
        "dry_run": True,
        "command": command,
        "request": request,
        "safety": safety,
    }
    if output_format == OutputFormat.TEXT:
        typer.echo(json.dumps(serialize_data(payload), indent=2))
        if ctas:
            render_cta_hints(ctas)
        return
    emit_data_with_cta(
        payload,
        ctas or [],
        output_format=output_format,
        include_cta=include_cta,
        fields=fields,
        allowed_fields=DRY_RUN_FIELDS,
    )


def require_confirmation(
    *,
    should_confirm: bool,
    confirmed: bool,
    dry_run: bool,
    message: str,
) -> None:
    if not should_confirm or confirmed or dry_run:
        return
    if sys.stdin.isatty():
        if typer.confirm(message):
            return
        raise CliError(
            code=ErrorCode.CONFIRMATION_REQUIRED,
            message="operation cancelled",
            exit_code=2,
        )
    raise CliError(
        code=ErrorCode.CONFIRMATION_REQUIRED,
        message=f"{message} Pass --yes or use --dry-run.",
        exit_code=2,
    )


def validate_structured_fields(
    *,
    output_format: OutputFormat,
    fields: tuple[str, ...] | None,
) -> None:
    if fields is None:
        return
    if output_format == OutputFormat.TEXT:
        raise CliError(
            code=ErrorCode.VALIDATION,
            message=(
                "--fields requires structured output;"
                " use --format json, --format jsonl, or --format toon"
            ),
            exit_code=2,
        )


def serialize_data(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return serialize_data(value.model_dump(mode="python"))
    if is_dataclass(value):
        return serialize_data(asdict(value))
    if isinstance(value, dict):
        return {str(key): serialize_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_data(item) for item in value]
    return str(value)


def project_fields(
    value: Any,
    *,
    fields: tuple[str, ...],
    allowed_fields: set[str] | None,
) -> Any:
    if allowed_fields is not None:
        unknown = sorted(set(fields) - allowed_fields)
        if unknown:
            raise CliError(
                code=ErrorCode.VALIDATION,
                message=f"unknown output fields: {', '.join(unknown)}",
                details={"fields": unknown},
                exit_code=2,
            )
    if isinstance(value, list):
        return [
            project_fields(item, fields=fields, allowed_fields=allowed_fields) for item in value
        ]
    if not isinstance(value, dict):
        raise CliError(
            code=ErrorCode.VALIDATION,
            message="--fields is only supported for object and array outputs",
            exit_code=2,
        )
    return {field: value[field] for field in fields if field in value}


# -- CTA (call-to-action) support --


@dataclass(frozen=True)
class Cta:
    """A suggested next command for workflow continuity."""

    command: str
    description: str


_CTA_ENVELOPE_FIELDS = {"data", "cta"}


def emit_data_with_cta(
    data: Any,
    ctas: list[Cta],
    *,
    output_format: OutputFormat,
    include_cta: bool,
    fields: tuple[str, ...] | None = None,
    allowed_fields: set[str] | None = None,
) -> None:
    """Emit structured data, optionally wrapped in a CTA envelope.

    When include_cta is True, wraps the response in
    {"data": <original>, "cta": [...]} per REQ-CLI-015.

    Field projection behavior with --cta:
    - If --fields contains envelope keys (data, cta), project the envelope.
    - Otherwise, project the data payload inside the envelope.
    """
    serialized = serialize_data(data)

    if include_cta:
        # Build envelope first, then decide where to apply --fields
        envelope: dict[str, Any] = {
            "data": serialized,
            "cta": [serialize_data(cta) for cta in ctas],
        }
        if fields is not None:
            requested = set(fields)
            if requested & _CTA_ENVELOPE_FIELDS:
                # Projecting envelope-level fields (data, cta)
                envelope = project_fields(
                    envelope,
                    fields=fields,
                    allowed_fields=_CTA_ENVELOPE_FIELDS,
                )
            else:
                # Projecting data-level fields inside the envelope
                envelope["data"] = project_fields(
                    serialized,
                    fields=fields,
                    allowed_fields=allowed_fields,
                )
        _emit_serialized(envelope, output_format=output_format)
    else:
        if fields is not None:
            serialized = project_fields(
                serialized,
                fields=fields,
                allowed_fields=allowed_fields,
            )
        _emit_serialized(serialized, output_format=output_format)


def render_cta_hints(ctas: list[Cta]) -> None:
    """Render CTA suggestions on stderr for TEXT mode."""
    if not ctas:
        return
    typer.echo("", err=True)
    for cta in ctas[:3]:
        typer.echo(f"  Next: {cta.command} -- {cta.description}", err=True)


def _emit_serialized(data: Any, *, output_format: OutputFormat) -> None:
    """Low-level serialized data emission for a specific format."""
    if output_format == OutputFormat.JSON:
        typer.echo(json.dumps(data, separators=(",", ":")))
    elif output_format == OutputFormat.JSONL:
        if isinstance(data, list):
            for item in data:
                typer.echo(json.dumps(item, separators=(",", ":")))
        else:
            typer.echo(json.dumps(data, separators=(",", ":")))
    elif output_format == OutputFormat.TOON:
        _emit_toon(data)
    else:
        raise RuntimeError(f"unsupported structured format for CTA emission: {output_format}")


# -- TOON encoding helpers --

# REQ-CLI-014: one-time warning when toon-format is unavailable
_toon_warning_emitted = False


def _emit_toon(data: Any) -> None:
    """Encode data as TOON, falling back to JSON if toon-format is unavailable."""
    global _toon_warning_emitted
    try:
        from toon_format import encode as toon_encode
    except ImportError as err:
        # Only fall back for a genuinely missing module, not a broken install
        if err.name != "toon_format":
            raise
        if not _toon_warning_emitted:
            typer.echo(
                "warning: toon-format not installed, falling back to JSON output",
                err=True,
            )
            _toon_warning_emitted = True
        typer.echo(json.dumps(data, separators=(",", ":")))
        return
    typer.echo(toon_encode(data))
