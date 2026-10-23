"""The responsive deadline hint is bounded metadata at the RPC boundary."""

import pytest
from pydantic import ValidationError
from recall.core.rpc_params import LiveParams, StatusParams


@pytest.mark.parametrize("schema", [LiveParams, StatusParams])
@pytest.mark.parametrize("value", [0, -1, 2001, True, False, 1.5, "1000", None, [], {}])
def test_remaining_duration_rejects_invalid_wire_values(schema, value):
    with pytest.raises(ValidationError):
        schema.model_validate({"remaining_read_ms": value})


@pytest.mark.parametrize("schema", [LiveParams, StatusParams])
def test_remaining_duration_accepts_endpoints_and_omission(schema):
    for value in (1, 2000):
        assert schema.model_validate({"remaining_read_ms": value}).remaining_read_ms == value
    assert schema.model_validate({}).remaining_read_ms == 2000
    # Existing peers send fields this build does not know yet.
    assert schema.model_validate({"future_hint": True}).remaining_read_ms == 2000
