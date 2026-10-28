from pathlib import Path

import pytest
from recall.core.index_request import resolve_index_request
from recall.core.rpc_params import IndexParams


@pytest.mark.parametrize("root", ["~", "~/existing-root"])
def test_index_rejects_unavailable_current_user_home(
    monkeypatch: pytest.MonkeyPatch, root: str
) -> None:
    def unavailable_home(path: Path) -> Path:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "expanduser", unavailable_home)

    with pytest.raises(ValueError) as error:
        resolve_index_request(IndexParams(root=root))

    assert str(error.value) == (f"cannot expand root: {root}: Could not determine home directory.")
