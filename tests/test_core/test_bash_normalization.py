from __future__ import annotations

import pytest
from recall.services.embeddings import normalize_bash_command


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "kubectl logs web-6dbc8bdc58-8cghv -c web --previous",
            "kubectl logs <k8s-name> -c web --previous",
        ),
        (
            "git show 9f3c2a1b4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a",
            "git show <git-sha>",
        ),
        (
            "cat /var/folders/ab/cd/T/tmp-12345/build.log",
            "cat <tmp-path>/build.log",
        ),
        (
            "curl https://example.test/jobs/123456789",
            "curl https://example.test/jobs/<num>",
        ),
    ],
)
def test_normalize_bash_command_rewrites_high_cardinality_tokens(raw: str, expected: str) -> None:
    assert normalize_bash_command(raw) == expected
