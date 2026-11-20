"""Interrupt actual historical migration bodies and audit effects inside observation."""

import sys
from unittest.mock import patch

from recall.core.config import AppConfig
from recall.db.connection import connect_owned
from recall.db.lease import owned_connection
from recall.services.recovery_admission import offline_file_lifetime

from tests.repair_process import checkpoint


def main():
    selected, fault, mode = sys.argv[1:]
    config = AppConfig.load()
    with offline_file_lifetime(config) as (owner, admission):
        call = owner.call_db

        def observed(operation, work):
            def effect():
                checkpoint("before:" + operation, selected, fault)
                value = work()
                checkpoint("after:" + operation, selected, fault)
                return value

            return call(operation, effect)

        with (
            patch.object(owner, "call_db", observed),
            owned_connection(
                owner,
                "historical_upgrade",
                lambda: connect_owned(
                    config,
                    lifetime=owner,
                    admission=admission.normal,
                    lenient_schema=mode == "lenient",
                ),
            ),
        ):
            pass
    assert selected == "none", "historical migration boundary was not reached"


if __name__ == "__main__":
    main()
