"""A Linux observer whose recursive watches also cover directories moved in.

watchdog's inotify backend arms a watch for a directory created under a
recursive watch, but not for one moved in from outside it: the move is reported
as a created directory and its files as created files, and then nothing under
it raises an event again (watchdog 6.0.0, the TODO in `Inotify.read_events`).
A transcript created there would wait for the discovery sweep (REQ-DAEMON-078).

Import only where `watchdog.observers.Observer` is the inotify observer.
"""

from __future__ import annotations

import errno
import logging
import os

from watchdog.events import DirCreatedEvent, FileSystemEvent
from watchdog.observers.api import BaseObserver
from watchdog.observers.inotify import InotifyEmitter

logger = logging.getLogger(__name__)


class _SubtreeInotifyEmitter(InotifyEmitter):
    """Arms each directory that arrives under a recursive watch as it is reported.

    A moved-in directory's subdirectories are reported one by one, each before
    its own files are listed, so a file created before its directory's watch
    exists is still reported by that listing.
    """

    def queue_event(self, event: FileSystemEvent) -> None:
        if self.watch.is_recursive and isinstance(event, DirCreatedEvent):
            self._watch_directory(os.fsencode(event.src_path))
        super().queue_event(event)

    def _watch_directory(self, path: bytes) -> None:
        buffer = self._inotify
        # A symlink to a directory is listed with the directories; watching it
        # would watch its target under another name, which watchdog never does.
        if buffer is None or os.path.islink(path):
            return
        try:
            # WAIVER(no private watchdog API): `InotifyBuffer` exposes no way to
            # add a watch to its inotify instance, so this reaches its `_inotify`.
            # `add_watch` is idempotent for a directory watchdog armed itself.
            # Debt: removed when watchdog watches moved-in directories (the TODO
            # in `Inotify.read_events`); `uv.lock` pins the watchdog it was read from.
            buffer._inotify.add_watch(path)
        except OSError as err:
            if err.errno in (errno.ENOENT, errno.ENOTDIR):
                # Gone again before it could be watched: nothing there to report.
                return
            logger.warning(
                "failed to watch directory path=%s reason=%s; "
                "new transcripts there wait for the discovery sweep",
                os.fsdecode(path),
                err,
            )


class SubtreeInotifyObserver(BaseObserver):
    """watchdog's inotify observer, with moved-in directories watched."""

    def __init__(self) -> None:
        super().__init__(_SubtreeInotifyEmitter)
