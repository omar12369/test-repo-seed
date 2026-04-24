# runtime/services/scheduler_service.py
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional, Union


class AsyncScheduler:
    """
    Minimal background scheduler that fires a 'tick' every `interval` seconds.

    Public API (stable):
      - start(interval: int = 60) -> None
      - await stop() -> None
      - running: bool (property)
      - interval: int
      - ticks: int

    Optional metadata (useful for /debug endpoints):
      - env: Optional[str]
      - data_dir: Optional[str]
    """

    def __init__(
        self,
        logger=None,
        *,
        env: Optional[str] = None,
        data_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.logger = logger
        self.env: Optional[str] = env
        self.data_dir: Optional[str] = str(data_dir) if data_dir is not None else None

        self.interval: int = 60
        self.ticks: int = 0

        self._task: Optional[asyncio.Task] = None
        self._running: bool = False

    # ------------------------ lifecycle ------------------------

    def start(self, interval: int = 60) -> None:
        """
        Start the background scheduler loop.

        Safe to call multiple times; subsequent calls are ignored
        while the scheduler is already running.
        """
        if self._running:
            return

        self.interval = max(1, int(interval))
        self._running = True

        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._run())

        if self.logger:
            self.logger.info("Scheduler background task started (interval=%ss)", self.interval)

    async def stop(self) -> None:
        """
        Stop the background scheduler loop and await its completion.
        Safe to call when not running.
        """
        if not self._running:
            return

        self._running = False

        if self._task:
            try:
                await self._task
            finally:
                self._task = None

        if self.logger:
            self.logger.info("Scheduler background task stopped")

    @property
    def running(self) -> bool:
        """True if the scheduler loop is running."""
        return self._running and self._task is not None and not self._task.done()

    # ------------------------ internals ------------------------

    async def _run(self) -> None:
        if self.logger:
            self.logger.info("Scheduler loop entered")

        try:
            while self._running:
                await asyncio.sleep(self.interval)
                self.ticks += 1
                if self.logger:
                    self.logger.info("Scheduler tick %s", self.ticks)
        except asyncio.CancelledError:
            # Cooperative cancellation
            if self.logger:
                self.logger.info("Scheduler loop cancelled")
            raise
        except Exception as e:  # pragma: no cover (best-effort logging)
            if self.logger:
                self.logger.warning("Scheduler loop error: %s", e)
        finally:
            if self.logger:
                self.logger.info("Scheduler loop exited")
