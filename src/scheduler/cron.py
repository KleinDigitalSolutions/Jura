"""Weekly scheduler for legal document ingestion updates."""
import asyncio
import time
from datetime import datetime

import schedule
from loguru import logger

from src.config import CRON_DAY, CRON_TIME, LOG_DIR


class IngestionScheduler:
    """Schedule weekly ingestion runs. Only processes new/updated documents."""

    def __init__(self):
        self._running = False
        self._last_run: datetime | None = None
        self._job_func = None

    def set_job(self, job_func):
        """Set the async ingestion function to call on schedule."""
        self._job_func = job_func

    def _run_job(self):
        """Wrapper to run the async job function synchronously."""
        if not self._job_func:
            logger.warning("No job function set, skipping scheduled run")
            return

        start = time.monotonic()
        logger.info(f"Scheduled ingestion starting at {datetime.now().isoformat()}")
        try:
            asyncio.run(self._job_func())
            self._last_run = datetime.now()
            elapsed = time.monotonic() - start
            logger.info(f"Scheduled ingestion complete ({elapsed:.1f}s)")
        except Exception as e:
            logger.error(f"Scheduled ingestion failed: {e}")

    def start(self) -> None:
        """Start the scheduler. Runs weekly at configured day/time."""
        day = CRON_DAY.lower()
        time_str = CRON_TIME

        day_map = {
            "monday": schedule.every().monday,
            "tuesday": schedule.every().tuesday,
            "wednesday": schedule.every().wednesday,
            "thursday": schedule.every().thursday,
            "friday": schedule.every().friday,
            "saturday": schedule.every().saturday,
            "sunday": schedule.every().sunday,
        }

        job = day_map.get(day, schedule.every().saturday)
        job.at(time_str).do(self._run_job)

        self._running = True
        logger.info(f"Scheduler started: every {day} at {time_str}")

        # Run once immediately on start
        self._run_job()

        try:
            while self._running:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            self._running = False

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        schedule.clear()
        logger.info("Scheduler stopped")
