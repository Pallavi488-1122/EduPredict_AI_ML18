"""Optional scheduled retraining. Run: python -m src.scheduler"""
import os
from apscheduler.schedulers.blocking import BlockingScheduler
from src.train import main

hour = int(os.getenv("RETRAIN_HOUR", "2"))
minute = int(os.getenv("RETRAIN_MINUTE", "0"))

if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone="Asia/Kolkata")
    scheduler.add_job(main, "cron", hour=hour, minute=minute, id="daily_retraining")
    print(f"Daily retraining scheduled at {hour:02d}:{minute:02d} IST. Press Ctrl+C to stop.")
    scheduler.start()
