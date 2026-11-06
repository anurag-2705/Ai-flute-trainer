"""Save lesson results so progress can be tracked over days."""
import csv
import time

import config


def save_results(lesson_name, results):
    config.DATA_DIR.mkdir(exist_ok=True)
    new = not config.PROGRESS_CSV.exists()
    with open(config.PROGRESS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["when", "lesson", "note", "seconds_to_succeed", "mean_abs_cents"])
        stamp = time.strftime("%Y-%m-%d %H:%M")
        for r in results:
            w.writerow([stamp, lesson_name, r.label, f"{r.seconds_to_succeed:.1f}",
                        f"{r.mean_abs_cents:.1f}"])


def history():
    """Return a list of dict rows, or [] if nothing saved yet."""
    if not config.PROGRESS_CSV.exists():
        return []
    with open(config.PROGRESS_CSV, newline="") as f:
        return list(csv.DictReader(f))
