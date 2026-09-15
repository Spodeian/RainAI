"""
Production-grade logging and telemetry system for RainAI.
Provides dual-destination logging (console + rotating disk log file),
structured metric tracking, GPU memory monitoring, and telemetry export.
"""

import os
import sys
import time
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

try:
    import torch
except ImportError:
    torch = None


class PlainFormatter(logging.Formatter):
    """Clean plain-text formatter for file logging with ISO timestamps."""
    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        return f"{timestamp} [{record.levelname:<7}] [{record.name}] {record.getMessage()}"


class ConsoleFormatter(logging.Formatter):
    """Console formatter with clean ASCII badges for cross-platform terminals."""
    BADGES = {
        logging.DEBUG: "[DEBUG]",
        logging.INFO: "[+]",
        logging.WARNING: "[!]",
        logging.ERROR: "[ERROR]",
        logging.CRITICAL: "[FATAL]",
    }

    def format(self, record):
        badge = self.BADGES.get(record.levelno, "[*]")
        return f"{badge} {record.getMessage()}"


class RainAILogger:
    """Wrapper providing structured logging, telemetry, and GPU resource tracking."""

    def __init__(self, logger: logging.Logger, log_file: Optional[Path] = None):
        self._logger = logger
        self.log_file = log_file
        self.metrics_history = []
        self.start_time = time.time()

    @property
    def name(self):
        return self._logger.name

    def info(self, msg: str, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def stage(self, current: int, total: int, title: str):
        banner = f"[STAGE {current}/{total}] {title}"
        bar = "-" * max(len(banner), 50)
        self._logger.info(f"\n{banner}\n{bar}")

    def banner(self, title: str, subtitle: Optional[str] = None):
        width = 80
        self._logger.info("=" * width)
        self._logger.info(f" {title}")
        if subtitle:
            self._logger.info(f" {subtitle}")
        self._logger.info("=" * width)

    def get_gpu_memory_mb(self) -> Dict[str, float]:
        """Returns allocated and reserved CUDA VRAM in MB."""
        if torch and torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024 * 1024)
            res = torch.cuda.memory_reserved() / (1024 * 1024)
            peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            return {"allocated_mb": round(alloc, 2), "reserved_mb": round(res, 2), "peak_mb": round(peak, 2)}
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}

    def log_metric(self, step: int, epoch: int, metrics: Dict[str, Any], prefix: str = "train"):
        """Record structured metrics with timestamp and hardware telemetry."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "step": step,
            "prefix": prefix,
            "metrics": {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in metrics.items()},
            "gpu_memory": self.get_gpu_memory_mb(),
            "elapsed_seconds": round(time.time() - self.start_time, 2)
        }
        self.metrics_history.append(record)
        self._logger.debug(f"Telemetry metric logged: {record}")

    def save_run_summary(self, summary_path: Path, extra_metadata: Optional[Dict[str, Any]] = None):
        """Persist structured telemetry and metric curves as JSON."""
        summary = {
            "run_id": self._logger.name,
            "completed_at": datetime.now().isoformat(),
            "total_elapsed_seconds": round(time.time() - self.start_time, 2),
            "gpu_telemetry": self.get_gpu_memory_mb(),
            "metadata": extra_metadata or {},
            "metrics_count": len(self.metrics_history),
            "metrics_history": self.metrics_history,
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        self._logger.info(f"Saved run summary and telemetry to {summary_path}")


def setup_logger(
    name: str = "rainai",
    log_dir: Optional[Path] = None,
    log_filename: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024, # 10 MB
    backup_count: int = 5
) -> RainAILogger:
    """
    Initialize and return a production-grade RainAILogger.
    
    Args:
        name: Logger identifier
        log_dir: Path to directory for persistent log files (default: PROJECT_ROOT/logs)
        log_filename: Custom filename for the log file (default: {name}_{timestamp}.log)
        console_level: Logging level for stdout
        file_level: Logging level for log file
        max_bytes: Max file size before rotating
        backup_count: Number of rotated backups to keep
    """
    base_logger = logging.getLogger(name)
    base_logger.setLevel(min(console_level, file_level))
    base_logger.handlers.clear()
    base_logger.propagate = False

    # Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(console_level)
    c_handler.setFormatter(ConsoleFormatter())
    base_logger.addHandler(c_handler)

    log_filepath = None
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if not log_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{name}_{timestamp}.log"
        
        log_filepath = log_dir / log_filename
        f_handler = RotatingFileHandler(
            str(log_filepath),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        f_handler.setLevel(file_level)
        f_handler.setFormatter(PlainFormatter())
        base_logger.addHandler(f_handler)

    return RainAILogger(base_logger, log_filepath)


def get_logger(name: str = "rainai") -> logging.Logger:
    """Retrieve an existing logger or root logger."""
    return logging.getLogger(name)
