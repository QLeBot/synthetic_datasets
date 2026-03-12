import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Import centralized config loader
import sys
_orchestrator_dir = Path(__file__).parent.parent.absolute()
if str(_orchestrator_dir) not in sys.path:
    sys.path.insert(0, str(_orchestrator_dir))
from orchestrator.core.config_loader import get_config

load_dotenv()

# Get centralized config
_config = get_config()

# Get the orchestrator directory (parent of extract directory)
_ORCHESTRATOR_DIR = Path(__file__).parent.parent.absolute()

@dataclass(frozen=True)
class Settings:
    """
    Settings class for backward compatibility.
    Now uses centralized config.yaml for most values.
    """
    sql_server: str = field(default_factory=lambda: _config.sql_server)
    sql_user: str | None = field(default_factory=lambda: _config.sql_user)
    sql_password: str | None = field(default_factory=lambda: _config.sql_password)

    local_export_dir: str = field(default_factory=lambda: _config.get_export_dir("daily"))
    local_bulk_export_dir: str = field(default_factory=lambda: _config.get_export_dir("bulk"))
    chunk_size: int = field(default_factory=lambda: _config.extraction.chunk_size)

def compute_window(days_back: Optional[int] = None):
    """
    Matches your current behavior:
    CURRENT_DATE = now - days_back days
    NEXT_DATE = CURRENT_DATE + 1 day
    
    Uses days_back from config.yaml if not provided.
    """
    if days_back is None:
        days_back = _config.daily.days_back
    current_date = datetime.now() - timedelta(days=days_back)
    next_date = current_date + timedelta(days=1)
    return current_date, next_date

def compute_bulk_window(period: str = "month"):
    """
    Compute date window for bulk extracts.
    
    Args:
        period: One of 'month', 'year', or 'full'
    
    Returns:
        Tuple of (start_date, end_date) or (None, None) for full table
    """
    now = datetime.now()
    
    if period == "month":
        # Last month (from first day of last month to first day of current month)
        if now.month == 1:
            start_date = datetime(now.year - 1, 12, 1)
        else:
            start_date = datetime(now.year, now.month - 1, 1)
        end_date = datetime(now.year, now.month, 1)
        return start_date, end_date
    
    elif period == "year":
        # Last year (from first day of last year to first day of current year)
        start_date = datetime(now.year - 1, 1, 1)
        end_date = datetime(now.year, 1, 1)
        return start_date, end_date
    
    elif period == "full":
        # Full table - no date filtering
        return None, None
    
    else:
        raise ValueError(f"Invalid period: {period}. Must be 'month', 'year', or 'full'")
