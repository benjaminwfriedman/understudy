from codecarbon import EmissionsTracker, OfflineEmissionsTracker
from typing import Optional, Dict
from datetime import datetime
from app.models import CarbonEmission
from app.models.database import AsyncSessionLocal
from app.core.config import settings
import logging
import os

logger = logging.getLogger(__name__)


class CarbonTracker:
    """Wrapper around CodeCarbon for tracking training and inference emissions."""
    
    def __init__(
        self,
        project_name: str,
        task_type: str = "training",  # "training" or "inference"
        country_iso_code: Optional[str] = None,
        output_dir: Optional[str] = None,
        region: Optional[str] = None  # For cloud providers
    ):
        self.project_name = project_name
        self.task_type = task_type
        self.country_iso_code = country_iso_code or settings.COUNTRY_ISO_CODE
        self.output_dir = output_dir or settings.CARBON_DATA_DIR
        self.region = region or os.getenv("AZURE_CARBON_REGION")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.tracker = None
        self.start_time = None
        self.emissions_data = {}
    
    def start(self):
        """Start tracking emissions."""
        try:
            # Use OfflineEmissionsTracker with country_iso_code
            # or EmissionsTracker without it (auto-detects location)
            if self.country_iso_code:
                self.tracker = OfflineEmissionsTracker(
                    project_name=f"{self.project_name}_{self.task_type}",
                    output_dir=self.output_dir,
                    save_to_file=True,
                    country_iso_code=self.country_iso_code,
                    log_level="warning",
                    measure_power_secs=10  # Measure power every 10 seconds
                )
            else:
                self.tracker = EmissionsTracker(
                    project_name=f"{self.project_name}_{self.task_type}",
                    output_dir=self.output_dir,
                    save_to_file=True,
                    log_level="warning",
                    measure_power_secs=10  # Measure power every 10 seconds
                )
            self.tracker.start()
            self.start_time = datetime.utcnow()
            logger.info(f"Started carbon tracking for {self.project_name}")
        except Exception as e:
            logger.error(f"Failed to start carbon tracking: {e}")
            # Continue without carbon tracking rather than failing
            self.tracker = None
    
    def stop(self) -> Dict[str, float]:
        """Stop tracking and return emissions data."""
        if not self.tracker:
            return {
                "emissions_kg": 0.0,
                "energy_consumed_kwh": 0.0,
                "duration_seconds": 0.0,
                "cpu_power_w": 0.0,
                "gpu_power_w": 0.0,
            }
        
        try:
            emissions_kg = self.tracker.stop()
            
            # Get detailed emissions data
            if hasattr(self.tracker, 'final_emissions_data'):
                data = self.tracker.final_emissions_data
                self.emissions_data = {
                    "emissions_kg": emissions_kg or 0.0,
                    "energy_consumed_kwh": getattr(data, 'energy_consumed', 0.0),
                    "duration_seconds": getattr(data, 'duration', 0.0),
                    "cpu_power_w": getattr(data, 'cpu_power', 0.0),
                    "gpu_power_w": getattr(data, 'gpu_power', 0.0),
                    "ram_power_w": getattr(data, 'ram_power', 0.0),
                    "cpu_energy_kwh": getattr(data, 'cpu_energy', 0.0),
                    "gpu_energy_kwh": getattr(data, 'gpu_energy', 0.0),
                    "ram_energy_kwh": getattr(data, 'ram_energy', 0.0),
                }
            else:
                # Fallback if detailed data not available
                duration = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0
                self.emissions_data = {
                    "emissions_kg": emissions_kg or 0.0,
                    "energy_consumed_kwh": 0.0,
                    "duration_seconds": duration,
                    "cpu_power_w": 0.0,
                    "gpu_power_w": 0.0,
                }
            
            logger.info(f"Carbon tracking stopped. Emissions: {self.emissions_data['emissions_kg']:.6f} kg CO2")
            return self.emissions_data
            
        except Exception as e:
            logger.error(f"Failed to stop carbon tracking: {e}")
            return {
                "emissions_kg": 0.0,
                "energy_consumed_kwh": 0.0,
                "duration_seconds": 0.0,
                "cpu_power_w": 0.0,
                "gpu_power_w": 0.0,
            }
    
    async def save_to_db(
        self,
        training_run_id: Optional[str] = None,
        inference_log_id: Optional[str] = None,
        remote_emissions_data: Optional[Dict] = None
    ):
        """Save emissions data to database."""
        # Use remote emissions data if provided (from Azure GPU)
        if remote_emissions_data:
            self.emissions_data = remote_emissions_data
        elif not self.emissions_data:
            self.stop()
        
        async with AsyncSessionLocal() as session:
            carbon_record = CarbonEmission(
                training_run_id=training_run_id,
                inference_log_id=inference_log_id,
                timestamp=datetime.utcnow(),
                duration_seconds=self.emissions_data.get("duration_seconds", 0),
                emissions_kg=self.emissions_data.get("emissions_kg", 0),
                energy_consumed_kwh=self.emissions_data.get("energy_consumed_kwh", 0),
                cpu_power_w=self.emissions_data.get("cpu_power_w", 0),
                gpu_power_w=self.emissions_data.get("gpu_power_w", 0),
                country_iso_code=self.country_iso_code,
                region=self.region if self.region else None,
                gpu_type=self.emissions_data.get("gpu_type", None)
            )
            session.add(carbon_record)
            await session.commit()
            
            logger.info(f"Saved carbon emissions to database: {self.emissions_data.get('emissions_kg', 0):.6f} kg CO2")
            return self.emissions_data
    
    def get_emissions_data(self) -> Dict[str, float]:
        """Get current emissions data without stopping tracker."""
        return self.emissions_data.copy()


class InferenceCarbonTracker:
    """Lightweight carbon tracker for inference operations."""
    
    @staticmethod
    async def track_inference(func, *args, **kwargs):
        """Decorator to track carbon emissions for a single inference."""
        if not settings.CARBON_TRACKING_ENABLED:
            return await func(*args, **kwargs)
        
        tracker = OfflineEmissionsTracker(
            country_iso_code=settings.COUNTRY_ISO_CODE,
            log_level="error"
        )
        
        tracker.start()
        try:
            result = await func(*args, **kwargs)
            emissions = tracker.stop()
            
            if isinstance(result, dict):
                result["carbon_emissions_kg"] = emissions
            
            return result
        except Exception as e:
            tracker.stop()
            raise e