import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Lightweight configuration manager for BBB-ASL.
    Centralizes JSON loading and prevents KeyError crashes.
    """

    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = {}

    def load(self):
        if not os.path.exists(self.config_path):
            logger.warning(f"{self.config_path} not found. Using empty configuration.")
            return {}

        try:
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            self.config = {}
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config = {}

        return self.config

    def get(self, *keys, default=None):
        """
        Safe nested access:
        config.get("physiological", "T1", default=1.6)
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value