"""
Configuration module for environment variables and project-level settings.
Loads paths from .env file and defines coordinate reference system.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# Root data directory from environment variable
ONEDRIVE = os.getenv('ONEDRIVE')

# New Zealand crs
CRS = 2193

# Main data directory and temporary folder locations
DATA = os.path.join(ONEDRIVE, '3_0_data')
DOWNLOADS = os.getenv("DOWNLOADS")
TEMP = os.path.join(DOWNLOADS, 'temp')