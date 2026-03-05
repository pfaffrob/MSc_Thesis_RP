"""
Field site path definitions.
Pre-built path objects for each research site with standardized directory structures.
Available sites: ESK, KAU, KEN, BUS, HAM, WAI
"""

from .build_paths import build_site
from tqdm import tqdm

# Build path objects for each field site
ESK = build_site('limited_extent', 'A_eskdale', key='ESK')
KAU = build_site('limited_extent', 'B_kauri_glen', key='KAU')
KEN = build_site('limited_extent', 'C_kendal_bay', key='KEN')
BUS = build_site('limited_extent', 'D_bushglen', key='BUS')
HAM = build_site('limited_extent', 'E_hammond_park', key='HAM')
WAI = build_site('limited_extent', 'F_waiwhakereke', key='WAI')


RESERVES = {'ESK': ESK, 'KAU': KAU, 'KEN': KEN, 'BUS': BUS, 'HAM': HAM, 'WAI': WAI}

def use(keys=None, progress=True):
    """
    Get list of site objects for processing.
    
    Args:
        keys: List of site keys to use (default: all sites)
        progress: Whether to show progress bar (currently unused)
        
    Returns:
        List of site objects
    """
    if keys is None:
        keys = list(RESERVES.keys())
    iterable = [RESERVES[k] for k in keys]
    return iterable