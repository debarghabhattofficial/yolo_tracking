# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

__version__ = '10.0.51'

from boxmot.postprocessing.gsi import gsi
from boxmot.tracker_zoo import create_tracker, get_tracker_config
from boxmot.trackers.botsort.bot_sort import BoTSORT
from boxmot.trackers.bytetrack.byte_tracker import BYTETracker
from boxmot.trackers.deepocsort.deep_ocsort import DeepOCSort as DeepOCSORT
from boxmot.trackers.hybridsort.hybridsort import HybridSORT
from boxmot.trackers.ocsort.ocsort import OCSort as OCSORT
from boxmot.trackers.strongsort.strong_sort import StrongSORT

# Following statements import modified versions 
# of the original trackers which also use the
# depth information from the RGBD camera for tracking.
# Added by Debargha Bhattacharjee for the Shadow Mode project.
# =================================================================================
from boxmot.trackers.ocsort.ocsort_rgbd import OCSORTRGBD as OCSORTRGBD
from boxmot.trackers.ocsort.ocsort_rgbd_cmc import OCSORTRGBDCMC as OCSORTRGBDCMC
from boxmot.trackers.botsort.bot_sort_rgbd import BoTSORTRGBD as BoTSORTRGBD
# =================================================================================

TRACKERS = [
    "bytetrack", 
    "botsort",
    "botsort_rgbd",  # DEB
    "strongsort", 
    "ocsort", 
    "ocsort_rgbd",  # DEB
    "ocsort_rgbd_cmc",  # DEB
    "deepocsort", 
    "hybridsort"
]

__all__ = (
    "__version__",
    "StrongSORT", 
    "OCSORT", 
    "OCSORTRGBD",  # DEB
    "OCSORTRGBDCMC",  # DEB
    "BYTETracker", 
    "BoTSORT", 
    "BoTSORTRGBD",  # DEB
    "DeepOCSORT", 
    "HybridSORT",
    "create_tracker", 
    "get_tracker_config", 
    "gsi"
)
