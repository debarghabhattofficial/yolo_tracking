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
from boxmot.trackers.ocsort.ocsort_d import OCSORT_D as OCSORT_D
from boxmot.trackers.ocsort.ocsort_dt import OCSORT_DT as OCSORT_DT
from boxmot.trackers.ocsort.ocsort_dtc import OCSORT_DTC as OCSORT_DTC
from boxmot.trackers.ocsort.ocsort_dtc_b import OCSORT_DTC_B as OCSORT_DTC_B
from boxmot.trackers.botsort.bot_sort_dc import BoTSORT_DC as BoTSORT_DC
from boxmot.trackers.botsort.bot_sort_dtc import BoTSORT_DTC as BoTSORT_DTC
# =================================================================================

TRACKERS = [
    "bytetrack", 
    "botsort",
    "botsort_dc",
    "botsort_dtc",
    "strongsort", 
    "ocsort", 
    "ocsort_d",
    "ocsort_dt",
    "ocsort_dtc",
    "ocsort_dtc_b",
    "deepocsort", 
    "hybridsort"
]

__all__ = (
    "__version__",
    "StrongSORT", 
    "OCSORT", 
    "OCSORT_D",
    "OCSORT_DT",
    "OCSORT_DTC",
    "OCSORT_DTC_B",
    "BYTETracker", 
    "BoTSORT", 
    "BoTSORT_DC",
    "BoTSORT_DTC",
    "DeepOCSORT", 
    "HybridSORT",
    "create_tracker", 
    "get_tracker_config", 
    "gsi"
)
