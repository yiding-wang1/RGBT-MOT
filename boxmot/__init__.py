# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

__version__ = '11.0.6'

from boxmot.postprocessing.gsi import gsi
from boxmot.tracker_zoo import create_tracker, get_tracker_config
from boxmot.trackers.botsort.botsort import BotSort
from boxmot.trackers.bytetrack.bytetrack import ByteTrack
from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
from boxmot.trackers.hybridsort.hybridsort import HybridSort
from boxmot.trackers.ocsort.ocsort import OcSort
from boxmot.trackers.strongsort.strongsort import StrongSort
from boxmot.trackers.imprassoc.imprassoctrack import ImprAssocTrack


TRACKERS = ['bytetrack', 'botsort', 'strongsort', 'ocsort', 'deepocsort', 'hybridsort', 'imprassoc',
            'rgbt_strongsort',
            'rgbt_deepocsort',
            'rgbt_bytetrack',
            'rgbt_hybridsort',
            'rgbt_boost',
            'rgbt_imprassoc',
            'rgbt_botsort',
            'rgbt_ocsort',
            ]

__all__ = ("__version__",
           "StrongSort", "OcSort", "ByteTrack", "BotSort", "DeepOcSort", "HybridSort", "ImprAssocTrack"
           "create_tracker", "get_tracker_config", "gsi")
