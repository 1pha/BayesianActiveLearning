from acquisition_function import (
    random_selection,
    least_confidence,
    margin_of_confidence,
    mnlp_confidence,
    bald_acquisition,
    # batch_bald_acquisition,
)

ACQUISITION_MAP = {
    "random": "RANDOM",
    "lc": "LC",
    "mc": "MARGIN",
    "entropy": "ENTROPY",
    "mnlp": "MNLP",
    "bald": "BALD",
    "batchbald": "B-BALD",
}


class AcquisitionTool:
    def __init__(self, config):

        self.config = config
        self.acquisition = config.acquisition
