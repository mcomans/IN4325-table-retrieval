from enum import Enum
from .RFR import RFR
from .SVR import SVR
from .AdaRank import AdaRank


class Approach(Enum):
    RFR = "RFR"
    SVR = "SVR"
    AdaRank = "AdaRank"
