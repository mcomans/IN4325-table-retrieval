from enum import Enum
from .RFR import RFR
from .SVR import SVR


class Approach(Enum):
    RFR = "RFR"
    SVR = "SVR"
