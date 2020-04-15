from enum import Enum
from .RFR import RFR
from .SVR import SVR
from .SVMrank import SVMrank


class Approach(Enum):
    RFR = "RFR"
    SVR = "SVR"
    SVM_RANK = "SVM_RANK"
