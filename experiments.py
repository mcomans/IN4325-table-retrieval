# Defines the experiments themselves, they can be called using parameters if
# applicable.

from methods.STR import run_str
from methods.str.extraction import ExtractionMethod
from methods.LTR import run_ltr
from methods.ltr import Approach

run_ltr(Approach.RFR)

run_str(ExtractionMethod.WORDS)
run_str(ExtractionMethod.ENTITY)
