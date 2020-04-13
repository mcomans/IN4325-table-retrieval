# Defines the experiments themselves, they can be called using parameters if
# applicable.

from methods.STR import run_str
from methods.str.extraction import ExtractionMethod

run_str(ExtractionMethod.WORDS)
run_str(ExtractionMethod.ENTITY)
