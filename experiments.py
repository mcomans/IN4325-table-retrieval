# Defines the experiments themselves, they can be called using parameters if
# applicable.

import argparse
from methods.STR import run_str
from methods.str.extraction import ExtractionMethod
from methods.LTR import run_ltr
from methods.ltr import Approach
from methods.str.semantic_matching import SemanticSpace
from methods.str.similarity import SimilarityMethod

parser = argparse.ArgumentParser(description="Runs the various experiments")
parser.add_argument("-l", "--ltr", action='store_true', help="Run LTR baseline experiment only")
parser.add_argument("-s", "--str", action='store_true', help="Run STR experiments only")

args = parser.parse_args()
run_all = len(vars(args)) == 0

if args.ltr or run_all:
    run_ltr(Approach.RFR)

if args.str or run_all:
    run_str(ExtractionMethod.ENTITY, SemanticSpace.WORD_EMBEDDINGS,
            SimilarityMethod.LATE_FUSION)
