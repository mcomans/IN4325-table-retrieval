# Defines the experiments themselves, they can be called using parameters if
# applicable.

import argparse
from methods.STR import run_str
from methods.str.extraction import ExtractionMethod
from methods.LTR import run_ltr
from methods.ltr import Approach

parser = argparse.ArgumentParser(description="Runs the various experiments")
parser.add_argument("--rfr", action='store_true', help="Run the RFR LTR baseline experiment only")
parser.add_argument("--svr", action='store_true', help="Run the SVR LTR baseline experiment only")
parser.add_argument("--svm-rank", action='store_true', help="Run SVMrank LTR baseline experiment only")
parser.add_argument("-s", "--str", action='store_true', help="Run STR experiments only")

args = parser.parse_args()
run_all = len(vars(args)) == 0

if args.rfr or run_all:
    run_ltr(Approach.RFR)

if args.svr or run_all:
    run_ltr(Approach.SVR)

if args.svm_rank or run_all:
    run_ltr(Approach.SVM_RANK)

if args.str or run_all:
    run_str(ExtractionMethod.WORDS)
    run_str(ExtractionMethod.ENTITY)
