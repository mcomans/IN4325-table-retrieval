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
parser.add_argument("-l", "--ltr", action='store_true',
                    help="Run LTR baseline experiment only")
parser.add_argument("-s", "--str", action='store_true',
                    help="Run STR experiments only")

args = parser.parse_args()
run_all = len(vars(args)) == 0

if args.ltr or run_all:
    run_ltr(Approach.RFR)

if args.str or run_all:
    print("=> Running STR (WORDS, WORD_EMBEDDINGS, EARLY_FUSION)")
    results_W_W_EF = run_str(ExtractionMethod.WORDS,
                             SemanticSpace.WORD_EMBEDDINGS,
                             SimilarityMethod.EARLY_FUSION)
    print("=> Running STR (WORDS, WORD_EMBEDDINGS, LATE_FUSION)")
    results_W_W_LF = run_str(ExtractionMethod.WORDS,
                             SemanticSpace.WORD_EMBEDDINGS,
                             SimilarityMethod.LATE_FUSION)
    print("=> Running STR (ENTITY, WORD_EMBEDDINGS, EARLY_FUSION)")
    results_E_W_EF = run_str(ExtractionMethod.ENTITY,
                             SemanticSpace.WORD_EMBEDDINGS,
                             SimilarityMethod.EARLY_FUSION)
    print("=> Running STR (ENTITY, WORD_EMBEDDINGS, LATE_FUSION)")
    results_E_W_LF = run_str(ExtractionMethod.ENTITY,
                             SemanticSpace.WORD_EMBEDDINGS,
                             SimilarityMethod.LATE_FUSION)
    # print("=> Running STR (ENTITY_SPACY, WORD_EMBEDDINGS, EARLY_FUSION)")
    # results_ES_W_EF = run_str(ExtractionMethod.ENTITY_SPACY,
    #                           SemanticSpace.WORD_EMBEDDINGS,
    #                           SimilarityMethod.EARLY_FUSION)
    # print("=> Running STR (ENTITY_SPACY, WORD_EMBEDDINGS, LATE_FUSION)")
    # results_ES_W_LF = run_str(ExtractionMethod.ENTITY_SPACY,
    #                           SemanticSpace.WORD_EMBEDDINGS,
    #                           SimilarityMethod.LATE_FUSION)
    with open("STR_features.csv", "w") as file:
        file.write("query_id,table_id,str_w_w_ef,str_w_w_lf_max,"
                   "str_w_w_lf_sum,str_w_w_lf_avg,str_e_w_ef,str_e_w_lf_max,"
                   "str_e_w_lf_sum,str_e_w_lf_avg")
        for (query_id, table_id, _), idx in enumerate(results_W_W_EF):
            file.write(f"{query_id},{table_id},"
                       f"{results_W_W_EF[idx]['score']},"
                       f"{results_W_W_LF[idx]['score'][0]},"
                       f"{results_W_W_LF[idx]['score'][1]},"
                       f"{results_W_W_LF[idx]['score'][2]},"
                       f"{results_E_W_EF[idx]['score']},"
                       f"{results_E_W_LF[idx]['score'][0]},"
                       f"{results_E_W_LF[idx]['score'][1]},"
                       f"{results_E_W_LF[idx]['score'][2]}")