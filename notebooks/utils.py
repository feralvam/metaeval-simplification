import pandas as pd
import math
from tqdm import tqdm

from easse.utils.constants import TEST_SETS_PATHS
from easse.utils.helpers import read_lines

def read_test_set(test_set, as_lists=False):
    orig_sents_path = TEST_SETS_PATHS[(test_set, "orig")]
    refs_sents_paths = TEST_SETS_PATHS[(test_set, "refs")]
    num_refs = len(refs_sents_paths)

    orig_sents = read_lines(orig_sents_path)
    refs_sents = [read_lines(ref_sents_path) for ref_sents_path in refs_sents_paths]

    if as_lists:
        return orig_sents, refs_sents

    fhs = [orig_sents] + refs_sents
    all_sent_id = []
    all_orig_sent = []
    all_ref_sents = []
    for sent_id, (orig_sent, *ref_sents) in enumerate(zip(*fhs), start=1):
        all_sent_id += [sent_id] * num_refs
        all_orig_sent += [orig_sent] * num_refs
        all_ref_sents += ref_sents
    return pd.DataFrame(
        list(zip(all_sent_id, all_orig_sent, all_ref_sents)),
        columns=["sent_id", "orig_sent", "ref_sent"],
    )


def collect_references(sent_ids, test_set_orig_sents, test_set_refs_sents, num_refs):
    orig_sents = []
    refs_sents = [[] for i in range(num_refs)]
    for sent_id in sent_ids:
        orig_sents.append(test_set_orig_sents[sent_id-1])
        for i, ref in enumerate(test_set_refs_sents):
            refs_sents[i].append(ref[sent_id-1])
            
    return orig_sents, refs_sents


def sigmoid(x):
    return 1 / (1 + math.exp(-x))
