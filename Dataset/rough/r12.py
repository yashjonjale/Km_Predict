from Bio import ExPASy, SwissProt # For sequence retrieval

emb_dim = 1280


def extract_sequence(uniprot_id):
    handle = ExPASy.get_sprot_raw(uniprot_id)
    # print(type(handle))

    # return handle
    record = SwissProt.read(handle)
    return record.sequence


def safe_extract_sequence(uniprot_id):
    try:
        return extract_sequence(uniprot_id)
    except Exception as e:
        print(f"Error retrieving sequence for {uniprot_id}: {e}")
        return None 

from Bio import Align

def similarity_seqs(seq1, seq2):
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    scores = [alignment.score for alignment in aligner.align(seq1, seq2)]
    print(scores)
    return max(scores) / max(len(seq1), len(seq2))

print(similarity_seqs("ACCGT", "ACG"))