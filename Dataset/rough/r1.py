import Bio

## sequence alignment

from Bio import pairwise2
from Bio.pairwise2 import format_alignment

alignments = pairwise2.align.globalxx("ACCGT", "ACG")
print(format_alignment(*alignments[0]))
print(alignments)


# extract score from the alignment
alignments = pairwise2.align.globalxx("ACCGT", "ACG")
print(alignments[0][2])


def similarity_seqs(seq1,seq2):
    alignments = pairwise2.align.globalxx(seq1, seq2)
    return alignments[0][2]/max(len(seq1),len(seq2))