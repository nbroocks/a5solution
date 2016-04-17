"""Build context vectors of word types from a corpus,
reduce dimensionality with singular value decomposition,
store vectors."""

import scipy
from utils import *
import argparse
import time

__author__='Sravana Reddy'

def count_context_vectors(filename, window, freqthresh):
    """Estimate count context vectors for each word,
    with at most window number of context tokens on either side of the sentence.
    Assume filename contains one tokenized sentence per line.
    Ignore words and contexts with less than freqthresh count
    Return a list consisting of
    1. a list of the m word types with at least freqthresh count, sorted in decreasing order of frequency.
    2. an array with m rows and m+2 columns,
    where row i is the vector for the ith word in the above list,
    column i is the ith word used as context (upto m),
    column m is <s> used as context, and column m+1 is </s> used as context.
    """

    text, wordcounts = parsetextfile(filename) # read text and get word frequencies

    #TODO: fill in the rest
    threshwords = filter(lambda (w,c):c>=freqthresh, wordcounts.items())   # threshold
    threshwords = sorted(threshwords, key=lambda x:x[1], reverse=True) # sort by frequency
    vocab = [w for (w,c) in threshwords]   # list of words to return
    word_indices = dict([(w, i) for (i, w) in enumerate(vocab)])  # for quick lookup of indices
    vocabsize = len(vocab)
    print 'Filtered vocabulary to', vocabsize, 'words'

    # build context vectors
    cv = scipy.zeros((vocabsize, vocabsize+2))
    for li, line in enumerate(text):
        line.insert(0, '<s>')
        line.append('</s>')
        progressbar(li+1, 2000, 50000)
        for i, word in enumerate(line):
            if word in word_indices:  # get context vectors
                left_context = line[max(0, i-window):i]
                right_context = line[i+1:i+window+1]
                for cword in left_context+right_context:
                    if cword in word_indices:
                        cv[word_indices[word], word_indices[cword]] += 1
                    if cword=='<s>':
                        cv[word_indices[word], -2] +=1
                    if cword=='</s>':
                        cv[word_indices[word], -1] +=1
    return vocab, cv

def ppmi(vectors):
    """Compute PPMI vectors from count vectors.
    """
    # Do not modify
    rowsum = scipy.sum(vectors, axis=0)  # sum each column across rows (count of context c)

    # remove all-zero columns
    nonzerocols = rowsum>0
    rowsum = rowsum[nonzerocols]
    vectors = vectors[:, nonzerocols]

    colsum = scipy.sum(vectors, axis=1)  # sum each row across columns (count of word w)
    allsum = scipy.sum(rowsum)  # sum all values in matrix

    # get p(x, y)/(p(x)*p(y))
    vectors/=colsum[:, scipy.newaxis] # count_ij/count_i*
    vectors/=rowsum # count_ij/(count_i* * count_j*)
    vectors *= allsum # prob_ij/(prob_i* * prob_j*)

    # get log, floored at 0
    vectors = scipy.log2(vectors) # will give runtime warning for log(0); ignore
    vectors[vectors<0] = 0  # get indices where value<0 and set them to 0

    return vectors

def main():
    # Do not modify
    start = time.time()

    parser = argparse.ArgumentParser(description='Build context vectors.')
    parser.add_argument('textfile', type=str, help='name of text file')
    parser.add_argument('window', type=int, help='context window')
    parser.add_argument('threshold', type=int, help='vocabulary minimum frequency')
    parser.add_argument('--ndims', type=int, default=100, help='number of SVD dimensions')
    parser.add_argument('--debug', type=bool, default=False)
    args = parser.parse_args()

    vocab, points = count_context_vectors(args.textfile+'.txt', args.window, args.threshold)
    print 'Estimated count context vectors'
    if not args.debug:  # compute PPMI and SVD before writing. if debug is True, just write the count vectors
        points = ppmi(points)
        print 'Converted to positive pointwise mutual information'
        points = dimensionality_reduce(points, args.ndims)
        print 'Reduced dimensionality'
        outfile = args.textfile+'.window'+str(args.window)+'.thresh'+str(args.threshold)
    else:
        outfile = args.textfile+'.window'+str(args.window)+'.thresh'+str(args.threshold)+'.todebug'

    with open(outfile+'.labels', 'w') as o:
        o.write('\n'.join(vocab)+'\n')
    scipy.savetxt(outfile+'.vecs', points, fmt='%.4e')
    print 'Saved to file'

    print time.time()-start, 'seconds'

if __name__=='__main__':
    main()
