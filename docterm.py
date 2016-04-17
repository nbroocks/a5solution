"""Build term-document matrix from collection of documents."""

import scipy
from utils import *
import argparse
import time

def tfidf_docterm(filename, freqthresh):
    """Estimate document-term TF-IDF vectors for each document (line in filename),
    where each column is a word, in decreasing order of frequency.
    Ignore words that appear fewer than freqthresh times.
    Return a list consisting of
    1. a list of the m word types with at least freqthresh count, sorted in decreasing order of frequency.
    2. an array with d rows and m columns,
    where row i is the vector for the ith document in filename,
    and col j represents the jth word in the above list.
    """
    #TODO: fill in
    text, wordcounts = parsetextfile(filename) # read text and get word frequencies

    threshwords = filter(lambda (w,c):c>=freqthresh, wordcounts.items())   # threshold
    threshwords = sorted(threshwords, key=lambda x:x[1], reverse=True) # sort by frequency
    vocab = [w for (w,c) in threshwords]   # list of words to return
    word_indices = dict([(w, i) for (i, w) in enumerate(vocab)])  # for quick lookup of indices
    vocabsize = len(vocab)
    print 'Filtered vocabulary to', vocabsize, 'terms'

    # build document-term vectors
    tf = scipy.zeros((len(text), vocabsize))
    for li, line in enumerate(text):
        progressbar(li+1, 2000, 50000)
        for term in line:
            if term in word_indices:
                tf[li, word_indices[term]] += 1

    numdocs, numterms = tf.shape
    makeones = scipy.zeros((numdocs, numterms))
    makeones[scipy.nonzero(tf)] = 1  # set non-zero indices to 1
    idf = scipy.sum(makeones, axis=0)  # sum each column across rows (num of documents where term c occurs)
    print 'Computed number of documents for each term'
    idf = -scipy.log2(idf/numdocs)  # log(N/idf)
    print 'Computed IDF'
    idf = scipy.tile(idf, (numdocs, 1))  # tile to same shape as vectors
    return vocab, scipy.multiply(tf, idf)  # element-wise multiplication

def main():
    # Do not modify
    start = time.time()

    parser = argparse.ArgumentParser(description='Build document-term vectors.')
    parser.add_argument('textfile', type=str, help='name of text file with documents on each line')
    parser.add_argument('threshold', type=int, help='term minimum frequency')
    parser.add_argument('--ndims', type=int, default=100, help='number of SVD dimensions')
    parser.add_argument('--debug', type=bool, default=False, help='debug mode?')
    args = parser.parse_args()

    terms, points = tfidf_docterm(args.textfile+'.txt', args.threshold)
    print 'Estimated document-term TF-IDF vectors'
    if not args.debug:  # compute PPMI and SVD before writing. if debug is True, just write the count vectors
        points = dimensionality_reduce(points, args.ndims)
        print 'Reduced dimensionality'
        outfile = args.textfile+'.tfidf'+'.thresh'+str(args.threshold)
    else:
        outfile = args.textfile+'.tfidf'+'.thresh'+str(args.threshold)+'.todebug'

    with open(outfile+'.dims', 'w') as o:
        o.write('\n'.join(terms)+'\n')
    scipy.savetxt(outfile+'.vecs', points, fmt='%.4e')
    print 'Saved to file'

    print time.time()-start, 'seconds'

if __name__=='__main__':
    main()
