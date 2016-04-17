"""Cluster vectors with k-means"""

import scipy
import argparse
from scipy.spatial.distance import cdist
import codecs
import time

from collections import defaultdict

__author__='Sravana Reddy'

def assign(points, means):
    """return a 1-d array assigning each point to the nearest mean
    (by Euclidean distance)"""
    cd = cdist(points, means, 'euclidean') # each row has distance to all means
    # get indices of closest mean
    return scipy.argmin(cd, axis=1)

def kmeans(points, labels, k, maxiter):
    """Cluster points into k groups.
    Return a dictionary mapping each cluster ID (from 0 through k-1)
    to a list of the labels of the points that belong to that cluster.
    """
    #TODO: fill in
    numvecs, numdims = points.shape
    # initialize means by picking first k vectors
    means = points[0:k, :]
    # assign points to nearest mean
    assignments = assign(points, means)
    # repeat till convergence (no points change assignments), for maxiter runs
    for i in range(maxiter):
        # compute new means from assignments
        means = scipy.zeros((k, numdims))
        for m in range(k):
            # get indices that have m in assignment
            indices = scipy.where(assignments == m)[0]
            if len(indices)==0:  # empty cluster
                # make the mean the mth point
                means[m, :] = points[m, :]
                print 'Empty cluster: assigning point to centroid'
            else:
                means[m, :] = scipy.mean(points[indices, :], axis=0)
        # get new assignments
        new_assignments = assign(points, means)
        if scipy.array_equal(assignments, new_assignments):
            print 'Converged in', i+1, 'iterations'
            break
        assignments = new_assignments

    # now assign corresponding words to clusters
    clusters = defaultdict(list)
    for i, c in enumerate(assignments):
        clusters[c].append(labels[i])
    return clusters

def main():
    # Do not modify
    start = time.time()

    parser = argparse.ArgumentParser(description='Cluster vectors with k-means.')
    parser.add_argument('vecfile', type=str, help='name of vector file (exclude extension)')
    parser.add_argument('k', type=int, help='number of clusters')
    parser.add_argument('--maxiter', type=int, default=100, help='maximum number of k-means iterations')
    args = parser.parse_args()

    points = scipy.loadtxt(args.vecfile+'.vecs')
    labels = codecs.open(args.vecfile+'.labels', 'r', 'utf8').read().split()

    clusters = kmeans(points, labels, args.k, args.maxiter)
    outfile = args.vecfile+'.cluster'+str(args.k)
    with codecs.open(outfile, 'w', 'utf8') as o:
        for c in clusters:
            o.write('CLUSTER '+str(c)+'\n')
            o.write(' '.join(clusters[c])+'\n')

    print time.time()-start, 'seconds'

if __name__=='__main__':
    main()
