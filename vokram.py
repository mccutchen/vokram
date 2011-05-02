#!/usr/bin/env python

"""
A simple, generic implementation of Markov chains in Python, with some
helpers for generating chains of words.

With inspiration from:
http://code.activestate.com/recipes/194364-the-markov-chain-algorithm/
"""

import os
import random
import sys
from collections import defaultdict


DEFAULT_NGRAM_SIZE = 2
MIN_SENTENCE_LENGTH = 5


##############################################################################
# Basic interface
##############################################################################

def markov_chain(model, length, start_key=None):
    """Generates a Markov chain with the given length based on the given
    model. The chain will be returned as a list. If a starting key (in the
    model) is not given, a random one will be chosen.
    """
    chain = []
    key = start_key or random.choice(model.keys())
    for _ in xrange(length):
        # Add a random selection from the value corresponding to the current
        # key to the chain.
        x = random.choice(model[key])
        chain.append(x)
        # Pick the next key by dropping the first item in the current key and
        # appending the current item (manually creating the n-gram that will
        # let us choose the next appropriate item for our chain)
        key = key[1:] + (x,)
    return chain

def build_model(xs, n=DEFAULT_NGRAM_SIZE):
    """Builds a model of the given sequence using n-grams of size n. The model
    is a dict mapping n-gram keys to lists of items appearing immediately
    after those n-grams.
    """
    model = defaultdict(list)
    for ngram in gen_ngrams(xs, n+1):
        key, item = ngram[:-1], ngram[-1]
        model[key].append(item)
    return dict(model)


##############################################################################
# Word-based interface
##############################################################################

def markov_words(model, length, start_key=None):
    """Generates a Markov chain of the given length. Attempts to be
    intelligent about generating chains made up of what (hopefully) look like
    complete sentences.
    """

    # An overly-simplistic heuristic to use to try to generate complete
    # sentences
    sentence_end = ('.', '!', '?', '"', "'")

    # Find a start key that (hopefully) indicates the end of a sentence, which
    # will make it more likely that our chain will start with a word from the
    # beginning of a sentence.
    if start_key is None:
        keys = model.keys()
        key = random.choice(keys)
        # Making sure the key ends in a period (instead of anything in
        # sentence_end) seems to yield better results at the start of the
        # chain.
        while not key[-1][-1] == '.':
            key = random.choice(keys)
        start_key = key

    # Make sure our chain seems to end at the end of a sentence, by dropping
    # any dangling words after the end of the last sentence in the chain.
    chain = markov_chain(model, length, start_key)
    if chain[-1][-1] not in sentence_end:
        for i in xrange(length-1, -1, -1):
            if chain[i][-1] in sentence_end:
                break
        chain = chain[:i+1]

    # Make sure we've got a reasonable-sized chain.
    if len(chain) < MIN_SENTENCE_LENGTH:
        return markov_words(model, length)
    else:
        return ' '.join(chain)

def build_word_model(corpus, n=DEFAULT_NGRAM_SIZE):
    """A special-case of build_model that knows how to build a model based on
    words from a corpus given as a string or a file-like object.
    """
    return build_model(gen_words(corpus), n=n)


##############################################################################
# Utility functions
##############################################################################

def gen_ngrams(xs, n=DEFAULT_NGRAM_SIZE):
    """Yields n-grams from the given sequence. Assumes len(xs) >= n. N-grams
    are yielded as tuples of length n.
    """
    # Explicitly capture an iterator over xs, because we'll need it twice
    it = iter(xs)

    # Build and yield the first n-gram. This is where the assumption of
    # len(xs) >= n needs to be true.
    n_gram = tuple(it.next() for _ in xrange(n))
    yield n_gram

    # Each successive n-gram is built by dropping the first item of the
    # previous n-gram and appending the current element
    for x in it:
        n_gram = n_gram[1:] + (x,)
        yield n_gram

def gen_words(corpus):
    """Yields each word from the given corpus, which can be either a string or
    a file-like object containing the words.
    """
    # If we're given the corpus as a string, split it into lines so that we
    # can iterate over it the same as we would an open file.
    if isinstance(corpus, basestring):
        corpus = corpus.splitlines()
    for line in corpus:
        for word in line.strip().split():
            yield word


if __name__ == '__main__':
    usage = """%s corpus [length]""" % sys.argv[0]
    try:
        corpus = sys.argv[1]
    except IndexError:
        print usage
        sys.exit(1)
    else:
        try:
            length = int(sys.argv[2])
        except (IndexError, ValueError):
            length = 30
        model = build_word_model(open(corpus))
        try:
            print markov_words(model, length)
        except RuntimeError, e:
            print 'Could not generate a chain with length %s.' % length,
            print 'Please consider increasing the length.'
