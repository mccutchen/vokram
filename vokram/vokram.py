#!/usr/bin/env python

"""
A simple, generic implementation of Markov chains in Python. A brief overview
of how this works:

 1. Build a Markov model from a given corpus, which can be a sequence of
    basically anything (e.g., numbers, words).

    In this implementation, a model is a dictionary that maps tuples of
    n-grams to lists of the items that appear after those n-grams in the input
    corpus.

    So, taking this simple corpus as an example (where `>>>` represents the
    interactive Python prompt):

        >>> corpus = [1, 1, 1, 2, 2, 2, 3, 3, 3]

    With an n-gram size of 2 and a `None` as the special `stop_token`, the
    model would look like this:

        >>> model = vokram.Model(n=2, stop_token=None)
        >>> model.add_sequence(corpus)
        >>> model.state
        {(None, None): [1],
         (None, 1): [1],
         (1, 1): [1, 2],
         (1, 2): [2],
         (2, 2): [2, 3],
         (2, 3): [3],
         (3, 3): [3, None]}

    **Note**: We preceed ever series of input tokens with `n` stop tokens. You
    can see this in the state above, where the key `(None, None)` points to the
    first token (`1`) in our input corpus. This makes it easy to pick a good
    "start" key when generating a new sequence of tokens.

    For reference, the model of the same corpus with an n-gram size of 3 would
    look like this:

        >>> model3 = vokram.Model(n=3, stop_token=None)
        >>> model3.add_sequence(corpus)
        >>> model3.state
        {(None, None, None): [1],
         (None, None, 1): [1],
         (None, 1, 1): [1],
         (1, 1, 1): [2],
         (1, 1, 2): [2],
         (1, 2, 2): [2],
         (2, 2, 2): [3],
         (2, 2, 3): [3],
         (2, 3, 3): [3],
         (3, 3, 3): [None]}

 2. Once the model is built, we can use it to generate a new, stastistically
    likely sequence of tokens based on the input tokens we've seen so far:

        >>> list(model.gen_sequence())
        [1, 1, 1, 2, 2, 3, 3]
        >>> list(model.gen_sequence())
        [1, 1, 2, 2, 2, 2, 3, 3]
        >>> list(model.gen_sequence())
        [1, 1, 2, 2, 3, 3]

    The process for building a new sequence of tokens works like this:

    1. Make a special "start" key composed of `n` `stop_token` values. We know
       that any tokens this key points to came from the beginning of a discrete
       sequence of input tokens.

    2. Pick a random token from the list that our chosen key points to and add
       it to our output sequence.

    3. Build a new key by dropping the first token in our current key and
       appending the token we chose in step 2.

    4. Start over at step 2, using our new key. Repeat this until we encounter
       a stop token or until the sequence has reached a specific length.
"""

from __future__ import unicode_literals

import itertools
import random


# We want a range function that will return a generator, but xrange is gone in
# Python 3. So, let's figure out what range function to use.
try:
    range_iter = xrange
except NameError:
    range_iter = range


UNDEFINED = object()


class Model(object):
    """
    A Markov model with a specific n-gram size, which can be fed sequences of
    tokens and which can then generate new sequences of tokens from its prior
    input.
    """

    def __init__(self, n, stop_token=UNDEFINED):
        """
        Initialize a model with an n-gram size of `n`, and a special "stop"
        token used to denote the end of a discrete sequence of input tokens.
        """
        assert n > 0
        self.n = n
        self.stop_token = object() if stop_token is UNDEFINED else stop_token
        self.state = self.make_state()

    def add_sequence(self, xs):
        """
        Add a sequence of tokens to a model.
        """
        xs = itertools.chain(self.make_start_key(), xs, [self.stop_token])
        for ngram in gen_ngrams(xs, self.n + 1):
            key, token = ngram[:-1], ngram[-1]
            self.add_token(key, token)

    def gen_sequence(self, max_size=None):
        """
        Yield a sequence of tokens generated from a model. The sequence will
        end when the model's stop value is encountered or, optionally, once a
        maximum number of tokens has been generated.
        """
        key = self.make_start_key()
        for i in itertools.count():
            x = self.get_token(key)
            if x == self.stop_token:
                break
            yield x
            if max_size and i >= max_size:
                break
            key = key[1:] + (x,)

    def make_start_key(self):
        """
        A start key is an n-gram containing only a model's stop tokens. Because
        we preceed every sequence of tokens fed into the model with `n` stop
        tokens, we know that the any token that follows this start key will be
        the start of some sequence of input.
        """
        return (self.stop_token,) * self.n

    def make_state(self):
        """Initialize the state for a model."""
        return {}

    def add_token(self, key, token):
        """Add a given token to the state for the given key."""
        self.state.setdefault(key, []).append(token)

    def get_token(self, key):
        """Choose a likely token from the state for the given key."""
        return random.choice(self.state[key])

    def __repr__(self):
        return 'Model(n={!r}, stop_token={!r}, state={!r})'.format(
            self.n, self.stop_token, self.state)


def gen_ngrams(xs, n):
    """
    Yields n-grams from the given sequence. Assumes `len(xs) >= n`. N-grams are
    yielded as tuples of length `n`.
    """
    # Explicitly capture an iterator over `xs`, because we'll need it twice
    it = iter(xs)

    # Build and yield the first n-gram. This is where the assumption of
    # `len(xs) >= n` needs to be true.
    ngram = tuple(next(it) for _ in range_iter(n))
    yield ngram

    # Each successive n-gram is built by dropping the first item of the
    # previous n-gram and appending the current element
    for x in it:
        ngram = ngram[1:] + (x,)
        yield ngram
