======
Vokram
======

Vokram is a toy `Markov chain`_ library that is most likely implemented
incorrectly and extremely inefficiently.


Installation
============

Use `pip`_ to install::

    pip install vokram


Usage
=====

    >>> import vokram
    >>> corpus = open('the_art_of_war.txt').read().split()
    >>> model = vokram.Model(2)
    >>> vokram.markov_words(model, 25))
    'Hence it is not supreme excellence; supreme excellence consists in breaking the enemy's few.'


Credits
=======

Vokram was made with inspiration from this simple and approachable
`Python implementation and explanation`_.

.. _Markov chain: http://en.wikipedia.org/wiki/Markov_chain
.. _Python implementation and explanation: http://code.activestate.com/recipes/194364-the-markov-chain-algorithm/
.. _pip: http://www.pip-installer.org/
