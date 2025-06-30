import numpy as np
from anyascii import anyascii

def string_match(word1: str, word2: str) -> tuple[bool, bool, bool, bool]:
    """Performs string comparison with multiple levels of tolerance

    Args:
        word1: a string
        word2: another string

    Returns:
        a tuple with booleans specifying respectively whether the raw strings, their lower-case counterparts, their
            anyascii counterparts and their lower-case anyascii counterparts match
    """
    raw_match = word1 == word2
    caseless_match = word1.lower() == word2.lower()
    anyascii_match = anyascii(word1) == anyascii(word2)

    # Warning: the order is important here otherwise the pair ("EUR", "€") cannot be matched
    unicase_match = anyascii(word1).lower() == anyascii(word2).lower()

    return raw_match, caseless_match, anyascii_match, unicase_match


class TextMatch:
    r"""Implements text match metric (word-level accuracy) for recognition task.

    The raw aggregated metric is computed as follows:

    .. math::
        \forall X, Y \in \mathcal{W}^N,
        TextMatch(X, Y) = \frac{1}{N} \sum\limits_{i=1}^N f_{Y_i}(X_i)

    with the indicator function :math:`f_{a}` defined as:

    .. math::
        \forall a, x \in \mathcal{W},
        f_a(x) = \left\{
            \begin{array}{ll}
                1 & \mbox{if } x = a \\
                0 & \mbox{otherwise.}
            \end{array}
        \right.

    where :math:`\mathcal{W}` is the set of all possible character sequences,
    :math:`N` is a strictly positive integer.

    >>> from receipt_cr.detection.recog_metrics import TextMatch
    >>> metric = TextMatch()
    >>> metric.update(['Hello', 'world'], ['hello', 'world'])
    >>> metric.summary()
    """

    def __init__(self) -> None:
        self.reset()

    def update(
        self,
        gt: list[str],
        pred: list[str],
    ) -> None:
        """Update the state of the metric with new predictions

        Args:
            gt: list of groung-truth character sequences
            pred: list of predicted character sequences
        """
        if len(gt) != len(pred):
            raise AssertionError("prediction size does not match with ground-truth labels size")

        for gt_word, pred_word in zip(gt, pred):
            _raw, _caseless, _anyascii, _unicase = string_match(gt_word, pred_word)
            self.raw += int(_raw)
            self.caseless += int(_caseless)
            self.anyascii += int(_anyascii)
            self.unicase += int(_unicase)

        self.total += len(gt)

    def summary(self) -> dict[str, float]:
        """Computes the aggregated metrics

        Returns:
            a dictionary with the exact match score for the raw data, its lower-case counterpart, its anyascii
            counterpart and its lower-case anyascii counterpart
        """
        if self.total == 0:
            raise AssertionError("you need to update the metric before getting the summary")

        return dict(
            raw=self.raw / self.total,
            caseless=self.caseless / self.total,
            anyascii=self.anyascii / self.total,
            unicase=self.unicase / self.total,
        )

    def reset(self) -> None:
        self.raw = 0
        self.caseless = 0
        self.anyascii = 0
        self.unicase = 0
        self.total = 0
