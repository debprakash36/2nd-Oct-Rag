"""Bounded retry policy shared by the hosted providers.

Extracted rather than written twice. The embedding and generation providers must
back off identically, and the cheapest way to guarantee that is for there to be one
copy of the policy: a second implementation would be free to drift, and a drift
here is invisible in tests because each provider tests its own constants.

**Retry is for transport faults only.** A retry is a bet that re-sending produces a
different outcome. That holds for a timeout, a refused connection, a rate limit or
a 502. It does not hold for a 401, a 404 or a malformed response, which come back
identically every time -- repeating them only delays the message that diagnoses
them.
"""

from __future__ import annotations

import random
from collections.abc import Callable

#: Statuses worth retrying. 429 is the rate limit both hosted APIs enforce, and the
#: reason this exists: a sweep or a burst of concurrent users trips it. 5xx is a
#: provider-side fault a retry usually clears. 408 and 425 are included because both
#: mean "the server did not take this one yet".
#:
#: Every other 4xx is deliberately excluded -- see the module docstring.
RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})

#: First backoff step, doubled per attempt and capped. The cap matters because the
#: exponent is applied per attempt: uncapped, a larger attempt count walks into a
#: multi-minute stall on a request that has already failed.
BASE_BACKOFF_SECONDS = 0.5
MAX_BACKOFF_SECONDS = 8.0


def backoff_seconds(
    attempt: int,
    *,
    jitter: Callable[[], float] = random.random,
) -> float:
    """Seconds to wait before attempt `attempt`+1, at equal jitter.

    Equal jitter: half of each ceiling is fixed and half is random, so the delay
    neither collapses toward zero -- which would defeat the backoff entirely -- nor
    exceeds the ceiling.

    The randomisation is not cosmetic. The failure mode being defended against is a
    *provider-wide* rate limit, so every caller backing off on an identical schedule
    arrives together and re-trips the limit on the same tick.
    """
    ceiling = min(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
    return ceiling * (0.5 + 0.5 * jitter())


__all__ = [
    "BASE_BACKOFF_SECONDS",
    "MAX_BACKOFF_SECONDS",
    "RETRYABLE_STATUS_CODES",
    "backoff_seconds",
]