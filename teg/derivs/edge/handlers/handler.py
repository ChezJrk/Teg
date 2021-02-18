"""
Base class for delta handlers
"""

from teg import (
    ITeg,
    Delta
)


class DeltaHandler():
    def can_rewrite(delta: Delta, not_ctx=set()) -> bool:
        raise NotImplementedError('can_rewrite() not implemented')

    def rewrite(delta: Delta, not_ctx=set()) -> ITeg:
        raise NotImplementedError('rewrite() not implemented')
