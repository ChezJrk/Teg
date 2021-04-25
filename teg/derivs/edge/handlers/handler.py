"""
Base class for delta handlers
"""

from teg import ITeg, Delta


class DeltaHandler:
    """Each delta handler adds compiler support for a specific family of discontinuities (e.g., affine). """

    def can_rewrite(delta: Delta, not_ctx=None) -> bool:
        """Check if this handler can simplify the given delta expression. """
        raise NotImplementedError('can_rewrite() not implemented')

    def rewrite(delta: Delta, not_ctx=None) -> ITeg:
        """Simplify this delta expression to another (hopefully simpler) delta expression. """
        raise NotImplementedError('rewrite() not implemented')
