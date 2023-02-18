import IPython

IPY_MAJOR = IPython.version_info[0]
if IPY_MAJOR < 3:
    raise ImportError("IPython version %d is not supported." % IPY_MAJOR)

IPY3 = (IPY_MAJOR == 3)

# IPython underwent a major refactor between versions 3 and 4.  Many of the
# imports in version 4 have aliases to their old locations in 3, but they raise
# noisy deprecation warnings.  By conditionally importing here, we can support
# older versions without triggering warnings for users on new versions.
if IPY3:
    from IPython.nbformat import read
else:
    from nbformat import read


__all__ = ['read']
