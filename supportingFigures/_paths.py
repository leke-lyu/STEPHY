"""Resolve the large data roots the figure scripts read from.

Neither root is stored in this repository. The trained models are archived on
Zenodo (see ../zenodo/README.md) and the Denmark case-study data lives in its
own repository (https://github.com/leke-lyu/stephy-denmark). Point the two
environment variables below at local copies, or pass the corresponding
command-line flag explicitly:

    export STEPHY_MODELS=/path/to/stephy-trained-models   # contains simu/, denmark/
    export DENMARK_CASE=/path/to/denmark_case             # contains nextstrain/, paper/

If a variable is unset, the argparse default becomes a literal placeholder such
as `<set STEPHY_MODELS>/simu/...`, so the resulting error names the variable
that needs setting rather than failing on a stale absolute path.
"""
import os

_VARS = {
    "models": "STEPHY_MODELS",
    "denmark": "DENMARK_CASE",
}


def root(kind: str) -> str:
    """Return the configured root for `kind`, or a self-describing placeholder."""
    var = _VARS[kind]
    return os.environ.get(var) or f"<set {var}>"


def under(kind: str, *parts: str) -> str:
    """Join `parts` beneath the configured root for `kind`."""
    return os.path.join(root(kind), *parts)
