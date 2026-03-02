#!/usr/bin/env python3
"""
Lightweight BEAST2 NEXUS tree file parser (regex-based).

Extracts or counts individual tree strings from BEAST2 ``*_beast2.trees``
files without importing heavy dependencies (torch, dgl, dendropy).  Used by
both ``analyze_trees.py`` (analysis only) and ``data.py`` (graph construction).
"""

import re

# Regex to extract individual tree strings from BEAST2 NEXUS files
# Matches: "tree STATE_0 = ((...));" and captures the Newick tree content
_TREE_CONTENT_PATTERN = r'tree STATE_\d+ = (.+?)(?=\ntree |\nEnd;|$)'

# Regex to count trees (faster than extracting full content)
_TREE_COUNT_PATTERN = r'tree STATE_\d+'


def parse_trees(content):
    """
    Extract tree strings from BEAST2 NEXUS file content.

    Args:
        content: String content of a BEAST2 .trees file

    Returns:
        List of tree strings (Newick format)
    """
    return re.findall(_TREE_CONTENT_PATTERN, content, re.DOTALL)


def count_trees(content):
    """
    Count number of trees in BEAST2 NEXUS file content.

    Faster than parse_trees() when you only need the count.

    Args:
        content: String content of a BEAST2 .trees file

    Returns:
        Number of trees in the file
    """
    return len(re.findall(_TREE_COUNT_PATTERN, content))
