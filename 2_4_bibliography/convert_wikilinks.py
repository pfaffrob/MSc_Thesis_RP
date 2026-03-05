#! /usr/bin/env python
# -*- coding: utf-8 -*-
# found on: https://forum.obsidian.md/t/have-pandoc-recognize-your-literature-links-as-citations/54604

from pandocfilters import toJSONFilter, Str, Plain, Para
import sys
import re

def replace_citation_links_with_citation(key, value, format, meta):
    # sys.stderr.write(f"=== {key} ===>\n")
    # sys.stderr.write(f"{value}\n\n")
    elements = []
    if key == "Plain" or key == "Para":
        for item_idx, item in enumerate(value):
            if item["t"] == "Str" and item["c"] == "[":
                if item_idx < len(value) - 1:
                    if value[item_idx + 1]["t"] == "Cite":
                        continue
            elif item["t"] == "Str" and item["c"] == "]":
                if item_idx > 0:
                    if value[item_idx - 1]["t"] == "Cite":
                        continue
            elements.append(item)
        if key == "Plain":
            return Plain(elements)
        elif key == "Para":
            return Para(elements)


if __name__ == "__main__":
    toJSONFilter(replace_citation_links_with_citation)