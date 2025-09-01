from collections import defaultdict
from typing import Dict, List, Tuple


def build_block_map(blocks):
    return {b["Id"]: b for b in blocks}


def _child_ids(block, rel_type="CHILD"):
    ids = []
    for rel in block.get("Relationships", []):
        if rel.get("Type") == rel_type:
            ids.extend(rel.get("Ids", []))
    return ids


def _text_from_block(block, block_map):
    parts = []
    for cid in _child_ids(block, "CHILD"):
        child = block_map.get(cid, {})
        btype = child.get("BlockType")
        if btype == "WORD":
            parts.append(child.get("Text", ""))
        elif btype == "SELECTION_ELEMENT":
            status = child.get("SelectionStatus", "NOT_SELECTED")
            parts.append("[x]" if status == "SELECTED" else "[ ]")
        elif btype == "MERGED_CELL":
            parts.append(_text_from_block(child, block_map))
    return " ".join([p for p in parts if p])


def parse_blocks(blocks):
    block_map = build_block_map(blocks)

    # Lines
    lines_by_page: Dict[int, List[Tuple[float, float, str]]] = defaultdict(list)
    for b in blocks:
        if b.get("BlockType") == "LINE":
            page = b.get("Page", 1)
            bb = b.get("Geometry", {}).get("BoundingBox", {})
            top = bb.get("Top", 0.0)
            left = bb.get("Left", 0.0)
            lines_by_page[page].append((top, left, b.get("Text", "")))
    for p in lines_by_page:
        lines_by_page[p].sort(key=lambda x: (round(x[0], 4), round(x[1], 4)))

    # Forms
    forms_by_page: Dict[int, List[str]] = defaultdict(list)
    kv_keys = [b for b in blocks if b.get("BlockType") == "KEY_VALUE_SET" and "KEY" in b.get("EntityTypes", [])]
    for key_block in kv_keys:
        page = key_block.get("Page", 1)
        ktxt = _text_from_block(key_block, block_map).strip()
        vtxts = []
        for vid in _child_ids(key_block, "VALUE"):
            vblock = block_map.get(vid)
            if vblock:
                vtxts.append(_text_from_block(vblock, block_map).strip())
        vtxt = " ".join(v for v in vtxts if v)
        if ktxt or vtxt:
            forms_by_page[page].append(f"{ktxt}: {vtxt}".strip(": "))

    # Tables
    tables_by_page: Dict[int, List[List[List[str]]]] = defaultdict(list)
    tables = [b for b in blocks if b.get("BlockType") == "TABLE"]
    for t in tables:
        page = t.get("Page", 1)
        cell_ids = [cid for cid in _child_ids(t, "CHILD")]
        cells = [block_map[cid] for cid in cell_ids if block_map.get(cid, {}).get("BlockType") == "CELL"]
        max_row = max([c.get("RowIndex", 1) for c in cells], default=1)
        max_col = max([c.get("ColumnIndex", 1) for c in cells], default=1)
        grid = [["" for _ in range(max_col)] for _ in range(max_row)]
        for c in cells:
            r = c.get("RowIndex", 1) - 1
            ci = c.get("ColumnIndex", 1) - 1
            grid[r][ci] = _text_from_block(c, block_map).strip()
        tables_by_page[page].append(grid)

    return lines_by_page, forms_by_page, tables_by_page
