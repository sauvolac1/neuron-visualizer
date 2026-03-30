#!/usr/bin/env python3
"""Add synapse data to existing HTML visualizations.

Extracts body IDs and normalization params from the embedded DATA,
fetches synapse positions from neuPrint, and patches them into the HTML.

Usage:
    python add_synapses.py Examples/Delta7_visualization.html
    python add_synapses.py --all           # All HTMLs missing synapse data
    python add_synapses.py --all --limit 5  # Only first 5 missing HTMLs
"""
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = SCRIPT_DIR / 'Examples'


def extract_data_json(html_text):
    """Extract the raw DATA JSON string from an HTML file (bracket counting)."""
    marker = 'const DATA = '
    idx = html_text.find(marker)
    if idx < 0:
        return None
    json_start = idx + len(marker)
    if html_text[json_start] != '{':
        return None
    depth = 0
    i = json_start
    while i < len(html_text):
        ch = html_text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return html_text[json_start:i + 1]
        elif ch == '"':
            i += 1
            while i < len(html_text) and html_text[i] != '"':
                if html_text[i] == '\\':
                    i += 1
                i += 1
        i += 1
    return None


def has_synapse_data(html_text):
    """Check if HTML already has non-null synapse data in the DATA bundle."""
    # The DATA JSON contains "synapseData":null or "synapseData":{"quantScale"...
    # The JS application code references DATA.synapseData (no quotes around key)
    # So we can safely search for the quoted form to match only the DATA bundle.
    if '"synapseData":null' in html_text:
        return False
    if '"synapseData":{"quantScale"' in html_text:
        return True
    # Key not present at all (old format)
    return False


def add_synapses_to_html(html_path, synapse_limit=None):
    """Fetch synapse data and patch it into an existing HTML."""
    import numpy as np
    import neuprint
    from generate_visualization import (
        get_client, fetch_synapse_positions, _build_synapse_data
    )

    html_path = Path(html_path)
    html_text = html_path.read_text(encoding='utf-8')

    # Check if already has synapse data
    if has_synapse_data(html_text):
        print(f"  SKIP: {html_path.name} (already has synapse data)")
        return False

    # Check for companion cache
    safe_name = html_path.stem.replace('_visualization', '')
    cache_path = html_path.parent / f'{safe_name}_synapses.json'
    if cache_path.exists():
        print(f"  Using cached synapse data: {cache_path.name}")
        synapse_json_str = cache_path.read_text(encoding='utf-8')
    else:
        # Ensure neuPrint client is available (may already be set by notebook)
        try:
            neuprint.default_client()
        except RuntimeError:
            get_client()

        # Extract DATA to get body IDs and norm params
        data_json_str = extract_data_json(html_text)
        if not data_json_str:
            print(f"  SKIP: {html_path.name} (no DATA bundle found)")
            return False

        print(f"  Parsing DATA bundle...")
        data = json.loads(data_json_str)

        # Extract body IDs and type lookup
        bid_type_map = data.get('bidTypeMap', {})
        if not bid_type_map:
            print(f"  SKIP: {html_path.name} (no bidTypeMap in DATA)")
            return False

        body_ids = [int(bid) for bid in bid_type_map.keys()]
        type_lookup = {int(bid): typ for bid, typ in bid_type_map.items()}
        norm_params = data.get('normParams', {})

        if not norm_params or 'cx' not in norm_params:
            print(f"  SKIP: {html_path.name} (no normParams in DATA)")
            return False

        print(f"  {len(body_ids)} neurons, fetching synapses...")

        # Fetch synapse positions
        t0 = time.time()
        syn_df, syn_bid_type_map = fetch_synapse_positions(
            body_ids, type_lookup,
            synapse_limit=synapse_limit,
        )
        elapsed = time.time() - t0

        if len(syn_df) == 0:
            print(f"  SKIP: {html_path.name} (no synapses found, {elapsed:.1f}s)")
            return False

        print(f"  {len(syn_df):,} synapses fetched in {elapsed:.1f}s")

        # Build synapse data
        synapse_data = _build_synapse_data(syn_df, syn_bid_type_map, norm_params)
        synapse_json_str = json.dumps(synapse_data, separators=(',', ':'))

        # Cache for future use
        cache_path.write_text(synapse_json_str, encoding='utf-8')
        cache_mb = os.path.getsize(cache_path) / 1e6
        print(f"  Cached to {cache_path.name} ({cache_mb:.1f} MB)")

    # Patch into HTML
    if '"synapseData":null' in html_text:
        html_text = html_text.replace('"synapseData":null', '"synapseData":' + synapse_json_str, 1)
    elif '"synapseData"' not in html_text:
        # Old format — need to add synapseData key to the DATA object
        # Find the end of the DATA object (last } before the APPLICATION marker or end of script)
        marker = ';\n\n// === APPLICATION ==='
        idx = html_text.find(marker)
        if idx < 0:
            # Try finding the closing of const DATA = {...};
            data_start = html_text.find('const DATA = ')
            if data_start < 0:
                print(f"  ERROR: Cannot patch {html_path.name} (unknown format)")
                return False
            # Find the matching close brace
            brace_start = html_text.index('{', data_start)
            depth = 0
            i = brace_start
            while i < len(html_text):
                ch = html_text[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        idx = i
                        break
                elif ch == '"':
                    i += 1
                    while i < len(html_text) and html_text[i] != '"':
                        if html_text[i] == '\\':
                            i += 1
                        i += 1
                i += 1
            else:
                print(f"  ERROR: Cannot find DATA end in {html_path.name}")
                return False
            # Insert synapseData before the closing }
            html_text = html_text[:idx] + ',"synapseData":' + synapse_json_str + html_text[idx:]
        else:
            # New format — insert before the closing } of DATA (which is at idx position)
            # The } is right before the ;\n\n marker
            close_brace = idx - 1
            while close_brace > 0 and html_text[close_brace] in ' \n\r\t':
                close_brace -= 1
            if html_text[close_brace] == '}':
                html_text = html_text[:close_brace] + ',"synapseData":' + synapse_json_str + html_text[close_brace:]
            else:
                print(f"  ERROR: Unexpected format at DATA end in {html_path.name}")
                return False

    html_path.write_text(html_text, encoding='utf-8')
    size_mb = os.path.getsize(html_path) / 1e6
    print(f"  OK: {html_path.name} ({size_mb:.1f} MB)")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Add synapse data to HTML visualizations')
    parser.add_argument('files', nargs='*', help='HTML files to process')
    parser.add_argument('--all', action='store_true', help='Process all HTMLs in Examples/')
    parser.add_argument('--limit', type=int, default=None, help='Max files to process')
    parser.add_argument('--synapse-limit', type=int, default=None, help='Max synapses per file')
    args = parser.parse_args()

    if args.all:
        targets = sorted(EXAMPLES_DIR.glob('*_visualization.html'))
    elif args.files:
        targets = [Path(f) for f in args.files]
    else:
        parser.print_help()
        sys.exit(1)

    if args.limit:
        targets = targets[:args.limit]

    print(f"Processing {len(targets)} file(s)...\n")
    t0 = time.time()
    processed = 0
    for path in targets:
        print(f"[{path.name}]")
        if add_synapses_to_html(path, synapse_limit=args.synapse_limit):
            processed += 1
        print()
    print(f"Done: {processed}/{len(targets)} files updated in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
