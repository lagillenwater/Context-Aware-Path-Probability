"""
Metapath utilities for Hetionet.

Functions to enumerate and validate metapaths.
"""

import json
import itertools
from typing import List, Tuple, Dict


def load_metagraph(metagraph_path: str = 'data/metagraph.json') -> Dict:
    """Load metagraph from JSON."""
    with open(metagraph_path, 'r') as f:
        return json.load(f)


def get_edge_abbreviation(source_type: str, target_type: str,
                           relation: str, direction: str,
                           abbrev_dict: Dict) -> str:
    """
    Get edge type abbreviation.

    Parameters:
    - source_type: Source node type (e.g., "Compound")
    - target_type: Target node type (e.g., "Gene")
    - relation: Relation type (e.g., "binds")
    - direction: Edge direction ("both" or "forward")
    - abbrev_dict: Dictionary from kind_to_abbrev

    Returns:
    - abbrev: Edge abbreviation (e.g., "CbG" or "Gr>G" for directed)
    """
    source_abbrev = abbrev_dict[source_type]
    target_abbrev = abbrev_dict[target_type]
    relation_abbrev = abbrev_dict[relation]

    if direction == "forward":
        return f"{source_abbrev}{relation_abbrev}>{target_abbrev}"
    else:
        return f"{source_abbrev}{relation_abbrev}{target_abbrev}"


def enumerate_2hop_metapaths(metagraph_path: str = 'data/metagraph.json') -> List[Dict]:
    """
    Enumerate all valid 2-hop metapaths in Hetionet.

    A valid 2-hop metapath requires:
    - edge1: (source_type, intermediate_type, relation1)
    - edge2: (intermediate_type, target_type, relation2)
    - The intermediate node type must match

    Returns:
    - metapaths: List of dicts with keys:
        - metapath: String like "CbGaD"
        - edge1: String like "CbG"
        - edge2: String like "GaD"
        - source_type: String like "Compound"
        - intermediate_type: String like "Gene"
        - target_type: String like "Disease"
        - description: Human-readable description
    """
    metagraph = load_metagraph(metagraph_path)

    metaedges = metagraph['metaedge_tuples']
    abbrev_dict = metagraph['kind_to_abbrev']

    edge_list = []
    for source_type, target_type, relation, direction in metaedges:
        edge_abbrev = get_edge_abbreviation(source_type, target_type,
                                              relation, direction, abbrev_dict)
        edge_list.append({
            'abbrev': edge_abbrev,
            'source_type': source_type,
            'target_type': target_type,
            'relation': relation,
            'direction': direction
        })

    metapaths = []

    for edge1 in edge_list:
        for edge2 in edge_list:
            if edge1['target_type'] == edge2['source_type']:
                metapath_abbrev = edge1['abbrev'] + edge2['abbrev'][1:]

                description = (
                    f"{edge1['source_type']}-{edge1['relation']}-"
                    f"{edge1['target_type']}-{edge2['relation']}-"
                    f"{edge2['target_type']}"
                )

                metapaths.append({
                    'metapath': metapath_abbrev,
                    'edge1': edge1['abbrev'],
                    'edge2': edge2['abbrev'],
                    'source_type': edge1['source_type'],
                    'intermediate_type': edge1['target_type'],
                    'target_type': edge2['target_type'],
                    'description': description
                })

    return metapaths


def save_metapath_list(output_path: str = 'data/2hop_metapaths.txt'):
    """Save list of all 2-hop metapaths to file for HPC job array."""
    metapaths = enumerate_2hop_metapaths()

    with open(output_path, 'w') as f:
        for mp in metapaths:
            f.write(f"{mp['metapath']}\t{mp['edge1']}\t{mp['edge2']}\t"
                    f"{mp['description']}\n")

    print(f"Saved {len(metapaths)} 2-hop metapaths to {output_path}")
    return metapaths


if __name__ == "__main__":
    metapaths = enumerate_2hop_metapaths()
    print(f"Found {len(metapaths)} valid 2-hop metapaths\n")

    print("First 10 metapaths:")
    for i, mp in enumerate(metapaths[:10]):
        print(f"{i+1}. {mp['metapath']}: {mp['description']}")

    print("\n...")
    print(f"\nTotal: {len(metapaths)} metapaths")

    save_metapath_list()
