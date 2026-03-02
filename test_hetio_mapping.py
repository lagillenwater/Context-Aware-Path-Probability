#!/usr/bin/env python
"""
Quick test to debug het.io identifier mapping
"""
import requests

def build_identifier_to_id_mapping(api_url="http://localhost:8015", max_attempts=1000):
    """Build mapping from (identifier, node_type) to integer node IDs."""
    mapping = {}
    url = f"{api_url}/v1/nodes/?limit=100"
    attempts = 0

    print("Building identifier to ID mapping from het.io API...")

    while url and attempts < max_attempts:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            for node in data.get('results', []):
                identifier = node.get('identifier')
                node_id = node.get('id')
                node_type = node.get('metanode')
                if identifier and node_id is not None and node_type:
                    mapping[(str(identifier), node_type)] = node_id

                    # Debug: Print specific nodes we're looking for
                    if identifier in ["DB00762", "43"]:
                        print(f"  Found: identifier={identifier}, type={node_type}, id={node_id}")

            url = data.get('next')
            attempts += 1

            if attempts % 100 == 0:
                print(f"  Fetched {len(mapping)} node mappings...")

        except Exception as e:
            print(f"Error fetching nodes: {e}")
            break

    print(f"\nBuilt mapping with {len(mapping)} node mappings")
    return mapping

# Test
mapping = build_identifier_to_id_mapping()

# Check our test cases
test_cases = [
    ("DB00762", "Compound"),
    ("43", "Gene"),
]

print("\nTest lookups:")
for identifier, node_type in test_cases:
    key = (identifier, node_type)
    if key in mapping:
        print(f"  {key} -> node_id={mapping[key]}")
    else:
        print(f"  {key} -> NOT FOUND")

# Also check what node IDs 625 and 33 actually map to
print("\nReverse lookup (what are node IDs 625 and 33?):")
for key, node_id in mapping.items():
    if node_id in [625, 33]:
        print(f"  node_id={node_id} <- {key}")
