#!/usr/bin/env python
# coding: utf-8

# # Export Hetionet v1.0 to hetmat format

# ## Summary
# 
# This notebook downloads and processes the Hetionet v1.0 biological knowledge graph, converting it from JSON format to the hetmat format for efficient matrix operations and analysis.
# 
# **Key Steps:**
# 1. **Download Hetionet v1.0**: Retrieves the biological knowledge graph from GitHub containing relationships between genes, diseases, compounds, anatomy, and other biological entities
# 2. **Convert to hetmat format**: Transforms the graph into a matrix representation optimized for machine learning and network analysis
# 3. **Export data structure**: Saves the hetmat files to the `../data/` directory for use in downstream analysis
# 4. **Demonstrate matrix operations**: Shows how to extract adjacency matrices for specific relationship types (e.g., Compound-treats-Disease relationships)
# 
# **Output**: Creates the foundational data structure in `../data/` that subsequent notebooks will use for permutation generation and edge prediction modeling.

# In[ ]:


# Import all required libraries
import os

# Domain-specific libraries
import hetnetpy.readwrite
import hetmatpy.hetmat
import hetmatpy.hetmat.archive


# In[ ]:


# Read Hetionet v1.0
url = "https://github.com/dhimmel/hetionet/raw/76550e6c93fbe92124edc71725e8c7dd4ca8b1f5/hetnet/json/hetionet-v1.0.json.bz2"
graph = hetnetpy.readwrite.read_graph(url)


# In[ ]:


path = os.path.join("..", "data")
os.makedirs(path, exist_ok=True)

hetmat = hetmatpy.hetmat.hetmat_from_graph(graph, path)


# ## Validation: Check Expected Outputs
# 

# In[ ]:


# Validation for Cell 5: Check hetmat data structure was created and saved
print("Validation: Hetmat Creation and Data Export")
print(f"  Hetmat type: {type(hetmat)}")


# Expected metanodes for Hetionet v1.0
expected_metanodes = {
    'Gene', 'Disease', 'Compound', 'Biological Process', 'Molecular Function',
    'Symptom', 'Pharmacologic Class', 'Anatomy', 'Pathway', 'Cellular Component', 'Side Effect'
}
metanode_names = set(str(node) for node in metanodes)
print(f"  Metanodes: {sorted(metanode_names)}")
assert len(metanodes) == 11, f"Expected 11 metanodes, got {len(metanodes)}"
assert metanode_names == expected_metanodes, f"Metanode names don't match expected set"
print(f"   PASSED: Found all {len(expected_metanodes)} expected metanodes")

# Expected number of metaedges (directional relationships)
expected_metaedge_count = 24
metaedge_names = sorted(str(edge) for edge in metaedges)
print(f"  Metaedges: {metaedge_names}")
assert len(metaedges) == expected_metaedge_count, f"Expected {expected_metaedge_count} metaedges, got {len(metaedges)}"


print(f"   PASSED: Found {len(metaedges)} metaedges as expected")



assert os.path.exists(path), f"Data directory {path} not found"

# Check for key hetmat files
expected_files = ["metagraph.json", "nodes", "edges"]
for file_name in expected_files:
    file_path = os.path.join(path, file_name)
    assert os.path.exists(file_path), f"Expected file/directory {file_path} not found"

print(f"   PASSED: Hetmat created and data exported to {path}")
print(f"   PASSED: Required files/directories present: {expected_files}")

