"""Structure-aware prediction of ligand-binding residues.

Layers, in dependency order:

``structure``   PDB/mmCIF parsing, ligand curation, geometric labelling
``features``    per-residue sequence, physicochemical and backbone features
``homology``    sequence-identity clustering (MMseqs2 or the built-in fallback)
``data``        datasets and homology-separated splits
``models``      the multimodal cross-attention transformer and baselines
``evaluate``    residue-level and protein-level metrics
``validation``  control experiments that establish what the metrics mean
"""

__version__ = "0.1.0"
