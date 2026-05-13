"""
Structure file loading and processing utilities.
Handles PDB/CIF files and extraction of backbone coordinates.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import warnings


def load_pdb_structure(
    pdb_path: str, chain_id: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Load backbone atom coordinates from a PDB or CIF file.

    Args:
        pdb_path: Path to PDB/CIF file.
        chain_id: Specific chain to load. If None, loads the first chain.

    Returns:
        Dictionary with:
            - 'N': N atom coordinates, shape (L, 3)
            - 'CA': CA atom coordinates, shape (L, 3)
            - 'C': C atom coordinates, shape (L, 3)
            - 'O': O atom coordinates, shape (L, 3)
            - 'sequence': amino acid sequence string
            - 'chain_id': chain identifier used
    """
    pdb_path = Path(pdb_path)
    suffix = pdb_path.suffix.lower()

    if suffix in (".pdb",):
        return _load_pdb(str(pdb_path), chain_id)
    elif suffix in (".cif", ".mmcif"):
        return _load_cif(str(pdb_path), chain_id)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _load_pdb(pdb_path: str, chain_id: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Load structure from PDB format using BioPython."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

    return _extract_backbone(model, chain_id)


def _load_cif(cif_path: str, chain_id: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Load structure from mmCIF format using BioPython."""
    from Bio.PDB import MMCIFParser

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_path)
    model = structure[0]

    return _extract_backbone(model, chain_id)


def _extract_backbone(model, chain_id: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Extract backbone coordinates from a BioPython model."""
    from Bio.PDB.Polypeptide import is_aa, three_to_one

    # Select chain
    chains = list(model.get_chains())
    if chain_id is not None:
        chain = model[chain_id]
    else:
        chain = chains[0]
        chain_id = chain.id

    # Extract residues (standard amino acids only)
    residues = [r for r in chain.get_residues() if is_aa(r, standard=True)]

    n_coords = []
    ca_coords = []
    c_coords = []
    o_coords = []
    sequence_chars = []

    backbone_atoms = {"N", "CA", "C", "O"}

    for residue in residues:
        atom_names = {a.get_name() for a in residue.get_atoms()}
        if not backbone_atoms.issubset(atom_names):
            warnings.warn(
                f"Residue {residue.get_id()} missing backbone atoms, skipping"
            )
            continue

        n_coords.append(residue["N"].get_vector().get_array())
        ca_coords.append(residue["CA"].get_vector().get_array())
        c_coords.append(residue["C"].get_vector().get_array())
        o_coords.append(residue["O"].get_vector().get_array())
        try:
            sequence_chars.append(three_to_one(residue.get_resname()))
        except KeyError:
            sequence_chars.append("X")

    return {
        "N": np.array(n_coords, dtype=np.float32),
        "CA": np.array(ca_coords, dtype=np.float32),
        "C": np.array(c_coords, dtype=np.float32),
        "O": np.array(o_coords, dtype=np.float32),
        "sequence": "".join(sequence_chars),
        "chain_id": chain_id,
    }


def get_backbone_coords(
    structure_data: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Stack backbone coordinates into a single array.

    Args:
        structure_data: Dictionary from load_pdb_structure.

    Returns:
        np.ndarray of shape (L, 4, 3) — [N, CA, C, O] for each residue.
    """
    coords = np.stack(
        [
            structure_data["N"],
            structure_data["CA"],
            structure_data["C"],
            structure_data["O"],
        ],
        axis=1,
    )
    return coords


def make_homodimer_coords(
    monomer_coords: np.ndarray,
) -> np.ndarray:
    """
    Create homodimer coordinates by duplicating monomer coordinates.
    The second copy is simply concatenated (actual symmetry operations
    would require the full oligomer PDB).

    Args:
        monomer_coords: shape (L, 4, 3)

    Returns:
        shape (2L, 4, 3)
    """
    return np.concatenate([monomer_coords, monomer_coords], axis=0)
