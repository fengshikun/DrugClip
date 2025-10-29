# lazy_sdf_dataset.py
from __future__ import annotations
import os, io, mmap, json
from functools import lru_cache
from typing import List, Dict, Any, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

class LazySDFDataset:
    """
    Lazy-loading SDF dataset with random access via byte-range index.

    Returns per item a dict:
    {
      'atoms': List[str],
      'coordinates': List[np.ndarray(shape=(N,3), dtype=np.float32)],
      'smi': str,
      'mol': rdkit.Chem.rdchem.Mol,
      'label': 0
    }
    """
    def __init__(
        self,
        sdf_path: str,
        sanitize: bool = True,
        remove_hs: bool = False,
        embed_if_missing: bool = False,
        keep_all_conformers: bool = True,
        index_path: str | None = None,
    ):
        assert os.path.isfile(sdf_path), f"{sdf_path} not found"
        self.sdf_path = sdf_path
        self.sanitize = sanitize
        self.remove_hs = remove_hs
        self.embed_if_missing = embed_if_missing
        self.keep_all_conformers = keep_all_conformers

        self.index_path = index_path or (sdf_path + ".idx.npy")
        self._offsets = self._load_or_build_index(self.sdf_path, self.index_path)

    # ---------- Indexing ----------
    @staticmethod
    def _scan_offsets(sdf_path: str) -> np.ndarray:
        """
        Scan the SDF file and return an array of shape (M, 2) with (start, end) byte offsets
        for each record (molecule). Records are delimited by lines with '$$$$'.
        """
        offsets: List[Tuple[int, int]] = []
        with open(sdf_path, "rb") as f:
            start = f.tell()  # start of current record
            while True:
                line = f.readline()
                if not line:  # EOF
                    # If the file doesn't end with $$$$, we still close the last record.
                    pos = f.tell()
                    if pos > start:
                        offsets.append((start, pos))
                    break
                if line.strip() == b"$$$$":
                    end = f.tell()
                    offsets.append((start, end))
                    start = f.tell()  # next record starts after $$$$

        return np.array(offsets, dtype=np.int64)

    def _load_or_build_index(self, sdf_path: str, index_path: str) -> np.ndarray:
        if os.path.isfile(index_path):
            try:
                arr = np.load(index_path)
                if arr.ndim == 2 and arr.shape[1] == 2:
                    return arr
            except Exception:
                pass  # fall back to rebuild
        arr = self._scan_offsets(sdf_path)
        np.save(index_path, arr)
        return arr

    # ---------- Dataset proto ----------
    def __len__(self) -> int:
        return int(self._offsets.shape[0])

    @staticmethod
    def _atoms(mol: Chem.Mol) -> List[str]:
        return [a.GetSymbol() for a in mol.GetAtoms()]

    def _coordinates(self, mol: Chem.Mol) -> List[np.ndarray]:
        coords_list: List[np.ndarray] = []
        n_conf = mol.GetNumConformers()
        if n_conf == 0:
            return coords_list
        conf_ids = range(n_conf) if self.keep_all_conformers else [0]
        N = mol.GetNumAtoms()
        for cid in conf_ids:
            conf = mol.GetConformer(cid)
            arr = np.empty((N, 3), dtype=np.float32)
            for i in range(N):
                p = conf.GetAtomPosition(i)
                arr[i, 0] = p.x
                arr[i, 1] = p.y
                arr[i, 2] = p.z
            coords_list.append(arr)
        return coords_list

    @staticmethod
    def _smiles(mol: Chem.Mol) -> str:
        try:
            smi = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
        except Exception:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        return smi

    def _ensure_3d(self, mol: Chem.Mol) -> Chem.Mol:
        # If no conformers (or 2D), optionally generate one with ETKDG
        if mol.GetNumConformers() == 0 and self.embed_if_missing:
            mH = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 17
            if AllChem.EmbedMolecule(mH, params=params) == 0:
                try:
                    AllChem.UFFOptimizeMolecule(mH, maxIters=200)
                except Exception:
                    pass
            mol = Chem.RemoveHs(mH) if self.remove_hs else mH
        elif self.remove_hs:
            mol = Chem.RemoveHs(mol)
        return mol

    def _parse_record_bytes(self, chunk: bytes) -> Chem.Mol | None:
        """
        Parse a single SDF record from raw bytes.
        Use ForwardSDMolSupplier on a BytesIO stream to respect SDF properties.
        """
        bio = io.BytesIO(chunk)
        sup = Chem.ForwardSDMolSupplier(bio, sanitize=self.sanitize, removeHs=False)
        try:
            mol = next(iter(sup), None)
        except Exception:
            mol = None
        return mol

    @lru_cache(maxsize=16)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        start, end = map(int, self._offsets[idx])
        # Read only the needed byte range:
        with open(self.sdf_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start)

        mol = self._parse_record_bytes(chunk)
        if mol is None:
            # 返回空壳，或抛错；这里选择抛错更容易暴露坏样本
            raise ValueError(f"Failed to parse molecule at index {idx} (bytes {start}:{end}).")

        # optional 3D embedding / H handling
        mol = self._ensure_3d(mol)

        data: Dict[str, Any] = {
            "atoms": self._atoms(mol),
            "coordinates": self._coordinates(mol),   # List[np.ndarray], 与你的LMDB风格一致
            "smi": self._smiles(mol),
            "mol": mol,
            "label": 0,
        }
        return data
