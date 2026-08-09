"""Build POTCAR-free inputs for the approved DFT finalization jobs."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

from analysis.dft_finalization_runner import authorized_jobs


MAGNDATA_185_VESTA_URL = (
    "https://www.cryst.ehu.es/magndata/dbfiles/alpha-Mn_1.85/1.85.alpha-Mn.vesta"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _block(lines: list[str], start: str, end: str | None = None) -> list[str]:
    try:
        index = next(i for i, line in enumerate(lines) if line.strip() == start) + 1
    except StopIteration as error:
        raise ValueError(f"missing VESTA section: {start}") from error
    result: list[str] = []
    for line in lines[index:]:
        if end is not None and line.strip().startswith(end):
            break
        result.append(line)
    return result


def expand_vesta_magnetic_structure(text: str) -> tuple[Structure, np.ndarray]:
    """Expand a MAGNDATA VESTA asymmetric unit and its axial moments."""

    lines = text.splitlines()
    cell_lines = _block(lines, "CELLP", "STRUC")
    cell = next((line.split() for line in cell_lines if len(line.split()) == 6), None)
    if cell is None:
        raise ValueError("VESTA CELLP section does not contain six lattice parameters")
    lattice = Lattice.from_parameters(*(float(value) for value in cell))

    operations: list[tuple[np.ndarray, np.ndarray, int]] = []
    for line in _block(lines, "SYMOP", "TRANM"):
        values = line.split()
        if not values:
            continue
        if values[0] == "-1.0":
            break
        if len(values) != 13:
            raise ValueError(f"invalid VESTA symmetry operation: {line}")
        translation = np.array([float(value) for value in values[:3]])
        rotation = np.array([int(value) for value in values[3:12]], dtype=int).reshape(3, 3)
        time_sign = int(values[12])
        if time_sign not in {-1, 1} or not math.isclose(abs(np.linalg.det(rotation)), 1.0):
            raise ValueError("invalid spatial or magnetic symmetry operation")
        operations.append((translation, rotation, time_sign))
    if not operations:
        raise ValueError("VESTA source contains no symmetry operations")

    sites: list[tuple[int, str, np.ndarray, int]] = []
    for line in _block(lines, "STRUC", "THERI"):
        values = line.split()
        if len(values) >= 9 and values[0].isdigit() and int(values[0]) > 0:
            sites.append(
                (
                    int(values[0]),
                    values[1],
                    np.array([float(value) for value in values[4:7]]),
                    int(values[7]),
                )
            )
    if not sites:
        raise ValueError("VESTA source contains no asymmetric sites")

    vectors: dict[int, np.ndarray] = {}
    valid_indices = {site[0] for site in sites}
    for line in _block(lines, "VECTR", "VECTT"):
        values = line.split()
        if len(values) == 5 and values[0].isdigit() and int(values[0]) in valid_indices:
            index = int(values[0])
            if index not in vectors:
                vectors[index] = np.array([float(value) for value in values[1:4]])
    if set(vectors) != valid_indices:
        raise ValueError("VESTA source is missing magnetic vectors for asymmetric sites")

    species: list[str] = []
    coordinates: list[np.ndarray] = []
    moments: list[np.ndarray] = []
    for index, element, coordinate, expected_multiplicity in sites:
        expanded: dict[tuple[float, float, float], np.ndarray] = {}
        for translation, rotation, time_sign in operations:
            transformed = np.mod(rotation @ coordinate + translation, 1.0)
            transformed[np.isclose(transformed, 1.0, atol=1e-8)] = 0.0
            key = tuple(np.round(transformed, 8))
            axial = time_sign * round(np.linalg.det(rotation)) * (rotation @ vectors[index])
            if key in expanded and not np.allclose(expanded[key], axial, atol=1e-7):
                raise ValueError(f"conflicting magnetic moments at expanded site {key}")
            expanded[key] = axial
        if len(expanded) != expected_multiplicity:
            raise ValueError(
                f"expanded multiplicity mismatch for site {index}: {len(expanded)} != {expected_multiplicity}"
            )
        for key, axial in expanded.items():
            species.append(element)
            coordinates.append(np.array(key))
            moments.append(axial)
    return Structure(lattice, species, coordinates), np.array(moments)


def _write_kpoints(path: Path, mesh: str) -> None:
    values = mesh.lower().split("x")
    if len(values) != 3 or not all(value.isdigit() and int(value) > 0 for value in values):
        raise ValueError(f"invalid explicit mesh: {mesh}")
    path.write_text(
        "Explicit Gamma mesh from frozen reciprocal spacing <= 0.15 A^-1\n"
        "0\nGamma\n"
        f"{' '.join(values)}\n",
        encoding="utf-8",
        newline="\n",
    )


def _validate_candidate_row(row: pd.Series) -> None:
    expected = {
        "ENCUT_eV": 520.0,
        "EDIFF": 1e-6,
        "EDIFFG": -0.05,
        "NELM": 160.0,
        "ISMEAR": 0.0,
        "SIGMA": 0.05,
        "IBRION": 2.0,
        "ISIF": 3.0,
        "NSW": 160.0,
    }
    for column, value in expected.items():
        if not math.isclose(float(row[column]), value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"candidate row violates frozen {column}: {row[column]}")
    if str(row["ALGO"]) != "Normal" or str(row["LDAUU"]).strip() != "0.0 3.7 0.0":
        raise ValueError("candidate row violates the frozen electronic/U protocol")
    expected_mesh = "15x9x9" if str(row["candidate_id"]) == "C120" else "15x10x9"
    if str(row["expected_mesh"]).lower() != expected_mesh:
        raise ValueError("candidate row violates the frozen structure-specific mesh")


def _candidate_incar(row: pd.Series) -> str:
    _validate_candidate_row(row)
    return "\n".join(
        [
            f"SYSTEM = {row['candidate_long_id']} {row['magnetic_initialization']} verification PBE+U relax",
            "ENCUT = 520",
            "EDIFF = 1E-6",
            "EDIFFG = -0.05",
            "IBRION = 2",
            "ISIF = 3",
            "NSW = 160",
            "NELM = 160",
            "ALGO = Normal",
            "ISMEAR = 0",
            "SIGMA = 0.05",
            "ISPIN = 2",
            f"MAGMOM = {row['MAGMOM_definition']}",
            "LREAL = .FALSE.",
            "LASPH = .TRUE.",
            "ADDGRID = .TRUE.",
            "LDAU = .TRUE.",
            "LDAUTYPE = 2",
            f"LDAUL = {row['LDAUL']}",
            f"LDAUU = {row['LDAUU']}",
            f"LDAUJ = {row['LDAUJ']}",
            "LMAXMIX = 4",
            "LWAVE = .FALSE.",
            "LCHARG = .FALSE.",
            "",
        ]
    )


def _validate_candidate_static_row(row: pd.Series) -> None:
    expected = {
        "ENCUT_eV": 520.0,
        "EDIFF": 1e-6,
        "NELM": 160.0,
        "ISMEAR": 0.0,
        "SIGMA": 0.05,
    }
    for column, value in expected.items():
        if not math.isclose(float(row[column]), value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"candidate static row violates frozen {column}: {row[column]}")
    if str(row["ALGO"]) != "Normal" or str(row["LDAUU"]).strip() != "0.0 3.7 0.0":
        raise ValueError("candidate static row violates the frozen electronic/U protocol")
    expected_mesh = "15x9x9" if str(row["candidate_id"]) == "C120" else "15x10x9"
    if str(row["expected_mesh"]).lower() != expected_mesh:
        raise ValueError("candidate static row violates the frozen structure-specific mesh")


def _candidate_static_incar(row: pd.Series) -> str:
    _validate_candidate_static_row(row)
    return "\n".join(
        [
            f"SYSTEM = {row['candidate_long_id']} {row['magnetic_initialization']} verification PBE+U frozen static",
            "ENCUT = 520",
            "EDIFF = 1E-6",
            "IBRION = -1",
            "NSW = 0",
            "NELM = 160",
            "ALGO = Normal",
            "ISMEAR = 0",
            "SIGMA = 0.05",
            "ISPIN = 2",
            f"MAGMOM = {row['MAGMOM_definition']}",
            "LREAL = .FALSE.",
            "LASPH = .TRUE.",
            "ADDGRID = .TRUE.",
            "LDAU = .TRUE.",
            "LDAUTYPE = 2",
            f"LDAUL = {row['LDAUL']}",
            f"LDAUU = {row['LDAUU']}",
            f"LDAUJ = {row['LDAUJ']}",
            "LMAXMIX = 4",
            "LWAVE = .FALSE.",
            "LCHARG = .FALSE.",
            "",
        ]
    )


def _config_hash(paths: Iterable[Path], metadata: dict[str, object]) -> str:
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode())
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def build_candidate_relaxation_inputs(
    manifest: pd.DataFrame,
    *,
    repo_root: Path,
    output_root: Path,
    git_commit: str,
) -> pd.DataFrame:
    """Create four immutable, POTCAR-free C120/C214 relaxation inputs."""

    selected = authorized_jobs(manifest, phase="candidate_relax")
    root = Path(output_root)
    if root.exists():
        raise FileExistsError(root)
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        source = (Path(repo_root) / str(row["input_structure_path"])).resolve()
        if not source.is_file() or _sha256(source) != str(row["input_structure_sha256"]).lower():
            raise ValueError(f"source structure hash mismatch: {source}")
        directory = inputs / str(row["job_id"])
        directory.mkdir()
        shutil.copyfile(source, directory / "POSCAR")
        shutil.copyfile(source, directory / "initial.POSCAR")
        structure = Structure.from_file(source)
        structure.to(filename=directory / "initial.cif", fmt="cif", symprec=None)
        (directory / "INCAR").write_text(_candidate_incar(row), encoding="utf-8", newline="\n")
        _write_kpoints(directory / "KPOINTS", str(row["expected_mesh"]))
        provenance = {
            "job_id": str(row["job_id"]),
            "candidate_id": str(row["candidate_id"]),
            "magnetic_initialization": str(row["magnetic_initialization"]),
            "calculation_scope": "reconstructed verification relaxation; not a historical result",
            "source_path": str(row["input_structure_path"]),
            "source_sha256": _sha256(source),
            "frozen_protocol_sha256": str(row.get("base_protocol_sha256", "")),
            "git_commit": git_commit,
        }
        provenance_path = directory / "input_provenance.json"
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        protected = [directory / name for name in ("INCAR", "KPOINTS", "POSCAR", "initial.POSCAR", "initial.cif")]
        records.append(
            {
                "job_id": str(row["job_id"]),
                "candidate_id": str(row["candidate_id"]),
                "magnetic_initialization": str(row["magnetic_initialization"]),
                "status": "PENDING",
                "input_dir": str(directory.resolve()),
                "config_hash": _config_hash(protected, provenance),
                "git_commit": git_commit,
            }
        )
    frame = pd.DataFrame(records)
    frame.to_csv(root / "candidate_relaxation_jobs.csv", index=False)
    return frame


def build_candidate_static_inputs(
    manifest: pd.DataFrame,
    *,
    dependency_results: Mapping[str, Mapping[str, object]],
    structural_review: Path,
    relaxation_metrics: pd.DataFrame,
    output_root: Path,
    git_commit: str,
) -> pd.DataFrame:
    """Build four POTCAR-free statics only after the generated relaxation gate passes."""

    selected = authorized_jobs(
        manifest,
        phase="candidate_static",
        dependency_results=dependency_results,
        structural_review=Path(structural_review),
    )
    metrics = relaxation_metrics.set_index("job_id", drop=False)
    dependencies = set(selected["dependency_job_ids"].astype(str))
    if set(metrics.index.astype(str)) != dependencies:
        raise ValueError("relaxation metrics do not exactly match the four static dependencies")
    root = Path(output_root)
    if root.exists():
        raise FileExistsError(root)
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        dependency = str(row["dependency_job_ids"])
        metric = metrics.loc[dependency]
        source = Path(str(metric["source_output_path"])).resolve()
        contcar = source / "CONTCAR"
        if not contcar.is_file() or _sha256(contcar) != str(metric["contcar_sha256"]):
            raise ValueError(f"dependency CONTCAR hash mismatch: {dependency}")
        recorded = dependency_results[dependency]
        if str(recorded.get("output_path")) != str(source) or str(recorded.get("contcar_sha256")) != _sha256(contcar):
            raise ValueError(f"dependency result provenance mismatch: {dependency}")
        directory = inputs / str(row["job_id"])
        directory.mkdir()
        shutil.copyfile(contcar, directory / "POSCAR")
        shutil.copyfile(contcar, directory / "initial.POSCAR")
        structure = Structure.from_file(contcar)
        structure.to(filename=directory / "initial.cif", fmt="cif", symprec=None)
        (directory / "INCAR").write_text(
            _candidate_static_incar(row), encoding="utf-8", newline="\n"
        )
        _write_kpoints(directory / "KPOINTS", str(row["expected_mesh"]))
        provenance = {
            "job_id": str(row["job_id"]),
            "candidate_id": str(row["candidate_id"]),
            "magnetic_initialization": str(row["magnetic_initialization"]),
            "calculation_scope": "frozen static on verification-relaxation final structure",
            "dependency_job_id": dependency,
            "dependency_output_path": str(source),
            "dependency_contcar_sha256": _sha256(contcar),
            "frozen_protocol_sha256": str(row.get("base_protocol_sha256", "")),
            "structural_review_path": str(Path(structural_review).resolve()),
            "structural_review_sha256": _sha256(Path(structural_review)),
            "git_commit": git_commit,
        }
        provenance_path = directory / "input_provenance.json"
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        protected = [
            directory / name
            for name in ("INCAR", "KPOINTS", "POSCAR", "initial.POSCAR", "initial.cif")
        ]
        records.append(
            {
                "job_id": str(row["job_id"]),
                "candidate_id": str(row["candidate_id"]),
                "magnetic_initialization": str(row["magnetic_initialization"]),
                "dependency_job_id": dependency,
                "status": "PENDING",
                "input_dir": str(directory.resolve()),
                "config_hash": _config_hash(protected, provenance),
                "git_commit": git_commit,
            }
        )
    frame = pd.DataFrame(records)
    frame.to_csv(root / "candidate_static_jobs.csv", index=False)
    return frame


def build_alpha_probe_input(*, vesta_path: Path, output_root: Path, git_commit: str) -> dict[str, object]:
    """Create the single approved non-scientific alpha-Mn NELM=1 probe."""

    root = Path(output_root)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    source = Path(vesta_path)
    structure, moments = expand_vesta_magnetic_structure(source.read_text(encoding="utf-8"))
    if len(structure) != 58 or moments.shape != (58, 3):
        raise ValueError("MAGNDATA 1.85 expansion did not produce 58 moment-bearing Mn sites")
    Poscar(structure).write_file(root / "POSCAR")
    Poscar(structure).write_file(root / "initial.POSCAR")
    structure.to(filename=root / "initial.cif", fmt="cif", symprec=None)
    shutil.copyfile(source, root / "source_MAGNDATA_1.85.vesta")
    magmom = " ".join(f"{value:.8f}" for vector in moments for value in vector)
    (root / "INCAR").write_text(
        "\n".join(
            [
                "SYSTEM = alpha-Mn MAGNDATA 1.85 noncollinear NELM=1 cost-memory probe",
                "ENCUT = 520",
                "EDIFF = 1E-6",
                "IBRION = -1",
                "NSW = 0",
                "NELM = 1",
                "ALGO = Normal",
                "ISMEAR = 1",
                "SIGMA = 0.2",
                "LNONCOLLINEAR = .TRUE.",
                "ISYM = 0",
                "SAXIS = 0 0 1",
                f"MAGMOM = {magmom}",
                "LREAL = .FALSE.",
                "LASPH = .TRUE.",
                "ADDGRID = .TRUE.",
                "LWAVE = .FALSE.",
                "LCHARG = .FALSE.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    _write_kpoints(root / "KPOINTS", "5x5x5")
    provenance = {
        "job_id": "dft_alpha_mn_cost_probe_pbe_noncollinear",
        "scientific_result": False,
        "NELM": 1,
        "source_url": MAGNDATA_185_VESTA_URL,
        "source_sha256": _sha256(source),
        "source_atom_count": 58,
        "magnetic_initialization": "MAGNDATA 1.85 noncollinear vectors",
        "git_commit": git_commit,
    }
    protected = [root / name for name in ("INCAR", "KPOINTS", "POSCAR", "initial.POSCAR", "initial.cif")]
    provenance["config_hash"] = _config_hash(protected, provenance)
    (root / "input_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return provenance
