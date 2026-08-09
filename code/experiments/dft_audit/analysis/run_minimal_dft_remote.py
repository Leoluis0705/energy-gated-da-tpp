#!/usr/bin/env python3
"""Execute the frozen five-candidate DFT batch on the authorized server.

The runner is intentionally self-contained. It assembles licensed PAW datasets
only inside each remote runtime directory and removes the assembled POTCAR after
every attempt. Probe candidates are completed and gated before the remaining
three candidates can start.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


MAGNETIC_STATES = ("FM", "AFM_or_ferri")
STAGES = ("relax", "static")
FORCE_THRESHOLD_EV_A = 0.05
SEVERE_MINIMUM_DISTANCE_A = 1.20
SEVERE_VOLUME_CHANGE_PERCENT = 30.0
DEFAULT_STAGE_WALL_LIMIT_SECONDS = 6 * 3600


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _last_float(pattern: str, text: str) -> float | None:
    values = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return float(values[-1]) if values else None


def _last_fmax(text: str) -> float | None:
    blocks = re.findall(
        r"TOTAL-FORCE\s*\(eV/Angst\)\s*\n\s*-+\s*\n(.*?)(?=\n\s*-{5,})",
        text,
        flags=re.DOTALL,
    )
    if not blocks:
        return None
    maximum = None
    for line in blocks[-1].splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        try:
            fx, fy, fz = (float(value) for value in fields[-3:])
        except ValueError:
            continue
        norm = math.sqrt(fx * fx + fy * fy + fz * fz)
        maximum = norm if maximum is None else max(maximum, norm)
    return maximum


def _last_local_moments(text: str) -> list[float]:
    blocks = re.findall(
        r"magnetization\s*\(x\)(.*?)(?=\n\s*magnetization\s*\([xyz]\)|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not blocks:
        return []
    moments: list[float] = []
    for line in blocks[-1].splitlines():
        fields = line.split()
        if not fields or not fields[0].isdigit() or len(fields) < 5:
            continue
        try:
            moments.append(float(fields[-1]))
        except ValueError:
            continue
    return moments


def parse_outcar_text(text: str) -> dict[str, Any]:
    version = re.search(r"\bvasp\.([0-9]+(?:\.[0-9]+)+)", text, re.IGNORECASE)
    return {
        "vasp_version": version.group(1) if version else "",
        "electronic_converged": "aborting loop because EDIFF is reached" in text,
        "ionic_marker_present": (
            "reached required accuracy - stopping structural energy minimisation"
            in text
        ),
        "timing_footer_present": "General timing and accounting" in text,
        "final_toten_eV": _last_float(
            r"free\s+energy\s+TOTEN\s*=\s*([-+0-9.Ee]+)", text
        ),
        "entropy_term_eV": _last_float(
            r"entropy\s+T\*S\s+EENTRO\s*=\s*([-+0-9.Ee]+)", text
        ),
        "Fmax_eV_A": _last_fmax(text),
        "local_moments_muB": _last_local_moments(text),
    }


def parse_oszicar_text(text: str) -> dict[str, Any]:
    steps = re.findall(r"^\s*(\d+)\s+F=", text, flags=re.MULTILINE)
    moments = re.findall(r"\bmag=\s*([-+0-9.Ee]+)", text)
    return {
        "last_ionic_step": int(steps[-1]) if steps else None,
        "final_total_moment_muB": float(moments[-1]) if moments else None,
    }


def _minimum_distance(structure: Structure) -> float:
    matrix = structure.distance_matrix
    values = [
        float(matrix[i][j])
        for i in range(len(structure))
        for j in range(i + 1, len(structure))
    ]
    return min(values)


def structure_metrics(initial: Structure, final: Structure) -> dict[str, Any]:
    if len(initial) != len(final):
        return {
            "initial_volume_A3": float(initial.volume),
            "final_volume_A3": float(final.volume),
            "relative_volume_change_percent": None,
            "minimum_interatomic_distance_A": None,
            "final_space_group": "",
            "structure_collapsed": True,
            "collapse_reasons": ["atom_count_changed"],
        }
    volume_change = 100.0 * (final.volume - initial.volume) / initial.volume
    minimum = _minimum_distance(final)
    reasons = []
    if minimum < SEVERE_MINIMUM_DISTANCE_A:
        reasons.append("minimum_distance_below_1p20_A")
    if abs(volume_change) > SEVERE_VOLUME_CHANGE_PERCENT:
        reasons.append("absolute_volume_change_above_30_percent")
    try:
        space_group = (
            f"{SpacegroupAnalyzer(final, symprec=0.1).get_space_group_symbol()} "
            f"({SpacegroupAnalyzer(final, symprec=0.1).get_space_group_number()})"
        )
    except Exception:
        space_group = ""
    return {
        "initial_volume_A3": float(initial.volume),
        "final_volume_A3": float(final.volume),
        "relative_volume_change_percent": float(volume_change),
        "minimum_interatomic_distance_A": minimum,
        "final_space_group": space_group,
        "structure_collapsed": bool(reasons),
        "collapse_reasons": reasons,
    }


def formation_energy_eV_atom(
    *,
    total_energy_eV: float,
    composition: Mapping[str, float],
    references_eV_atom: Mapping[str, float],
) -> float:
    missing = sorted(set(composition) - set(references_eV_atom))
    if missing:
        raise ValueError(f"missing elemental references: {missing}")
    atom_count = sum(float(value) for value in composition.values())
    reference_sum = sum(
        float(amount) * float(references_eV_atom[element])
        for element, amount in composition.items()
    )
    return (float(total_energy_eV) - reference_sum) / atom_count


def _parse_resource_usage(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"peak_rss_kb": None}
    text = path.read_text(encoding="utf-8", errors="replace")
    maximum = _last_float(
        r"Maximum resident set size \(kbytes\):\s*([0-9]+)", text
    )
    return {"peak_rss_kb": int(maximum) if maximum is not None else None}


def candidate_gate_passes(
    rows: list[dict[str, Any]],
    *,
    memory_limit_kb: int,
    wall_limit_seconds: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    expected = {(state, stage) for state in MAGNETIC_STATES for stage in STAGES}
    observed = {
        (str(row.get("magnetic_state")), str(row.get("stage"))) for row in rows
    }
    if observed != expected:
        reasons.append(f"incomplete_stage_set:{sorted(expected - observed)}")
    for row in rows:
        identity = f"{row.get('magnetic_state')}/{row.get('stage')}"
        if int(row.get("exit_code", -1)) != 0:
            reasons.append(f"{identity}:nonzero_exit")
        if not bool(row.get("electronic_converged")):
            reasons.append(f"{identity}:electronic_nonconvergence")
        if not bool(row.get("timing_footer_present")):
            reasons.append(f"{identity}:missing_timing_footer")
        if bool(row.get("potcar_retained")):
            reasons.append(f"{identity}:potcar_retained")
        if bool(row.get("structure_collapsed")):
            reasons.append(f"{identity}:structure_collapse")
        if row.get("stage") == "relax" and not (
            bool(row.get("ionic_converged"))
            or bool(row.get("nsw_limit_reached"))
        ):
            reasons.append(f"{identity}:relaxation_not_converged_or_at_nsw_limit")
        if row.get("stage") == "static" and not bool(
            row.get("formation_energy_recomputed")
        ):
            reasons.append(f"{identity}:formation_energy_not_recomputed")
        peak = row.get("peak_rss_kb")
        if peak is not None and int(peak) > int(memory_limit_kb):
            reasons.append(f"{identity}:memory_budget_exceeded")
        elapsed = row.get("elapsed_seconds")
        if elapsed is not None and float(elapsed) > float(wall_limit_seconds):
            reasons.append(f"{identity}:wall_budget_exceeded")
    return not reasons, reasons


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _reference_channels(path: Path) -> dict[str, dict[str, float]]:
    channels: dict[str, dict[str, float]] = {}
    for row in _read_csv(path):
        if str(row.get("electronic_converged", "")).lower() != "true":
            continue
        functional = str(row["functional"])
        channels.setdefault(functional, {})[str(row["element"])] = float(
            row["energy_per_atom_eV"]
        )
    return channels


def _assemble_potcar(
    *,
    destination: Path,
    element_order: list[str],
    paths: Mapping[str, str],
    expected_hashes: Mapping[str, str],
) -> dict[str, Any]:
    components = []
    with destination.open("xb") as output:
        for element in element_order:
            source = Path(paths[element])
            observed_hash = sha256(source)
            if observed_hash.lower() != str(expected_hashes[element]).lower():
                raise RuntimeError(f"server POTCAR hash mismatch for {element}")
            with source.open("rb") as handle:
                shutil.copyfileobj(handle, output)
            components.append(
                {
                    "element": element,
                    "path": str(source),
                    "sha256": observed_hash,
                    "size_bytes": source.stat().st_size,
                }
            )
    return {
        "components": components,
        "assembled_sha256": sha256(destination),
        "assembled_size_bytes": destination.stat().st_size,
    }


def _run_process(
    *,
    command: list[str],
    cwd: Path,
    log_path: Path,
    resource_path: Path,
    timeout_seconds: int,
) -> tuple[int, bool, float]:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "1",
        }
    )
    time_binary = Path("/usr/bin/time")
    timed_command = (
        [str(time_binary), "-v", "-o", str(resource_path), *command]
        if time_binary.is_file() and os.access(time_binary, os.X_OK)
        else command
    )
    started = time.monotonic()
    timed_out = False
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            timed_command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            exit_code = int(process.wait(timeout=timeout_seconds))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                exit_code = int(process.wait(timeout=60))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                exit_code = int(process.wait(timeout=60))
    if not resource_path.is_file():
        import resource

        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        resource_path.write_text(
            "resource_source: getrusage_RUSAGE_CHILDREN\n"
            f"Maximum resident set size (kbytes): {int(usage.ru_maxrss)}\n",
            encoding="utf-8",
        )
    return exit_code, timed_out, time.monotonic() - started


def _band_gap(vasprun: Path) -> tuple[float | None, bool | None]:
    if not vasprun.is_file():
        return None, None
    try:
        from pymatgen.io.vasp.outputs import Vasprun

        parsed = Vasprun(
            vasprun,
            parse_dos=False,
            parse_eigen=True,
            parse_projected_eigen=False,
            exception_on_bad_xml=False,
        )
        gap, _, _, direct = parsed.eigenvalue_band_properties
        return float(gap), bool(direct)
    except Exception:
        return None, None


def _stage_row(
    *,
    output: Path,
    metadata: dict[str, Any],
    exit_code: int,
    timed_out: bool,
    elapsed_seconds: float,
    references: dict[str, dict[str, float]],
) -> dict[str, Any]:
    outcar_path = output / "OUTCAR"
    oszicar_path = output / "OSZICAR"
    outcar_text = (
        outcar_path.read_text(encoding="utf-8", errors="ignore")
        if outcar_path.is_file()
        else ""
    )
    oszicar_text = (
        oszicar_path.read_text(encoding="utf-8", errors="ignore")
        if oszicar_path.is_file()
        else ""
    )
    parsed = parse_outcar_text(outcar_text)
    oszicar = parse_oszicar_text(oszicar_text)
    metrics = {
        "structure_collapsed": True,
        "collapse_reasons": ["missing_or_unparsable_structure"],
    }
    try:
        initial = Structure.from_file(output / "POSCAR")
        final = Structure.from_file(output / "CONTCAR")
        metrics = structure_metrics(initial, final)
    except Exception:
        pass
    nsw = int(
        _last_float(
            r"^\s*NSW\s*=\s*([0-9]+)",
            (output / "INCAR").read_text(encoding="utf-8", errors="ignore"),
        )
        or 0
    )
    ionic_converged = bool(
        parsed["ionic_marker_present"]
        and parsed["Fmax_eV_A"] is not None
        and float(parsed["Fmax_eV_A"]) <= FORCE_THRESHOLD_EV_A + 1e-9
    )
    nsw_limit = bool(
        metadata["stage"] == "relax"
        and nsw > 0
        and oszicar["last_ionic_step"] is not None
        and int(oszicar["last_ionic_step"]) >= nsw
        and not ionic_converged
    )
    formation = None
    formation_recomputed = False
    if (
        metadata["stage"] == "static"
        and parsed["final_toten_eV"] is not None
        and metadata["functional"] in references
    ):
        composition = Composition(metadata["formula"]).get_el_amt_dict()
        formation = formation_energy_eV_atom(
            total_energy_eV=float(parsed["final_toten_eV"]),
            composition=composition,
            references_eV_atom=references[metadata["functional"]],
        )
        formation_recomputed = True
    band_gap, direct_gap = (
        _band_gap(output / "vasprun.xml")
        if metadata["stage"] == "static"
        else (None, None)
    )
    resource = _parse_resource_usage(output / "resource_usage.txt")
    atom_count = sum(Composition(metadata["formula"]).get_el_amt_dict().values())
    entropy = parsed["entropy_term_eV"]
    row = {
        "candidate_id": metadata["candidate_id"],
        "candidate_stratum": metadata["candidate_stratum"],
        "formula": metadata["formula"],
        "element": metadata["element"],
        "functional": metadata["functional"],
        "magnetic_state": metadata["magnetic_state"],
        "stage": metadata["stage"],
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed_seconds,
        "output_path": str(output),
        "electronic_converged": bool(parsed["electronic_converged"]),
        "ionic_converged": ionic_converged,
        "nsw_limit_reached": nsw_limit,
        "last_ionic_step": oszicar["last_ionic_step"],
        "Fmax_eV_A": parsed["Fmax_eV_A"],
        "timing_footer_present": bool(parsed["timing_footer_present"]),
        "final_toten_eV": parsed["final_toten_eV"],
        "formation_energy_eV_atom": formation,
        "formation_energy_recomputed": formation_recomputed,
        "entropy_term_eV": entropy,
        "entropy_term_abs_eV_atom": (
            abs(float(entropy)) / float(atom_count) if entropy is not None else None
        ),
        "smearing_review_required": (
            entropy is not None and abs(float(entropy)) / float(atom_count) > 0.001
        ),
        "band_gap_eV": band_gap,
        "direct_band_gap": direct_gap,
        "final_total_moment_muB": oszicar["final_total_moment_muB"],
        "local_moments_muB_json": json.dumps(parsed["local_moments_muB"]),
        "vasp_version": parsed["vasp_version"],
        "potcar_retained": (output / "POTCAR").exists(),
        "outcar_sha256": sha256(outcar_path) if outcar_path.is_file() else "",
        "contcar_sha256": (
            sha256(output / "CONTCAR") if (output / "CONTCAR").is_file() else ""
        ),
        **resource,
        **metrics,
    }
    return row


def _is_retryable_operational_failure(row: dict[str, Any]) -> bool:
    return bool(
        int(row["exit_code"]) != 0
        and not row["outcar_sha256"]
        and not row["electronic_converged"]
    )


def run_stage(
    *,
    run_root: Path,
    inputs_root: Path,
    stage_manifest_row: dict[str, str],
    vasp: Path,
    potcar_paths: Mapping[str, str],
    potcar_hashes: Mapping[str, str],
    references: dict[str, dict[str, float]],
    timeout_seconds: int,
) -> dict[str, Any]:
    relative = Path(stage_manifest_row["relative_stage_dir"])
    source = inputs_root / relative
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    stage_root = run_root / "results" / relative
    stage_root.mkdir(parents=True, exist_ok=True)
    completed_result = stage_root / "stage_result.json"
    if completed_result.is_file():
        return json.loads(completed_result.read_text(encoding="utf-8"))

    last_row = None
    for attempt in (1, 2):
        output = stage_root / f"attempt_{attempt}"
        if output.exists():
            raise FileExistsError(f"unrecorded stage attempt already exists: {output}")
        output.mkdir(parents=True)
        for name in ("INCAR", "KPOINTS"):
            shutil.copy2(source / name, output / name)
        if metadata["stage"] == "static":
            dependency_stage_root = (
                run_root
                / "results"
                / Path(str(metadata["poscar_dependency"])).parent
            )
            dependency_result = dependency_stage_root / "stage_result.json"
            if not dependency_result.is_file():
                raise FileNotFoundError(
                    f"relaxation stage result is unavailable: {dependency_result}"
                )
            dependency_payload = json.loads(
                dependency_result.read_text(encoding="utf-8")
            )
            dependency = (
                dependency_stage_root
                / f"attempt_{int(dependency_payload['attempt'])}"
                / "CONTCAR"
            )
            if not dependency.is_file():
                raise FileNotFoundError(
                    f"static dependency CONTCAR is unavailable: {metadata['poscar_dependency']}"
                )
            shutil.copy2(dependency, output / "POSCAR")
        else:
            shutil.copy2(source / "POSCAR", output / "POSCAR")
        potcar = output / "POTCAR"
        provenance = _assemble_potcar(
            destination=potcar,
            element_order=list(metadata["element_order"]),
            paths=potcar_paths,
            expected_hashes=potcar_hashes,
        )
        write_json(output / "POTCAR_provenance.json", provenance)
        started_at = utc_now()
        try:
            exit_code, timed_out, elapsed = _run_process(
                command=[str(vasp)],
                cwd=output,
                log_path=output / "vasp.stdout_stderr.log",
                resource_path=output / "resource_usage.txt",
                timeout_seconds=timeout_seconds,
            )
        finally:
            potcar.unlink(missing_ok=True)
        row = _stage_row(
            output=output,
            metadata=metadata,
            exit_code=exit_code,
            timed_out=timed_out,
            elapsed_seconds=elapsed,
            references=references,
        )
        row["attempt"] = attempt
        row["started_at_utc"] = started_at
        row["ended_at_utc"] = utc_now()
        write_json(output / "task_result.json", row)
        last_row = row
        if attempt == 1 and _is_retryable_operational_failure(row):
            continue
        break
    if last_row is None:
        raise RuntimeError("stage produced no result")
    write_json(completed_result, last_row)
    return last_row


def run_candidate(
    *,
    candidate_id: str,
    stage_rows: list[dict[str, str]],
    run_root: Path,
    inputs_root: Path,
    vasp: Path,
    potcar_paths: Mapping[str, str],
    potcar_hashes: Mapping[str, str],
    references: dict[str, dict[str, float]],
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    rows = [row for row in stage_rows if row["candidate_id"] == candidate_id]
    if len(rows) != 4:
        raise ValueError(f"{candidate_id}: expected four frozen stages, found {len(rows)}")
    indexed = {(row["magnetic_state"], row["stage"]): row for row in rows}
    results = []
    for state in MAGNETIC_STATES:
        relax = run_stage(
            run_root=run_root,
            inputs_root=inputs_root,
            stage_manifest_row=indexed[(state, "relax")],
            vasp=vasp,
            potcar_paths=potcar_paths,
            potcar_hashes=potcar_hashes,
            references=references,
            timeout_seconds=timeout_seconds,
        )
        results.append(relax)
        if (
            int(relax["exit_code"]) != 0
            or not relax["electronic_converged"]
            or not relax["timing_footer_present"]
            or relax["structure_collapsed"]
            or not (relax["ionic_converged"] or relax["nsw_limit_reached"])
        ):
            continue
        results.append(
            run_stage(
                run_root=run_root,
                inputs_root=inputs_root,
                stage_manifest_row=indexed[(state, "static")],
                vasp=vasp,
                potcar_paths=potcar_paths,
                potcar_hashes=potcar_hashes,
                references=references,
                timeout_seconds=timeout_seconds,
            )
        )
    return results


def execute(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root).resolve()
    package = Path(args.package_root).resolve()
    status_path = run_root / "status.json"
    run_root.mkdir(parents=True, exist_ok=True)
    if status_path.exists():
        existing = json.loads(status_path.read_text(encoding="utf-8"))
        if existing.get("protocol_sha256") != args.protocol_sha256:
            raise RuntimeError("existing remote run has a different frozen protocol hash")
    protocol = json.loads((package / "dft_protocol_frozen.json").read_text(encoding="utf-8"))
    if protocol.get("overall_status") != "PASS":
        raise RuntimeError("frozen protocol is not PASS")
    if sha256(package / "dft_protocol_frozen.json") != args.protocol_sha256:
        raise RuntimeError("uploaded frozen protocol hash mismatch")
    stage_rows = _read_csv(package / "inputs" / "stage_manifest.csv")
    references = _reference_channels(package / "elemental_references.csv")
    potcar_paths = json.loads(args.potcar_paths_json)
    potcar_hashes = json.loads(args.potcar_hashes_json)
    probes = [value for value in args.probe_candidates.split(",") if value]
    remaining = [value for value in args.remaining_candidates.split(",") if value]
    if set(probes).intersection(remaining) or len(probes) != 2 or len(remaining) != 3:
        raise ValueError("expected two distinct probes and three distinct remaining candidates")
    state = {
        "status": "PROBES_RUNNING",
        "started_at_utc": utc_now(),
        "protocol_sha256": args.protocol_sha256,
        "same_scale_status": protocol["same_scale_status"],
        "probe_candidates": probes,
        "remaining_candidates": remaining,
        "completed_candidates": [],
        "current_candidate": None,
        "current_phase": "probes",
    }
    write_json(status_path, state)
    all_rows: list[dict[str, Any]] = []
    memory_limit_kb = int(args.memory_limit_kb)
    for candidate_id in probes:
        state["current_candidate"] = candidate_id
        write_json(status_path, state)
        candidate_rows = run_candidate(
            candidate_id=candidate_id,
            stage_rows=stage_rows,
            run_root=run_root,
            inputs_root=package / "inputs",
            vasp=Path(args.vasp),
            potcar_paths=potcar_paths,
            potcar_hashes=potcar_hashes,
            references=references,
            timeout_seconds=args.stage_timeout_seconds,
        )
        all_rows.extend(candidate_rows)
        state["completed_candidates"].append(candidate_id)
        write_csv(run_root / "stage_results.csv", all_rows)
        write_json(status_path, state)
    probe_gate = {}
    for candidate_id in probes:
        candidate_rows = [
            row for row in all_rows if row["candidate_id"] == candidate_id
        ]
        passed, reasons = candidate_gate_passes(
            candidate_rows,
            memory_limit_kb=memory_limit_kb,
            wall_limit_seconds=args.stage_timeout_seconds,
        )
        probe_gate[candidate_id] = {"pass": passed, "reasons": reasons}
    write_json(run_root / "probe_gate.json", probe_gate)
    if not all(row["pass"] for row in probe_gate.values()):
        state.update(
            {
                "status": "STOPPED_BY_PROBE_GATE",
                "current_candidate": None,
                "current_phase": "terminal",
                "ended_at_utc": utc_now(),
                "probe_gate": probe_gate,
            }
        )
        write_json(status_path, state)
        write_json(run_root / "final_status.json", state)
        return 2

    state.update(
        {
            "status": "REMAINING_RUNNING",
            "current_phase": "remaining",
            "probe_gate": probe_gate,
        }
    )
    write_json(status_path, state)
    for candidate_id in remaining:
        state["current_candidate"] = candidate_id
        write_json(status_path, state)
        candidate_rows = run_candidate(
            candidate_id=candidate_id,
            stage_rows=stage_rows,
            run_root=run_root,
            inputs_root=package / "inputs",
            vasp=Path(args.vasp),
            potcar_paths=potcar_paths,
            potcar_hashes=potcar_hashes,
            references=references,
            timeout_seconds=args.stage_timeout_seconds,
        )
        all_rows.extend(candidate_rows)
        state["completed_candidates"].append(candidate_id)
        write_csv(run_root / "stage_results.csv", all_rows)
        write_json(status_path, state)
    candidate_gates = {}
    for candidate_id in [*probes, *remaining]:
        candidate_rows = [
            row for row in all_rows if row["candidate_id"] == candidate_id
        ]
        passed, reasons = candidate_gate_passes(
            candidate_rows,
            memory_limit_kb=memory_limit_kb,
            wall_limit_seconds=args.stage_timeout_seconds,
        )
        candidate_gates[candidate_id] = {"pass": passed, "reasons": reasons}
    state.update(
        {
            "status": "COMPLETE",
            "current_candidate": None,
            "current_phase": "terminal",
            "ended_at_utc": utc_now(),
            "candidate_gates": candidate_gates,
        }
    )
    write_json(run_root / "candidate_gates.json", candidate_gates)
    write_json(status_path, state)
    write_json(run_root / "final_status.json", state)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--vasp", required=True)
    parser.add_argument("--potcar-paths-json", required=True)
    parser.add_argument("--potcar-hashes-json", required=True)
    parser.add_argument("--probe-candidates", required=True)
    parser.add_argument("--remaining-candidates", required=True)
    parser.add_argument(
        "--stage-timeout-seconds",
        type=int,
        default=DEFAULT_STAGE_WALL_LIMIT_SECONDS,
    )
    parser.add_argument("--memory-limit-kb", type=int, default=50 * 1024 * 1024)
    return execute(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
