import json
import sys
from pathlib import Path

from analysis import run_vasp_benchmark_task as vasp_benchmark


def _input_set(root: Path) -> Path:
    root.mkdir()
    for name in ("INCAR", "KPOINTS", "POSCAR"):
        (root / name).write_text(f"{name} benchmark input\n", encoding="utf-8")
    (root / "POTCAR").write_text("licensed test fixture\n", encoding="utf-8")
    return root


def test_run_vasp_task_records_energy_hashes_and_removes_potcar(tmp_path):
    inputs = _input_set(tmp_path / "inputs")
    fake = tmp_path / "fake_vasp.py"
    fake.write_text(
        "from pathlib import Path\n"
        "Path('OUTCAR').write_text('free  energy   TOTEN  =      -3.12345678 eV\\n'"
        "+ 'aborting loop because EDIFF is reached\\nGeneral timing and accounting\\n')\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"

    result = vasp_benchmark.run_vasp_task(inputs, output, [sys.executable, str(fake)])

    assert result["status"] == "DONE"
    assert result["exit_code"] == 0
    assert result["final_toten_ev"] == -3.12345678
    assert result["electronic_converged"] is True
    assert result["timing_footer_present"] is True
    assert not (output / "POTCAR").exists()
    manifest = json.loads((output / "input_manifest.json").read_text(encoding="utf-8"))
    assert manifest["POTCAR"]["sha256"]
    assert json.loads((output / "task_result.json").read_text(encoding="utf-8"))["status"] == "DONE"


def test_run_vasp_task_retains_failure_and_removes_potcar(tmp_path):
    inputs = _input_set(tmp_path / "inputs")
    fake = tmp_path / "fail_vasp.py"
    fake.write_text("raise SystemExit(7)\n", encoding="utf-8")
    output = tmp_path / "output"

    result = vasp_benchmark.run_vasp_task(inputs, output, [sys.executable, str(fake)])

    assert result["status"] == "FAILED"
    assert result["exit_code"] == 7
    assert result["final_toten_ev"] is None
    assert (output / "vasp.stdout_stderr.log").is_file()
    assert not (output / "POTCAR").exists()


def test_run_vasp_task_accepts_separate_server_side_potcar_source(tmp_path):
    inputs = _input_set(tmp_path / "inputs")
    potcar = tmp_path / "licensed" / "POTCAR"
    potcar.parent.mkdir()
    inputs.joinpath("POTCAR").replace(potcar)
    fake = tmp_path / "fake_vasp.py"
    fake.write_text(
        "from pathlib import Path\n"
        "assert Path('POTCAR').read_text() == 'licensed test fixture\\n'\n"
        "Path('OUTCAR').write_text('free  energy   TOTEN  = -1.0 eV\\n'"
        "+ 'aborting loop because EDIFF is reached\\nGeneral timing and accounting\\n')\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"

    result = vasp_benchmark.run_vasp_task(
        inputs,
        output,
        [sys.executable, str(fake)],
        potcar_source=potcar,
    )

    assert result["status"] == "DONE"
    assert not (output / "POTCAR").exists()
    assert potcar.is_file()
    manifest = json.loads((output / "input_manifest.json").read_text(encoding="utf-8"))
    assert manifest["POTCAR"]["source_path"] == str(potcar.resolve())
