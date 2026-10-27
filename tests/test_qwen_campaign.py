"""Run the real Bash campaign against tiny local stand-ins for training/checkpoints."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("step,status", [(100000, 0), (100000, 17), (99999, 0)])
def test_campaign_advances_only_after_successful_complete_run(tmp_path, step, status):
    tools = tmp_path / "tools"
    tools.mkdir()
    script = tools / "run_qwen_molt_campaign.sh"
    shutil.copy(Path(__file__).resolve().parents[1] / "tools" / script.name, script)
    (tmp_path / ".env").touch()
    bin_dir = tmp_path / ".venv" / "bin"
    bin_dir.mkdir(parents=True)

    def executable(name, body):
        path = bin_dir / name
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + body)
        path.chmod(0o755)

    executable("pgrep", "exit 1\n")
    executable("sleep", "/bin/sleep 0.05\n")
    executable("python", 'cat "$2"\n')
    executable("clt", """
while (( $# )); do
    if [[ "$1" == --model.init_args.layer ]]; then layer="$2"; break; fi
    shift
done
echo "$layer" >> launched-layers
checkpoint_dir="checkpoints/molt-qwen3-4b/layer-$layer/N80-rank512to32-lr4e-5-sp7.5e-5-native-ramp80"
mkdir -p "$checkpoint_dir"
echo STEP > "$checkpoint_dir/training.ckpt"
echo 'Fake training finished'
/bin/sleep 0.1
exit STATUS
""".replace("STEP", str(step)).replace("STATUS", str(status)))

    result = subprocess.run(
        ["bash", str(script), "--first-five"], capture_output=True, text=True, timeout=15,
        env={**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]},
    )
    launched = (tmp_path / "launched-layers").read_text().splitlines()
    if step == 100000 and status == 0:
        assert result.returncode == 0, result.stdout + result.stderr
        assert launched == ["22", "0", "1", "10", "30"]
        assert "Campaign complete" in result.stdout
    else:
        assert result.returncode == (status or 1), result.stdout + result.stderr
        assert launched == ["22"]
