"""Do not load a training framework in the lightweight benchmark/evaluator."""
import subprocess
import sys


def test_deployment_frontends_do_not_import_torch():
    subprocess.run([sys.executable, "-c", "import sys; "
                    "import parseq_ocr.waterec_cli; "
                    "import parseq_ocr.waterec_benchmark; "
                    "import parseq_ocr.waterec_orientation; "
                    "import parseq_ocr.evaluation; "
                    "assert 'torch' not in sys.modules"], check=True)
