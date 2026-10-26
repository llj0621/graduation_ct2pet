"""Standalone PET lesion evaluation from this source checkout."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ct2pet_cervical_cancer.pipelines.evaluate_lesions import main

if __name__ == "__main__":
    main()
