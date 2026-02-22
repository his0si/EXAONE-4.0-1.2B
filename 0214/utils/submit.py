"""Create competition submission zip."""
import os
import sys
import shutil
import tempfile
from pathlib import Path


def create_submission(model_dir, output_name, submissions_dir=None):
    if submissions_dir is None:
        submissions_dir = os.path.join(os.path.dirname(__file__), "..", "submissions")
    os.makedirs(submissions_dir, exist_ok=True)

    output_path = os.path.join(submissions_dir, output_name)
    tmp = tempfile.mkdtemp()
    try:
        model_dst = os.path.join(tmp, "model")
        shutil.copytree(model_dir, model_dst)
        shutil.make_archive(output_path, "zip", tmp)
        size_mb = os.path.getsize(f"{output_path}.zip") / (1024**2)
        print(f"[INFO] Created {output_path}.zip ({size_mb:.1f} MB)")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python submit.py <model_dir> <output_name>")
        sys.exit(1)
    create_submission(sys.argv[1], sys.argv[2])
