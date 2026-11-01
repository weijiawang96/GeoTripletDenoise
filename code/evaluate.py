"""Evaluate the original PUNet protocol (CD/P2F x 1e4; squared HD x 1e3)."""

import argparse
from pathlib import Path

from utils.evaluate import Evaluator
from utils.paths import PUNET_ROOT, writable_output
from utils.misc import get_logger, seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--dataset_root", default=str(PUNET_ROOT))
    parser.add_argument("--exp_name", default="GeoTripletDenoise_latest")
    args = parser.parse_args()
    writable_output(args.results_root)
    seed_all(2023)
    for folder in sorted(Path(args.results_root).iterdir()):
        parts = folder.name.rsplit("_", 1)
        if not folder.is_dir() or len(parts) != 2 or not parts[0].endswith("_poisson"):
            continue
        Evaluator(
            output_pcl_dir=str(folder),
            dataset_root=args.dataset_root,
            dataset="",
            summary_dir=args.results_root,
            experiment_name=f"{args.exp_name}_{folder.name}",
            res_gts=parts[0],
            logger=get_logger("test", str(folder)),
        ).run()


if __name__ == "__main__":
    main()
