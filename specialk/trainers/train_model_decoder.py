"""Publications decoder trainer (compatibility wrapper).

Prefer the unified entry point:

    python -m specialk.trainers.train --dataset publications --prod
"""

import sys

from specialk.core.utils import check_torch_device
from specialk.trainers.train import main as train_main


def main(argv=None):
    # Preserve the old behaviour: CUDA machines train on the full corpus.
    prod = check_torch_device() == "cuda"
    defaults = ["--dataset", "publications" if prod else "dummy"]
    if prod:
        defaults.append("--prod")
    return train_main([*defaults, *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    main()
