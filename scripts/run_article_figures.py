from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil


CURRENT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = CURRENT_DIR / "figures"


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, CURRENT_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Regenerate only the figures currently used in the manuscript."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    publication = load_module(
        "09_generate_publication_quality_figures.py",
        "publication_quality_figures",
    )
    figure_jobs = [
        ("Training convergence", publication.fig01_training_convergence),
        ("Latency and energy performance comparison", publication.fig02_latency_energy_comparison),
        ("Dynamic trustworthiness under R1--R5 scenarios", publication.fig05_dynamic_trustworthiness),
        ("Digital-twin ablation study", publication.fig06_dt_ablation),
    ]

    print("Regenerating manuscript experiment figures...")
    for name, func in figure_jobs:
        outputs = func()
        print(f"\n{name}:")
        for path in outputs:
            print(f"  {path}")
            if path.suffix.lower() in {".png", ".pdf"}:
                shutil.copy2(path, FIGURES_DIR / path.name)

    print("\nDone. Figures are saved under outputs/")
    print(f"Copies are also saved under {FIGURES_DIR}")


if __name__ == "__main__":
    main()


