"""Load config.yaml and resolve project paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_NAME = "config.yaml"


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def data_dir(self) -> Path:
        return self.root / self.raw.get("data_dir", "data")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def output_dir(self) -> Path:
        return self.root / self.raw.get("output_dir", "outputs")

    @property
    def components(self) -> list[str]:
        return list(self.raw["components"])

    @property
    def divisions(self) -> list[str]:
        return list(self.raw.get("divisions", {}))

    @property
    def breakdowns(self) -> dict[str, list[str]]:
        """The two partitions of the headline: main aggregates and COICOP ver.2 divisions."""
        out = {"Main aggregates": self.components}
        if self.divisions:
            out["COICOP ver.2 divisions"] = self.divisions
        return out

    @property
    def energy(self) -> str:
        return self.raw.get("energy", "NRG")

    @property
    def geos(self) -> list[str]:
        return [self.raw["area"], *self.raw["countries"]]

    @property
    def required_codes(self) -> list[str]:
        return [self.raw["codes"]["headline"], *self.components, *self.divisions]

    @property
    def optional_codes(self) -> list[str]:
        return [c for k, c in self.raw["codes"].items() if k != "headline" and c]

    @property
    def windows(self) -> dict[str, tuple[str | None, str | None]]:
        return {k: (v[0], v[1]) for k, v in self.raw["evaluation"].items()}


def find_config(start: Path | None = None) -> Path:
    """Walk up from `start` (default: cwd) until config.yaml is found, so the notebook in
    notebooks/ and scripts run from the project root both work."""
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        candidate = folder / CONFIG_NAME
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{CONFIG_NAME} not found in {here} or any parent folder")


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else find_config()
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, root=path.resolve().parent)
