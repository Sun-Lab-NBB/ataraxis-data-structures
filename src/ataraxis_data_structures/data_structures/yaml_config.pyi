from typing import Self
from pathlib import Path
from dataclasses import dataclass

@dataclass
class YamlConfig:
    def to_yaml(self, file_path: Path) -> None: ...
    @classmethod
    def from_yaml(cls, file_path: Path) -> Self: ...
