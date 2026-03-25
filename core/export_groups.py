from __future__ import annotations

import shutil
from pathlib import Path


def ensure_clean_output_dir(output_folder: str | Path) -> Path:
    """
    Создает выходную папку, если ее нет.
    Ничего не удаляет.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder


def export_groups(
    groups: list[list[Path | str]],
    output_folder: str | Path,
    copy_files: bool = True,
) -> None:
    """
    Сохраняет группы по подпапкам:
        grouped_images/group_001/...
        grouped_images/group_002/...

    groups:
        [
            [Path("a.jpg"), Path("b.jpg")],
            [Path("c.jpg")],
            ...
        ]

    copy_files=True  -> копировать
    copy_files=False -> перемещать
    """
    output_folder = ensure_clean_output_dir(output_folder)

    for group_idx, group in enumerate(groups, start=1):
        group_dir = output_folder / f"group_{group_idx:03d}"
        group_dir.mkdir(parents=True, exist_ok=True)

        for src in group:
            src = Path(src)
            dst = group_dir / src.name

            if copy_files:
                shutil.copy2(src, dst)
            else:
                shutil.move(str(src), str(dst))