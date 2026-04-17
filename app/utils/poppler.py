import platform
from pathlib import Path

from app.config import settings


def _contains_pdftoppm(bin_dir: Path) -> bool:
    executable = "pdftoppm.exe" if platform.system() == "Windows" else "pdftoppm"
    return (bin_dir / executable).exists()


def _normalize_poppler_bin(path_value: Path) -> str | None:
    candidates = [
        path_value,
        path_value / "Library" / "bin",
        path_value / "bin",
    ]

    for candidate in candidates:
        if candidate.exists() and _contains_pdftoppm(candidate):
            return str(candidate)

    return None


def resolve_poppler_path() -> str | None:
    """Resolve Poppler bin directory without relying on global PATH."""
    if settings.POPPLER_PATH:
        normalized = _normalize_poppler_bin(Path(settings.POPPLER_PATH))
        if normalized:
            return normalized

    if platform.system() != "Windows":
        return None

    winget_root = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    if not winget_root.exists():
        return None

    package_dirs = sorted(
        winget_root.glob("oschwartz10612.Poppler_*"),
        reverse=True,
    )

    for package_dir in package_dirs:
        bin_candidates = sorted(package_dir.glob("poppler-*/*/bin"), reverse=True)
        for bin_dir in bin_candidates:
            if _contains_pdftoppm(bin_dir):
                return str(bin_dir)

    return None
