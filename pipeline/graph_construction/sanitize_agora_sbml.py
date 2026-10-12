
import os
from pathlib import Path
from tqdm import tqdm

# Path to this script's folder (~/Real_Data)
HERE = Path(__file__).resolve().parent

# Input directory (raw AGORA models)
SRC_DIR = HERE / "AGORA2_SBML"

# Output directory (clean models)
DST_DIR = HERE / "AGORA2_SBML_clean"
DST_DIR.mkdir(exist_ok=True)


def valid_xml_char(ch: str) -> bool:
    """Return True if ch is allowed in XML 1.0."""
    code = ord(ch)
    return (
        code == 0x9 or code == 0xA or code == 0xD or
        0x20 <= code <= 0xD7FF or
        0xE000 <= code <= 0xFFFD or
        0x10000 <= code <= 0x10FFFF
    )


def sanitize_file(src_path: Path, dst_path: Path) -> None:
    # Read raw bytes
    with open(src_path, "rb") as f:
        data = f.read()

    # Decode as UTF-8 but ignore any invalid bytes
    text = data.decode("utf-8", errors="ignore")

    # Remove illegal XML characters
    cleaned = "".join(ch for ch in text if valid_xml_char(ch))

    # Write back as clean UTF-8 text
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(cleaned)


def main() -> None:
    if not SRC_DIR.exists():
        raise SystemExit(f"Source directory not found: {SRC_DIR}")

    xml_files = list(SRC_DIR.glob("*.xml"))
    print(f"[*] Found {len(xml_files)} SBML models to sanitize.")

    for src in tqdm(xml_files, desc="Sanitizing SBML files"):
        dst = DST_DIR / src.name
        sanitize_file(src, dst)


if __name__ == "__main__":
    main()

