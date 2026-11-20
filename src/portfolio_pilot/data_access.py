"""Get the raw SCF file automatically, so the project runs without a manual download.

If data/raw/SCFP2025.csv is missing, the summary extract is downloaded once from the
Federal Reserve (scf.url in config.yaml), unzipped, and saved there. The raw file is
then only read, never changed. A SHA-256 checksum is printed so any later revision of the
public file by the Federal Reserve can be detected (set scf.sha256 in config.yaml to check it).
"""
import hashlib
import io
import urllib.request
import zipfile


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_scf(cfg):
    """Return the local SCF file, downloading it from paths.scf_url first if it is missing."""
    path = cfg["paths"]["scf_file"]
    if not path.exists():
        url = cfg["scf"]["url"]
        print("SCF file not found locally; downloading", url)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (PortfolioPilot course project)"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csvs:
                raise RuntimeError(f"No CSV file inside {url}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(z.read(csvs[0]))
        print("Saved", path)

    digest = sha256(path)
    expected = cfg["scf"].get("sha256")
    if expected and digest != expected:
        print("WARNING: the SCF file differs from the version used for the report "
              f"(sha256 {digest[:12]}... vs {expected[:12]}...). The Federal Reserve may have revised it.")
    else:
        print("SCF file sha256:", digest)
    return path
