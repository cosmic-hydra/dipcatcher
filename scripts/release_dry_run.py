#!/usr/bin/env python3
"""Local release dry-run: versioning, wheel, SBOM, checksums, optional container.

Does not create tags, sign with Sigstore, mint SLSA attestations, push images,
or publish to PyPI / GitHub Releases. Owner tags after this passes and CI is
green on the candidate SHA.

Usage::

    uv run python scripts/release_dry_run.py
    uv run python scripts/release_dry_run.py --skip-container
    make release-dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


def _write_sbom(dist_dir: Path) -> None:
    sbom = dist_dir / "sbom.cdx.json"
    # uv export may still emit the document on stdout; keep the log clean.
    print(
        "+ uv export --format cyclonedx1.5 --frozen --no-dev --all-extras "
        f"--no-emit-project --output-file {sbom}",
        flush=True,
    )
    subprocess.run(
        [
            "uv",
            "export",
            "--format",
            "cyclonedx1.5",
            "--frozen",
            "--no-dev",
            "--all-extras",
            "--no-emit-project",
            "--output-file",
            str(sbom),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    doc = json.loads(sbom.read_text(encoding="utf-8"))
    if doc.get("bomFormat") != "CycloneDX" or doc.get("specVersion") != "1.5":
        raise SystemExit(f"unexpected SBOM header: {doc.get('bomFormat')} {doc.get('specVersion')}")
    if not (doc.get("components") or []):
        raise SystemExit("SBOM has no components")
    print(f"CycloneDX {doc['specVersion']} components={len(doc['components'])}")


def _checksums(dist_dir: Path) -> None:
    names = sorted(
        [p.name for p in dist_dir.glob("*.whl")]
        + [p.name for p in dist_dir.glob("*.tar.gz")]
        + ["sbom.cdx.json"]
    )
    if not any(n.endswith(".whl") for n in names) or not any(n.endswith(".tar.gz") for n in names):
        raise SystemExit("expected at least one wheel and one sdist in dist/")
    lines = []
    for name in names:
        digest = subprocess.check_output(["sha256sum", name], cwd=dist_dir, text=True).split()[0]
        lines.append(f"{digest}  {name}")
    (dist_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((dist_dir / "SHA256SUMS").read_text(encoding="utf-8"), end="")


def _container_health(image: str) -> None:
    name = "dipcatcher-release-dry-run"
    subprocess.run(["docker", "rm", "--force", name], check=False, capture_output=True)
    _run(["docker", "run", "--detach", "--name", name, image])
    try:
        for _ in range(30):
            status = subprocess.check_output(
                ["docker", "inspect", "--format={{.State.Health.Status}}", name],
                text=True,
            ).strip()
            if status == "healthy":
                print(f"container health: {status}")
                return
            if status == "unhealthy":
                subprocess.run(["docker", "logs", name], check=False)
                raise SystemExit("container became unhealthy")
            import time

            time.sleep(2)
        subprocess.run(["docker", "logs", name], check=False)
        raise SystemExit("container health check timed out")
    finally:
        subprocess.run(["docker", "rm", "--force", name], check=False, capture_output=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-container",
        action="store_true",
        help="Skip docker build + HEALTHCHECK (wheel/SBOM only)",
    )
    parser.add_argument(
        "--keep-dist",
        action="store_true",
        help="Leave artifacts under ./dist instead of a temp directory",
    )
    args = parser.parse_args(argv)

    _run([sys.executable, str(ROOT / "scripts" / "verify_release_readiness.py"), "--local-only"])

    if args.keep_dist:
        dist_dir = ROOT / "dist"
        if dist_dir.exists():
            shutil.rmtree(dist_dir)
        dist_dir.mkdir(parents=True)
        cleanup: tempfile.TemporaryDirectory[str] | None = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="dipcatcher-release-dist-")
        dist_dir = Path(cleanup.name)

    try:
        _run(["uv", "build", "--out-dir", str(dist_dir)])
        _run(
            [
                "uv",
                "sync",
                "--frozen",
                "--all-groups",
                "--no-install-project",
                "--no-build",
            ]
        )
        wheels = sorted(dist_dir.glob("*.whl"))
        if not wheels:
            raise SystemExit("uv build produced no wheel")
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(ROOT / ".venv" / "bin" / "python"),
                "--no-deps",
                str(wheels[0]),
            ]
        )
        _run(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                str(ROOT / "scripts" / "smoke_installed_wheel.py"),
            ]
        )
        _write_sbom(dist_dir)
        _checksums(dist_dir)
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "verify_release_artifacts.py"),
                str(dist_dir),
            ]
        )

        if not args.skip_container:
            if shutil.which("docker") is None:
                raise SystemExit("docker not found; pass --skip-container or install docker")
            image = "dipcatcher-release-dry-run:local"
            _run(["docker", "build", "--pull", "--tag", image, "."])
            _container_health(image)

        print("release dry-run: OK (no tag, no Sigstore, no publish)")
        print(
            "Owner next: green CI on the candidate SHA, then tag "
            f"v$(uv run python -c 'from fx1 import __version__; print(__version__)') "
            "to run release.yml for real."
        )
        return 0
    finally:
        if cleanup is not None:
            cleanup.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
