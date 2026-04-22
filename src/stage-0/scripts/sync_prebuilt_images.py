from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from config import (
    EXPERIMENTS_BUILD_IMAGES_DIR,
    PREBUILT_IMAGE_INSTANCES_DIR,
    PREBUILT_IMAGE_MANIFEST,
)

DIR_NAME_RE = re.compile(r"^sweb\.eval\.[^.]+\.(?P<instance_id>.+)__latest$")
GIT_CLONE_RE = re.compile(
    r"git clone(?:\s+-o\s+origin)?\s+--single-branch\s+https://github\.com/(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\s+/testbed"
)
GIT_RESET_RE = re.compile(r"git reset --hard (?P<sha>[0-9a-fA-F]{7,40})")
BUILD_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - INFO - Building image (?P<tag>sweb\.eval\.[^\s:]+:[^\s]+)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sync prebuilt image artifacts from experiments into Stage 0 outputs "
            "and write a repository-to-image manifest."
        )
    )
    p.add_argument(
        "--source-dir",
        type=Path,
        default=EXPERIMENTS_BUILD_IMAGES_DIR,
        help="Source directory containing per-image artifact folders.",
    )
    p.add_argument(
        "--dest-dir",
        type=Path,
        default=PREBUILT_IMAGE_INSTANCES_DIR,
        help="Destination directory under Stage 0 outputs.",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=PREBUILT_IMAGE_MANIFEST,
        help="JSON manifest path to write.",
    )
    p.add_argument(
        "--operation",
        choices=["copy", "move"],
        default="move",
        help="How to ingest newly discovered artifacts from --source-dir.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output to a single summary line.",
    )
    return p.parse_args()


def log(message: str, *, quiet: bool) -> None:
    if not quiet:
        print(message)


def parse_setup_repo(setup_repo_path: Path) -> tuple[str, str]:
    if not setup_repo_path.exists():
        return "", ""

    text = setup_repo_path.read_text(encoding="utf-8", errors="ignore")
    repo_match = GIT_CLONE_RE.search(text)
    sha_match = GIT_RESET_RE.search(text)
    github_repo = repo_match.group("repo").lower() if repo_match else ""
    commit = sha_match.group("sha") if sha_match else ""
    return github_repo, commit


def parse_build_log(build_log_path: Path, fallback_tag: str) -> tuple[str, str]:
    if not build_log_path.exists():
        return "", fallback_tag

    first_line = ""
    with build_log_path.open("r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()

    match = BUILD_LINE_RE.match(first_line)
    if not match:
        return "", fallback_tag

    timestamp = match.group("ts")
    image_tag = match.group("tag")

    # Timestamps in build logs are local time; keep as plain ISO-like text.
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S,%f")
        parsed_timestamp = dt.isoformat(timespec="seconds")
    except ValueError:
        parsed_timestamp = ""

    return parsed_timestamp, image_tag


def image_tag_from_dir_name(dir_name: str) -> tuple[str, str]:
    match = DIR_NAME_RE.match(dir_name)
    if not match:
        return "", ""
    instance_id = match.group("instance_id")
    image_tag = f"{dir_name.removesuffix('__latest')}:latest"
    return instance_id, image_tag


def list_local_docker_images() -> tuple[bool, set[str]]:
    if shutil.which("docker") is None:
        return False, set()

    proc = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return False, set()

    tags = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    return True, tags


def sync_artifacts(
    source_dir: Path, dest_dir: Path, operation: str, *, quiet: bool
) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved_or_copied = 0

    if not source_dir.exists():
        log(f"Source directory not found, skipping sync: {source_dir}", quiet=quiet)
        return moved_or_copied

    source_dirs = sorted(p for p in source_dir.iterdir() if p.is_dir())
    if not source_dirs:
        log(f"No source artifacts found under {source_dir}", quiet=quiet)
        return moved_or_copied

    for source_path in source_dirs:
        dest_path = dest_dir / source_path.name
        if dest_path.exists():
            shutil.rmtree(dest_path)

        if operation == "copy":
            shutil.copytree(source_path, dest_path)
        else:
            shutil.move(str(source_path), str(dest_path))
        moved_or_copied += 1

    log(
        f"Synced {moved_or_copied} artifact directories from {source_dir} to {dest_dir} ({operation})",
        quiet=quiet,
    )
    return moved_or_copied


def build_manifest(dest_dir: Path, manifest_path: Path, *, quiet: bool) -> dict:
    docker_cli_available, local_tags = list_local_docker_images()

    entries = []
    if dest_dir.exists():
        for artifact_dir in sorted(p for p in dest_dir.iterdir() if p.is_dir()):
            instance_id, tag_from_name = image_tag_from_dir_name(artifact_dir.name)
            if not instance_id or not tag_from_name:
                continue

            setup_repo_path = artifact_dir / "setup_repo.sh"
            build_log_path = artifact_dir / "build_image.log"
            dockerfile_path = artifact_dir / "Dockerfile"

            github_repo, commit = parse_setup_repo(setup_repo_path)
            build_timestamp, image_tag = parse_build_log(build_log_path, tag_from_name)

            entries.append(
                {
                    "instance_id": instance_id,
                    "image_tag": image_tag,
                    "github_repo": github_repo,
                    "commit": commit,
                    "build_timestamp": build_timestamp,
                    "artifact_dir": str(artifact_dir.resolve()),
                    "files_present": {
                        "setup_repo_sh": setup_repo_path.exists(),
                        "build_image_log": build_log_path.exists(),
                        "dockerfile": dockerfile_path.exists(),
                    },
                    "local_docker_image_present": bool(image_tag in local_tags)
                    if docker_cli_available
                    else False,
                }
            )

    entries.sort(key=lambda x: (x["github_repo"], x["instance_id"], x["image_tag"]))

    repo_to_images: dict[str, list[str]] = {}
    repo_entries: dict[str, list[dict]] = {}
    for item in entries:
        repo = item["github_repo"]
        if not repo:
            continue
        repo_to_images.setdefault(repo, [])
        if item["image_tag"] not in repo_to_images[repo]:
            repo_to_images[repo].append(item["image_tag"])
        repo_entries.setdefault(repo, []).append(item)

    recommended: dict[str, str] = {}
    for repo, items in repo_entries.items():
        items_sorted = sorted(
            items,
            key=lambda x: (
                x.get("build_timestamp", ""),
                x.get("instance_id", ""),
                x.get("image_tag", ""),
            ),
            reverse=True,
        )
        recommended[repo] = items_sorted[0]["image_tag"]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "destination_dir": str(dest_dir.resolve()),
        "image_count": len(entries),
        "repository_count": len(repo_to_images),
        "docker_cli_available": docker_cli_available,
        "images": entries,
        "repo_to_images": repo_to_images,
        "recommended_image_tag_by_repo": recommended,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote prebuilt image manifest: {manifest_path}", quiet=quiet)
    return payload


def main() -> int:
    args = parse_args()

    synced_count = sync_artifacts(
        source_dir=args.source_dir,
        dest_dir=args.dest_dir,
        operation=args.operation,
        quiet=args.quiet,
    )
    manifest = build_manifest(args.dest_dir, args.manifest, quiet=args.quiet)

    print(
        "Prebuilt image sync complete: "
        f"synced_dirs={synced_count}, images={manifest['image_count']}, "
        f"repos={manifest['repository_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
