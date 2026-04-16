from __future__ import annotations
import sys
if sys.version_info < (3, 7):
   sys.exit("hfi-export.py requires Python 3.7 or newer (you have %s)" % sys.version.split()[0])
import argparse
import datetime
import glob
import io
import json
import os
import socket
import subprocess
import tarfile
import tempfile




# Skipped in the config dir — the marketplace clone is tens of MB and
# contains zero session data.
CONFIG_DIR_EXCLUDES = {"plugins"}




# -------- discovery --------


def find_tmpdir_hfi() -> list[str]:
   """Return every directory that looks like a claude-hfi tmp IPC root."""
   candidates: list[str] = []


   # 1. What Python thinks tmpdir is (honours $TMPDIR, matches Node's os.tmpdir())
   d = os.path.join(tempfile.gettempdir(), "claude-hfi")
   if os.path.isdir(d):
       candidates.append(d)


   # 2. Explicit $TMPDIR in case the interpreter resolved something different
   env_tmp = os.environ.get("TMPDIR")
   if env_tmp:
       d = os.path.join(env_tmp.rstrip("/"), "claude-hfi")
       if os.path.isdir(d):
           candidates.append(d)


   # 3. macOS per-user temp dir is under /var/folders — scan for claude-hfi at
   #    depth 4 to catch sessions started from a shell that had a different
   #    TMPDIR than this script (e.g. launch agents vs Terminal).
   if sys.platform == "darwin":
       for hit in glob.glob("/private/var/folders/*/*/T/claude-hfi"):
           if os.path.isdir(hit):
               candidates.append(hit)


   # 4. Plain /tmp for Linux
   for p in ("/tmp/claude-hfi", "/var/tmp/claude-hfi"):
       if os.path.isdir(p):
           candidates.append(p)


   # Dedup by realpath — /var/folders/… vs /private/var/folders/… collapse,
   # as does TMPDIR vs gettempdir().
   seen: set[str] = set()
   out: list[str] = []
   for c in candidates:
       r = os.path.realpath(c)
       if r not in seen:
           seen.add(r)
           out.append(c)
   return out




def find_config_dir() -> str | None:
   d = os.path.expanduser("~/.claude-hfi")
   return d if os.path.isdir(d) else None




def discover_base_repos() -> list[str]:
   """Find the git repos that HFI worktrees point back to, so we can record
   their hfi/* branch list without shipping the worktree code itself."""
   xdg = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
   worktree_cache = os.path.join(xdg, "claude-hfi")
   if not os.path.isdir(worktree_cache):
       return []
   repos: set[str] = set()
   for repo_slug in os.listdir(worktree_cache):
       slug_dir = os.path.join(worktree_cache, repo_slug)
       if not os.path.isdir(slug_dir):
           continue
       for side in ("A", "B"):
           gitfile = os.path.join(slug_dir, side, ".git")
           try:
               with open(gitfile, "r") as f:
                   line = f.read().strip()
               # "gitdir: /path/to/main/.git/worktrees/…"
               if line.startswith("gitdir:"):
                   gitdir = line.split(":", 1)[1].strip()
                   # …/.git/worktrees/<name> → repo root
                   idx = gitdir.find("/.git/")
                   if idx != -1:
                       repos.add(gitdir[:idx])
           except (FileNotFoundError, IsADirectoryError, PermissionError):
               pass
   return sorted(repos)




# -------- collection --------


def add_tree(tar: tarfile.TarFile, src: str, arcname_root: str,
            *, excludes: set[str] | None = None,
            skip_file=None,  # callable(name) -> bool
            stats: dict) -> None:
   excludes = excludes or set()
   for root, dirs, files in os.walk(src, topdown=True, followlinks=False):
       dirs[:] = [d for d in dirs if d not in excludes]
       rel = os.path.relpath(root, src)
       arc_dir = arcname_root if rel == "." else os.path.join(arcname_root, rel)
       for f in files:
           if skip_file and skip_file(f):
               stats["skipped_files"] += 1
               continue
           full = os.path.join(root, f)
           try:
               st = os.lstat(full)
           except OSError:
               stats["errors"] += 1
               continue
           arc = os.path.join(arc_dir, f)
           try:
               tar.add(full, arcname=arc, recursive=False)
               stats["files"] += 1
               stats["bytes"] += st.st_size
           except (OSError, tarfile.TarError) as e:
               stats["errors"] += 1
               stats.setdefault("error_samples", []).append(f"{full}: {e}")




def git_hfi_branches(repo: str) -> str:
   def run(*cmd: str) -> str:
       try:
           return subprocess.run(
               cmd, cwd=repo, capture_output=True, text=True, timeout=15
           ).stdout
       except Exception as e:
           return f"<error: {e}>"
   lines = [
       f"# repo: {repo}",
       "## git branch -a | grep hfi/",
       *(l for l in run("git", "branch", "-a").splitlines() if "hfi/" in l),
       "## git worktree list",
       run("git", "worktree", "list"),
       "",
   ]
   return "\n".join(lines)




def add_bytes(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
   info = tarfile.TarInfo(name=arcname)
   info.size = len(data)
   info.mtime = int(datetime.datetime.now().timestamp())
   tar.addfile(info, io.BytesIO(data))




# -------- main --------


def human(n: int) -> str:
   x = float(n)
   for unit in ("B", "KB", "MB", "GB"):
       if x < 1024:
           return f"{x:.0f}{unit}" if unit == "B" else f"{x:.1f}{unit}"
       x /= 1024
   return f"{x:.1f}TB"




def main() -> int:
   parser = argparse.ArgumentParser(
       description="Collect all on-disk claude-hfi metadata into a .tar.gz."
   )
   parser.add_argument("--out", help="output .tar.gz path (default: ~/Desktop/claude-hfi-export-<host>-<ts>.tar.gz)")
   parser.add_argument("--redact-auth", action="store_true",
                       help="skip .claude.json* (contains Auth0 bearer token)")
   parser.add_argument("-v", "--verbose", action="store_true",
                       help="print per-source file/byte breakdown")
   args = parser.parse_args()
   vprint = print if args.verbose else (lambda *a, **k: None)


   hostname = socket.gethostname()
   ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
   prefix = f"claude-hfi-export-{hostname}-{ts}"
   # Default to Desktop (or home if no Desktop) so labelers don't have to
   # guess where their cwd was.
   desktop = os.path.expanduser("~/Desktop")
   default_dir = desktop if os.path.isdir(desktop) else os.path.expanduser("~")
   out_path = args.out or os.path.join(default_dir, f"{prefix}.tar.gz")


   tmpdirs = find_tmpdir_hfi()
   config_dir = find_config_dir()
   base_repos = discover_base_repos()


   manifest = {
       "hostname": hostname,
       "created_at": datetime.datetime.now().isoformat(),
       "platform": sys.platform,
       "python": sys.version.split()[0],
       "sources": {
           "tmpdir": tmpdirs,
           "config_dir": config_dir,
           "base_repos": base_repos,
       },
       "notes": {
           "config_dir_excludes": sorted(CONFIG_DIR_EXCLUDES),
           "redact_auth": args.redact_auth,
       },
       "stats": {},
       "warnings": [],
   }


   if not tmpdirs and not config_dir:
       print("No claude-hfi data found on this machine.", file=sys.stderr)
       print("Checked: $TMPDIR/claude-hfi, /var/folders/*/*/T/claude-hfi, "
             "/tmp/claude-hfi, ~/.claude-hfi",
             file=sys.stderr)
       return 1


   vprint(f"Writing {out_path}")
   with tarfile.open(out_path, "w:gz") as tar:
       # tmpdir IPC dirs — the ephemeral gold
       for i, d in enumerate(tmpdirs):
           label = f"tmpdir_{i}" if len(tmpdirs) > 1 else "tmpdir"
           stats = {"files": 0, "bytes": 0, "errors": 0, "skipped_files": 0}
           add_tree(tar, d, f"{prefix}/{label}", stats=stats)
           manifest["stats"][label] = {**stats, "source_path": d}
           vprint(f"  {label:<14} {stats['files']:>6} files  {human(stats['bytes']):>10}  ← {d}")
           if stats["files"] == 0:
               manifest["warnings"].append(f"{d} exists but is empty")


       # config dir — durable transcripts live here
       if config_dir:
           stats = {"files": 0, "bytes": 0, "errors": 0, "skipped_files": 0}
           if args.redact_auth:
               excludes = CONFIG_DIR_EXCLUDES | {"backups"}
               skip = lambda name: name.startswith(".claude.json")
           else:
               excludes = CONFIG_DIR_EXCLUDES
               skip = None
           add_tree(tar, config_dir, f"{prefix}/config-dir",
                    excludes=excludes, skip_file=skip, stats=stats)
           manifest["stats"]["config_dir"] = {**stats, "source_path": config_dir}
           vprint(f"  {'config_dir':<14} {stats['files']:>6} files  {human(stats['bytes']):>10}  ← {config_dir}")
       else:
           manifest["warnings"].append("~/.claude-hfi does not exist — no transcripts on this machine")


       # git branch listing — so hfiSessionId mapping survives
       if base_repos:
           branch_dump = "\n".join(git_hfi_branches(r) for r in base_repos)
           add_bytes(tar, f"{prefix}/git-branches.txt", branch_dump.encode())
           vprint(f"  {'git_branches':<14} {len(base_repos):>6} repos")


       # manifest last, so it reflects everything above
       add_bytes(tar, f"{prefix}/manifest.json",
                 json.dumps(manifest, indent=2).encode())


   size = os.path.getsize(out_path)
   total_files = sum(s["files"] for s in manifest["stats"].values())
   for w in manifest["warnings"]:
       print(f"  ⚠ {w}")
   abs_out = os.path.abspath(out_path)
   banner = "=" * max(60, len(abs_out) + 4)
   print()
   print(banner)
   print(f"  DONE — {total_files} files, {human(size)} — upload this file:")
   print()
   print(f"  {abs_out}")
   print(banner)
   return 0




if __name__ == "__main__":
   sys.exit(main())

