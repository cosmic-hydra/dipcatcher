"""Owner-only mode for live desk files.

POSIX chmod 0o600 is the contract: the mode bits must be exactly that.
Windows CPython reports 0o666 for every writable file, so chmod cannot
satisfy the check. There the DACL is rewritten to the current user alone.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def restrict_to_owner(path: Path) -> None:
    """Leave ``path`` readable and writable by its owner only."""
    os.chmod(path, 0o600)
    if os.name == "nt":
        _windows_owner_only(path)


def is_owner_only(path: Path) -> bool:
    """True when the POSIX mode is 0600, or the Windows DACL is owner-only."""
    if os.name != "nt":
        return os.stat(path).st_mode & 0o777 == 0o600
    return _windows_dacl_owner_only(path)


def _windows_owner_only(path: Path) -> None:
    """Drop inherited ACEs and grant the current user read and write."""
    script = (
        "$path = $args[0]\n"
        "$acl = Get-Acl -LiteralPath $path\n"
        "$acl.SetAccessRuleProtection($true, $false)\n"
        "foreach ($rule in @($acl.Access)) { [void]$acl.RemoveAccessRule($rule) }\n"
        "$me = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name\n"
        "$rule = New-Object System.Security.AccessControl.FileSystemAccessRule("
        "$me, 'Read,Write', 'Allow')\n"
        "$acl.AddAccessRule($rule)\n"
        "Set-Acl -LiteralPath $path -AclObject $acl\n"
    )
    _powershell(script, path)


def _windows_dacl_owner_only(path: Path) -> bool:
    script = (
        "$path = $args[0]\n"
        "$acl = Get-Acl -LiteralPath $path\n"
        "$me = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name\n"
        "$rules = @($acl.Access)\n"
        "if ($rules.Count -ne 1) { exit 1 }\n"
        "$rule = $rules[0]\n"
        "if ($rule.IdentityReference.Value -ne $me) { exit 1 }\n"
        "if ($rule.AccessControlType -ne 'Allow') { exit 1 }\n"
        "$rights = [string]$rule.FileSystemRights\n"
        "if ($rights -notmatch 'Read' -or $rights -notmatch 'Write') { exit 1 }\n"
        "if ($rights -match 'FullControl|Modify') { exit 1 }\n"
        "exit 0\n"
    )
    completed = _powershell(script, path, check=False)
    return completed.returncode == 0


def _powershell(script: str, path: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script, os.fspath(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise OSError(f"owner-only ACL failed for {path}: {detail}")
    return completed
