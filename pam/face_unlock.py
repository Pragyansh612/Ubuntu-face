"""
face_unlock.py — PAM Python module

Called by pam_python.so during GDM / login authentication.

Returns:
    PAM_SUCCESS   (0)  → biometric auth passed; user is logged in
    PAM_AUTH_ERR  (7)  → biometric auth failed; PAM continues to next module
                          (i.e. password prompt)

PAM configuration line (/etc/pam.d/gdm-password):
    auth sufficient pam_python.so /FULL/PATH/TO/REPO/pam/face_unlock.py
"""

import os
import sys
import subprocess

# Absolute path to auth_controller.py
_REPO_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_AUTH_CONTROLLER = os.path.join(_REPO_ROOT, "auth", "auth_controller.py")
_PYTHON = "/home/pragyansh612/Desktop/codinggggg/me/face-gesture-unlock/venv/bin/python"

# PAM return codes
PAM_SUCCESS  = 0
PAM_AUTH_ERR = 7


def pam_sm_authenticate(pamh, flags, argv):
    """
    Entry point called by pam_python.so.

    pamh  – PAM handle (provided by libpam)
    flags – PAM flags (unused)
    argv  – additional arguments from PAM config (unused)
    """
    try:
        result = subprocess.run(
            [_PYTHON, _AUTH_CONTROLLER],
            timeout=10,        # hard ceiling; scan itself is 3 s
            capture_output=True,
        )
        if result.returncode == 0:
            return PAM_SUCCESS
        else:
            return PAM_AUTH_ERR

    except subprocess.TimeoutExpired:
        print("[PAM] Biometric auth timed out.", file=sys.stderr)
        return PAM_AUTH_ERR

    except Exception as exc:
        print(f"[PAM] Unexpected error: {exc}", file=sys.stderr)
        return PAM_AUTH_ERR


def pam_sm_setcred(pamh, flags, argv):
    """Required by PAM — not used for authentication."""
    return PAM_SUCCESS


def pam_sm_acct_mgmt(pamh, flags, argv):
    """Required by PAM — not used here."""
    return PAM_SUCCESS
