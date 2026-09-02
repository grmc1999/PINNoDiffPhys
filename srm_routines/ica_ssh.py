#!/usr/bin/env python3
"""Reusable paramiko SSH helper for PINNoDiffPhys cluster operations.

Usage (typical):
    from srm_routines.ica_ssh import ICA

    ica = ICA()
    out = ica.run("hostname")                 # single command
    ica.sftp_put(local, remote)
    ica.sftp_get(remote, local)

Credentials are read from ../cluster-credentials/ (ICA_config.txt) or env.
"""
import os
import paramiko


def _read_ica_creds():
    # Try env overrides first
    for var in ("ICA_HOST", "ICA_USER", "ICA_PASS"):
        if os.environ.get(var):
            continue
    try:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        home = os.path.expanduser("~")
        candidates = [
            os.path.join(base, "cluster-credentials", "ICA_config.txt"),
            os.path.join(home, "cluster-credentials", "ICA_config.txt"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), candidates[0])
        with open(path) as f:
            creds = {}
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":", 1)
                    creds[k.strip().lower()] = v.strip()
        return {
            "host": os.environ.get("ICA_HOST", creds.get("ip", "139.82.152.10")),
            "user": os.environ.get("ICA_USER", creds.get("user", "gmorenoc")),
            "pass": os.environ.get("ICA_PASS", creds.get("password")),
            "port": 22,
        }
    except FileNotFoundError:
        return {
            "host": os.environ.get("ICA_HOST", "139.82.152.10"),
            "user": os.environ.get("ICA_USER", "gmorenoc"),
            "pass": os.environ.get("ICA_PASS"),
            "port": 22,
        }


class ICA:
    """Paramiko client wrapper for the ICA cluster."""

    def __init__(self, host=None, user=None, password=None, port=22):
        creds = _read_ica_creds()
        self.host = host or creds["host"]
        self.user = user or creds["user"]
        self.password = password or creds["pass"]
        self.port = port
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            self.host, port=self.port, username=self.user,
            password=self.password, timeout=20,
            look_for_keys=False, allow_agent=False,
        )

    def run(self, cmd, timeout=60, check=True):
        """Run a command and return (exit_status, stdout, stderr)."""
        stdin, stdout, stderr = self.client.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        status = stdout.channel.recv_exit_status()
        if check and status != 0 and err:
            raise RuntimeError(f"cmd failed ({status}): {cmd}\nERR: {err[:500]}")
        return status, out, err

    def sftp_put(self, local, remote):
        with self.client.open_sftp() as sftp:
            sftp.put(local, remote)

    def sftp_get(self, remote, local):
        with self.client.open_sftp() as sftp:
            sftp.get(remote, local)

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == "__main__":
    with ICA() as ica:
        st, out, err = ica.run("hostname && whoami")
        print(out)
