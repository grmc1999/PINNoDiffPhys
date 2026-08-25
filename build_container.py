#!/usr/bin/env python3
"""Build and deploy PINNoDiffPhys Apptainer containers.

Run this on a cluster node with apptainer/singularity installed.
It builds both CPU and CUDA .sif files and copies them to $PATH_ENV.

Usage:
    python build_container.py --cluster ICA     # CPU only
    python build_container.py --cluster SD2     # CPU + CUDA
    python build_container.py --deploy          # build + deploy to $PATH_ENV
"""
import argparse
import os
import subprocess
import sys


CONTAINER_DIR = os.path.join(os.path.dirname(__file__), "container")


def build(def_file, output_sif, sudo=False):
    cmd = ["apptainer", "build"]
    if sudo:
        cmd.insert(1, "--fakeroot")
    cmd.extend([output_sif, def_file])
    print(f"Building: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        sys.exit(1)
    print(f"Built: {output_sif}")


def deploy(sif_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(sif_path))
    cmd = ["cp", sif_path, dest]
    subprocess.run(cmd, check=True)
    print(f"Deployed: {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, default="ICA",
                        choices=["ICA", "SD2_h100", "SD2_gh200"])
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--sudo", action="store_true")
    args = parser.parse_args()

    build_dir = os.path.join(os.path.dirname(__file__), "container", "build")
    os.makedirs(build_dir, exist_ok=True)

    # CPU container (all clusters)
    cpu_def = os.path.join(CONTAINER_DIR, "PINNoDiffPhys.def")
    cpu_sif = os.path.join(build_dir, "PINNoDiffPhys_cpu.sif")
    build(cpu_def, cpu_sif, sudo=args.sudo)

    if args.cluster in ("SD2_h100", "SD2_gh200"):
        cuda_def = os.path.join(CONTAINER_DIR, "PINNoDiffPhys_cuda.def")
        cuda_sif = os.path.join(build_dir, "PINNoDiffPhys_cuda.sif")
        build(cuda_def, cuda_sif, sudo=args.sudo)

    if args.deploy:
        path_env = os.environ.get("PATH_ENV", "")
        if not path_env:
            print("ERROR: PATH_ENV not set")
            sys.exit(1)

        if args.cluster == "ICA":
            dest = os.path.join(path_env, "envs")
        else:
            user = "guillermo.carrillo" if "SD" in args.cluster else "gmorenoc"
            dest = os.path.join(path_env, "users", user, "Ambientes")

        deploy(cpu_sif, dest)
        if args.cluster in ("SD2_h100", "SD2_gh200"):
            # rename CUDA variant for SDumont convention
            cuda_sif = os.path.join(build_dir, "PINNoDiffPhys_cuda.sif")
            suffix = "h100_v3.sif" if "h100" in args.cluster else "gh200_v2.sif"
            cuda_dest = os.path.join(dest, f"PINNoDiffPhys_{suffix}")
            subprocess.run(["cp", cuda_sif, cuda_dest], check=True)
            print(f"Deployed: {cuda_dest}")

    print("Done.")


if __name__ == "__main__":
    main()
