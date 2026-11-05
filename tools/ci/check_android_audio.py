#!/usr/bin/env python3
"""Compile the changed player and mixer for Android arm64, without private game inputs.

This is a compilation check, not an APK build. Its expectation header is a
compile-only fixture; the objects must never be packaged as a playable build.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.request


ROOT = Path(__file__).resolve().parents[2]


def run(*args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ndk", required=True, type=Path)
    args = parser.parse_args()
    compiler = args.ndk.resolve() / "toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++"
    if not compiler.is_file():
        parser.error("the Android NDK compiler is missing")
    out = ROOT / "private/generated/ci/android-audio"
    out.mkdir(parents=True, exist_ok=True)
    pins = json.loads((ROOT / "tools/pins.json").read_text())["tools"]
    sdl = pins["sdl3"]
    archive = ROOT / sdl["path"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        temporary = archive.with_suffix(".part")
        with urllib.request.urlopen(sdl["url"], timeout=120) as response:
            temporary.write_bytes(response.read())
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != sdl["sha256"]:
            raise RuntimeError("SDL source download does not match tools/pins.json")
        temporary.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != sdl["sha256"]:
        raise RuntimeError("SDL source archive does not match tools/pins.json")
    with tarfile.open(archive) as tar:
        tar.extractall(out, filter="data")
    headers = out / f"SDL3-{sdl['version']}" / "include"

    dxvk = ROOT / pins["dxvk"]["path"]
    if not dxvk.exists():
        run("git", "clone", "--depth", "1", "--branch", pins["dxvk"]["tag"], pins["dxvk"]["url"], dxvk)
    commit = subprocess.check_output(["git", "-C", str(dxvk), "rev-parse", "HEAD"], text=True).strip()
    if commit != pins["dxvk"]["commit"]:
        raise RuntimeError("DXVK checkout does not match tools/pins.json")
    run("git", "-C", dxvk, "submodule", "update", "--init", "--depth", "1", "include/native/directx")

    spec = importlib.util.spec_from_file_location("android_app", ROOT / "tools/s8/android_app.py")
    android = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(android)
    (out / "sa2_host_expectations.h").write_text(
        "// Compile-only fixture. Never use in a playable build.\n"
        "#pragma once\nnamespace sa2_host_expected {\n"
        "inline constexpr int sdl_version = SDL_VERSION;\n"
        "inline constexpr char sdl_revision[] = SDL_REVISION;\n"
        'inline constexpr char sdl_sha256[] = "' + "0" * 64 + '";\n}\n')
    flags = ["-std=c++17", "-O2", "-fPIC", "-Wall", "-Wextra", "-Wno-unused-parameter",
             "-DSA2_PLAYER_GPU=1", "-DSDL_MAIN_HANDLED", "-Dmain=sa2_player_main",
             f"-I{headers}", f"-I{out}", f"-I{ROOT / 'host'}", f"-I{ROOT / 'host/player'}",
             f"-I{dxvk / 'include/native/windows'}", f"-I{dxvk / 'include/native/directx'}"]
    for name in ("mixer", "player"):
        rel = f"host/player/{name}.cpp"
        source = out / f"{name}.cpp"
        source.write_text(android.arm_portable(rel, (ROOT / rel).read_text()))
        print(f"Compiling {rel} for Android arm64", flush=True)
        run(compiler, *flags, "-c", source, "-o", out / f"{name}.o")
    print("Android arm64 compilation passed (player and mixer objects; no APK).")


if __name__ == "__main__":
    main()
