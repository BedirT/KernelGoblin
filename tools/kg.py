#!/usr/bin/env python3
"""Dependency-free entry point for selecting and exercising kernel ports."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRELLIS_NATIVE_ROOTS = {
    "Sources/KernelGoblinTrellis2",
    "Sources/KernelGoblinTrellis2CLI",
}
TRELLIS_NATIVE_EXTENSIONS = {".metal", ".swift"}
TRELLIS_NATIVE_TARGETS = {
    "KernelGoblinTrellis2",
    "KernelGoblinTrellis2CLI",
}
TRELLIS_NATIVE_IMPORTS = {
    "CoreFoundation",
    "CryptoKit",
    "Darwin",
    "Foundation",
    "KernelGoblinTrellis2",
    "Metal",
}
TRELLIS_FORBIDDEN_RUNTIME_APIS = (
    re.compile(r"\bProcess\s*[.(]"),
    re.compile(r"\b(?:posix_spawn|system|dlopen|dlsym)\s*\("),
)


def manifests() -> dict[str, dict]:
    found: dict[str, dict] = {}
    for path in sorted((ROOT / "kernels").glob("*/*/kernel.toml")):
        with path.open("rb") as stream:
            manifest = tomllib.load(stream)
        manifest["_path"] = path
        found[manifest["id"]] = manifest
    return found


def kernel(value: str) -> dict:
    available = manifests()
    try:
        return available[value]
    except KeyError:
        choices = ", ".join(available) or "none"
        raise SystemExit(f"unknown kernel {value!r}; available: {choices}")


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def native_source_errors() -> list[str]:
    errors: list[str] = []
    for source_root in sorted(TRELLIS_NATIVE_ROOTS):
        root = ROOT / source_root
        if not root.is_dir():
            errors.append(f"native source root does not exist: {source_root}")
            continue
        for source in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = source.relative_to(ROOT)
            if source.suffix not in TRELLIS_NATIVE_EXTENSIONS:
                errors.append(f"native source has forbidden extension: {relative}")
                continue
            if source.suffix != ".swift":
                continue
            text = source.read_text()
            for module in re.findall(r"(?m)^\s*import\s+([A-Za-z_][A-Za-z0-9_]*)", text):
                if module not in TRELLIS_NATIVE_IMPORTS:
                    errors.append(f"native source imports undeclared module {module}: {relative}")
            for pattern in TRELLIS_FORBIDDEN_RUNTIME_APIS:
                if pattern.search(text):
                    errors.append(f"native source uses forbidden runtime-loading API: {relative}")
                    break
    return errors


def native_package_errors(description: dict) -> list[str]:
    errors: list[str] = []
    targets = {target["name"]: target for target in description.get("targets", [])}
    products = {
        product["name"]: product for product in description.get("products", [])
    }
    executable = products.get("kg-trellis2")
    if executable is None:
        return ["Swift package does not declare the kg-trellis2 product"]

    pending = list(executable.get("targets", []))
    closure: set[str] = set()
    while pending:
        name = pending.pop()
        if name in closure:
            continue
        target = targets.get(name)
        if target is None:
            errors.append(f"kg-trellis2 depends on unresolved local target {name}")
            continue
        closure.add(name)
        pending.extend(target.get("target_dependencies", []))

    if closure != TRELLIS_NATIVE_TARGETS:
        errors.append(
            "kg-trellis2 local target closure must be exactly "
            + ", ".join(sorted(TRELLIS_NATIVE_TARGETS))
        )
    for name in sorted(closure):
        target = targets.get(name)
        if target is None:
            continue
        if target.get("module_type") != "SwiftTarget":
            errors.append(f"native target {name} is not a Swift target")
        path = target.get("path")
        if path not in TRELLIS_NATIVE_ROOTS:
            errors.append(f"native target {name} has forbidden source root {path!r}")
        for source in target.get("sources", []):
            if Path(source).suffix != ".swift":
                errors.append(f"native target {name} has non-Swift source {source}")
    return errors


def require_backend(manifest: dict) -> None:
    if "metal" in manifest["backends"] and platform.system() != "Darwin":
        raise SystemExit(f"{manifest['id']} currently requires macOS for its Metal backend")


def configure(manifest: dict) -> None:
    require_backend(manifest)
    run(["cmake", "--preset", manifest["cmake_preset"]])
    run(["cmake", "--build", "--preset", manifest["cmake_preset"]])


def command_list(_: argparse.Namespace) -> None:
    for item in manifests().values():
        backends = " -> ".join(item["backends"])
        print(f"{item['id']:<24} {backends:<16} {item['description']}")


def command_doctor(_: argparse.Namespace) -> None:
    checks = {
        "system": f"{platform.system()} {platform.machine()}",
        "cmake": shutil.which("cmake"),
        "ninja": shutil.which("ninja"),
        "xcrun": shutil.which("xcrun"),
    }
    failed = False
    for name, value in checks.items():
        print(f"{name:<10} {value or 'MISSING'}")
        failed |= value is None
    if platform.system() == "Darwin" and shutil.which("xcrun"):
        result = subprocess.run(
            ["xcrun", "-f", "metal"], text=True, capture_output=True
        )
        metal = result.stdout.strip() if result.returncode == 0 else "MISSING"
        print(f"{'metal':<10} {metal}")
        failed |= result.returncode != 0
    if failed:
        raise SystemExit(1)


def command_validate(_: argparse.Namespace) -> None:
    errors: list[str] = []
    available = manifests()
    preset_file = ROOT / "CMakePresets.json"
    preset_text = preset_file.read_text()
    full_sha = re.compile(r"^[0-9a-f]{40}$")

    if not available:
        errors.append("no kernel manifests found")
    for kernel_id, item in available.items():
        expected = item["_path"].parent.relative_to(ROOT / "kernels").as_posix()
        if kernel_id != expected:
            errors.append(f"{item['_path']}: id {kernel_id!r} must match {expected!r}")
        for field in ("description", "model", "operation", "port", "backends", "cmake_preset", "benchmark"):
            if not item.get(field):
                errors.append(f"{item['_path']}: missing {field}")
        upstream = item.get("upstream", {})
        for field in ("repository", "revision", "sources", "license"):
            if not upstream.get(field):
                errors.append(f"{item['_path']}: missing upstream.{field}")
        if upstream.get("revision") and not full_sha.match(upstream["revision"]):
            errors.append(f"{item['_path']}: upstream.revision must be a full lowercase commit SHA")
        if item.get("cmake_preset") and f'"name": "{item["cmake_preset"]}"' not in preset_text:
            errors.append(f"{item['_path']}: CMake preset is not declared")
        if not (item["_path"].parent / "README.md").is_file():
            errors.append(f"{item['_path'].parent}: missing README.md")

    for path in sorted((ROOT / "ports").glob("*/model.toml")):
        with path.open("rb") as stream:
            model = tomllib.load(stream)
        expected_id = path.parent.name
        for field in (
            "id", "model", "description", "production_runtime",
            "reference_runtime", "platforms",
        ):
            if not model.get(field):
                errors.append(f"{path}: missing {field}")
        if model.get("id") != expected_id:
            errors.append(f"{path}: id must match directory {expected_id!r}")
        source = model.get("source", {})
        for field in ("repository", "revision", "license"):
            if not source.get(field):
                errors.append(f"{path}: missing source.{field}")
        revisions = [source.get("revision")] + [
            checkpoint.get("revision") for checkpoint in model.get("checkpoints", [])
        ]
        for revision in revisions:
            if revision and not full_sha.match(revision):
                errors.append(f"{path}: revision {revision!r} must be a full lowercase Git SHA")

        if model.get("production_runtime") == "swift-metal":
            native = model.get("native", {})
            for field in (
                "package", "executable", "source_roots",
                "allowed_source_extensions", "forbidden_imports",
                "external_package_dependencies", "torch_policy",
            ):
                if field not in native:
                    errors.append(f"{path}: missing native.{field}")
            if native.get("torch_policy") != "oracle-only":
                errors.append(f"{path}: native.torch_policy must be 'oracle-only'")
            if set(native.get("source_roots", [])) != TRELLIS_NATIVE_ROOTS:
                errors.append(f"{path}: native.source_roots must match the enforced roots")
            if set(native.get("allowed_source_extensions", [])) != TRELLIS_NATIVE_EXTENSIONS:
                errors.append(
                    f"{path}: native.allowed_source_extensions must be exactly .swift and .metal"
                )
            if native.get("external_package_dependencies") != []:
                errors.append(f"{path}: native external package dependencies must remain empty")
            errors.extend(f"{path}: {error}" for error in native_source_errors())

    agent_dir = ROOT / ".codex" / "agents"
    for path in sorted(agent_dir.glob("*.toml")):
        with path.open("rb") as stream:
            agent = tomllib.load(stream)
        for field in ("name", "description", "developer_instructions"):
            if not agent.get(field):
                errors.append(f"{path}: missing {field}")

    skill_file = ROOT / ".agents" / "skills" / "port-gpu-kernel" / "SKILL.md"
    skill_text = skill_file.read_text() if skill_file.is_file() else ""
    if not skill_text.startswith("---\n") or "\nname: port-gpu-kernel\n" not in skill_text:
        errors.append(f"{skill_file}: invalid or missing skill frontmatter")
    for required in (ROOT / "AGENTS.md", ROOT / "THIRD_PARTY_NOTICES.md", ROOT / ".codex" / "config.toml"):
        if not required.is_file():
            errors.append(f"missing required harness file: {required}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"PASS: {len(available)} kernel manifest(s), agent roles, and repo skill are valid")


def command_setup(args: argparse.Namespace) -> None:
    configure(kernel(args.kernel))


def command_test(args: argparse.Namespace) -> None:
    item = kernel(args.kernel)
    configure(item)
    run(["ctest", "--preset", item["cmake_preset"]])


def command_benchmark(args: argparse.Namespace) -> None:
    item = kernel(args.kernel)
    configure(item)
    executable = ROOT / "build" / item["cmake_preset"] / item["benchmark"]
    run([str(executable)])


def trellis_python() -> Path:
    return ROOT / "build" / "trellis2" / ".venv" / "bin" / "python"


def command_model_setup(args: argparse.Namespace) -> None:
    if args.model != "trellis2":
        raise SystemExit(f"unknown model runtime {args.model!r}; available: trellis2")
    run([sys.executable, str(ROOT / "ports" / "trellis2" / "setup_runtime.py")])


def command_model_native_setup(args: argparse.Namespace) -> None:
    if args.model != "trellis2":
        raise SystemExit(f"unknown model runtime {args.model!r}; available: trellis2")
    run(["swift", "build", "-c", "release", "--product", "kg-trellis2"])


def command_model_native_test(args: argparse.Namespace) -> None:
    if args.model != "trellis2":
        raise SystemExit(f"unknown model runtime {args.model!r}; available: trellis2")
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else (
        Path.home() / ".cache" / "huggingface" / "hub"
        / "models--microsoft--TRELLIS.2-4B" / "snapshots"
        / "af44b45f2e35a493886929c6d786e563ec68364d" / "ckpts"
        / "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors"
    )
    if not checkpoint.is_file():
        raise SystemExit(
            "native TRELLIS.2 conformance requires the pinned shape-flow checkpoint; "
            "pass --checkpoint FILE.safetensors"
        )
    texture_checkpoint = (
        Path(args.texture_checkpoint).expanduser().resolve() if args.texture_checkpoint else (
            checkpoint.parent
            / "slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors"
        )
    )
    if not texture_checkpoint.is_file():
        raise SystemExit(
            "native TRELLIS.2 conformance requires the pinned texture-flow checkpoint; "
            "pass --texture-checkpoint FILE.safetensors"
        )
    sparse_structure_checkpoint = (
        Path(args.sparse_structure_checkpoint).expanduser().resolve()
        if args.sparse_structure_checkpoint else (
            checkpoint.parent / "ss_flow_img_dit_1_3B_64_bf16.safetensors"
        )
    )
    if not sparse_structure_checkpoint.is_file():
        raise SystemExit(
            "native TRELLIS.2 conformance requires the pinned sparse-structure "
            "checkpoint; pass --sparse-structure-checkpoint FILE.safetensors"
        )
    sparse_structure_decoder_checkpoint = (
        Path(args.sparse_structure_decoder_checkpoint).expanduser().resolve()
        if args.sparse_structure_decoder_checkpoint else (
            Path.home() / ".cache" / "huggingface" / "hub"
            / "models--microsoft--TRELLIS-image-large" / "snapshots"
            / "25e0d31ffbebe4b5a97464dd851910efc3002d96" / "ckpts"
            / "ss_dec_conv3d_16l8_fp16.safetensors"
        )
    )
    if not sparse_structure_decoder_checkpoint.is_file():
        raise SystemExit(
            "native TRELLIS.2 conformance requires the pinned sparse-structure "
            "decoder; pass --sparse-structure-decoder-checkpoint FILE.safetensors"
        )
    dino_checkpoint = (
        Path(args.dino_checkpoint).expanduser().resolve() if args.dino_checkpoint else (
            Path.home() / ".cache" / "huggingface" / "hub"
            / "models--facebook--dinov3-vitl16-pretrain-lvd1689m" / "snapshots"
            / "ea8dc2863c51be0a264bab82070e3e8836b02d51" / "model.safetensors"
        )
    )
    if not dino_checkpoint.is_file():
        raise SystemExit(
            "native TRELLIS.2 conformance requires the separately gated pinned DINOv3 "
            "checkpoint; pass --dino-checkpoint FILE.safetensors"
        )
    environment = os.environ.copy()
    environment["KG_TRELLIS2_SHAPE_FLOW_CHECKPOINT"] = str(checkpoint)
    environment["KG_TRELLIS2_TEXTURE_FLOW_CHECKPOINT"] = str(texture_checkpoint)
    environment["KG_TRELLIS2_SPARSE_STRUCTURE_FLOW_CHECKPOINT"] = str(
        sparse_structure_checkpoint
    )
    environment["KG_TRELLIS2_SPARSE_STRUCTURE_DECODER_CHECKPOINT"] = str(
        sparse_structure_decoder_checkpoint
    )
    environment["KG_TRELLIS2_DINO_CHECKPOINT"] = str(dino_checkpoint)
    # Physical GPU conformance and memory peaks are not meaningful when the
    # heavyweight stage tests contend on independent Metal queues.
    run(["swift", "test", "--no-parallel"], env=environment)


def command_model_native_audit(args: argparse.Namespace) -> None:
    if args.model != "trellis2":
        raise SystemExit(f"unknown model runtime {args.model!r}; available: trellis2")
    if platform.system() != "Darwin":
        raise SystemExit("native TRELLIS.2 binary audit requires macOS")

    source_errors = native_source_errors()
    if source_errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in source_errors))

    run(["swift", "build", "-c", "release", "--product", "kg-trellis2"])
    dependency_result = subprocess.run(
        ["swift", "package", "show-dependencies", "--format", "json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    dependency_graph = json.loads(dependency_result.stdout)
    dependencies = dependency_graph.get("dependencies", [])
    if dependencies:
        names = ", ".join(item.get("name", "unknown") for item in dependencies)
        raise SystemExit(f"native Swift package has external dependencies: {names}")

    description_result = subprocess.run(
        ["swift", "package", "describe", "--type", "json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    package_errors = native_package_errors(json.loads(description_result.stdout))
    if package_errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in package_errors))

    bin_result = subprocess.run(
        ["swift", "build", "-c", "release", "--show-bin-path"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    executable = Path(bin_result.stdout.strip()) / "kg-trellis2"
    linkage = subprocess.run(
        ["otool", "-L", str(executable)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    forbidden = re.compile(r"(?i)(python|torch|libc10|mlx)")
    linked_libraries = [
        line.strip().split(" (", 1)[0]
        for line in linkage.splitlines()[1:]
        if line.strip()
    ]
    bad_libraries = [
        library for library in linked_libraries
        if forbidden.search(library)
        or not library.startswith(("/System/Library/", "/usr/lib/"))
    ]
    if bad_libraries:
        raise SystemExit("native binary has forbidden linkage: " + ", ".join(bad_libraries))

    print(
        "PASS: kg-trellis2 is Swift + Metal, has zero external Swift packages, "
        f"and links {len(linked_libraries)} Apple/Swift system libraries"
    )


def command_model_test(args: argparse.Namespace) -> None:
    command_model_setup(args)
    run([
        str(trellis_python()), "-m", "unittest", "discover",
        "-s", "ports/trellis2/tests", "-v",
    ])


def command_model_native_benchmark(args: argparse.Namespace) -> None:
    if args.model != "trellis2":
        raise SystemExit(f"unknown model runtime {args.model!r}; available: trellis2")
    run([
        "swift", "run", "-c", "release", "kg-trellis2-dense-bench",
        "--warmup", str(args.warmup), "--iterations", str(args.iterations),
    ])


def command_model_run(args: argparse.Namespace) -> None:
    if args.model != "trellis2":
        raise SystemExit(f"unknown model runtime {args.model!r}; available: trellis2")
    if not trellis_python().is_file():
        command_model_setup(args)
    command = [
        str(trellis_python()), str(ROOT / "ports" / "trellis2" / "run.py"),
        "--input", args.input, "--output", args.output, "--seed", str(args.seed),
        "--pipeline-type", args.pipeline_type,
    ]
    if args.no_preprocess:
        command.append("--no-preprocess")
    if args.steps is not None:
        command.extend(["--steps", str(args.steps)])
    if args.experimental_pbr:
        command.append("--experimental-pbr")
    command.extend([
        "--texture-size", str(args.texture_size),
        "--decimation-target", str(args.decimation_target),
        "--alpha-mode", args.alpha_mode,
    ])
    run(command)


def command_model_texture(args: argparse.Namespace) -> None:
    if args.model != "trellis2":
        raise SystemExit(f"unknown model runtime {args.model!r}; available: trellis2")
    if not trellis_python().is_file():
        command_model_setup(args)
    command = [
        str(trellis_python()), str(ROOT / "ports" / "trellis2" / "texture.py"),
        "--mesh", args.mesh, "--input", args.input, "--output", args.output,
        "--seed", str(args.seed), "--resolution", str(args.resolution),
        "--texture-size", str(args.texture_size), "--steps", str(args.steps),
        "--uv-policy", args.uv_policy, "--alpha-mode", args.alpha_mode,
    ]
    if args.no_preprocess:
        command.append("--no-preprocess")
    run(command)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="kg", description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list available kernel ports").set_defaults(func=command_list)
    commands.add_parser("doctor", help="check native build tools").set_defaults(func=command_doctor)
    commands.add_parser("validate", help="validate manifests and agent harness").set_defaults(func=command_validate)
    for name, help_text, function in (
        ("setup", "configure and build one kernel", command_setup),
        ("test", "build and run one kernel's tests", command_test),
        ("benchmark", "build and benchmark one kernel", command_benchmark),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("kernel")
        sub.set_defaults(func=function)
    model = commands.add_parser("model", help="set up, test, or run a full model runtime")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    for name, help_text, function in (
        ("setup", "prepare an isolated model runtime", command_model_setup),
        ("test", "run reference compatibility and primitive tests", command_model_test),
        ("native-setup", "build the no-Torch Swift/Metal runtime", command_model_native_setup),
        ("native-test", "run native Swift/Metal conformance tests", command_model_native_test),
        ("native-audit", "audit the release binary for a Swift/Metal-only runtime", command_model_native_audit),
        ("native-benchmark", "benchmark native Metal model primitives", command_model_native_benchmark),
    ):
        sub = model_commands.add_parser(name, help=help_text)
        sub.add_argument("model")
        if name == "native-test":
            sub.add_argument("--checkpoint")
            sub.add_argument("--texture-checkpoint")
            sub.add_argument("--sparse-structure-checkpoint")
            sub.add_argument("--sparse-structure-decoder-checkpoint")
            sub.add_argument("--dino-checkpoint")
        if name == "native-benchmark":
            sub.add_argument("--warmup", type=int, default=5)
            sub.add_argument("--iterations", type=int, default=20)
        sub.set_defaults(func=function)
    model_run = model_commands.add_parser("run", help="run real model inference")
    model_run.add_argument("model")
    model_run.add_argument("--input", required=True)
    model_run.add_argument("--output", required=True)
    model_run.add_argument("--seed", type=int, default=42)
    model_run.add_argument(
        "--pipeline-type",
        choices=("512", "1024", "1024_cascade", "1536_cascade"),
        default="512",
    )
    model_run.add_argument("--steps", type=int)
    model_run.add_argument("--texture-size", type=int, default=2048)
    model_run.add_argument("--decimation-target", type=int, default=1_000_000)
    model_run.add_argument("--alpha-mode", choices=("OPAQUE", "BLEND", "MASK"), default="OPAQUE")
    model_run.add_argument("--no-preprocess", action="store_true")
    model_run.add_argument("--experimental-pbr", action="store_true")
    model_run.set_defaults(func=command_model_run)
    model_texture = model_commands.add_parser(
        "texture", help="texture an existing mesh with TRELLIS.2"
    )
    model_texture.add_argument("model")
    model_texture.add_argument("--mesh", required=True)
    model_texture.add_argument("--input", required=True)
    model_texture.add_argument("--output", required=True)
    model_texture.add_argument("--seed", type=int, default=42)
    model_texture.add_argument("--resolution", type=int, choices=(512, 1024, 1536), default=512)
    model_texture.add_argument("--texture-size", type=int, choices=(1024, 2048, 4096), default=2048)
    model_texture.add_argument("--steps", type=int, default=12)
    model_texture.add_argument("--uv-policy", choices=("preserve", "regenerate"), default="preserve")
    model_texture.add_argument("--alpha-mode", choices=("OPAQUE", "BLEND", "MASK"), default="OPAQUE")
    model_texture.add_argument("--no-preprocess", action="store_true")
    model_texture.set_defaults(func=command_model_texture)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        args.func(args)
    except FileNotFoundError as error:
        raise SystemExit(f"required tool not found: {error.filename}") from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error


if __name__ == "__main__":
    main()
