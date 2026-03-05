import argparse
import glob
import os
import shutil
import subprocess
import sys


def run_command(command, cwd=None, env=None):
    print(f"Running: {command}")
    subprocess.check_call(command, shell=True, cwd=cwd, env=env)


def main():
    parser = argparse.ArgumentParser(
        description="Update IDE support files (compile_commands.json, stubs)"
    )
    parser.add_argument("--release", action="store_true", help="Build in Release mode")
    parser.add_argument("--debug", action="store_true", help="Build in Debug mode")
    parser.add_argument(
        "--build-type", help="Explicit build type (e.g. RelWithDebInfo)"
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    # Prepare environment
    env = os.environ.copy()
    if args.release:
        env["CMAKE_BUILD_TYPE"] = "Release"
    elif args.debug:
        env["CMAKE_BUILD_TYPE"] = "Debug"
    elif args.build_type:
        env["CMAKE_BUILD_TYPE"] = args.build_type

    print(f"--- 1. Triggering Build ({env.get('CMAKE_BUILD_TYPE', 'Default')}) ---")

    # We use --no-build-isolation to use the installed build backend
    run_command(
        f"{sys.executable} -m pip install -e . --no-build-isolation -v", env=env
    )

    print("\n--- 2. Linking compile_commands.json ---")
    # Find compile_commands.json in the build directory
    # The build dir is configured as build/{wheel_tag} in pyproject.toml
    # We'll search recursively in build/
    compile_commands_candidates = glob.glob(
        "build/**/compile_commands.json", recursive=True
    )

    if not compile_commands_candidates:
        print("Error: Could not find compile_commands.json in build/ directory.")
        # Try to suggest why
        print("Make sure cmake ran and CMAKE_EXPORT_COMPILE_COMMANDS=ON was effective.")
        sys.exit(1)

    # Pick the most recent one if multiple
    compile_commands_src = sorted(compile_commands_candidates, key=os.path.getmtime)[-1]
    compile_commands_dst = "compile_commands.json"

    if os.path.exists(compile_commands_dst):
        os.remove(compile_commands_dst)

    # Symlink if possible, else copy
    try:
        os.symlink(compile_commands_src, compile_commands_dst)
        print(f"Symlinked {compile_commands_src} -> {compile_commands_dst}")
    except OSError:
        shutil.copy2(compile_commands_src, compile_commands_dst)
        print(f"Copied {compile_commands_src} -> {compile_commands_dst}")

    print("\n--- 3. Verifying Stub Generation ---")
    stub_path = os.path.join("src", "task4feedback", "fastsim2.pyi")
    if os.path.exists(stub_path):
        print(f"Success: {stub_path} exists.")
    else:
        print(f"Error: {stub_path} was not generated.")
        print("Check the CMake custom command execution.")
        sys.exit(1)

    print("\n--- IDE Support Setup Complete ---")
    print("Restart your editor or reload the window to apply changes.")


if __name__ == "__main__":
    main()
