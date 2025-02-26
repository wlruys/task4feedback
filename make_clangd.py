import os
import yaml


def generate_clangd_config(output_file):
    current_dir = os.getcwd()
    tabulate_path = os.path.join(current_dir, "external", "tabulate", "include")
    doctest_path = os.path.join(current_dir, "external", "doctest")
    spdlog_path = os.path.join(current_dir, "external")
    backend_path = os.path.join(current_dir, "src", "backend")
    backend_inc_path = os.path.join(current_dir, "src", "backend", "include")
    nanobind_inc_path = "-I/home/popdesk/micromamba/envs/py311/lib/python3.11/site-packages/nanobind/include"

    config = {
        "CompileFlags": {
            "Add": [
                "-std=c++20",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                "-Wshadow",
                "-Wconversion",
                f"-I{tabulate_path}",
                f"-I{doctest_path}",
                f"-I{backend_path}",
                f"-I{backend_inc_path}",
                f"-I{spdlog_path}",
                f"{nanobind_inc_path}",
            ]
        },
        "Diagnostics": {
            "UnusedIncludes": "Strict",
            "ClangTidy": {
                "Add": ["modernize*", "performance*", "readability*", "bugprone*"],
                "Remove": ["modernize-use-trailing-return-type"],
                "CheckOptions": {
                    "readability-identifier-naming.IgnoreMainLikeFunctions": True,
                    "readability-identifier-length.MinimumVariableNameLength": "1",
                    "readability-identifier-length.MinimumParameterNameLength": "1",
                },
            },
        },
        "Index": {"Background": "Build"},
        "InlayHints": {
            "Enabled": True,
            "ParameterNames": False,
            "DeducedTypes": True,
            "Designators": True,
        },
        "Hover": {"ShowAKA": True},
        "Completion": {"AllScopes": True},
    }

    with open(output_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    output_file = ".clangd"
    generate_clangd_config(output_file)
    print(
        f".clangd configuration file has been generated in {os.path.abspath(output_file)}"
    )
