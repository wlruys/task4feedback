import sys
from pathlib import Path
from typing import List, Set, Dict, Tuple
import re


CONF_DIR = Path(__file__).parent.parent / "conf"


def extract_defaults(yaml_path: Path) -> List[str]:
    """
    Extract defaults list from a YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        List of default references (e.g., ["feature: cnn_batch", "models: gnn"])
    """
    defaults = []

    try:
        with open(yaml_path) as f:
            in_defaults = False
            for line in f:
                stripped = line.strip()

                # Start of defaults section
                if stripped.startswith("defaults:"):
                    in_defaults = True
                    continue

                # End of defaults section (next top-level key)
                if in_defaults and not line.startswith((" ", "\t", "-")):
                    break

                # Parse default entry
                if in_defaults and stripped.startswith("- "):
                    # Remove leading "- " and comments
                    entry = stripped[2:].split("#")[0].strip()
                    if entry and entry != "_self_":
                        defaults.append(entry)

    except Exception as e:
        print(f"# Warning: Failed to read {yaml_path}: {e}", file=sys.stderr)

    return defaults


def parse_default_entry(entry: str) -> Tuple[str, str]:
    """
    Parse a default entry into (category, name) or (path, target).

    Examples:
        "feature: cnn_batch" -> ("feature", "cnn_batch")
        "/normalization/standard" -> ("/normalization", "standard")
        "architecture/vector@layers.state" -> ("architecture/vector", "@layers.state")

    Args:
        entry: Default entry string

    Returns:
        Tuple of (source, target)
    """
    # Handle package path syntax (@)
    if "@" in entry:
        source, target = entry.split("@", 1)
        return (source.strip(), f"@{target.strip()}")

    # Handle category: name syntax
    if ":" in entry:
        category, name = entry.split(":", 1)
        return (category.strip(), name.strip())

    # Handle absolute paths
    if entry.startswith("/"):
        parts = entry.split("/")
        if len(parts) > 2:
            return ("/".join(parts[:-1]), parts[-1])
        return (entry, "")

    # Simple path
    return (entry, "")


def sanitize_node_name(name: str) -> str:
    """Convert file path to valid Graphviz node name."""
    return name.replace("/", "_").replace(".", "_").replace(":", "_").replace("@", "_at_")


def generate_graph():
    """Generate Graphviz DOT graph of configuration dependencies."""
    print("digraph ConfigDeps {")
    print('  rankdir=LR;')
    print('  node [shape=box, style=rounded];')
    print()

    # Collect all config files and their dependencies
    edges: Set[Tuple[str, str]] = set()
    nodes: Set[str] = set()

    for yaml_file in sorted(CONF_DIR.rglob("*.yaml")):
        rel_path = str(yaml_file.relative_to(CONF_DIR))
        node_name = sanitize_node_name(rel_path)
        nodes.add((node_name, rel_path))

        defaults = extract_defaults(yaml_file)
        for default in defaults:
            source, target = parse_default_entry(default)

            # Skip package mappings for clarity
            if target.startswith("@"):
                continue

            # Create edge
            if ":" in default:
                # Category reference (e.g., feature: cnn_batch)
                category, name = default.split(":", 1)
                target_path = f"{category.strip()}/{name.strip()}.yaml"
                target_node = sanitize_node_name(target_path)
                edges.add((node_name, target_node))
            elif default.startswith("/"):
                # Absolute path
                target_path = default[1:] + ".yaml"
                target_node = sanitize_node_name(target_path)
                edges.add((node_name, target_node))

    # Output nodes with labels
    print("  // Nodes")
    for node_name, rel_path in sorted(nodes):
        label = rel_path.replace(".yaml", "")
        # Color root configs differently
        if "/" not in rel_path and not rel_path.startswith("_"):
            print(f'  {node_name} [label="{label}", style="rounded,filled", fillcolor=lightblue];')
        else:
            print(f'  {node_name} [label="{label}"];')

    print()
    print("  // Edges")
    for source, target in sorted(edges):
        print(f'  {source} -> {target};')

    print("}")


def main():
    """Main entry point."""
    if not CONF_DIR.exists():
        print(f"Error: Config directory not found: {CONF_DIR}", file=sys.stderr)
        sys.exit(1)

    generate_graph()


if __name__ == "__main__":
    main()
