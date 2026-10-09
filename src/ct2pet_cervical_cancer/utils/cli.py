"""JSON configuration defaults with explicit command-line overrides."""
import argparse
import json
from pathlib import Path
from .paths import project_root


def parse_config_args(parser: argparse.ArgumentParser, name: str) -> argparse.Namespace:
    root = project_root()
    parser.add_argument("--config", type=Path, default=root / "configs" / name,
                        help="JSON config file; CLI options override its values. Relative data/output paths use the project root.")
    initial, _ = parser.parse_known_args()
    config_path = initial.config.expanduser().resolve()
    try:
        values = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot read config {config_path}: {exc}")
    if not isinstance(values, dict):
        parser.error("Config must be a JSON object")
    actions = {action.dest: action for action in parser._actions if action.dest not in {"help", "config"}}
    unknown = values.keys() - actions.keys()
    if unknown:
        parser.error(f"Unknown config keys: {sorted(unknown)}")
    for key, value in values.items():
        action = actions[key]
        if value is None:
            continue
        if isinstance(action, (argparse.BooleanOptionalAction, argparse._StoreTrueAction, argparse._StoreFalseAction)):
            if not isinstance(value, bool):
                parser.error(f"Config {key} must be boolean")
        elif action.type:
            try:
                values[key] = action.type(value)
            except (TypeError, ValueError) as exc:
                parser.error(f"Invalid config {key}: {exc}")
        if action.choices is not None and values[key] not in action.choices:
            parser.error(f"Config {key} must be one of {action.choices}")
    parser.set_defaults(**values)
    args = parser.parse_args()
    args.config = config_path
    for key, action in actions.items():
        value = getattr(args, key)
        if action.type is Path and value is not None:
            value = Path(value).expanduser()
            setattr(args, key, value.resolve() if value.is_absolute() else (root / value).resolve())
    return args
