import sys

SUBCOMMANDS = {
    "train": "mt_training.train",
    "eval": "mt_training.eval",
    "infer": "mt_training.inference",
    "backtranslate": "mt_training.backtranslate",
    "convert-ct2": "mt_training.convert_ct2",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in SUBCOMMANDS:
        print("Usage: mt-training <subcommand> [args...]\n")
        print("Subcommands:")
        for name in SUBCOMMANDS:
            print(f"  {name}")
        sys.exit(1 if len(sys.argv) < 2 else 0)

    # Remove the subcommand so the underlying parser sees clean argv
    cmd = sys.argv.pop(1)
    module_name = SUBCOMMANDS[cmd]

    import importlib
    mod = importlib.import_module(module_name)
    mod.main()
