import sys

import typer

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

# Pass-through context so draccus/argparse in each module sees its own args
_PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True, "max_content_width": 120}


@app.command(context_settings=_PASSTHROUGH, add_help_option=False)
def train(ctx: typer.Context) -> None:
    """Train a seq2seq model."""
    sys.argv = [sys.argv[0]] + ctx.args
    from mt_training.train import main
    main()


@app.command(context_settings=_PASSTHROUGH, add_help_option=False)
def eval(ctx: typer.Context) -> None:
    """Run evaluation (BLEU/chrF/TER) on a dataset."""
    sys.argv = [sys.argv[0]] + ctx.args
    from mt_training.eval import main
    main()


@app.command(context_settings=_PASSTHROUGH, add_help_option=False)
def infer(ctx: typer.Context) -> None:
    """Translate text interactively or from a single input."""
    sys.argv = [sys.argv[0]] + ctx.args
    from mt_training.inference import main
    main()


@app.command(context_settings=_PASSTHROUGH, add_help_option=False)
def backtranslate(ctx: typer.Context) -> None:
    """Generate synthetic data via back-translation (fr→en→fr)."""
    sys.argv = [sys.argv[0]] + ctx.args
    from mt_training.backtranslate import main
    main()


@app.command("convert-ct2", context_settings=_PASSTHROUGH, add_help_option=False)
def convert_ct2(ctx: typer.Context) -> None:
    """Convert a HuggingFace model to CTranslate2 format."""
    sys.argv = [sys.argv[0]] + ctx.args
    from mt_training.convert_ct2 import main
    main()


def main() -> None:
    app()
