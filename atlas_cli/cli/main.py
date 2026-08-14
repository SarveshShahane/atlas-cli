import typer
from atlas_cli.core.utils import print_banner

from atlas_cli.cli.commands.init_cmd import init
from atlas_cli.cli.commands.analyze_cmd import analyze
from atlas_cli.cli.commands.clean_cmd import clean
from atlas_cli.cli.commands.plan_cmd import plan
from atlas_cli.cli.commands.experiment_cmd import experiment
from atlas_cli.cli.commands.compare_cmd import compare
from atlas_cli.cli.commands.review_cmd import review
from atlas_cli.cli.commands.explain_cmd import explain
from atlas_cli.cli.commands.report_cmd import report
from atlas_cli.cli.commands.deploy_cmd import deploy
from atlas_cli.cli.commands.replay_cmd import replay
from atlas_cli.cli.commands.ask_cmd import ask
from atlas_cli.cli.commands.export_cmd import export
from atlas_cli.cli.commands.run_cmd import run_pipeline

app = typer.Typer(
    name="atlas",
    help="Atlas CLI: Autonomous Data Science Operating System",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def main_callback(ctx: typer.Context) -> None:
    """Atlas CLI — Autonomous Data Science Operating System."""
    if ctx.invoked_subcommand is not None:
        print_banner()


app.command("run",        help="Run entire autonomous data science pipeline end-to-end.")(run_pipeline)
app.command("init",       help="Initialize local workspace and SQLite database.")(init)
app.command("analyze",    help="Analyze dataset: schema, profiling, risk assessment.")(analyze)
app.command("clean",      help="Clean dataset: Winsorization, drop collinear features & noise filtering.")(clean)
app.command("plan",       help="Generate LLM-driven preprocessing and model plan.")(plan)
app.command("experiment", help="Execute parallel multi-model training experiments.")(experiment)
app.command("compare",    help="Compare experiments with multi-objective scoring.")(compare)
app.command("review",     help="AI Reviewer: auto-diagnose model issues & execute critique-and-refine retry loop.")(review)
app.command("explain",    help="SHAP feature importance and model explanation.")(explain)
app.command("report",     help="Generate Markdown and HTML executive reports.")(report)
app.command("deploy",     help="Scaffold production FastAPI inference microservice.")(deploy)
app.command("replay",     help="Replay an experiment from its reproducibility snapshot.")(replay)
app.command("ask",        help="Ask natural language questions about your workspace.")(ask)
app.command("export",     help="Export cleaned, feature engineered, and train/test split datasets.")(export)


if __name__ == "__main__":
    app()
