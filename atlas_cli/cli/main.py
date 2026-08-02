import typer
from atlas_cli.core.utils import print_banner
from atlas_cli.cli.commands.init_cmd import app as init_app
from atlas_cli.cli.commands.analyze_cmd import app as analyze_app
from atlas_cli.cli.commands.plan_cmd import app as plan_app
from atlas_cli.cli.commands.experiment_cmd import app as experiment_app
from atlas_cli.cli.commands.compare_cmd import app as compare_app
from atlas_cli.cli.commands.explain_cmd import app as explain_app
from atlas_cli.cli.commands.report_cmd import app as report_app
from atlas_cli.cli.commands.deploy_cmd import app as deploy_app
from atlas_cli.cli.commands.replay_cmd import app as replay_app
from atlas_cli.cli.commands.ask_cmd import app as ask_app

app = typer.Typer(
    name="atlas",
    help="Atlas CLI: Autonomous Data Science Operating System",
    add_completion=False,
    no_args_is_help=True
)

@app.callback()
def main_callback():
    """Atlas CLI Autonomous Data Science Operating System."""
    print_banner()

app.add_typer(init_app, name="init")
app.add_typer(analyze_app, name="analyze")
app.add_typer(plan_app, name="plan")
app.add_typer(experiment_app, name="experiment")
app.add_typer(compare_app, name="compare")
app.add_typer(explain_app, name="explain")
app.add_typer(report_app, name="report")
app.add_typer(deploy_app, name="deploy")
app.add_typer(replay_app, name="replay")
app.add_typer(ask_app, name="ask")

if __name__ == "__main__":
    app()
