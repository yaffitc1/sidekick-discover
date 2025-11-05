"""CLI entry points for Discovery.

Commands:
- discovery add snowflake --database <DB> --schema <SCHEMA> --tables <T1,T2> --name <alias>
- discovery agent --source <alias> [--sample-rows N] [--output ./outputs]
- discovery dashboard <path>
"""

import os
import webbrowser
from pathlib import Path
from typing import Optional
import typer
from dotenv import load_dotenv

from discovery.io.registry import upsert_source, get_source
from discovery.agent.orchestrator import run_snowflake, run_multi_source

# Load environment variables from .env file
load_dotenv()

app = typer.Typer(help="Discovery CLI")


@app.command("dashboard")
def dashboard_open(path: str = typer.Argument(..., help="Path to dashboard HTML")) -> None:
    """Open a generated dashboard HTML in the default browser."""
    html_path = Path(path).expanduser().resolve()
    if not html_path.exists():
        typer.secho(f"Dashboard not found: {html_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    webbrowser.open_new_tab(html_path.as_uri())
    typer.secho(f"Opened {html_path}", fg=typer.colors.GREEN)


@app.command("add")
def add(
    name: str = typer.Option(..., "--name", help="Alias for the source"),
    database: str = typer.Option(..., "--database", help="Snowflake database name"),
    schema: str = typer.Option(..., "--schema", help="Snowflake schema name"),
    tables: Optional[str] = typer.Option(None, "--tables", help="Comma-separated list of tables (optional)"),
):
    """Register a Snowflake source alias in the local registry (read-only, no table changes)."""
    table_list = [t.strip() for t in tables.split(",")] if tables else None
    src = {
        "type": "snowflake",
        "database": database,
        "schema": schema,
        "tables": table_list,
    }
    upsert_source(name, src)
    tables_str = f" tables: {', '.join(table_list)}" if table_list else " (all tables)"
    typer.secho(f"Registered Snowflake source '{name}' -> {database}.{schema}{tables_str}", fg=typer.colors.GREEN)


@app.command("agent")
def agent_run(
    source: str = typer.Option(None, "--source", help="Registered source name (single source)"),
    sources: str = typer.Option(None, "--sources", help="Comma-separated list of source names (multiple sources)"),
    output: str = typer.Option("./outputs", "--output", help="Base output directory"),
    sample_rows: int = typer.Option(100000, "--sample-rows", help="Max rows to sample"),
    goal: Optional[str] = typer.Option(None, "--goal", help="Optional exploration goal"),
):
    """Run the agent pipeline on registered Snowflake source(s) and write artifacts.
    
    Use --source for single source or --sources for multi-source analysis.
    """
    if not source and not sources:
        typer.secho("Either --source or --sources must be provided", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    
    if source and sources:
        typer.secho("Cannot use both --source and --sources", fg=typer.colors.RED)
        raise typer.Exit(code=2)
    
    # Multi-source mode
    if sources:
        source_list = [s.strip() for s in sources.split(",")]
        source_configs = []
        
        for src_name in source_list:
            src = get_source(src_name)
            if not src:
                typer.secho(f"Source '{src_name}' not found. Use 'discovery add snowflake ...' first.", fg=typer.colors.RED)
                raise typer.Exit(code=3)
            
            if src.get("type") != "snowflake":
                typer.secho(f"Only Snowflake sources are supported. Source '{src_name}' has type '{src.get('type')}'", fg=typer.colors.RED)
                raise typer.Exit(code=4)
            
            source_configs.append({
                "name": src_name,
                "database": src["database"],
                "schema": src["schema"],
                "tables": src.get("tables"),
            })
        
        run_multi_source(
            sources=source_configs,
            output_dir=str(Path(output) / "multi_source_analysis"),
            sample_rows=sample_rows,
            goal=goal,
        )
        typer.secho(f"Multi-source analysis completed. Outputs at {Path(output) / 'multi_source_analysis'}", fg=typer.colors.GREEN)
        return
    
    # Single source mode
    src = get_source(source)
    if not src:
        typer.secho(f"Source '{source}' not found. Use 'discovery add snowflake ...' first.", fg=typer.colors.RED)
        raise typer.Exit(code=3)
    
    if src.get("type") != "snowflake":
        typer.secho(f"Only Snowflake sources are supported. Source '{source}' has type '{src.get('type')}'", fg=typer.colors.RED)
        raise typer.Exit(code=4)
    
    run_snowflake(
        database=src["database"],
        schema=src["schema"],
        source_name=source,
        output_dir=str(Path(output) / source),
        tables=src.get("tables"),
        sample_rows=sample_rows,
        goal=goal,
    )
    typer.secho(f"Agent run completed. Outputs at {Path(output) / source}", fg=typer.colors.GREEN)


def main() -> None:
    app()


if __name__ == "__main__":
    main()


