import os
import typer
import time
import subprocess
from rich.console import Console
from rich.prompt import Confirm

app = typer.Typer()
console = Console()

def find_env_files(base_dir: str):
    """Recursively find all .env files starting from the base directory."""
    env_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == ".env":
                env_files.append(os.path.join(root, file))
    return env_files

def update_is_test(env_file: str, value: str):
    """Update the IS_TEST value in a given .env file."""
    updated_lines = []
    with open(env_file, "r") as file:
        lines = file.readlines()
        is_test_found = False
        for line in lines:
            if line.startswith("IS_TEST="):
                updated_lines.append(f"IS_TEST={value}\n")
                is_test_found = True
            else:
                updated_lines.append(line)
        if not is_test_found:
            updated_lines.append(f"IS_TEST={value}\n")
    
    # Write the updated lines back to the .env file
    with open(env_file, "w") as file:
        file.writelines(updated_lines)
    
    console.print(f"[green]Updated IS_TEST to {value} in [magenta]{env_file}[/magenta].[/green]")

def run_pytest():
    """Run pytest and display output."""
    console.print("[bold green]Running pytest...[/bold green]")
    try:
        result = subprocess.run(["pytest"], capture_output=True, text=True)
        console.print(result.stdout)
        if result.returncode == 0:
            console.print("[bold cyan]Tests completed successfully![/bold cyan]")
        else:
            console.print(f"[bold red]Tests failed. Check the logs:[/bold red]\n{result.stderr}")
    except FileNotFoundError:
        console.print("[red]pytest not found. Please ensure pytest is installed in your environment.[/red]")

@app.command()
def run_tests(base_dir: str = typer.Argument(".", help="The base directory to search for .env files")):
    """Search for .env files in subdirectories and update IS_TEST."""
    
    # Start a spinner while finding .env files
    with console.status("[cyan]Searching for .env files...[/cyan]", spinner="dots"):
        env_files = find_env_files(base_dir)

    if not env_files:
        console.print("[red]No .env files found.[/red]")
        raise typer.Exit()

    # List found .env files
    console.print(f"[green]Found {len(env_files)} .env files:[/green]")
    for file in env_files:
        console.print(f"  • [cyan]{file}[/cyan]")

    # Ask user if they want to run tests
    run_tests = Confirm.ask("Do you want to run tests?", default=True)
    
    if run_tests:
        # Update IS_TEST to true and run pytest
        for env_file in env_files:
            update_is_test(env_file, "true")
        run_pytest()
    else:
        # Update IS_TEST to false
        for env_file in env_files:
            update_is_test(env_file, "false")
        console.print("[bold yellow]Tests will not be executed. IS_TEST set to false.[/bold yellow]")

if __name__ == "__main__":
    app()
