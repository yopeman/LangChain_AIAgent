import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich.prompt import Prompt

from ai_agent import AIAgent

# --------------------------------------------------
# Environment & Console
# --------------------------------------------------
load_dotenv()
console = Console()

# --------------------------------------------------
# Workspace (SANDBOX)
# --------------------------------------------------
workspace = Path("code_editor_agent").resolve()
workspace.mkdir(exist_ok=True, parents=True)
os.chdir(workspace)

WORKSPACE_ROOT = Path.cwd().resolve()

def resolve_path(user_path: str) -> Path:
    """
    Resolve paths safely inside WORKSPACE_ROOT.
    Prevents path traversal and ambiguity.
    """
    resolved = (WORKSPACE_ROOT / user_path).resolve()
    if not str(resolved).startswith(str(WORKSPACE_ROOT)):
        raise ValueError("Path escapes workspace")
    return resolved

# --------------------------------------------------
# File System Tools (REAL DISK OPS)
# --------------------------------------------------

@tool
def create_file(file_name: str, content: str = "") -> str:
    """Create a new file with optional content."""
    try:
        path = resolve_path(file_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        return (
            f"✅ File created\n"
            f"- Path: {path.relative_to(WORKSPACE_ROOT)}\n"
            f"- Size: {path.stat().st_size} bytes"
        )
    except Exception as e:
        return f"❌ Failed to create file: {e}"


@tool
def write_to_file(file_name: str, content: str, mode: str = "w") -> str:
    """Write or append content to a file."""
    try:
        path = resolve_path(file_name)

        if not path.exists():
            return f"⚠️ File not found: {file_name}"

        if mode not in {"w", "a"}:
            mode = "w"

        with path.open(mode, encoding="utf-8") as f:
            f.write(content)

        return (
            f"✍️ File updated\n"
            f"- Path: {path.relative_to(WORKSPACE_ROOT)}\n"
            f"- Mode: {mode}\n"
            f"- Size: {path.stat().st_size} bytes"
        )
    except Exception as e:
        return f"❌ Write error: {e}"


@tool
def read_file(file_name: str) -> str:
    """Read a file safely."""
    try:
        path = resolve_path(file_name)

        if not path.exists() or not path.is_file():
            return f"❌ File not found: {file_name}"

        size = path.stat().st_size
        if size > 1_000_000:
            return f"⚠️ File too large ({size} bytes)"

        content = path.read_text(encoding="utf-8")

        return (
            f"📄 File: {path.relative_to(WORKSPACE_ROOT)}\n"
            f"Size: {size} bytes\n\n"
            f"{content}"
        )
    except Exception as e:
        return f"❌ Read error: {e}"


@tool
def delete_file(file_name: str) -> str:
    """Delete a file safely."""
    protected = {"main.py", "ai_agent.py", ".env", "requirements.txt"}
    try:
        path = resolve_path(file_name)

        if not path.exists():
            return "❌ File does not exist"

        if path.name in protected:
            return "⚠️ Protected file. Manual deletion required."

        path.unlink()
        return f"🗑️ File deleted: {path.relative_to(WORKSPACE_ROOT)}"
    except Exception as e:
        return f"❌ Delete error: {e}"


@tool
def create_dir(dir_name: str) -> str:
    """Create a directory."""
    try:
        path = resolve_path(dir_name)
        path.mkdir(parents=True, exist_ok=True)
        return f"📁 Directory created: {path.relative_to(WORKSPACE_ROOT)}"
    except Exception as e:
        return f"❌ Directory error: {e}"


@tool
def read_dir(dir_name: str = ".") -> str:
    """List directory contents."""
    try:
        path = resolve_path(dir_name)

        if not path.exists() or not path.is_dir():
            return f"❌ Directory not found: {dir_name}"

        table = Table(title=f"📂 {path.relative_to(WORKSPACE_ROOT)}")
        table.add_column("Type")
        table.add_column("Name")
        table.add_column("Size", justify="right")

        lines = []
        for item in sorted(path.iterdir()):
            if item.is_dir():
                table.add_row("DIR", f"[bold blue]{item.name}/[/]", "")
                lines.append(f"DIR  {item.name}/")
            else:
                size = item.stat().st_size
                table.add_row("FILE", item.name, f"{size} bytes")
                lines.append(f"FILE {item.name} ({size} bytes)")

        console.print(table)

        return (
            f"📂 Directory: {path.relative_to(WORKSPACE_ROOT)}\n"
            f"Items:\n" + "\n".join(lines)
        )
    except Exception as e:
        return f"❌ Read dir error: {e}"


@tool
def delete_dir(dir_name: str) -> str:
    """Delete an empty directory."""
    try:
        path = resolve_path(dir_name)

        if any(path.iterdir()):
            return "⚠️ Directory not empty"

        path.rmdir()
        return f"🗑️ Directory deleted: {path.relative_to(WORKSPACE_ROOT)}"
    except Exception as e:
        return f"❌ Delete dir error: {e}"


@tool
def get_current_directory() -> str:
    """Return the current workspace directory."""
    return str(WORKSPACE_ROOT)

# --------------------------------------------------
# Agent Setup
# --------------------------------------------------

class ResponseOutput(BaseModel):
    result: str = Field(..., description="Result of operation")


SYSTEM_PROMPT = """
You are YopeAI, a professional Code Editor Agent.

Rules:
- All files are created inside the workspace directory.
- File operations persist to disk.
- Always create files before writing.
- Use read_dir to verify structure.
- Return clear results of every operation.

You can:
- Create, read, write, and delete files
- Create and list directories
- Search the web for documentation
""".strip()


llm = ChatGroq(model="llama-3.3-70b-versatile")

tools = [
    create_file,
    write_to_file,
    read_file,
    delete_file,
    create_dir,
    read_dir,
    delete_dir,
    get_current_directory,
    TavilySearch(max_results=5),
]

agent = AIAgent(
    llm=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    response_format=ResponseOutput,
)

# --------------------------------------------------
# Context Memory
# --------------------------------------------------

class Context:
    def __init__(self, max_len: int = 8):
        self.items: List[dict] = []
        self.max_len = max_len

    def append(self, item: dict):
        if len(self.items) >= self.max_len:
            self.items.pop(0)
        self.items.append(item)

history = Context()

# --------------------------------------------------
# Runtime Loop
# --------------------------------------------------

console.print(Panel.fit(
    "[bold cyan]🤖 YopeAI Code Editor Agent[/bold cyan]\n"
    "Workspace: [bold]" + str(WORKSPACE_ROOT) + "[/bold]\n"
    "Type [bold yellow].q[/bold yellow] to quit",
    border_style="cyan"
))

while True:
    user_input = Prompt.ask("[bold green]Enter your prompt[/bold green]")

    if user_input.strip() == ".q":
        console.print("👋 Goodbye!", style="bold red")
        break

    prompt = PromptTemplate.from_template(
        "Context:\n{context}\n\nAvailable tools:{tools}\n\nPrompt:\n{prompt}"
    ).format(
        context=history.items,
        prompt=user_input,
        tools=tools
    )

    improved = agent.improve(prompt)
    result = agent.invoke(improved)

    if isinstance(result, BaseModel):
        result = result.result

    history.append({"user": improved, "agent": result})

    console.print(Panel(
        Text(user_input, style="bold white"),
        title="🧑 User Input",
        border_style="green",
    ))

    console.print(Panel(
        Syntax(improved, "markdown", word_wrap=True),
        title="✨ Improved Prompt",
        border_style="blue",
    ))

    console.print(Panel(
        Syntax(result, "markdown", word_wrap=True),
        title="🤖 Agent Result",
        border_style="magenta",
    ))
