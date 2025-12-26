import os
import sys
import json
from typing import List, Dict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ai_agent import AIAgent

# Load environment variables
load_dotenv()

# Initialize Rich console for better output
console = Console()

# Define response format models
class FileOperationResponse(BaseModel):
    """Response model for file operations"""
    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="Detailed message about the operation")
    file_path: str = Field(default="", description="Path to the file/directory")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class FileContentResponse(BaseModel):
    """Response model for file content reading"""
    content: str = Field(description="Content of the file")
    file_name: str = Field(description="Name of the file")
    size: int = Field(description="Size of the file in bytes")
    exists: bool = Field(description="Whether the file exists")

class DirectoryListingResponse(BaseModel):
    """Response model for directory listing"""
    path: str = Field(description="Directory path")
    items: List[str] = Field(description="List of items in the directory")
    total_items: int = Field(description="Total number of items")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ConversationMemory:
    """Simple conversation memory manager"""
    def __init__(self, max_history: int = 20, memory_file: str = "conversation_memory.json"):
        self.max_history = max_history
        self.memory_file = Path(memory_file)
        self.history = []
        self.load_memory()
    
    def add(self, user_input: str, agent_response: str, timestamp: str = None):
        """Add a conversation turn to memory"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        self.history.append({
            "user": user_input,
            "agent": agent_response,
            "timestamp": timestamp
        })
        
        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        self.save_memory()
    
    def get_context(self, last_n: int = 5) -> str:
        """Get recent conversation context"""
        if not self.history:
            return ""
        
        recent = self.history[-last_n:] if len(self.history) > last_n else self.history
        context_lines = []
        
        for i, turn in enumerate(recent, 1):
            context_lines.append(f"Turn {i} (User): {turn['user']}")
            # Truncate long agent responses for context
            agent_response = turn['agent']
            if len(agent_response) > 500:
                agent_response = agent_response[:500] + "..."
            context_lines.append(f"Turn {i} (Agent): {agent_response}")
        
        return "\n".join(context_lines)
    
    def get_full_history(self) -> List[Dict]:
        """Get full conversation history"""
        return self.history
    
    def clear(self):
        """Clear conversation history"""
        self.history = []
        self.save_memory()
    
    def save_memory(self):
        """Save memory to file"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save memory: {e}[/yellow]")
    
    def load_memory(self):
        """Load memory from file"""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                console.print(f"[dim]Loaded {len(self.history)} conversation turns from memory[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load memory: {e}[/yellow]")
            self.history = []

# Initialize LLM with better parameters
# llm = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     temperature=0.3,  # Lower temperature for more consistent file operations
#     max_tokens=2048,
# )

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash-lite",
    temperature=0.2,
)

# File system tools with better error handling and actual implementation
@tool
def create_file(file_name: str, content: str = "") -> str:
    """Create a new file with optional content.
    
    Args:
        file_name: Name of the file to create (including path if needed)
        content: Optional content to write to the file
        
    Returns:
        Success message or error
    """
    try:
        # Ensure the directory exists
        file_path = Path(file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create the file
        file_path.touch(exist_ok=True)
        
        # Write content if provided
        if content:
            file_path.write_text(content)
            
        return f"File '{file_name}' created successfully with {len(content)} characters."
    except Exception as e:
        return f"Error creating file '{file_name}': {str(e)}"

@tool
def write_to_file(file_name: str, content: str, mode: str = "w") -> str:
    """Write content to a file.
    
    Args:
        file_name: Name of the file to write to
        content: Content to write to the file
        mode: Write mode - 'w' for write (overwrite), 'a' for append
        
    Returns:
        Success message or error
    """
    try:
        file_path = Path(file_name)
        
        if not file_path.exists():
            return f"File '{file_name}' does not exist. Use create_file first."
        
        # Validate mode
        if mode not in ["w", "a"]:
            mode = "w"
            
        # Write content
        if mode == "w":
            file_path.write_text(content)
        else:  # mode == "a"
            with file_path.open("a", encoding="utf-8") as f:
                f.write(content)
                
        return f"Successfully wrote {len(content)} characters to '{file_name}' in mode '{mode}'."
    except Exception as e:
        return f"Error writing to file '{file_name}': {str(e)}"

@tool
def read_file(file_name: str) -> str:
    """Read content from a file.
    
    Args:
        file_name: Name of the file to read
        
    Returns:
        File content or error message
    """
    try:
        file_path = Path(file_name)
        
        if not file_path.exists():
            return f"Error: File '{file_name}' does not exist."
            
        if not file_path.is_file():
            return f"Error: '{file_name}' is not a file."
            
        # Check file size to prevent reading huge files
        if file_path.stat().st_size > 1024 * 1024:  # 1MB limit
            return f"Error: File '{file_name}' is too large ({file_path.stat().st_size} bytes)."
            
        content = file_path.read_text(encoding="utf-8")
        return f"Content of '{file_name}':\n\n{content}"
    except Exception as e:
        return f"Error reading file '{file_name}': {str(e)}"

@tool
def delete_file(file_name: str) -> str:
    """Delete a file.
    
    Args:
        file_name: Name of the file to delete
        
    Returns:
        Success message or error
    """
    try:
        file_path = Path(file_name)
        
        if not file_path.exists():
            return f"Error: File '{file_name}' does not exist."
            
        if not file_path.is_file():
            return f"Error: '{file_name}' is not a file."
            
        # For safety, ask for confirmation for important files
        if file_name in ["main.py", "ai_agent.py", ".env", "requirements.txt"]:
            return f"Warning: '{file_name}' is an important file. Manual deletion recommended."
            
        file_path.unlink()
        return f"File '{file_name}' deleted successfully."
    except Exception as e:
        return f"Error deleting file '{file_name}': {str(e)}"

@tool
def create_dir(dir_name: str) -> str:
    """Create a new directory.
    
    Args:
        dir_name: Name of the directory to create
        
    Returns:
        Success message or error
    """
    try:
        dir_path = Path(dir_name)
        
        if dir_path.exists():
            return f"Directory '{dir_name}' already exists."
            
        dir_path.mkdir(parents=True, exist_ok=True)
        return f"Directory '{dir_name}' created successfully."
    except Exception as e:
        return f"Error creating directory '{dir_name}': {str(e)}"

@tool
def read_dir(dir_name: str = ".") -> str:
    """List contents of a directory.
    
    Args:
        dir_name: Directory path (defaults to current directory)
        
    Returns:
        Formatted directory listing or error
    """
    try:
        dir_path = Path(dir_name)
        
        if not dir_path.exists():
            return f"Error: Directory '{dir_name}' does not exist."
            
        if not dir_path.is_dir():
            return f"Error: '{dir_name}' is not a directory."
            
        items = []
        for item in dir_path.iterdir():
            if item.is_dir():
                items.append(f"📁 {item.name}/")
            else:
                size = item.stat().st_size
                items.append(f"📄 {item.name} ({size} bytes)")
                
        items.sort()
        listing = "\n".join(items)
        return f"Contents of '{dir_name}':\n\n{listing}\n\nTotal: {len(items)} items"
    except Exception as e:
        return f"Error reading directory '{dir_name}': {str(e)}"

@tool
def delete_dir(dir_name: str) -> str:
    """Delete a directory (must be empty).
    
    Args:
        dir_name: Name of the directory to delete
        
    Returns:
        Success message or error
    """
    try:
        dir_path = Path(dir_name)
        
        if not dir_path.exists():
            return f"Error: Directory '{dir_name}' does not exist."
            
        if not dir_path.is_dir():
            return f"Error: '{dir_name}' is not a directory."
            
        # Check if directory is empty
        if any(dir_path.iterdir()):
            return f"Error: Directory '{dir_name}' is not empty. Cannot delete non-empty directories."
            
        dir_path.rmdir()
        return f"Directory '{dir_name}' deleted successfully."
    except Exception as e:
        return f"Error deleting directory '{dir_name}': {str(e)}"

@tool
def get_current_directory() -> str:
    """Get the current working directory."""
    return f"Current directory: {os.getcwd()}"

@tool
def list_available_tools() -> str:
    """List all available tools with descriptions."""
    tools_list = [
        "📝 create_file(file_name: str, content: str = '') - Create a new file",
        "✍️ write_to_file(file_name: str, content: str, mode: str = 'w') - Write to a file",
        "📖 read_file(file_name: str) - Read content from a file",
        "🗑️ delete_file(file_name: str) - Delete a file",
        "📁 create_dir(dir_name: str) - Create a new directory",
        "📋 read_dir(dir_name: str = '.') - List directory contents",
        "🗑️ delete_dir(dir_name: str) - Delete an empty directory",
        "📍 get_current_directory() - Get current working directory",
        "🛠️ list_available_tools() - List all available tools",
    ]
    return "Available Tools:\n\n" + "\n".join(tools_list)

# Collect all tools
my_tools = [
    create_file,
    write_to_file,
    read_file,
    delete_file,
    create_dir,
    read_dir,
    delete_dir,
    get_current_directory,
    list_available_tools,
]

def display_banner():
    """Display a fancy banner."""
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]🤖 AI Agent File System Manager[/bold cyan]\n"
        "[italic]Powered by Groq + LangChain[/italic]",
        border_style="cyan"
    ))
    
def display_help():
    """Display help information."""
    table = Table(title="Available Commands", show_header=True, header_style="bold magenta")
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Example", style="green")
    
    commands = [
        (".q", "Quit the program", ".q"),
        (".h", "Show this help", ".h"),
        (".t", "Show available tools", ".t"),
        (".i", "Show agent info", ".i"),
        (".c", "Clear screen", ".c"),
        (".m", "Toggle prompt improvement mode", ".m"),
        (".s", "Show session stats", ".s"),
        (".p", "Print last response", ".p"),
        (".mem", "Show conversation memory", ".mem"),
        (".clearmem", "Clear conversation memory", ".clearmem"),
        ("any text", "Process as command to agent", "create a file named test.txt with hello world"),
    ]
    
    for cmd, desc, example in commands:
        table.add_row(cmd, desc, example)
    
    console.print(table)
    
def display_agent_info(agent: AIAgent):
    """Display agent configuration information."""
    info = agent.get_agent_info()
    
    table = Table(title="🤖 Agent Configuration", show_header=True, header_style="bold blue")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in info.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(table)

def display_memory(memory: ConversationMemory):
    """Display conversation memory."""
    history = memory.get_full_history()
    
    if not history:
        console.print("[yellow]No conversation history yet.[/yellow]")
        return
    
    table = Table(title="🧠 Conversation Memory", show_header=True, header_style="bold purple")
    table.add_column("#", style="cyan")
    table.add_column("User Input", style="white", width=40)
    table.add_column("Agent Response", style="green", width=40)
    table.add_column("Timestamp", style="dim")
    
    for i, turn in enumerate(history, 1):
        # Truncate long responses for display
        user_input = turn['user'][:50] + "..." if len(turn['user']) > 50 else turn['user']
        agent_response = turn['agent'][:50] + "..." if len(turn['agent']) > 50 else turn['agent']
        
        table.add_row(
            str(i),
            user_input,
            agent_response,
            turn['timestamp'][:19]  # Show only date and time without microseconds
        )
    
    console.print(table)
    console.print(f"[dim]Total turns in memory: {len(history)}[/dim]")

def test_basic_operations(agent: AIAgent, memory: ConversationMemory):
    """Test basic agent operations."""
    console.print("[bold yellow]🧪 Testing Basic Operations...[/bold yellow]")
    
    test_cases = [
        "List the current directory contents",
        "Create a test directory called 'agent_test'",
        "Create a file named 'agent_test/test.txt' with content 'Hello from AI Agent'",
        "Read the content of 'agent_test/test.txt'",
        "Append ' - Additional text' to 'agent_test/test.txt'",
        "Read 'agent_test/test.txt' again to see changes",
        "List contents of 'agent_test' directory",
        "Show me what tools are available",
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        console.print(f"\n[bold]Test {i}:[/bold] {test_input}")
        try:
            # Add context from memory
            context = memory.get_context()
            full_input = test_input
            if context:
                full_input = f"Previous conversation context:\n{context}\n\nCurrent request: {test_input}"
            
            result = agent.invoke(full_input)
            
            # Add to memory
            memory.add(test_input, str(result))
            
            # Display result
            if isinstance(result, dict):
                console.print(Panel(str(result), title=f"Result {i}", border_style="dim"))
            else:
                console.print(Panel(str(result), title=f"Result {i}", border_style="green"))
                
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    console.print("\n[bold green]✅ Basic tests completed![/bold green]")

def main():
    """Main interactive loop."""
    display_banner()
    
    # Create test directory
    test_dir = Path(f"agent_files/file_@_{datetime.now()}")
    test_dir.mkdir(exist_ok=True, parents=True)
    os.chdir(test_dir)
    
    # Initialize conversation memory
    memory = ConversationMemory(max_history=50)
    
    # Initialize agent with response format
    console.print("[yellow]Initializing AI Agent...[/yellow]")
    try:
        agent = AIAgent(
            llm=llm,
            tools=my_tools,
            response_format=FileOperationResponse,  # Can change this for different operations
            system_prompt="You are a helpful file system assistant. "
                         "Use the available tools to perform file operations. "
                         "Be precise and confirm operations when needed. "
                         "You can remember previous interactions to provide better assistance.",
            verbose=True,  # Show agent thinking process
            max_iterations=10,
        )
        console.print("[green]✅ Agent initialized successfully![/green]")
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize agent: {e}[/red]")
        return
    
    # Display initial info
    display_agent_info(agent)
    console.print(f"[dim]Memory loaded: {len(memory.get_full_history())} conversation turns[/dim]")
    console.print("\n[italic]Type .h for help, .q to quit[/italic]\n")
    
    # Session variables
    session_stats = {
        "queries": 0,
        "improved_prompts": 0,
        "errors": 0,
        "start_time": datetime.now(),
    }
    use_improved_prompt = True
    use_memory_context = True
    last_response = None
    
    # Run tests if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_basic_operations(agent, memory)
        return
    
    # Main interactive loop
    while True:
        try:
            # Get user input with custom prompt
            prompt_icon = "✨" if use_improved_prompt else "➡️"
            memory_icon = "🧠" if use_memory_context else "📝"
            user_input = console.input(f"[bold cyan]{prompt_icon}{memory_icon} Agent> [/bold cyan]").strip()
            
            # Handle special commands
            if not user_input:
                continue
                
            if user_input == ".q":
                console.print("[yellow]👋 Goodbye![/yellow]")
                # Save memory before exit
                memory.save_memory()
                break
                
            elif user_input == ".h":
                display_help()
                continue
                
            elif user_input == ".t":
                console.print(Panel(agent.get_tool_names(), title="Available Tools", border_style="blue"))
                continue
                
            elif user_input == ".i":
                display_agent_info(agent)
                continue
                
            elif user_input == ".c":
                console.clear()
                display_banner()
                continue
                
            elif user_input == ".m":
                use_improved_prompt = not use_improved_prompt
                status = "ENABLED" if use_improved_prompt else "DISABLED"
                console.print(f"[yellow]Prompt improvement mode: {status}[/yellow]")
                continue
                
            elif user_input == ".mem":
                display_memory(memory)
                continue
                
            elif user_input == ".clearmem":
                memory.clear()
                console.print("[green]✅ Conversation memory cleared.[/green]")
                continue
                
            elif user_input == ".s":
                duration = datetime.now() - session_stats["start_time"]
                table = Table(title="Session Statistics")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                table.add_row("Queries", str(session_stats["queries"]))
                table.add_row("Improved Prompts", str(session_stats["improved_prompts"]))
                table.add_row("Errors", str(session_stats["errors"]))
                table.add_row("Memory Turns", str(len(memory.get_full_history())))
                table.add_row("Duration", str(duration).split(".")[0])
                console.print(table)
                continue
                
            elif user_input == ".p":
                if last_response:
                    console.print(Panel(str(last_response), title="Last Response", border_style="green"))
                else:
                    console.print("[yellow]No previous response to show.[/yellow]")
                continue
            
            # Process user input
            session_stats["queries"] += 1
            console.print(f"[dim]Processing query #{session_stats['queries']}...[/dim]")
            
            # Prepare input with memory context if enabled
            agent_input = user_input
            if use_memory_context:
                context = memory.get_context()
                if context:
                    agent_input = f"Previous conversation context:\n{context}\n\nCurrent request: {user_input}"
            
            # Optionally improve prompt
            if use_improved_prompt and user_input not in [".q", ".h", ".t", ".i", ".c", ".m", ".s", ".p", ".mem", ".clearmem"]:
                try:
                    improved_prompt = agent.improve_prompt(
                        agent_input,
                        context="You are a file system assistant. Use available tools and consider conversation history.",
                        improvement_guidelines={
                            "clarity": True,
                            "specificity": True,
                            "constraints": True,
                        }
                    )
                    if improved_prompt != agent_input:
                        session_stats["improved_prompts"] += 1
                        console.print(f"[dim]Improved prompt: {improved_prompt[:100]}...[/dim]" if len(improved_prompt) > 100 else f"[dim]Improved prompt: {improved_prompt}[/dim]")
                        agent_input = improved_prompt
                except Exception as e:
                    console.print(f"[yellow]⚠️ Prompt improvement failed: {e}[/yellow]")
            
            # Invoke agent
            try:
                with console.status("[bold green]Thinking...", spinner="dots"):
                    result = agent.invoke(user_input=agent_input)
                
                # Store response
                last_response = result
                
                # Add to memory
                memory.add(user_input, str(result))
                
                # Display result
                if isinstance(result, BaseModel):
                    # Pretty print Pydantic models
                    result_dict = result.dict()
                    table = Table(title="✅ Agent Response", show_header=False)
                    for key, value in result_dict.items():
                        table.add_row(key.title(), str(value))
                    console.print(table)
                elif isinstance(result, dict):
                    # Display dictionary nicely
                    console.print(Panel(
                        "\n".join([f"[cyan]{k}:[/cyan] {v}" for k, v in result.items()]),
                        title="Agent Response",
                        border_style="green"
                    ))
                else:
                    # Display string or other types
                    console.print(Panel(str(result), title="Agent Response", border_style="green"))
                    
            except Exception as e:
                session_stats["errors"] += 1
                memory.add(user_input, f"Error: {str(e)}")
                console.print(f"[red]❌ Agent error: {e}[/red]")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]⏸️ Interrupted. Press .q to quit or continue.[/yellow]")
        except EOFError:
            console.print("\n[yellow]👋 Goodbye![/yellow]")
            memory.save_memory()
            break
        except Exception as e:
            session_stats["errors"] += 1
            console.print(f"[red]❌ Unexpected error: {e}[/red]")
    
    # Cleanup
    try:
        os.chdir("..")
        if test_dir.exists():
            # Optional: Clean up test files
            # import shutil
            # shutil.rmtree(test_dir)
            console.print(f"[dim]Test files saved in: {test_dir}[/dim]")
    except:
        pass

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
