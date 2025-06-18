from rich.console import Console
from rich.theme import Theme

# Define a monochrome theme
monochrome_theme = Theme({
    "info": "dim",
    "warning": "bold",
    "error": "bold",
    "danger": "bold",
    "repr.url": "dim",
    
    # Custom styles for our runner
    "title": "bold",
    "progress.description": "", # default color
    "progress.percentage": "dim",
    "progress.remaining": "dim",
    "spinner": "", # default color
    "bar.back": "dim",
    "bar.complete": "white",
    "bar.finished": "white",
    "table.header": "bold",
    "table.cell": "", # default color
})

# A central console instance to be used throughout the application
console = Console(theme=monochrome_theme, highlight=False) 