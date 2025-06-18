import logging
from rich.logging import RichHandler
from affine.theme import console # Import the themed console

class AppLogHandler(RichHandler):
    """A rich log handler using the application's central theme."""
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            console=console, # Use the central themed console
            show_path=False,
            show_level=True,
            show_time=False,
            rich_tracebacks=True,
            # Let rich use its default, which is better for monochrome
            markup=True
        )
        # Use a simple formatter, letting Rich handle the styling
        self.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]")) 