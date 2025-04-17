from xdsl.ir import Block
from xdsl.printer import Printer
from io import StringIO

def tuline(text: str) -> str:
    """
    Insert terminal escape characters for underline
    """
    return "\033[4m" + text + "\033[0m"

def tbold(text: str) -> str:
    """
    Insert terminal escape characters for bold
    """
    return "\033[1m" + text + "\033[0m"

def titalic(text: str) -> str:
    """
    Insert terminal escape characters for bold
    """
    return "\033[3m" + text + "\033[0m"

def print_fallback(print_input) -> str:
    """
    Print print_input with xdsl's printer.
    If something fails, it falls back to the default printer, 
    and the pretty assembly stream is discarded.
    """
    try:
        # Create entirely new stream and Printer on purpose
        string_stream = StringIO()
        p = Printer(stream=string_stream, print_generic_format=False)
        p.print(print_input)
        return string_stream.getvalue()
    #something went wrong during printing, fall back to generic printer
    except: 
        # The old stream has to be discarded, create a new one
        g_string_stream = StringIO()
        gp = Printer(stream=g_string_stream, print_generic_format=True)
        gp.print(print_input)
        return g_string_stream.getvalue()

def remove_name_hints(block: Block) -> None:
    """
    Removes name hints in place for block arguments
    """
    #for block_arg in operation.body.block.args:
    for block_arg in block.args:
        block_arg.name_hint = None

