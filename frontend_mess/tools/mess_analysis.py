import argparse
from typing import Sequence, Callable
from xdsl.context import Context
from xdsl.tools.command_line_tool import CommandLineTool
from xdsl.dialects.builtin import ModuleOp
from frontend_mess.analyses import get_all_analyses
from frontend_mess.analyses.analysis import ModuleAnalysis


class MessAnalysisMain(CommandLineTool):

    def __init__(
        self,
        description: str = "Minimal xdsl-based analysis tool",
        args: Sequence[str] | None = None,
    ):
        self.available_frontends = {}
        self.available_targets = {}
        self.available_analyses = {}

        self.ctx = Context()
        self.register_all_dialects()
        self.register_all_frontends()
        self.register_all_analyses()

        # arg handling
        arg_parser = argparse.ArgumentParser(description=description)
        self.register_all_arguments(arg_parser)
        arg_parser.add_argument("analysis", type=str, choices=self.available_analyses.keys())
        self.args = arg_parser.parse_args(args=args)
        self.analysis = self.available_analyses[self.args.analysis]()
        self.ctx.allow_unregistered = self.args.allow_unregistered_dialect


    def run(self):
        """
        Executes an analysis.
        """
        chunk, file_extension = self.get_input_stream()
        try:
            module = self.parse_chunk(chunk, file_extension)
            if module is not None:
                self.apply_analysis(module)
        finally:
            chunk.close()

    def apply_analysis(self, prog: ModuleOp) -> bool:
        self.analysis().apply(self.ctx, prog)
        return True

    def register_analysis(
        self, pass_name: str, analysis_factory: Callable[[], type[ModuleAnalysis]]
    ):
        self.available_analyses[pass_name] = analysis_factory

    def register_all_analyses(self):
        """
        Register all passes that can be used.

        Add other/additional passes by overloading this function.
        """
        for analysis_name, analysis_factory in get_all_analyses().items():
            self.register_analysis(analysis_name, analysis_factory)

def main():
    return MessAnalysisMain().run()

if __name__ == "__main__":
    main()
