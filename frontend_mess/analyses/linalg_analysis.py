from collections import Counter
from frontend_mess.analyses.analysis import ModuleAnalysis
from xdsl.context import Context
from xdsl.dialects import builtin, linalg
from pprint import pprint
from xdsl.printer import Printer
from io import StringIO

class LinalgAnalysis(ModuleAnalysis):
    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        """
        This is a simple analysis that tries to see how similar linalg
        generics are by looking at how their strings are printed.
        """
        map_keeper : list[str] = [] 
        iterator_keeper : list[str] = [] 
        body_keeper : list[str] = []
        body_ops_keeper : list[str] = []
        for operation in op.walk():
            if isinstance(operation, linalg.GenericOp):
                # Keep linalg generic indexing maps

                map_keeper.append(str(operation.indexing_maps))

                # Keep linalg generic iterators
                
                iterator_keeper.append(str(operation.iterator_types))

                # Compare linalg generic body's (full body)

                # Create entirely new stream and Printer
                string_stream = StringIO()
                p = Printer(stream=string_stream, print_generic_format=True)
                # Remove SSAValue name hints to allow similarity check
                for block_arg in operation.body.block.args:
                    block_arg.name_hint = None
                p.print(operation.body.block)
                body_keeper.append(string_stream.getvalue())


                # Compare linalg generic body's (individual ops)

                for body_op in operation.body.block.ops:
                    # Get another fresh printer
                    string_stream = StringIO()
                    p = Printer(stream=string_stream, print_generic_format=True)
                    p.print(body_op)
                    body_ops_keeper.append(string_stream.getvalue())

                # Count all unique occurences with str's __hash__ and store counts

        pprint(Counter(map_keeper))
        pprint(Counter(iterator_keeper))
        pprint(Counter(body_keeper))
        pprint(Counter(body_ops_keeper))



