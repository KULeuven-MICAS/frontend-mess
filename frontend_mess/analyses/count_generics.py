from frontend_mess.analyses.analysis import ModuleAnalysis
from xdsl.context import Context
from xdsl.dialects import builtin, linalg

class CountGenerics(ModuleAnalysis):
    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        count = 0
        for operation in op.walk():
            if isinstance(operation, linalg.GenericOp):
                count += 1
        print(count)


