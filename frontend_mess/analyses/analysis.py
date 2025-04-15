from abc import ABC, abstractmethod
from xdsl.dialects import builtin
from xdsl.context import Context

class ModuleAnalysis(ABC):

    @abstractmethod
    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None: ...
