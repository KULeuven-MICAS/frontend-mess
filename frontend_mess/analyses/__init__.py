from typing import Callable
from xdsl.dialects.builtin import ModuleOp
from frontend_mess.analyses.analysis import ModuleAnalysis


def get_count_generics():
    from frontend_mess.analyses.count_generics import CountGenerics
    return CountGenerics

def get_linalg_analysis():
    from frontend_mess.analyses.linalg_analysis import LinalgAnalysis
    return LinalgAnalysis

def get_iterator_first():
    from frontend_mess.analyses.iterator_first import IteratorFirst
    return IteratorFirst

def get_all_analyses() -> dict[str, Callable[[], type[ModuleAnalysis]]]:
    return {"count_generics": get_count_generics,
            "iterator_first": get_iterator_first,
            "linalg_analysis": get_linalg_analysis}
