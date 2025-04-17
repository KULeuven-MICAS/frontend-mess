from frontend_mess.analyses.analysis import ModuleAnalysis
from frontend_mess.utils.printing import remove_name_hints, print_fallback, tuline, tbold, titalic
from xdsl.context import Context
from xdsl.dialects import builtin, linalg
from collections import defaultdict
from pprint import pprint


class IteratorFirst(ModuleAnalysis):
    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        """
        This analysis tries to piece together different bodies of 
        Linalg Generics that have the same iterator types.
        """
        # Map iterator to bodies
        iterator_to_body : dict[str,list[str]] = defaultdict(list)
        # Count occurences
        iterator_count : dict[str,int] = defaultdict(int)
        body_count: dict[str, int] = defaultdict(int)

        for operation in op.walk():
            if isinstance(operation, linalg.GenericOp):
                remove_name_hints(operation.body.block)
                body : str = print_fallback(operation.body.block)
                iterator : str = str(operation.iterator_types)
                # Count all occurences
                iterator_count[iterator] += 1
                body_count[iterator+body] += 1
                # Add body if not present yet
                if body not in iterator_to_body[iterator]:
                    iterator_to_body[iterator].append(body)

        for num, (iterator,bodies) in enumerate(sorted(iterator_to_body.items(), key=lambda x : iterator_count[x[0]], reverse=True)):
            print(tbold(f"It{num:^3} (uses: {iterator_count[iterator]}) → {len(bodies)} body types  ⬎ \n") + tuline(iterator))
            for num, body in enumerate(sorted(bodies, key=lambda x : body_count[iterator+x], reverse=True)):
                print(tbold(titalic(f"body {num} (uses: {body_count[iterator+body]}):")))
                print(body)


