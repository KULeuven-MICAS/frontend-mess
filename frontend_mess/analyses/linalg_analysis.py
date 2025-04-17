from collections import Counter
from frontend_mess.analyses.analysis import ModuleAnalysis
from frontend_mess.utils.printing import print_fallback, remove_name_hints
from xdsl.context import Context
from xdsl.dialects import builtin, linalg

class LinalgAnalysis(ModuleAnalysis):
    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        """
        This is a simple analysis that tries to see how similar linalg
        generics are by looking at how their strings are printed.
        """
        amount_of_generics = 0
        map_keeper : list[str] = [] 
        iterator_keeper : list[str] = [] 
        types_keeper : list[str] = [] 
        element_types_keeper : list[str] = [] 
        body_keeper : list[str] = []
        body_ops_keeper : list[str] = []
        for operation in op.walk():
            if isinstance(operation, linalg.GenericOp):
                # Count number of linalgs
                amount_of_generics += 1

                # Keep linalg generic indexing maps

                map_keeper.append(str(operation.indexing_maps))

                # Keep linalg generic iterators
                
                iterator_keeper.append(str(operation.iterator_types))

                # Keep linalg generic operand types 

                types_keeper.extend([str(oper.type) for oper in operation.operands])

                # Keep linalg generic element types

                for operand in operation.operands:
                    if isinstance(operand.type, builtin.TensorType):
                        element_types_keeper.append(str(operand.type.element_type))
                    else:
                        element_types_keeper.append(str(operand.type))

                # Compare linalg generic body's (full body)

                # Remove SSAValue name hints to allow similarity check
                remove_name_hints(operation.body.block)
                body_keeper.append(print_fallback(operation.body.block))

                # Compare linalg generic body's (individual ops)

                for body_op in operation.body.block.ops:
                    body_ops_keeper.append(print_fallback(body_op))

                # Count all unique occurences with str's __hash__ and store counts
        keepers = [map_keeper, iterator_keeper, types_keeper, 
                   element_types_keeper, body_keeper, body_ops_keeper]
        counters = [Counter(keeper) for keeper in keepers]
        names = ["maps", "iterators", "operand types","element types","bodies", "body ops"]
        unique_lens = []
        for counter,name in zip(counters,names):
            print("="*19)
            print(name)
            print("="*19)
            for key, val in sorted(counter.items(), key=lambda x : x[1], reverse=True):
                print(f"\033[1m\033[4m{val:^3}\033[0m → {key}")
            unique_lens.append(len(counter))
        print("="*19)
        print(" Analysis Overview ")
        print("="*19)
        print(f"Linalg generics      : {amount_of_generics}")
        for name, number in zip(names, unique_lens, strict=True):
            print(f"Unique {name:13} : {number}")

