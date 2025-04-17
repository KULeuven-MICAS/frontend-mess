from frontend_mess.analyses.analysis import ModuleAnalysis
from frontend_mess.utils.printing import remove_name_hints, print_fallback, tuline, tbold, titalic
from xdsl.context import Context
from xdsl.ir import Block
from xdsl.dialects import builtin, linalg
from collections import defaultdict
from pprint import pprint
from graphviz import Digraph


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

        dot = Digraph()
        dot.attr(fontname='SF Mono')
        dot.node_attr.update(fontname='SF Mono')
        dot.edge_attr.update(fontname='SF Mono')


        for operation in op.walk():
            if isinstance(operation, linalg.GenericOp):
                remove_name_hints(operation.body.block)
                visualize_bodies(operation.body.block, dot)
                body : str = print_fallback(operation.body.block)
                iterator : str = str(operation.iterator_types)
                # Count all occurences
                iterator_count[iterator] += 1
                body_count[iterator+body] += 1
                # Add body if not present yet
                if body not in iterator_to_body[iterator]:
                    iterator_to_body[iterator].append(body)
                count=1

        for num, (iterator,bodies) in enumerate(sorted(iterator_to_body.items(), key=lambda x : iterator_count[x[0]], reverse=True)):
            print(tbold(f"It{num:^3} (uses: {iterator_count[iterator]}) → {len(bodies)} body types  ⬎ \n") + tuline(iterator))
            for num, body in enumerate(sorted(bodies, key=lambda x : body_count[iterator+x], reverse=True)):
                print(tbold(titalic(f"body {num} (uses: {body_count[iterator+body]}):")))
                print(body)

        print(dot.source)
        dot.render(directory='.').replace('\\', '/')


def visualize_bodies(block: Block, graph: Digraph):
    cluster_name = f'cluster_{id(block)}'
    with graph.subgraph(name=cluster_name) as sub:
        yield_op_id : int = 0
        for arg in block.args:
            if len(arg.uses) == 0:
                continue
            sub.node(str(id(arg)),f"%{arg.index}")
            for use in arg.uses:
                sub.edge(str(id(arg)),str(id(use.operation)))
        for op in block.ops:
            match op.name.split(".")[0]:
                case "linalg":
                    color = "lightblue"
                case "arith":
                    color = "lightgreen"
                case _:
                    color = "lightcoral"
            sub.node(str(id(op)),str(op.name),shape='box', color=color ,style='rounded,filled')
            for result in op.results:
                for use in result.uses:
                    sub.edge(str(id(op)),str(id(use.operation)))
            if "linalg.yield" in op.name:
                yield_op_id = id(op)
        # FIXME: Not sure this is always okay for final op
        final_id = f"{cluster_name}_final_op"
        sub.node(final_id,f"%{len(block.args)-1}")
        sub.edge(str(yield_op_id),final_id)
