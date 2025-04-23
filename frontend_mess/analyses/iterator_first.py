from frontend_mess.analyses.analysis import ModuleAnalysis
from frontend_mess.utils.printing import remove_name_hints, print_fallback, tuline, tbold, titalic
from xdsl.context import Context
from xdsl.ir import Block
from xdsl.dialects import builtin, linalg, arith
from collections import defaultdict
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
                body : str = print_fallback(operation.body.block)
                iterator : str = str(operation.iterator_types)
                # Count all occurences
                iterator_count[iterator] += 1
                body_count[iterator+body] += 1
                # Add body and visualize if not present yet
                if body not in iterator_to_body[iterator]:
                    iterator_to_body[iterator].append(body)
                    visualize_bodies(operation, dot)

        for num, (iterator,bodies) in enumerate(sorted(iterator_to_body.items(), key=lambda x : iterator_count[x[0]], reverse=True)):
            print(tbold(f"It{num:^3} (uses: {iterator_count[iterator]}) → {len(bodies)} body types  ⬎ \n") + tuline(iterator))
            for num, body in enumerate(sorted(bodies, key=lambda x : body_count[iterator+x], reverse=True)):
                print(tbold(titalic(f"body {num} (uses: {body_count[iterator+body]}):")))
                print(body)

        #print(dot.source)
        dot.render(directory='.', format="pdf").replace('\\', '/')

def get_element_type_fallback(inpt):
    if isinstance(inpt.type, builtin.TensorType):
        element_type = inpt.type.element_type
    else:
        element_type = inpt.type
    return element_type


def visualize_bodies(operation: linalg.GenericOp, graph: Digraph):
    block = operation.body.block
    cluster_name = f'cluster_{id(block)}'
    round_box_fill = {'shape':'box', 'style':'rounded,filled'}
    round_box = {'shape':'box', 'style':'rounded'}
    all_gray = {'color':"gray34", 'fontcolor':"gray34"}
    with graph.subgraph(name=cluster_name) as sub:
        yield_op : Operation | None = None
        for op in block.ops:
            # Keep track of the linalg.yield op (there can only be one!)
            if "linalg.yield" in op.name:
                yield_op = op
        assert yield_op is not None
        no_generic_results = len(yield_op.operands)
        # In linalg on tensors, both linalg ins and outs are operands
        generic_inputs = operation.operands[:-no_generic_results]
        block_inputs = operation.body.block.args[:-no_generic_results]
        generic_outputs = operation.operands[-no_generic_results:]
        block_outputs = operation.body.block.args[-no_generic_results:]
        for index, inpt in enumerate(generic_inputs):
            element_type = get_element_type_fallback(inpt)
            # operands can actually be used in multiple places, so create unique id.
            node_id = str(id(inpt))+cluster_name
            sub.node(node_id,f"in{index}:{element_type}", **round_box_fill)
            sub.edge(node_id, str(id(block_inputs[index])))

        for index, outpt in enumerate(generic_outputs):
            element_type = get_element_type_fallback(outpt)
            # operands can actually be used in multiple places, so create unique id.
            node_id = str(id(outpt))+cluster_name
            sub.node(node_id,f"out{index}:{element_type}", **round_box_fill)
            sub.edge(str(id(block_outputs[index])),node_id)

        for index in range(no_generic_results):
            sub.edge(str(id(yield_op)),str(id(block_outputs[index])))


        for arg in block.args:
            sub.node(str(id(arg)),f"%{arg.index}:{arg.type}", **all_gray , **round_box)
            for use in arg.uses:
                sub.edge(str(id(arg)),str(id(use.operation)))
        # Add nodes for all block arguments
        ext_count, cst_count = (0,0)
        for op in block.ops:
            # Choose a color, based on dialect name
            match op.name.split(".")[0]:
                case "linalg":
                    color = "lightblue"
                case "arith":
                    color = "lightgreen"
                case _:
                    color = "lightcoral"
            if len(op.result_types) > 0:
                sub.node(str(id(op)),f"{op.name}:{op.result_types[0]}",color=color, **round_box_fill)
            else:
                sub.node(str(id(op)),f"{op.name}",color=color, **round_box_fill)
            for result in op.results:
                if "linalg.yield" in op.name:
                    continue
                for use in result.uses:
                    sub.edge(str(id(op)),str(id(use.operation)))
            # Add extra nodes for the SSAValues coming outside of the block
            for operand in op.operands:
                if operand.owner.parent_region() != operation.regions[0]:
                    # Add unique id based on this op
                    if isinstance(operand.owner, arith.ConstantOp):
                        color="lavender"
                        name = f"cst{cst_count}\n{operand.owner.properties['value']}"
                        node_id = str(id(op))+f"cst{cst_count}"
                        cst_count += 1
                    else:
                        color = "lightcoral"
                        name = f"ext{ext_count}:{operand.type}"
                        node_id = str(id(op))+f"ext{ext_count}"
                        ext_count += 1
                    sub.node(node_id, name, color=color, **round_box_fill)
                    sub.edge(node_id, str(id(op)))
