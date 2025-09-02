from typing import List, Tuple
from qonnx.transformation.base import Transformation
from onnx import helper
from copy import deepcopy

def decompose_to_swaps_with_last_flag(target: List[int]) -> Tuple[List[List[int]], List[Tuple[int,int]], List[bool]]:
    """
    Greedy left-to-right decomposition of `target` (permutation list) into single transpositions.
    Returns (P_list, transpositions, touches_last_list).
    - P_list: list of full permutation lists P so that new = prev[P].
    - transpositions: list of swapped index pairs (i, j) performed at each step.
    - touches_last_list: list of booleans whether the swap touched last dimension (index n-1).
    """
    n = len(target)
    state = list(range(n))
    P_list = []
    transpositions = []
    touches_last = []

    while state != target:
        for i in range(n):
            if state[i] != target[i]:
                break
        j = state.index(target[i]) 

        state[i], state[j] = state[j], state[i]
        prev = state.copy()
        prev[i], prev[j] = prev[j], prev[i]
        curr = state
        prev_index = {val: idx for idx, val in enumerate(prev)}
        P = [prev_index[val] for val in curr]

        P_list.append(P)
        transpositions.append((i, j))
        touches_last.append((i == n-1) or (j == n-1))

    return P_list, transpositions, touches_last

class TransposeDecomposition(Transformation):
    """
    Transformation that decomposes Transpose nodes into
    a chain of single-swap Transpose nodes. Uses a snapshot of the original node list
    so newly created nodes aren't reprocessed during the same pass.
    """

    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        self._name_counter = 0

    def _unique(self, base):
        self._name_counter += 1
        return f"{base}_{self._name_counter}"

    def get_perm(self, node) -> List[int]:
        for a in node.attribute:
            if a.name == "perm":
                return list(a.ints)
        raise RuntimeError("Unable to determine the permutations from the Transpose node")
    
    def apply(self, model):
        g = model.graph
        original_nodes = list(g.node)

        for node in original_nodes:
            if node.op_type != "Transpose":
                continue

            perm = self.get_perm(node)

            P_list, swaps, flags = decompose_to_swaps_with_last_flag(perm)
            if len(P_list) == 0:
                print("\tNo swaps necessary (identity permutation).")
                continue
            orig_input = list(node.input)
            orig_output = list(node.output)

            if len(orig_input) != 1 or len(orig_output) != 1:
                # Transpose usually has one input and one output; if not, skip replacement
                print(f"\tSkipping node {node.name}: unexpected number of inputs/outputs.")
                continue

            prev_tensor = orig_input[0]
            new_nodes = []

            for step_idx, (P, (i, j), flag) in enumerate(zip(P_list, swaps, flags), start=1):
                step_name = self._unique(f"{node.name}_decomp_step{step_idx}")
                if step_idx < len(P_list):
                    out_tensor = self._unique(f"{node.output[0]}_step{step_idx}")
                else:
                    out_tensor = orig_output[0]

                perm_attr = helper.make_attribute("perm", P)
                transpose_node = helper.make_node(
                    op_type="Transpose",
                    inputs=[prev_tensor],
                    outputs=[out_tensor],
                    name=step_name,
                )
                transpose_node.attribute.extend([perm_attr])
                new_nodes.append(transpose_node)

                prev_tensor = out_tensor

            for nnode in new_nodes:
                g.node.append(nnode)

            try:
                g.node.remove(node)
            except ValueError:
                for idx, gn in enumerate(list(g.node)):
                    if gn.name == node.name:
                        del g.node[idx]
                        break

        return model, False

