from typing import List, Tuple, Optional
from collections import deque
from qonnx.transformation.base import Transformation
from onnx import helper, TensorProto
from copy import deepcopy
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes

def apply_pT_operation(perm: List[int]) -> List[int]:
    """
    Apply pT operation: swap the last two positions
    (..., a, b) -> (..., b, a)

    Also referred to as a parallel Transpose
    """
    if len(perm) < 2:
        return perm[:]
    
    result = perm[:]
    result[-2], result[-1] = result[-1], result[-2]
    return result


def apply_iG_operation(perm: List[int], i: int, j: int) -> Optional[List[int]]:
    """
    Apply iG operation: swap positions i and j
    Constraint: cannot move the very last dimension

    Also referred to as a input_generator
    """
    n = len(perm)
    if n < 2:
        return None
    
    # Check constraints for iG operation - cannot move the very last dimension
    if i == n - 1 or j == n - 1:
        return None
    
    if i == j or i < 0 or j < 0 or i >= n or j >= n:
        return None
    
    result = perm[:]
    result[i], result[j] = result[j], result[i]
    return result


def get_all_possible_moves(perm: List[int]) -> List[Tuple[List[int], str, Optional[Tuple[int, int]]]]:
    """
    Get all possible moves from current permutation.
    Returns list of (new_permutation, operation_type, operation_params) tuples.
    operation_type is either 'pT' or 'iG'
    operation_params is None for pT, (i, j) for iG
    """
    moves = []
    n = len(perm)
    
    # Try pT operation
    new_perm = apply_pT_operation(perm)
    if new_perm != perm:
        moves.append((new_perm, "pT", None))
    
    # Try all valid iG operations
    for i in range(n):
        for j in range(i + 1, n):
            new_perm = apply_iG_operation(perm, i, j)
            if new_perm is not None and new_perm != perm:
                moves.append((new_perm, "iG", (i, j)))
    
    return moves


def is_valid_hardware_permutation(perm_array: List[int]) -> bool:
    """
    Check if a permutation array represents a valid hardware operation.
    Valid operations are:
    - pT: swap last two elements  
    - iG: swap any two elements except the last two
    """
    n = len(perm_array)
    if n < 2:
        return True
        
    identity = list(range(n))
    if perm_array == identity:
        return True  # Identity is always valid
    
    expected_pT = identity[:]
    expected_pT[-2], expected_pT[-1] = expected_pT[-1], expected_pT[-2] 
    if perm_array == expected_pT:
        return True
    
    diff_count = sum(1 for i in range(n) if perm_array[i] != identity[i])
    if diff_count == 2:  # Exactly one swap
        diff_positions = [i for i in range(n) if perm_array[i] != identity[i]]
        if len(diff_positions) == 2:
            pos1, pos2 = diff_positions
            if pos1 != n-1 and pos2 != n-1:
                if perm_array[pos1] == pos2 and perm_array[pos2] == pos1:
                    return True
    
    return False


def find_minimal_operation_sequence(start_perm: List[int], target_perm: List[int]) -> Optional[List[Tuple[str, Optional[Tuple[int, int]]]]]:
    """
    Find minimal sequence of operations to transform start_perm into target_perm.
    Uses BFS to find shortest path, ensuring all intermediate permutations are hardware-valid.
    Returns list of (operation_type, operation_params) tuples.


    TODO: We want this to be cost based and include a buffer size cost model.
    """
    if start_perm == target_perm:
        return []
    
    queue = deque([(start_perm, [])])
    visited = {tuple(start_perm)}
    
    while queue:
        current_perm, operations = queue.popleft()
        
        for next_perm, op_type, op_params in get_all_possible_moves(current_perm):
            test_operations = operations + [(op_type, op_params)]
            test_perms = convert_operations_to_permutations(list(range(len(start_perm))), test_operations)
            
            if not is_valid_hardware_permutation(test_perms[-1]):
                continue  
            
            if next_perm == target_perm:
                return test_operations
            
            next_tuple = tuple(next_perm)
            if next_tuple not in visited:
                visited.add(next_tuple)
                queue.append((next_perm, test_operations))
    
    return None


def convert_operations_to_permutations(start_perm: List[int], operations: List[Tuple[str, Optional[Tuple[int, int]]]]) -> List[List[int]]:
    """
    Convert a sequence of operations to a list of permutation arrays.
    Each permutation represents the transformation for that step.
    """
    current_perm = start_perm[:]
    permutations = []
    
    for op_type, op_params in operations:
        if op_type == "pT":
            new_perm = apply_pT_operation(current_perm)
        elif op_type == "iG" and op_params is not None:
            i, j = op_params
            new_perm = apply_iG_operation(current_perm, i, j)
            if new_perm is None:
                raise RuntimeError(f"Invalid iG operation: ({i}, {j})")
        else:
            raise RuntimeError(f"Unknown operation: {op_type}")
        
        perm_array = [0] * len(current_perm)
        for new_idx, val in enumerate(new_perm):
            old_idx = current_perm.index(val)
            perm_array[new_idx] = old_idx
        
        permutations.append(perm_array)
        current_perm = new_perm
    
    return permutations


def can_be_single_operation(target_perm: List[int]) -> Optional[Tuple[str, Optional[Tuple[int, int]]]]:
    """
    Check if the target permutation can be achieved with a single operation.
    i.e. no decomposition is required.
    Returns (operation_type, operation_params) or None if not possible.
    """
    n = len(target_perm)
    start_perm = list(range(n))
    
    if target_perm == start_perm:
        return None  
    
    # Check if it's a simple pT operation (swap last two)
    if n >= 2:
        expected_pT = apply_pT_operation(start_perm)
        if target_perm == expected_pT:
            return ("pT", None)
    
    # Check if it's a simple iG operation (single swap of any valid positions)
    for i in range(n):
        for j in range(i + 1, n):
            expected_iG = apply_iG_operation(start_perm, i, j)
            if expected_iG is not None and target_perm == expected_iG:
                return ("iG", (i, j))
    
    return None


def decompose_transpose_with_constraints(target_perm: List[int]) -> Tuple[List[List[int]], List[str]]:
    """
    Decompose a target permutation into a sequence of hardware-constrained operations.
    Returns (permutations, operation_types).
    - permutations: list of permutation arrays for each step
    - operation_types: list of operation types ('pT' or 'iG') for each step
    """
    n = len(target_perm)
    start_perm = list(range(n))
    
    if target_perm == start_perm:
        return [], []  # Identity permutation
    
    # First check if this can be done with a single operation
    single_op = can_be_single_operation(target_perm)
    if single_op is not None:
        op_type, op_params = single_op
        # Create the permutation array for this single operation
        permutations = convert_operations_to_permutations(start_perm, [single_op])
        return permutations, [op_type]
    
    # If not a single operation, find minimal sequence using BFS
    operations = find_minimal_operation_sequence(start_perm, target_perm)
    
    if operations is None:
        raise RuntimeError(f"No solution found for permutation: {target_perm}")
    
    if len(operations) == 0:
        return [], []  # Identity permutation
    
    # Convert operations to permutation arrays
    permutations = convert_operations_to_permutations(start_perm, operations)
    operation_types = [op[0] for op in operations]
    
    return permutations, operation_types

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

            try:
                P_list, operation_types = decompose_transpose_with_constraints(perm)
                if len(P_list) == 0:
                    print("\tNo swaps necessary (identity permutation).")
                    continue
            except RuntimeError as e:
                print(f"\tSkipping node {node.name}: {e}")
                continue
            orig_input = list(node.input)
            orig_output = list(node.output)

            if len(orig_input) != 1 or len(orig_output) != 1:
                # Transpose usually has one input and one output; if not, skip replacement
                # TODO: Should this raise an exception? Probably need to be handled.
                print(f"\tSkipping node {node.name}: unexpected number of inputs/outputs.")
                continue

            prev_tensor = orig_input[0]
            new_nodes = []
            
            # Create decomposed transposes using hardware-constrained operations
            for step_idx, (P, op_type) in enumerate(zip(P_list, operation_types), start=1):
                step_name = self._unique(f"{node.name}_{op_type}_step{step_idx}")
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

        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        return model, False

