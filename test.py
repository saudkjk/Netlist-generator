import networkx as nx
import ast

def parse_input(edge_list):
    """
    Parses an input list of strings in the format:
    "Resistor_1, (1, 2)" -> {"Resistor": [1, 2]}
    and returns a dictionary {component_type: [neighbors]}.
    """
    adjacency_dict = {}
    
    for line in edge_list:
        parts = line.split(", ", maxsplit=1)  # Ensure we only split once
        node_raw = parts[0].strip()  # Extract the full component name

        # Extract base component name (ignore number after last '_')
        if "_" in node_raw:
            node = "_".join(node_raw.split("_")[:-1])  # Remove last part after '_'
        else:
            node = node_raw  # If no underscore, keep as is

        try:
            neighbors = ast.literal_eval(parts[1].strip())  # Safely parse tuple/list
        except (SyntaxError, ValueError):
            print(f"Error parsing line: {line}")
            continue

        # Ensure it's a list (handles both single and multiple neighbors)
        if isinstance(neighbors, int):
            neighbors = [neighbors]

        # Store in adjacency dictionary
        if node not in adjacency_dict:
            adjacency_dict[node] = []
        adjacency_dict[node].extend(neighbors)
    
    return adjacency_dict

def check_isomorphism_with_component_types(input1, input2):
    """
    Parses two input graphs and checks if they are isomorphic, ignoring component numbers.
    :param input1: List of edge strings for Graph 1
    :param input2: List of edge strings for Graph 2
    :return: True if the graphs are structurally identical AND have the same component types.
    """
    G1 = nx.Graph()
    G2 = nx.Graph()

    # Parse inputs, ignoring component numbers
    adj_dict1 = parse_input(input1)
    adj_dict2 = parse_input(input2)

    # Ensure component types match (ignoring numbering)
    if set(adj_dict1.keys()) != set(adj_dict2.keys()):
        return False  # Component types must be identical

    # Add edges to Graph 1
    for node, neighbors in adj_dict1.items():
        for neighbor in neighbors:
            G1.add_edge(node, neighbor)

    # Add edges to Graph 2
    for node, neighbors in adj_dict2.items():
        for neighbor in neighbors:
            G2.add_edge(node, neighbor)

    # Check isomorphism while enforcing identical component types
    return nx.is_isomorphic(G1, G2)

# ✅ These should be considered isomorphic (same structure & same component types)
graph1 = [
    "Resistor_1, (1, 2)",
    "Resistor_2, (2, 3)",
    "Capacitor_1, (3, 4)",
    "Capacitor_2, (1, 4)"
]

graph2_correct = [
    "Resistor_2, (11, 12)",
    "Resistor_1, (12, 13)",
    "Capacitor_1, (13, 14)",
    "Capacitor_2, (11, 14)"
]

# ❌ These should NOT be considered isomorphic (different component types)
graph2_wrong = [
    "Resistor_2, (11, 12)",
    "Capacitor_1, (12, 13)",  # Changed from Resistor to Capacitor (should not be isomorphic)
    "Capacitor_1, (13, 14)",
    "Capacitor_2, (11, 14)"
]

# Run isomorphism checks
result_correct = check_isomorphism_with_component_types(graph1, graph2_correct)  # Should be True
result_wrong = check_isomorphism_with_component_types(graph1, graph2_wrong)  # Should be False

# Print results
print(f"Graph 1 and Graph 2 (correct) are isomorphic: {result_correct}")
print(f"Graph 1 and Graph 2 (wrong) are isomorphic: {result_wrong}")
