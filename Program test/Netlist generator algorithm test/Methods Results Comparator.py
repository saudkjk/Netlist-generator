import os
from collections import defaultdict, Counter
from tkinter import Tk, filedialog
import networkx as nx
import numpy as np


class Component:
    def __init__(self, name, nodes):
        self.name = name
        self.nodes = nodes

    def __repr__(self):
        return f"{self.name} {self.nodes}"


def is_set_match(true_netlist_component, test_component):
    """
    Check if components have the same nodes regardless of order.
    """
    return set(true_netlist_component.nodes) == set(test_component.nodes)


def compare_netlists(true_netlist, test_netlist):
    """
    Compare two netlists to determine if they are equivalent.
    """
    locked_nodes = {}  # Tracks nodes that have been verified as correct
    node_mapping = {}  # Tracks current mapping from test nodes to true netlist nodes
    used_components = set()  # Tracks test components that are already matched

    def backtrack(true_netlist_index, current_mapping, locked_nodes):
        """
        Recursive function to match true netlist components with test netlist components.
        """
        if true_netlist_index >= len(true_netlist):
            # All components have been successfully matched
            return True

        true_netlist_component = true_netlist[true_netlist_index]
        # print(f"\nProcessing true netlist component: {true_netlist_component}")

        # Step 1: Check for set matches
        for test_component in test_netlist:
            if test_component in used_components:
                continue  # Skip already used test components

            # print(f"Checking set match for test component: {test_component}")

            if is_set_match(true_netlist_component, test_component):
                # print(f"Set match found: {true_netlist_component} == {test_component}")
                # Mark as used
                used_components.add(test_component)

                # Add set match to node_mapping
                original_mapping = current_mapping.copy()
                for i in range(len(true_netlist_component.nodes)):
                    true_netlist_node = true_netlist_component.nodes[i]
                    current_mapping[true_netlist_node] = true_netlist_node

                # Recurse to the next component
                if backtrack(true_netlist_index + 1, current_mapping, locked_nodes):
                    return True

                # Revert changes if this path fails
                # print(f"Backtracking from set match: {true_netlist_component} and {test_component}")
                current_mapping = original_mapping
                used_components.remove(test_component)

        # Step 2: Attempt dynamic mapping if no set match is found
        for test_component in test_netlist:
            if test_component in used_components:
                continue  # Skip already used test components

            # print(f"Attempting dynamic mapping for test component: {test_component}")

            if len(test_component.nodes) != len(true_netlist_component.nodes):
                # print(f"Skipping {test_component}: Node count mismatch with {true_netlist_component}")
                continue  # Skip components with mismatched node counts

            # Assume this test component matches (attempt dynamic mapping)
            used_components.add(test_component)
            original_mapping = current_mapping.copy()

            # Try mapping this component's nodes
            valid_mapping = True
            for i in range(len(true_netlist_component.nodes)):
                true_netlist_node = true_netlist_component.nodes[i]
                test_node = test_component.nodes[i]

                if true_netlist_node in locked_nodes:
                    if locked_nodes[true_netlist_node] != test_node:
                        # print(f"Conflict in locked nodes: {true_netlist_node} -> {locked_nodes[true_netlist_node]} vs {test_node}")
                        valid_mapping = False
                        break  # Inconsistent mapping, skip this match

                if true_netlist_node not in current_mapping:
                    current_mapping[true_netlist_node] = test_node

            if valid_mapping:
                # print(f"Dynamic mapping successful: {true_netlist_component} -> {test_component}")
                # Recursive call to match the next component
                if backtrack(true_netlist_index + 1, current_mapping, locked_nodes):
                    return True

            # Revert changes if this path fails
            # print(f"Backtracking from dynamic mapping: {true_netlist_component} and {test_component}")
            current_mapping = original_mapping
            used_components.remove(test_component)

        return False  # No valid mapping found

    # Main comparison loop
    # print("\nStarting netlist comparison...")
    return backtrack(0, node_mapping, locked_nodes)


def read_netlist(file_path):
    """
    Reads a netlist file and returns a list of Component objects.
    """
    components = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            name = parts[0]
            nodes = list(map(int, parts[1:]))
            components.append(Component(name, nodes))
    return components

# Parse a netlist file
def parse_netlist(file_content):
    """
    Parse a netlist file content into components and their connections.
    """
    component_to_nodes = {}
    node_to_components = defaultdict(list)

    for line in file_content.strip().split("\n"):
        parts = line.split()
        component = parts[0]
        connections = list(map(int, parts[1:]))
        component_to_nodes[component] = connections
        for node in connections:
            node_to_components[node].append(component)

    # print(f"Parsed components to nodes: {component_to_nodes}")  # Debug: Components and nodes
    # print(f"Parsed nodes to components: {node_to_components}")  # Debug: Nodes and their associated components

    return component_to_nodes, node_to_components

def process_the_folder(folder_path):
    """Reads all files in a folder and parses them into netlists."""
    netlists = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):  # Ensure it's a file
            netlists[file_name] = read_netlist(file_path)
    return netlists

# Count component types connected to each node
def count_component_types(node_to_components):
    node_component_type_counts = {}
    for node, components in node_to_components.items():
        component_type_counts = Counter(comp.split('_')[0] for comp in components)
        node_component_type_counts[node] = component_type_counts
    return node_component_type_counts

# Process two netlist files and calculate type counts
def process_netlist_files(folder_path, true_netlist_dir, filename):
    generated_file_path = os.path.join(folder_path, filename)
    true_file_path = os.path.join(true_netlist_dir, filename)

    if not os.path.exists(true_file_path):
        print(f"\nSkipping {filename}: True file not found.")
        return None, None
    print(f"\nProcessing {filename}...")
    # Read the netlist files
    with open(true_file_path) as f:
        true_netlist_content = f.read()

    with open(generated_file_path) as f:
        generated_netlist_content = f.read()

    # Parse the netlist files
    _, true_node_to_components = parse_netlist(true_netlist_content)
    _, generated_node_to_components = parse_netlist(generated_netlist_content)

    # Count component types connected to each node
    true_node_type_counts = count_component_types(true_node_to_components)
    generated_node_type_counts = count_component_types(generated_node_to_components)

    return true_node_type_counts, generated_node_type_counts

# Check netlist equivalence and calculate metrics
def check_netlist_equivalence_by_type(true_node_map, generated_node_map, num_files = 0, verbose=True):
    matched_nodes = 0
    false_generated_nodes = 0

    # Calculate matched nodes and false generated nodes
    for true_node, true_counts in true_node_map.items():
        matched = False
        for generated_node, generated_counts in generated_node_map.items():
            if true_counts == generated_counts:
                matched = True
                matched_nodes += 1
                break
        if not matched:
            false_generated_nodes += 1

    # Calculate the remaining metrics
    total_true_nodes = len(true_node_map)
    total_generated_nodes = len(generated_node_map)
    unmatched_actual_nodes = total_true_nodes - matched_nodes
    false_generated_nodes = total_generated_nodes - matched_nodes

    # Print metrics
    if verbose:
        print("\ncircuit ",num_files," Performance Metrics:")
        print(f"Number of actual nodes: {total_true_nodes}")
        print(f"Number of nodes generated by the method: {total_generated_nodes}")
        print(f"Number of nodes that match the actual nodes: {matched_nodes}")
        print(f"Number of actual nodes with no match: {unmatched_actual_nodes}")
        print(f"Number of false generated nodes: {false_generated_nodes}")

    return matched_nodes, total_true_nodes

def parse_input(component_list):
    """
    Parses a list of Component objects and returns a dictionary {component_type: [neighbors]}.
    """
    adjacency_dict = {}

    for component in component_list:
        if not isinstance(component, Component):
            print(f"Error: Expected Component object, got {type(component)}")
            continue

        node_raw = component.name  # Extract component name
        neighbors = component.nodes  # Extract list of connected nodes

        # Ensure it's a list (handles cases where nodes are not in list format)
        if isinstance(neighbors, int):
            neighbors = [neighbors]

        # Extract base component name (ignore number after last '_')
        if "_" in node_raw:
            node = "_".join(node_raw.split("_")[:-1])  # Remove last part after '_'
        else:
            node = node_raw  # If no underscore, keep as is

        # Store in adjacency dictionary
        if node not in adjacency_dict:
            adjacency_dict[node] = []
        adjacency_dict[node].extend(neighbors)

    return adjacency_dict

def check_isomorphism(input1, input2):
    """
    Checks if two netlists are isomorphic.
    :param input1: List of Component objects for Graph 1 (True Netlist)
    :param input2: List of Component objects for Graph 2 (Generated Netlist)
    :return: Boolean (True if isomorphic, False otherwise)
    """
    G1 = nx.Graph()
    G2 = nx.Graph()

    # Parse inputs into adjacency lists
    adj_dict1 = parse_input(input1)
    adj_dict2 = parse_input(input2)

    # Add edges to Graph 1
    for node, neighbors in adj_dict1.items():
        for neighbor in neighbors:
            G1.add_edge(str(node), str(neighbor))

    # Add edges to Graph 2
    for node, neighbors in adj_dict2.items():
        for neighbor in neighbors:
            G2.add_edge(str(node), str(neighbor))

    # Check isomorphism
    return nx.is_isomorphic(G1, G2)

def spectral_similarity_fixed(G1, G2):
    A1 = nx.adjacency_matrix(G1).todense()
    A2 = nx.adjacency_matrix(G2).todense()

    # Compute eigenvalues and convert to numpy arrays
    eig1 = np.sort(np.linalg.eigvals(A1))
    eig2 = np.sort(np.linalg.eigvals(A2))

    # Pad the shorter eigenvalue list with zeros to match the lengths
    max_length = max(len(eig1), len(eig2))
    eig1 = np.pad(eig1, (0, max_length - len(eig1)), mode='constant')
    eig2 = np.pad(eig2, (0, max_length - len(eig2)), mode='constant')

    # Compute Euclidean distance between eigenvalues
    return np.linalg.norm(eig1 - eig2)

def explain_graph_differences(input1, input2):
    """
    Uses only Graph Edit Distance (GED) and Spectral Similarity to explain why two graphs 
    are not isomorphic and what needs to be changed to make them isomorphic.
    """

    G1 = nx.Graph()
    G2 = nx.Graph()

    # Parse inputs into adjacency lists
    adj_dict1 = parse_input(input1)
    adj_dict2 = parse_input(input2)

    # Add edges to Graph 1
    for node, neighbors in adj_dict1.items():
        for neighbor in neighbors:
            G1.add_edge(str(node), str(neighbor))

    # Add edges to Graph 2
    for node, neighbors in adj_dict2.items():
        for neighbor in neighbors:
            G2.add_edge(str(node), str(neighbor))
    
    # Compute Graph Edit Distance
    ged = nx.algorithms.similarity.graph_edit_distance(G1, G2)
    
    # Compute Spectral Similarity
    spectral_sim = spectral_similarity_fixed(G1, G2)

    # Generate an explanation
    explanation = f"**Graph Analysis:**\n"
    explanation += f"- **Graph Edit Distance (GED):** {ged:.1f} changes needed.\n"
    explanation += f"- **Spectral Similarity Score:** {spectral_sim:.2f} (Lower means more similar).\n"

    # Combined Interpretation
    severity_score = (ged / 3) + (spectral_sim / 5)  # Normalize both metrics

    if severity_score < 0.01:
        explanation += "✅ The graphs are already isomorphic! No changes are needed.\n"
    elif severity_score < 0.5:
        explanation += "🔄 **The graphs are very close to being isomorphic!** Only small modifications are needed.\n"
    elif severity_score < 1.5:
        explanation += "⚠️ **The graphs have some structural differences but share a similar shape.** Moderate changes are needed.\n"
    else:
        explanation += "❌ **The graphs are structurally quite different.** Significant modifications are required to make them isomorphic.\n"

    return explanation


# Process a folder and calculate overall metrics
def process_folder(folder_path, true_netlist_dir, verbose=True):
    total_matched_nodes = 0
    total_true_nodes = 0
    total_generated_nodes = 0
    total_unmatched_actual_nodes = 0
    total_false_generated_nodes = 0
    num_files = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            # Process the netlist files
            true_node_type_counts, generated_node_type_counts = process_netlist_files(folder_path, true_netlist_dir, filename)

            if true_node_type_counts is None or generated_node_type_counts is None:
                continue  # Skip files without a corresponding true file

            # Check equivalence and calculate metrics
            matched_nodes, total_nodes = check_netlist_equivalence_by_type(true_node_type_counts, generated_node_type_counts, num_files, verbose)

            # Update overall metrics
            total_matched_nodes += matched_nodes
            total_true_nodes += total_nodes
            total_generated_nodes += len(generated_node_type_counts)
            total_unmatched_actual_nodes += (total_nodes - matched_nodes)
            total_false_generated_nodes += (len(generated_node_type_counts) - matched_nodes)
            num_files += 1

    # Print overall metrics
    if verbose:
        print("\nOverall Metrics Across All Files:")
        print(f"Total number of actual nodes: {total_true_nodes}")
        print(f"Total number of nodes generated by the method: {total_generated_nodes}")
        print(f"Total number of nodes that match the actual nodes: {total_matched_nodes}")
        print(f"Number of actual nodes with no match: {total_unmatched_actual_nodes}")
        print(f"Total number of false generated nodes: {total_false_generated_nodes}")

    return total_matched_nodes, total_true_nodes, total_generated_nodes, num_files

# Function 1: Compare Methods
def compare_methods():
    current_dir = os.getcwd()
    method_1_folder = os.path.join(current_dir, 'Method 1/Test results/')
    method_2_folder = os.path.join(current_dir, 'Method 2/Test results/')
    true_netlist_dir = os.path.join(current_dir, 'True netlists/')

    method_1_metrics = process_folder(method_1_folder, true_netlist_dir, verbose=False)
    method_2_metrics = process_folder(method_2_folder, true_netlist_dir, verbose=False)

    # Extract metrics for accuracy calculation
    method_1_matched_nodes, method_1_total_true_nodes, method_1_total_generated_nodes, _ = method_1_metrics
    method_2_matched_nodes, method_2_total_true_nodes, method_2_total_generated_nodes, _ = method_2_metrics

    # Calculate false nodes for both methods
    method_1_false_nodes = method_1_total_generated_nodes - method_1_matched_nodes
    method_2_false_nodes = method_2_total_generated_nodes - method_2_matched_nodes

    # Calculate accuracy for both methods (with false nodes penalty)
    method_1_accuracy = (method_1_matched_nodes / (method_1_total_true_nodes + method_1_false_nodes)) * 100 if method_1_total_true_nodes > 0 else 0
    method_2_accuracy = (method_2_matched_nodes / (method_2_total_true_nodes + method_2_false_nodes)) * 100 if method_2_total_true_nodes > 0 else 0

    # Display results
    print("\nOverall Performance Metrics for Method 1:")
    print(f"Accuracy: {method_1_accuracy:.2f}%")
    print(f"Total Matched Nodes: {method_1_matched_nodes}")
    print(f"Total Nodes in all True Netlist: {method_1_total_true_nodes}")
    print(f"Total False Nodes Generated: {method_1_false_nodes}")

    print("\nOverall Performance Metrics for Method 2:")
    print(f"Accuracy: {method_2_accuracy:.2f}%")
    print(f"Total Matched Nodes: {method_2_matched_nodes}")
    print(f"Total Nodes in all True Netlist: {method_2_total_true_nodes}")
    print(f"Total False Nodes Generated: {method_2_false_nodes}")

    # Compare and determine the better method
    print("\nComparison:")
    if method_1_accuracy > method_2_accuracy:
        print("Method 1 performed better in terms of accuracy.")
    elif method_1_accuracy < method_2_accuracy:
        print("Method 2 performed better in terms of accuracy.")
    else:
        print("Both methods performed equally in terms of accuracy.")

# Function 2: Test Method on All netlists
def test_method_on_all_netlists(test_netlists_path):
    current_dir = os.getcwd()
    true_netlist_dir = os.path.join(current_dir, 'True netlists/')

    print("Processing Method 2...")
    total_matched_nodes, total_true_nodes, total_generated_nodes, num_files = process_folder(test_netlists_path, true_netlist_dir, False)

    # Calculate false nodes
    total_false_nodes = total_generated_nodes - total_matched_nodes

    # Calculate accuracy with false node penalty
    avg_accuracy = (total_matched_nodes / (total_true_nodes + total_false_nodes)) * 100 if total_true_nodes > 0 else 0

    # Print overall performance metrics
    print("\nOverall Performance Metrics:")
    print(f"Accuracy: {avg_accuracy:.2f}%")
    print(f"Total Matched Nodes: {total_matched_nodes}")
    print(f"Total Nodes in all True Netlists: {total_true_nodes}")
    print(f"Total False Nodes Generated: {total_false_nodes}")
    print(f"Number of Circuits Processed: {num_files}")

# Function 3: Test Method on One netlist
def test_method_on_one_netlist(test_netlists_path):
    # Specify folder paths
    current_dir = os.getcwd()
    true_netlist_dir = os.path.join(current_dir, 'True netlists/')

    # Open a file selection dialog for the user
    print("Please select a file from the Test Results folder.")
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    root.lift()  # Bring the dialog to the front
    root.attributes("-topmost", True)  # Ensure the dialog appears on top

    # Let the user select a file
    selected_file = filedialog.askopenfilename(initialdir=test_netlists_path, title="Select a file",
                                               filetypes=(("Text files", "*.txt"), ("All files", "*.*")))

    # Destroy the root window after use
    root.destroy()

    # If the user cancels the selection, exit the function
    if not selected_file:
        print("No file selected. Exiting...")
        return

    # Extract the filename from the selected file
    filename = os.path.basename(selected_file)
    true_netlist_path = os.path.join(true_netlist_dir, filename)

    # Check if the true file exists
    if not os.path.exists(true_netlist_path):
        print(f"Error: The netlist file for '{filename}' was not found in the True netlists folder.")
        return

    # Process the netlist files
    true_node_type_counts, generated_node_type_counts = process_netlist_files(test_netlists_path, true_netlist_dir, filename)

    if true_node_type_counts is None or generated_node_type_counts is None:
        print(f"Error: Could not process the file '{filename}'.")
        return

    # Calculate metrics
    matched_nodes = 0
    total_generated_nodes = len(generated_node_type_counts)
    total_true_nodes = len(true_node_type_counts)

    # Calculate matched nodes and false nodes
    for true_node, true_counts in true_node_type_counts.items():
        for generated_node, generated_counts in generated_node_type_counts.items():
            if true_counts == generated_counts:
                matched_nodes += 1
                break

    false_nodes_generated = total_generated_nodes - matched_nodes

    # Calculate accuracy with false nodes penalty
    accuracy = (matched_nodes / (total_true_nodes + false_nodes_generated)) * 100 if total_true_nodes > 0 else 0

    # Print results for the specific file
    print(f"\nResults for {filename}:")
    print("Performance Metrics:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Matched Nodes: {matched_nodes}")
    print(f"Total Nodes in all True Netlists: {total_true_nodes}")
    print(f"Total False Nodes Generated: {false_nodes_generated}\n")

# Function 4: Compare all netlists in Method 2 Test results with true netlists
def compare_all_netlists_in_method_2():
    current_dir = os.getcwd()
    method_2_netlist_path = os.path.join(current_dir, 'Method 2/Test results/')
    true_netlist_dir = os.path.join(current_dir, 'True netlists/')
    true_netlist_files = process_the_folder(true_netlist_dir)
    test_netlist_files = process_the_folder(method_2_netlist_path)

    # Compare netlists file by file
    missing_files = set()
    for file_name in test_netlist_files:
        if file_name in true_netlist_files:
            true_netlist = true_netlist_files[file_name]
            test_netlist = test_netlist_files[file_name]
            print(true_netlist)
            print(test_netlist)
            result = compare_netlists(true_netlist, test_netlist)
            print(f"Match Result for {file_name}: {result}\n")
        else:
            missing_files.add(file_name)
    
    for file_name in missing_files:
        print(f"\nTrue netlist file missing for {file_name}\n")

# Function 5: Compare all images using Graph Theory
def compare_all_images_using_graph_theory():
    current_dir = os.getcwd()
    method_2_netlist_path = os.path.join(current_dir, 'Method 2/Test results/')
    true_netlist_dir = os.path.join(current_dir, 'True netlists/')
    true_netlist_files = process_the_folder(true_netlist_dir)
    test_netlist_files = process_the_folder(method_2_netlist_path)

    missing_files = set()
    for file_name in test_netlist_files:
        if file_name in true_netlist_files:
            true_netlist = true_netlist_files[file_name]
            test_netlist = test_netlist_files[file_name]
            print(true_netlist)
            print(test_netlist)
            # Check isomorphism
            isomorphic = check_isomorphism(true_netlist, test_netlist)
            print(f"Comparison for {file_name}:")
            print(f"Isomorphic: {isomorphic}\n")
            if not isomorphic:
                print(explain_graph_differences(true_netlist, test_netlist))
        else:
            missing_files.add(file_name)

    for file_name in missing_files:
        print(f"\nTrue netlist file missing for {file_name}\n")



# Main Function
def main():
    # Function to get the method choice from the user
    def get_method_choice():
        current_dir = os.getcwd()
        method_1_netlists_path = os.path.join(current_dir, 'Method 1/Test results/')
        method_2_netlists_path = os.path.join(current_dir, 'Method 2/Test results/')
        print("Select the method you want to test:")
        print("1. Method 1")
        print("2. Method 2")
        method_choice = input("Enter your choice (1/2): ")
        if method_choice == '1':
            return method_1_netlists_path
        elif method_choice == '2':
            return method_2_netlists_path
        else:
            print("Invalid choice. Exiting...")
            return None

    print("Select an option:")
    print("1. Compare Methods")
    print("2. Test Method on All Netlists")
    print("3. Test Method on One Netlist")
    # print("4. Compare all netlists in Method 2 Test results with true netlists")
    print("5. Compare all images using Graph Theory")
    choice = input("Enter your choice (1/2/3/4/5): ")

    if choice == '1':
        compare_methods()
    elif choice == '2':
        test_method_on_all_netlists(get_method_choice())
    elif choice == '3':
        test_method_on_one_netlist(get_method_choice())
    # elif choice == '4':
    #     compare_all_netlists_in_method_2()
    elif choice == '5':
        compare_all_images_using_graph_theory()
    else:
        print("Invalid choice. Please select 1, 2, 3, 4, or 5.")

if __name__ == "__main__":
    main()
