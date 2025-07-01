import json
import os
import random
import re
import string
import sys
from collections import OrderedDict
from functools import wraps
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

import libsbml
import libsedml
import numpy as np
import pandas as pd

from scigym.api import (
    ExperimentAction,
    ExperimentConstraint,
    ModifyReactionAction,
    ModifySpeciesAction,
    NullifyReactionAction,
    NullifySpeciesAction,
    ParseExperimentActionError,
)

T = TypeVar("T")


def find_latest_timestamp_folder(directory_path):
    """
    Find the latest created folder in a directory based on timestamp folder names.
    Expects folder names in format YYYYMMDD_HHMMSS (e.g., 20250508_084713)

    Args:
        directory_path (str): Path to the directory containing timestamp folders

    Returns:
        str: Name of the latest timestamp folder, or None if no valid folders found
    """
    # Get all subdirectories
    subdirs = [
        d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))
    ]

    # Filter for timestamp-formatted folders (YYYYMMDD_HHMMSS)
    timestamp_dirs = [d for d in subdirs if re.match(r"\d{8}_\d{6}", d)]

    if not timestamp_dirs:
        return None

    # Sort by the actual timestamp in the folder name (not creation time)
    # Later timestamps will come last in the sorted list
    timestamp_dirs.sort(reverse=True)  # Sort in descending order for latest first

    # Return the latest folder
    return timestamp_dirs[0]


def compute_smape(a, b):
    """
    Compute the Symmetric Mean Absolute Percentage Error (SMAPE) between two 2D NumPy arrays.

    Parameters:
    -----------
    a, b : numpy.ndarray
        Two 2D arrays of the same shape.

    Returns:
    --------
    float
        The SMAPE value.
    """
    # Ensure the arrays are of the same shape
    if a.shape != b.shape:
        raise ValueError("Arrays must have the same shape")

    # Calculate the absolute difference and sum
    numerator = np.abs(a - b)
    denominator = np.abs(a) + np.abs(b)

    # Handle division by zero - when both a and b are zero
    zero_indices = denominator == 0

    # Calculate SMAPE for each element
    smape_values = np.zeros_like(a, dtype=float)
    non_zero_indices = ~zero_indices
    smape_values[non_zero_indices] = numerator[non_zero_indices] / denominator[non_zero_indices]
    # For elements where both a and b are zero, smape is already set to 0

    # Average over all elements
    total_smape = np.sum(smape_values) / a.size

    return total_smape


def perturb_concentration_proportional(concentration, relative_noise, min_conc=1e-10):
    """
    Add noise that is proportional to concentration, with special handling for zeros

    Parameters:
    concentration (float): Original concentration value(s)
    relative_noise (float): Relative magnitude of noise (as a fraction of concentration)
    min_conc (float): Minimum concentration to consider for zero values

    Returns:
    float or array: Perturbed concentration
    """
    # Generate noise proportional to the concentration
    cap_concentration = max(concentration, min_conc)
    noise = np.random.normal(0, relative_noise * cap_concentration)

    # Add noise to original concentration
    perturbed = concentration + noise

    # Ensure non-negative
    perturbed = np.maximum(perturbed, 0)

    return perturbed


def print_chat_history_with_tokens(chat_history, output_file="chat_history.txt"):
    """
    Prints chat history to a file using special tokens to mark user and assistant content.

    Args:
        chat_history: Dictionary containing conversation history
        output_file: Path to the output file
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for key in chat_history:
            # Print the iteration identifier

            # Print user message with tokens
            print(chat_history[key]["user"], file=f)

            # Print assistant message with tokens
            print(chat_history[key]["assistant"], file=f)

    print(f"Chat history saved to {output_file}")


class SkipNonSerializableEncoder(json.JSONEncoder):
    def default(self, o):
        try:
            return super().default(o)
        except TypeError:
            # Skip this field by returning None, which is JSON serializable
            return None


def read_file(file_path):
    """Read a file and return its contents as a string."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)


def replace_section(content, section_name, replacement):
    """
    Replace the content between the BEGIN and END markers for a section with the replacement text.
    If remove_markers is True, also remove the markers themselves.

    Args:
        content (str): The full content of the template file
        section_name (str): Name of the section to replace (e.g., "EXPERIMENTAL_ACTIONS")
        replacement (str): New content to insert into the section

    Returns:
        str: Updated content with the section replaced
    """
    begin_marker = f"<!-- BEGIN {section_name} -->"
    end_marker = f"<!-- END {section_name} -->"

    # Check if markers exist
    if begin_marker not in content or end_marker not in content:
        print(f"Warning: Could not find markers for section {section_name}")
        return content

    # Create pattern to match content between markers including the markers
    pattern = re.compile(f"{re.escape(begin_marker)}(.*?){re.escape(end_marker)}", re.DOTALL)

    # Replace the content between the markers without preserving the markers
    updated_content = pattern.sub(replacement.strip(), content)

    return updated_content


def update_markdown(template_path, **section_files):
    """
    Update a template markdown file by replacing sections with content from other files.

    Args:
        template_path (str): Path to the template markdown file
        output_path (str): Path for the output markdown file
        **section_files: Keyword arguments where keys are section names and values are file paths
    """
    # Read the template file
    template_content = read_file(template_path)
    updated_content = template_content

    # Replace each section if a file is provided
    for section_name, file_path in section_files.items():
        if file_path:
            section_content = read_file(file_path)
            updated_content = replace_section(updated_content, section_name, section_content)

    # Write the updated content to the output file
    return updated_content


def calculate_stats_dict(dict_list):
    """
    Calculate mean and max for each field in a list of dictionaries.

    Args:
        dict_list: A list of dictionaries, each containing 'pred_mse', 'inco_mse',
                  and 'normalized_mse' keys

    Returns:
        A dictionary with mean and max values for each field (6 keys total)
    """
    if not dict_list:
        return {}

    # Initialize variables to store sums
    sum_pred_mse = 0
    sum_inco_mse = 0
    sum_normalized_mse = 0

    # Initialize variables to store max values
    max_pred_mse = float("-inf")
    max_inco_mse = float("-inf")
    max_normalized_mse = float("-inf")

    # Calculate sums and find max values
    for d in dict_list:
        # Update sums
        sum_pred_mse += d["pred_mse"]
        sum_inco_mse += d["inco_mse"]
        sum_normalized_mse += d["normalized_mse"]

        # Update max values
        max_pred_mse = max(max_pred_mse, d["pred_mse"])
        max_inco_mse = max(max_inco_mse, d["inco_mse"])
        max_normalized_mse = max(max_normalized_mse, d["normalized_mse"])

    # Calculate means
    n = len(dict_list)
    mean_pred_mse = sum_pred_mse / n
    mean_inco_mse = sum_inco_mse / n
    mean_normalized_mse = sum_normalized_mse / n

    # Create and return the result dictionary with 6 keys
    return {
        "mean_pred_mse": mean_pred_mse,
        "mean_inco_mse": mean_inco_mse,
        "mean_normalized_mse": mean_normalized_mse,
        "max_pred_mse": max_pred_mse,
        "max_inco_mse": max_inco_mse,
        "max_normalized_mse": max_normalized_mse,
    }


def display_sample_rows(df, n=3):
    """
    Display first n rows, middle n rows, and last n rows of a DataFrame as a formatted string.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to sample from
    n : int, default 3
        Number of rows to display from each section

    Returns:
    --------
    str
        Formatted string representation of the sampled rows
    """
    # Get total number of rows
    total_rows = len(df)

    # Calculate middle index
    mid_start = (total_rows // 2) - (n // 2)

    # Get the rows
    first_rows = df.head(n)
    middle_rows = df.iloc[mid_start : mid_start + n]
    last_rows = df.tail(n)

    # Format as string
    result = "=== First {} rows ===\n".format(n)
    result += first_rows.to_string()
    result += "\n\n=== Middle {} rows ===\n".format(n)
    result += middle_rows.to_string()
    result += "\n\n=== Last {} rows ===\n".format(n)
    result += last_rows.to_string()

    return result


def create_dataframe(time: List[float], result: Dict[str, List[float]]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from time and result variables.

    Parameters:
    - time: List of float values representing timestamps, or None
    - result: Dictionary with string keys and lists of float values, or None

    Returns:
    - pandas DataFrame with time as a regular column
    """
    # Validate that all result lists have the same length as time
    for key, values in result.items():
        if len(values) != len(time):
            raise ValueError(
                f"Length mismatch: 'time' has {len(time)} elements, but '{key}' has {len(values)} elements"
            )

    # Create DataFrame with time as a regular column
    data = result.copy()  # Copy the result dictionary
    data["Time"] = time  # Add time as a regular column

    # Create the DataFrame
    df = pd.DataFrame(data)

    return df


def compute_mean_error(
    list_a: List[Dict[str, List[float]]],
    list_b: List[Dict[str, List[float]]],
    species_ids: List[str] | None = None,
):
    """
    Compute the mean error between two lists of dictionaries with common keys.
    Each dictionary maps string keys to lists of float values.

    Args:
        list_a (List[Dict[str, List[float]]]): First list of dictionaries
        list_b (List[Dict[str, List[float]]]): Second list of dictionaries
        species_ids (List[str] | None): List of species ids to consider, use maximum in common if None

    Returns:
        float: The mean error between the lists for common keys
    """

    total_diff = 0.0
    count = 0

    # Iterate through each pair of dictionaries
    for dict_a, dict_b in zip(list_a, list_b):
        # Find common keys
        if species_ids is None:
            common_keys = set(dict_a.keys()) & set(dict_b.keys())
        else:
            common_keys = set(species_ids) & set(dict_b.keys() & set(dict_a.keys()))

        # For each common key, compute the difference between the lists
        for key in common_keys:
            list_a_values = dict_a[key]
            list_b_values = dict_b[key]

            # Find the minimum length to avoid index errors
            min_length = min(len(list_a_values), len(list_b_values))

            # Calculate difference for each position in the lists
            for i in range(min_length):
                total_diff += abs(list_a_values[i] - list_b_values[i])
                count += 1

    # Return the mean error
    return total_diff / count if count > 0 else 0.0


def find_json(string) -> str:
    start_idx = string.find("{")
    json_str = ""
    if start_idx != -1:
        # Track brace depth to find matching closing brace
        depth = 0
        for i in range(start_idx, len(string)):
            if string[i] == "{":
                depth += 1
            elif string[i] == "}":
                depth -= 1
                if depth == 0:
                    # Found the matching closing brace
                    json_str = string[start_idx : i + 1]
    return json_str


def conversation_to_ordered_dict(messages):
    """
    Convert a list of message dictionaries to an ordered dictionary where
    keys follow the pattern "user round X" and "assistant round X".

    Args:
        messages: List of dictionaries with format {"role": "user"|"assistant", "content": str}

    Returns:
        An ordered dictionary with keys following the conversation sequence
    """
    ordered_dict = OrderedDict()
    current_round = 0
    messages = messages[1:]
    for i, message in enumerate(messages):
        role = message["role"]
        try:
            content = find_json(message["content"])
        except:
            content = message["content"]

        # Determine the round number based on the message position
        # This assumes strict alternation between user and assistant
        key = f"{role} message {i}"

        ordered_dict[key] = content

    return ordered_dict


def dict_to_array(input_dict: Dict[str, List[float]]):
    """
    Convert a dictionary with string keys and list values into a single 2D numpy array
    containing only the values.

    Args:
        input_dict (dict): A dictionary where keys are strings and values are lists

    Returns:
        numpy.ndarray: A 2D numpy array containing all values from the dictionary
    """
    # Flatten all the values into a single list
    all_values = []
    for val_list in input_dict.values():
        all_values.extend(val_list)

    # Convert to numpy array and reshape to 2D
    values_array = np.array(all_values).reshape(-1, 1)

    return values_array


def default_model_parameter(method: Callable[..., T]) -> Callable[..., T]:
    @wraps(method)
    def wrapper(self, object=None, *args, **kwargs):
        if object is None:
            object = self.model
        return method(self, object, *args, **kwargs)

    return wrapper


def default_document_parameter(method: Callable[..., T]) -> Callable[..., T]:
    @wraps(method)
    def wrapper(self, object=None, *args, **kwargs):
        if object is None:
            object = self.document
        return method(self, object, *args, **kwargs)

    return wrapper


def get_experimental_constraint(object: libsbml.SBase) -> ExperimentConstraint:
    can_modify = False
    can_nullify = False

    if object.isSetNotes():
        notes_string = libsbml.XMLNode.convertXMLNodeToString(object.getNotes())
        if "<p>nullable: true</p>" in notes_string:
            can_nullify = True
        if "<p>modifiable: true</p>" in notes_string:
            can_modify = True

    return ExperimentConstraint(
        type_code=object.getTypeCode(),
        id=object.getId(),
        can_modify=can_modify,
        can_nullify=can_nullify,
    )


def generate_new_id(prefix="id_", ban_list=None):
    if ban_list is None:
        ban_list = []
    random_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    while f"{prefix}{random_str}" in ban_list:
        random_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{prefix}{random_str}"


def load_sedml_from_string_or_file(sedml_string_or_file) -> libsedml.SedDocument:
    doc: libsedml.SedDocument
    if os.path.exists(sedml_string_or_file):
        doc = libsedml.readSedMLFromFile(str(sedml_string_or_file))
    else:
        doc = libsedml.readSedMLFromString(sedml_string_or_file)

    if not isinstance(doc, libsedml.SedDocument):
        raise ValueError("Failed to load SedML document")

    error_log: libsedml.SedErrorLog = doc.getErrorLog()
    if doc.getNumErrors() > 0:
        raise ValueError(error_log.toString())

    return doc.clone()


def check_validity(
    document: libsbml.SBMLDocument,
    model: libsbml.Model,
    original_counts: Dict[str, int] | None = None,
):
    """
    Checks the validity of the SBML file after modifications to ensure
    functional integrity is maintained for simulation purposes.

    Returns:
        tuple: (bool, list) - (is_valid, list_of_issues)
    """
    issues = []

    # 1. Basic SBML validation using libSBML validator
    num_errors = document.validateSBML()
    if num_errors > 0:
        error_log: libsbml.SBMLErrorLog = document.getErrorLog()
        for i in range(error_log.getNumErrors()):
            error: libsbml.SBMLError = error_log.getError(i)
            # Only treat errors and fatal errors as issues, not warnings
            if error.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
                issues.append(f"SBML Error: {error.getMessage()}")

    # Ensure all essential components exist
    if model.getNumCompartments() == 0:
        issues.append("Model has no compartments")

    if model.getNumSpecies() == 0:
        issues.append("Model has no species")

    if model.getNumReactions() == 0:
        issues.append("Model has no reactions")

    # 3. Check for dangling references
    # Check species compartments
    for i in range(model.getNumSpecies()):
        species: libsbml.Species = model.getSpecies(i)
        comp_id = species.getCompartment()
        if model.getCompartment(comp_id) is None:
            issues.append(
                f"Species {species.getId()} references non-existent compartment {comp_id}"
            )

    # Check reaction species references
    for i in range(model.getNumReactions()):
        reaction: libsbml.Reaction = model.getReaction(i)

        # Check reactants
        for j in range(reaction.getNumReactants()):
            reactant: libsbml.SpeciesReference = reaction.getReactant(j)
            species_id = reactant.getSpecies()
            if model.getSpecies(species_id) is None:
                issues.append(
                    f"Reaction {reaction.getId()} references non-existent reactant species {species_id}"
                )

        # Check products
        for j in range(reaction.getNumProducts()):
            product: libsbml.SpeciesReference = reaction.getProduct(j)
            species_id = product.getSpecies()
            if model.getSpecies(species_id) is None:
                issues.append(
                    f"Reaction {reaction.getId()} references non-existent product species {species_id}"
                )

        # Check modifiers
        for j in range(reaction.getNumModifiers()):
            modifier: libsbml.SpeciesReference = reaction.getModifier(j)
            species_id = modifier.getSpecies()
            if model.getSpecies(species_id) is None:
                issues.append(
                    f"Reaction {reaction.getId()} references non-existent modifier species {species_id}"
                )

    # 4. Check math consistency
    # Ensure all math expressions are well-formed
    math_containers = []

    # Collect all elements with math
    for i in range(model.getNumReactions()):
        reaction = model.getReaction(i)
        if reaction.isSetKineticLaw():
            math_containers.append((reaction.getId(), "KineticLaw", reaction.getKineticLaw()))

    for i in range(model.getNumRules()):
        rule = model.getRule(i)
        math_containers.append((f"Rule{i}", "Rule", rule))

    for i in range(model.getNumEvents()):
        event: libsbml.Event = model.getEvent(i)
        if event.isSetTrigger():
            math_containers.append((event.getId(), "Trigger", event.getTrigger()))

        for j in range(event.getNumEventAssignments()):
            ea = event.getEventAssignment(j)
            math_containers.append((f"{event.getId()}_assignment_{j}", "EventAssignment", ea))

    # Check math in collected containers
    for element_id, element_type, container in math_containers:
        container: libsbml.KineticLaw | libsbml.Rule | libsbml.Trigger | libsbml.EventAssignment
        if hasattr(container, "isSetMath") and container.isSetMath():
            math_obj: libsbml.ASTNode = container.getMath()

            # Check if math is well-formed
            if math_obj is None:
                issues.append(f"{element_type} in {element_id} has NULL math")
            elif not math_obj.isWellFormedASTNode():
                issues.append(f"{element_type} in {element_id} has malformed math")

    # 5. Check for unit consistency if supported
    if document.getLevel() >= 2:
        consistency_check = document.checkConsistency()
        if consistency_check != 0:
            issues.append(f"Model consistency check failed with code {consistency_check}")

    # 6. Check for conservation of reactions and species
    # This is a basic check to ensure scrambling hasn't lost any elements
    if original_counts:
        if model.getNumCompartments() != original_counts.get(
            "compartments", model.getNumCompartments()
        ):
            issues.append("Number of compartments changed during modification")

        if model.getNumSpecies() != original_counts.get("species", model.getNumSpecies()):
            issues.append("Number of species changed during modification")

        if model.getNumReactions() != original_counts.get("reactions", model.getNumReactions()):
            issues.append("Number of reactions changed during modification")

        if model.getNumParameters() != original_counts.get("parameters", model.getNumParameters()):
            issues.append("Number of parameters changed during modification")

    # Return validity status and issues
    is_valid = len(issues) == 0
    return is_valid, issues


def compare_dicts(dict1: dict, dict2: dict, abs_tolerance=1e-12, rel_tolerance=1e-9):
    if dict1 == dict2:
        return True

    # Find keys in dict1 but not in dict2
    only_in_dict1 = {k: dict1[k] for k in dict1 if k not in dict2}
    if only_in_dict1:
        print("Keys only in first dictionary:")
        for k, v in only_in_dict1.items():
            print(f"  {k}: {v}")

    # Find keys in dict2 but not in dict1
    only_in_dict2 = {k: dict2[k] for k in dict2 if k not in dict1}
    if only_in_dict2:
        print("Keys only in second dictionary:")
        for k, v in only_in_dict2.items():
            print(f"  {k}: {v}")

    # Find keys in both but with different values (considering tolerance for numeric values)
    common_keys = set(dict1.keys()) & set(dict2.keys())
    diff_values = {}

    for k in common_keys:
        v1, v2 = dict1[k], dict2[k]

        # Check if both values are numeric
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            # Calculate absolute and relative differences
            abs_diff = abs(v1 - v2)

            # Handle special case where both values are zero or very close to zero
            if abs(v1) < abs_tolerance and abs(v2) < abs_tolerance:
                if abs_diff > abs_tolerance:
                    diff_values[k] = (v1, v2)
            else:
                # Use relative tolerance for larger values
                max_val = max(abs(v1), abs(v2))
                rel_diff = abs_diff / max_val

                if rel_diff > rel_tolerance and abs_diff > abs_tolerance:
                    diff_values[k] = (v1, v2)
        elif v1 != v2:
            diff_values[k] = (v1, v2)

    if diff_values:
        print(
            f"Keys with values differing beyond tolerances (abs: {abs_tolerance}, rel: {rel_tolerance}):"
        )
        for k, (v1, v2) in diff_values.items():
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                abs_diff = abs(v1 - v2)
                if abs(v1) < abs_tolerance and abs(v2) < abs_tolerance:
                    print(f"  {k}: {v1} != {v2} (abs diff: {abs_diff})")
                else:
                    rel_diff = abs_diff / max(abs(v1), abs(v2))
                    print(f"  {k}: {v1} != {v2} (abs diff: {abs_diff}, rel diff: {rel_diff:.2e})")
            else:
                print(f"  {k}: {v1} != {v2}")

    # Return True if there are no differences or all differences are within tolerance
    are_equal = not (only_in_dict1 or only_in_dict2 or diff_values)
    return are_equal


def subsample_dict(d, n=None):
    """Randomly sample n items from dictionary d"""
    n = n or len(d) // 2  # Default to half if n not specified
    keys = random.sample(list(d.keys()), min(n, len(d)))
    return {k: d[k] for k in keys}


def subsample_array(d, n=None, seed=42):
    """Randomly sample n items from array d"""
    n = n or len(d) // 2  # Default to half if n not specified
    random.seed(seed)
    return random.sample(d, min(n, len(d)))


# Helper function to check if an ASTNode references a species
def ast_references_species(ast_node: Optional[libsbml.ASTNode], species_id: str) -> bool:
    if ast_node is None:
        return False

    # If it's a name and matches the species ID, return True
    if ast_node.isName():
        # Get the name and compare it with the species ID
        name: Optional[str] = ast_node.getName()
        if name is not None and name == species_id:
            return True

    # Recursively check all children
    for i in range(ast_node.getNumChildren()):
        child_node: libsbml.ASTNode = ast_node.getChild(i)
        if ast_references_species(child_node, species_id):
            return True
    return False


def find_species_references(model: libsbml.Model, species_id: str) -> List[libsbml.SBase]:
    """
    Find all SBase objects that reference a particular species ID in an SBML model.

    Args:
        model (libsbml.Model): The SBML model object
        species_id (str): ID of the species to find references for

    Returns:
        List[libsbml.SBase]: List of SBase objects that reference the species
    """
    # List to store all SBase objects referencing the species
    sbase_objects: List[libsbml.SBase] = []

    # Check if the species exists
    species: Optional[libsbml.Species] = model.getSpecies(species_id)
    if species is None:
        print(f"Species with ID '{species_id}' not found in the model")
        return sbase_objects

    # Find references in reactions
    reactions: libsbml.ListOfReactions = model.getListOfReactions()
    for i in range(reactions.size()):
        reaction: libsbml.Reaction = reactions.get(i)
        reaction_references_species: bool = False

        # Check reactants
        reactants: libsbml.ListOfSpeciesReferences = reaction.getListOfReactants()
        for j in range(reactants.size()):
            reactant: libsbml.SpeciesReference = reactants.get(j)
            if reactant.getSpecies() == species_id:
                reaction_references_species = True
                break

        if reaction_references_species:
            sbase_objects.append(reaction)
            continue

        # Check products
        products: libsbml.ListOfSpeciesReferences = reaction.getListOfProducts()
        for j in range(products.size()):
            product: libsbml.SpeciesReference = products.get(j)
            if product.getSpecies() == species_id:
                reaction_references_species = True
                break

        if reaction_references_species:
            sbase_objects.append(reaction)
            continue

        # Check modifiers
        modifiers: libsbml.ListOfSpeciesReferences = reaction.getListOfModifiers()
        for j in range(modifiers.size()):
            modifier: libsbml.ModifierSpeciesReference = modifiers.get(j)
            if modifier.getSpecies() == species_id:
                reaction_references_species = True
                break

        if reaction_references_species:
            sbase_objects.append(reaction)
            continue

        # Check kinetic laws for species references in math expressions
        kinetic_law: Optional[libsbml.KineticLaw] = reaction.getKineticLaw()
        if kinetic_law:
            # Check using AST
            math_node: Optional[libsbml.ASTNode] = kinetic_law.getMath()
            if ast_references_species(math_node, species_id):
                sbase_objects.append(reaction)

    # Find references in rules
    rules: libsbml.ListOfRules = model.getListOfRules()
    for i in range(rules.size()):
        rule: libsbml.Rule = rules.get(i)
        rule_references_species: bool = False

        # Check assignment and rate rules for variable
        if rule.isAssignment() or rule.isRate():
            variable: str = rule.getVariable()
            if variable == species_id:
                rule_references_species = True

        if rule_references_species:
            sbase_objects.append(rule)
            continue

        # Check all rule types including algebraic rules via AST
        math_node: Optional[libsbml.ASTNode] = rule.getMath()
        if ast_references_species(math_node, species_id):
            sbase_objects.append(rule)

    # Find references in initial assignments
    initial_assignments: libsbml.ListOfInitialAssignments = model.getListOfInitialAssignments()
    for i in range(initial_assignments.size()):
        i_assignment: libsbml.InitialAssignment = initial_assignments.get(i)

        symbol: str = i_assignment.getSymbol()
        if symbol == species_id:
            sbase_objects.append(i_assignment)
            continue

        # Check using AST
        math_node: Optional[libsbml.ASTNode] = i_assignment.getMath()
        if ast_references_species(math_node, species_id):
            sbase_objects.append(i_assignment)

    # Find references in events
    events: libsbml.ListOfEvents = model.getListOfEvents()
    for i in range(events.size()):
        event: libsbml.Event = events.get(i)
        event_references_species: bool = False

        # Check trigger
        trigger: Optional[libsbml.Trigger] = event.getTrigger()
        if trigger:
            math_node: Optional[libsbml.ASTNode] = trigger.getMath()
            if ast_references_species(math_node, species_id):
                event_references_species = True

        # Check delay
        delay: Optional[libsbml.Delay] = event.getDelay()
        if delay and not event_references_species:
            math_node: Optional[libsbml.ASTNode] = delay.getMath()
            if ast_references_species(math_node, species_id):
                event_references_species = True

        # Check priority
        priority: Optional[libsbml.Priority] = event.getPriority()
        if priority and not event_references_species:
            math_node: Optional[libsbml.ASTNode] = priority.getMath()
            if ast_references_species(math_node, species_id):
                event_references_species = True

        # Check event assignments
        if not event_references_species:
            event_assignments: libsbml.ListOfEventAssignments = event.getListOfEventAssignments()
            for j in range(event_assignments.size()):
                assignment: libsbml.EventAssignment = event_assignments.get(j)

                variable: str = assignment.getVariable()
                if variable == species_id:
                    event_references_species = True
                    break

                math_node: Optional[libsbml.ASTNode] = assignment.getMath()
                if ast_references_species(math_node, species_id):
                    event_references_species = True
                    break

        # If the event references the species, add it
        if event_references_species:
            sbase_objects.append(event)

    # Find references in constraints
    constraints: libsbml.ListOfConstraints = model.getListOfConstraints()
    for i in range(constraints.size()):
        constraint: libsbml.Constraint = constraints.get(i)
        math_node: Optional[libsbml.ASTNode] = constraint.getMath()
        if ast_references_species(math_node, species_id):
            sbase_objects.append(constraint)

    # Find references in function definitions
    function_defs: libsbml.ListOfFunctionDefinitions = model.getListOfFunctionDefinitions()
    for i in range(function_defs.size()):
        func: libsbml.FunctionDefinition = function_defs.get(i)
        math_node: Optional[libsbml.ASTNode] = func.getMath()
        if ast_references_species(math_node, species_id):
            sbase_objects.append(func)

    return sbase_objects


def find_reaction_references(model: libsbml.Model, reaction_id: str) -> List[libsbml.SBase]:
    """
    Find all SBase objects that functionally reference a particular reaction ID in an SBML model.
    This only includes references that affect simulation results, not references in notes or annotations.

    Args:
        model (libsbml.Model): The SBML model object
        reaction_id (str): ID of the reaction to find references for

    Returns:
        List[libsbml.SBase]: List of SBase objects that reference the reaction
    """
    # List to store all SBase objects referencing the reaction
    sbase_objects: List[libsbml.SBase] = []

    # Check if the reaction exists
    reaction: Optional[libsbml.Reaction] = model.getReaction(reaction_id)
    if reaction is None:
        print(f"Reaction with ID '{reaction_id}' not found in the model")
        return sbase_objects

    # Helper function to check if an ASTNode references a reaction
    def ast_references_reaction(ast_node: Optional[libsbml.ASTNode], reaction_id: str) -> bool:
        if ast_node is None:
            return False

        # If it's a name and matches the reaction ID, return True
        if ast_node.isName():
            # Get the name and compare it with the reaction ID
            name: Optional[str] = ast_node.getName()
            if name is not None and name == reaction_id:
                return True

        # Recursively check all children
        for i in range(ast_node.getNumChildren()):
            child_node: libsbml.ASTNode = ast_node.getChild(i)
            if ast_references_reaction(child_node, reaction_id):
                return True

        return False

    # Find references in rules
    # Rules can use reaction IDs in mathematical expressions to establish relationships
    # between model components based on reaction rates
    rules: libsbml.ListOfRules = model.getListOfRules()
    for i in range(rules.size()):
        rule: libsbml.Rule = rules.get(i)
        math_node: Optional[libsbml.ASTNode] = rule.getMath()
        if ast_references_reaction(math_node, reaction_id):
            sbase_objects.append(rule)

    # Find references in initial assignments
    # Initial assignments can reference reaction IDs in their formulas
    initial_assignments: libsbml.ListOfInitialAssignments = model.getListOfInitialAssignments()
    for i in range(initial_assignments.size()):
        i_assignment: libsbml.InitialAssignment = initial_assignments.get(i)
        math_node: Optional[libsbml.ASTNode] = i_assignment.getMath()
        if ast_references_reaction(math_node, reaction_id):
            sbase_objects.append(i_assignment)

    # Find references in events
    # Events can reference reaction IDs in triggers, delays, priorities, and assignments
    events: libsbml.ListOfEvents = model.getListOfEvents()
    for i in range(events.size()):
        event: libsbml.Event = events.get(i)
        event_references_reaction: bool = False

        # Check trigger
        trigger: Optional[libsbml.Trigger] = event.getTrigger()
        if trigger:
            math_node: Optional[libsbml.ASTNode] = trigger.getMath()
            if ast_references_reaction(math_node, reaction_id):
                event_references_reaction = True

        # Check delay
        delay: Optional[libsbml.Delay] = event.getDelay()
        if delay and not event_references_reaction:
            math_node: Optional[libsbml.ASTNode] = delay.getMath()
            if ast_references_reaction(math_node, reaction_id):
                event_references_reaction = True

        # Check priority
        priority: Optional[libsbml.Priority] = event.getPriority()
        if priority and not event_references_reaction:
            math_node: Optional[libsbml.ASTNode] = priority.getMath()
            if ast_references_reaction(math_node, reaction_id):
                event_references_reaction = True

        # Check event assignments
        if not event_references_reaction:
            event_assignments: libsbml.ListOfEventAssignments = event.getListOfEventAssignments()
            for j in range(event_assignments.size()):
                assignment: libsbml.EventAssignment = event_assignments.get(j)
                math_node: Optional[libsbml.ASTNode] = assignment.getMath()
                if ast_references_reaction(math_node, reaction_id):
                    event_references_reaction = True
                    break

        # If the event references the reaction, add it
        if event_references_reaction:
            sbase_objects.append(event)

    # Find references in constraints
    # Constraints can reference reaction IDs in their mathematical expressions
    constraints: libsbml.ListOfConstraints = model.getListOfConstraints()
    for i in range(constraints.size()):
        constraint: libsbml.Constraint = constraints.get(i)
        math_node: Optional[libsbml.ASTNode] = constraint.getMath()
        if ast_references_reaction(math_node, reaction_id):
            sbase_objects.append(constraint)

    # Find references in function definitions
    # Function definitions can use reaction IDs in their mathematical expressions
    function_defs: libsbml.ListOfFunctionDefinitions = model.getListOfFunctionDefinitions()
    for i in range(function_defs.size()):
        func: libsbml.FunctionDefinition = function_defs.get(i)
        math_node: Optional[libsbml.ASTNode] = func.getMath()
        if ast_references_reaction(math_node, reaction_id):
            sbase_objects.append(func)

    # Find references in kinetic laws of other reactions
    # One reaction's rate law might depend on another reaction's flux
    reactions: libsbml.ListOfReactions = model.getListOfReactions()
    for i in range(reactions.size()):
        other_reaction: libsbml.Reaction = reactions.get(i)

        # Skip the reaction we're searching for
        if other_reaction.getId() == reaction_id:
            continue

        # Check kinetic law for references to our reaction
        kinetic_law: Optional[libsbml.KineticLaw] = other_reaction.getKineticLaw()
        if kinetic_law:
            math_node: Optional[libsbml.ASTNode] = kinetic_law.getMath()
            if ast_references_reaction(math_node, reaction_id):
                sbase_objects.append(other_reaction)

    # Find references in rate rules for parameters used in the reaction
    # If a parameter in the reaction is controlled by a rate rule, changing the reaction
    # will affect how that parameter is used
    if reaction:
        kinetic_law: Optional[libsbml.KineticLaw] = reaction.getKineticLaw()
        if kinetic_law:
            parameters: libsbml.ListOfParameters = kinetic_law.getListOfParameters()
            for i in range(parameters.size()):
                param: libsbml.Parameter = parameters.get(i)
                param_id: str = param.getId()

                # Check if this parameter is used in any rate rules
                for j in range(rules.size()):
                    rule: libsbml.Rule = rules.get(j)
                    if rule.isRate() and rule.getVariable() == param_id:
                        # Only add if we haven't already added this rule
                        if rule not in sbase_objects:
                            sbase_objects.append(rule)

    return sbase_objects


def find_dangling_objects(model: libsbml.Model) -> Sequence[libsbml.SBase]:
    """
    Finds global parameters that are not used anywhere in the model and can be deleted
    without affecting the functionality of the model under simulation.

    Returns a list of dangling Parameter objects.
    """
    # Get all IDs referenced in math expressions throughout the model
    referenced_ids = set()

    # Helper function to extract IDs from ASTNode
    def collect_ids(node: libsbml.ASTNode):
        if node is None:
            return

        if node.isName():
            referenced_ids.add(node.getName())

        for i in range(node.getNumChildren()):
            collect_ids(node.getChild(i))

    # 1. Check reaction kinetics
    for i in range(model.getNumReactions()):
        reaction: libsbml.Reaction = model.getReaction(i)
        if reaction.isSetKineticLaw():
            klaw: libsbml.KineticLaw = reaction.getKineticLaw()
            if klaw.isSetMath():
                collect_ids(klaw.getMath())

    # 2. Check rules
    for i in range(model.getNumRules()):
        rule: libsbml.Rule = model.getRule(i)
        if rule.isSetMath():
            collect_ids(rule.getMath())

    # 3. Check initial assignments
    for i in range(model.getNumInitialAssignments()):
        ia: libsbml.InitialAssignment = model.getInitialAssignment(i)
        if ia.isSetMath():
            collect_ids(ia.getMath())

    # 4. Check constraints
    for i in range(model.getNumConstraints()):
        constraint: libsbml.Constraint = model.getConstraint(i)
        if constraint.isSetMath():
            collect_ids(constraint.getMath())

    # 5. Check events
    for i in range(model.getNumEvents()):
        event: libsbml.Event = model.getEvent(i)
        # Check trigger
        if event.isSetTrigger():
            trigger: libsbml.Trigger = event.getTrigger()
            if trigger.isSetMath():
                collect_ids(trigger.getMath())

        # Check delay
        if event.isSetDelay():
            delay: libsbml.Delay = event.getDelay()
            if delay.isSetMath():
                collect_ids(delay.getMath())

        # Check event assignments
        for j in range(event.getNumEventAssignments()):
            ea: libsbml.EventAssignment = event.getEventAssignment(j)
            if ea.isSetMath():
                collect_ids(ea.getMath())

    # 6. Check function definitions
    for i in range(model.getNumFunctionDefinitions()):
        fd: libsbml.FunctionDefinition = model.getFunctionDefinition(i)
        if fd.isSetMath():
            collect_ids(fd.getMath())

    # Find parameters that are targets of rules or initial assignments
    target_ids = set()
    for i in range(model.getNumRules()):
        rule = model.getRule(i)
        if rule.isSetVariable():
            target_ids.add(rule.getVariable())

    for i in range(model.getNumInitialAssignments()):
        ia = model.getInitialAssignment(i)
        if ia.isSetSymbol():
            target_ids.add(ia.getSymbol())

    # Combine sets - parameters are used if they're referenced in math or are targets
    used_ids = referenced_ids.union(target_ids)

    # Find dangling parameters
    dangling_params: List[libsbml.Parameter] = []
    for i in range(model.getNumParameters()):
        param: libsbml.Parameter = model.getParameter(i)
        if param.getId() not in used_ids:
            dangling_params.append(param)

    return dangling_params


def find_parameter_initializations(model: libsbml.Model, parameter_id: str) -> List[libsbml.SBase]:
    """
    Identifies all definitions of a parameter's initial value in an SBML model.

    This function finds:
    1. Any initialAssignment targeting the parameter
    2. Any assignmentRule targeting the parameter
    3. Any rateRule targeting the parameter
    4. Any eventAssignment that sets the parameter value

    Parameters:
    -----------
    model : libsbml.Model
        The libSBML model object
    parameter_id : str
        The ID of the parameter to unset

    Returns:
    --------
    tuple[list[libsbml.SBase], bool]
        A list of SBase objects that define the parameter's initialization,
        and a boolean indicating success (True) or failure (False)
    """
    # Check if the parameter exists
    parameter: libsbml.Parameter | None = model.getParameter(parameter_id)
    if parameter is None:
        print(f"Parameter '{parameter_id}' not found in the model")
        return []

    objects_to_delete = []

    # Find any initial assignment for this parameter
    initial_assignments: libsbml.ListOfInitialAssignments = model.getListOfInitialAssignments()
    for i in range(initial_assignments.size()):
        ia: libsbml.InitialAssignment = initial_assignments.get(i)
        if ia.getSymbol() == parameter_id:
            objects_to_delete.append(ia)
            # print(f"Found initialAssignment for {parameter_id}")

    # Find any assignment rule or rate rule for this parameter
    rules: libsbml.ListOfRules = model.getListOfRules()
    for i in range(rules.size()):
        rule: libsbml.Rule = rules.get(i)

        if (rule.isAssignment() or rule.isRate()) and rule.getVariable() == parameter_id:
            objects_to_delete.append(rule)
            rule_type = "assignmentRule" if rule.isAssignment() else "rateRule"
            # print(f"Found {rule_type} for {parameter_id}")

    # Find event assignments that set the parameter
    events: libsbml.ListOfEvents = model.getListOfEvents()
    for i in range(events.size()):
        event: libsbml.Event = events.get(i)
        event_assignments: libsbml.ListOfEventAssignments = event.getListOfEventAssignments()

        for j in range(event_assignments.size()):
            ea: libsbml.EventAssignment = event_assignments.get(j)
            if ea.getVariable() == parameter_id:
                objects_to_delete.append(ea)
                # print(f"Found eventAssignment for {parameter_id} in event {event.getId()}")

    return objects_to_delete


def find_species_knockout_references(model: libsbml.Model, species_id: str) -> List[libsbml.SBase]:
    """
    Finds all SBase objects that can affect the concentration of this species
    """
    sbase_objects: List[libsbml.SBase] = []

    # Check if the species exists
    species: Optional[libsbml.Species] = model.getSpecies(species_id)
    if species is None:
        print(f"Species with ID '{species_id}' not found in the model")
        return sbase_objects

    # Find references in reactions
    reactions: libsbml.ListOfReactions = model.getListOfReactions()
    for i in range(reactions.size()):
        reaction: libsbml.Reaction = reactions.get(i)
        reaction_references_species: bool = False

        # Check reactants
        reactants: libsbml.ListOfSpeciesReferences = reaction.getListOfReactants()
        for j in range(reactants.size()):
            reactant: libsbml.SpeciesReference = reactants.get(j)
            if reactant.getSpecies() == species_id:
                reaction_references_species = True
                break

        if reaction_references_species:
            sbase_objects.append(reaction)
            continue

        # Check products
        products: libsbml.ListOfSpeciesReferences = reaction.getListOfProducts()
        for j in range(products.size()):
            product: libsbml.SpeciesReference = products.get(j)
            if product.getSpecies() == species_id:
                reaction_references_species = True
                break

        # Check modifiers
        # modifiers: libsbml.ListOfSpeciesReferences = reaction.getListOfModifiers()
        # for j in range(modifiers.size()):
        #     modifier: libsbml.ModifierSpeciesReference = modifiers.get(j)
        #     if modifier.getSpecies() == species_id:
        #         reaction_references_species = True
        #         break

        if reaction_references_species:
            sbase_objects.append(reaction)
            continue

    # Find references in rules
    rules: libsbml.ListOfRules = model.getListOfRules()
    for i in range(rules.size()):
        rule: libsbml.Rule = rules.get(i)
        rule_references_species: bool = False

        # Check assignment and rate rules for variable
        if rule.isAssignment() or rule.isRate():
            variable: str = rule.getVariable()
            if variable == species_id:
                rule_references_species = True

        # Check for algebraic rule
        if rule.isAlgebraic():
            math_node: Optional[libsbml.ASTNode] = rule.getMath()
            if ast_references_species(math_node, species_id):
                rule_references_species = True

        if rule_references_species:
            sbase_objects.append(rule)
            continue

    # Find references in initial assignments
    initial_assignments: libsbml.ListOfInitialAssignments = model.getListOfInitialAssignments()
    for i in range(initial_assignments.size()):
        i_assignment: libsbml.InitialAssignment = initial_assignments.get(i)

        symbol: str = i_assignment.getSymbol()
        if symbol == species_id:
            sbase_objects.append(i_assignment)
            continue

    return sbase_objects


def process_annotation(annotation: libsbml.XMLNode, rm_prefix=["celldesigner"]):
    """
    Process an annotation node to remove all CellDesigner tags.

    Parameters:
    annotation (XMLNode): The annotation XML node
    """
    # Remove all CellDesigner children (traversing in reverse to safely remove)
    for i in range(annotation.getNumChildren() - 1, -1, -1):
        child: libsbml.XMLNode = annotation.getChild(i)

        # Check if this child is a CellDesigner element
        prefix = child.getPrefix()
        name = child.getName()

        if prefix in rm_prefix or any([m in name for m in rm_prefix]):
            annotation.removeChild(i)
        else:
            # Recursively process any remaining children
            if child.getNumChildren() > 0:
                process_annotation(child)


def parse_experiment_action(
    experiment_action: str,
    valid_species_ids: List[str],
    valid_reaction_ids: List[str],
) -> ExperimentAction:
    """
    Parse a string representation of an experiment action into an ExperimentAction object.

    Args:
        experiment_action (str): The string representation of the experiment action
        valid_species_ids (List[str]): List of valid species IDs
        valid_reaction_ids (List[str]): List of valid reaction IDs

    Returns:
        ExperimentAction: The parsed ExperimentAction object

    Raises:
        ParseExperimentActionError: If the action string is invalid or cannot be parsed
    """
    # Remove any whitespace
    experiment_action = experiment_action.strip()

    # Regular expressions for parsing different action formats
    change_reaction_regex = (
        r'change_reaction_rate\((?:\\"|")([^\\"]*)(?:\\"|")\s*,\s*([0-9.]+(?:[eE][+-]?[0-9]+)?)\)'
    )
    nullify_reaction_regex = r'deactivate_reaction\((?:\\"|")([^\\"]*)(?:\\"|")\)'
    change_concentration_regex = r'change_initial_concentration\((?:\\"|")([^\\"]*)(?:\\"|")\s*,\s*([0-9.]+(?:[eE][+-]?[0-9]+)?)\)'
    nullify_species_regex = r'knockout_species\((?:\\"|")([^\\"]*)(?:\\"|")\)'

    # Try to match change_reaction_rate
    match = re.match(change_reaction_regex, experiment_action)
    if match:
        reaction_id = match.group(1)
        multiply_factor = float(match.group(2))

        if reaction_id not in valid_reaction_ids:
            raise ParseExperimentActionError(
                f"Reaction ID '{reaction_id}' is not valid. Valid IDs are: {valid_reaction_ids}"
            )

        return ModifyReactionAction(reaction_id=reaction_id, multiply_factor=multiply_factor)

    # Try to match deactivate_reaction
    match = re.match(nullify_reaction_regex, experiment_action)
    if match:
        reaction_id = match.group(1)

        if reaction_id not in valid_reaction_ids:
            raise ParseExperimentActionError(
                f"Reaction ID '{reaction_id}' is not valid. Valid IDs are: {valid_reaction_ids}"
            )

        return NullifyReactionAction(reaction_id=reaction_id)

    # Try to match change_initial_concentration
    match = re.match(change_concentration_regex, experiment_action)
    if match:
        species_id = match.group(1)

        if species_id not in valid_species_ids:
            raise ParseExperimentActionError(
                f"Species ID '{species_id}' is not valid. Valid IDs are: {valid_species_ids}"
            )

        value = float(match.group(2))
        return ModifySpeciesAction(species_id=species_id, value=value)

    # Try to match knockout_species
    match = re.match(nullify_species_regex, experiment_action)
    if match:
        species_id = match.group(1)

        if species_id not in valid_species_ids:
            raise ParseExperimentActionError(
                f"Species ID '{species_id}' is not valid. Valid IDs are: {valid_species_ids}"
            )

        return NullifySpeciesAction(species_id=species_id)

    # If no match was found, raise an error
    raise ParseExperimentActionError(f"Failed to parse experiment action: {experiment_action}")


def make_random_sequence_of_actions_strings(
    can_nullify_rids: List[str],
    can_nullify_sids: List[str],
    can_modify_rids: List[str],
    can_modify_sids: List[str],
    num_actions: int = 5,
    seed: int = 42,
) -> List[str]:
    """
    Generate a list of random action strings that can be parsed into ExperimentAction objects.

    Args:
        can_nullify_rids: List of reaction IDs that can be nullified
        can_nullify_sids: List of species IDs that can be nullified
        can_modify_rids: List of reaction IDs that can be modified
        can_modify_sids: List of species IDs that can be modified
        initial_sid_concentrations: Dictionary mapping species IDs to their initial concentrations
        num_actions: Number of random actions to generate (default: 5)
        seed: Random seed for reproducibility (default: None)

    Returns:
        List of action strings that can be parsed by parse_experiment_action
    """
    if seed is not None:
        random.seed(seed)

    action_strings = set()
    action_types = []

    if can_nullify_rids:
        action_types.append("deactivate_reaction")

    if can_nullify_sids:
        action_types.append("knockout_species")

    if can_modify_rids:
        action_types.append("change_reaction_rate")

    if can_modify_sids:
        action_types.append("change_initial_concentration")

    if not action_types:
        return []  # No valid action types available

    chosen_ids = []

    # Generate random actions
    for _ in range(num_actions):
        action_type = random.choice(action_types)

        if action_type == "deactivate_reaction":
            sid = random.choice(can_nullify_rids)
            if sid in chosen_ids:
                continue
            action = f'deactivate_reaction("{sid}")'

        elif action_type == "knockout_species":
            sid = random.choice(can_nullify_sids)
            if sid in chosen_ids:
                continue
            action = f'knockout_species("{sid}")'

        elif action_type == "change_reaction_rate":
            sid = random.choice(can_modify_rids)
            if sid in chosen_ids:
                continue
            multiplier = round(random.uniform(0.5, 1.5), 2)
            action = f'change_reaction_rate("{sid}", {multiplier})'

        elif action_type == "change_initial_concentration":
            sid = random.choice(can_modify_sids)
            if sid in chosen_ids:
                continue
            multiplier = round(random.uniform(0.5, 1.5), 2)
            action = f'change_initial_concentration("{sid}", {multiplier})'

        else:
            raise ValueError(f"Unexpected action type: {action_type}")

        chosen_ids.append(sid)
        action_strings.add(action)

    return list(action_strings)


def generate_all_possible_action_strings(
    can_nullify_rids: list[str],
    can_nullify_sids: list[str],
    can_modify_rids: list[str],
    can_modify_sids: list[str],
    initial_sid_concentrations: dict[str, float],
    default_rate_multiplier: float = 1.25,
    default_concentration_multiplier: float = 2.0,
) -> list[str]:
    """
    Generate a list of all possible action strings that can be parsed into ExperimentAction objects.
    For actions requiring values (like rate changes or concentration settings),
    uses one default value per ID.

    Args:
        can_nullify_rids: List of reaction IDs that can be nullified
        can_nullify_sids: List of species IDs that can be nullified
        can_modify_rids: List of reaction IDs that can be modified
        can_modify_sids: List of species IDs that can be modified
        initial_sid_concentrations: Dictionary mapping species IDs to their initial concentrations
        default_rate_multiplier: Default multiplier to use for reaction rate changes (default: 1.25)
        default_concentration_multiplier: Default multiplier for concentration changes (default: 2.0)

    Returns:
        List of all possible action strings that can be parsed by parse_experiment_action
    """
    all_action_strings = []

    # Add all possible reaction nullifications
    for rid in can_nullify_rids:
        all_action_strings.append(f'deactivate_reaction("{rid}")')

    # Add all possible species nullifications
    for sid in can_nullify_sids:
        all_action_strings.append(f'knockout_species("{sid}")')

    # Add all possible reaction rate changes (one per ID)
    for rid in can_modify_rids:
        all_action_strings.append(f'change_reaction_rate("{rid}", {default_rate_multiplier})')

    # Add all possible initial concentration changes (one per ID)
    for sid in can_modify_sids:
        # Use the initial concentration to calculate a new value
        all_action_strings.append(
            f'change_initial_concentration("{sid}", {default_rate_multiplier})'
        )

    return all_action_strings


def generate_invalid_action_strings(valid_actions) -> List[Tuple[str, str]]:
    """
    Generate invalid action strings from valid ones to test parser robustness.

    Args:
        valid_actions: List of valid action strings

    Returns:
        List of tuples (invalid_action, description_of_issue)
    """
    invalid_actions = []

    for action in valid_actions:
        # Missing quotes around IDs
        if 'deactivate_reaction("' in action:
            rid = action.split('"')[1]
            invalid_actions.append(
                (f"deactivate_reaction({rid})", "Missing quotes around reaction ID")
            )

        if 'knockout_species("' in action:
            sid = action.split('"')[1]
            invalid_actions.append((f"knockout_species({sid})", "Missing quotes around species ID"))

        if 'change_reaction_rate("' in action:
            rid = action.split('"')[1]
            multiplier = action.split(", ")[1].rstrip(")")
            invalid_actions.append(
                (f"change_reaction_rate({rid}, {multiplier})", "Missing quotes around reaction ID")
            )

        if 'change_initial_concentration("' in action:
            sid = action.split('"')[1]
            multiplier = action.split(", ")[1].rstrip(")")
            invalid_actions.append(
                (
                    f"change_initial_concentration({sid}, {multiplier})",
                    "Missing quotes around species ID",
                )
            )

        # Wrong quote types
        if 'deactivate_reaction("' in action:
            invalid_actions.append(
                (action.replace('"', "'"), "Using single quotes instead of double quotes")
            )

        # Missing parentheses
        if "deactivate_reaction" in action:
            invalid_actions.append((action.rstrip(")"), "Missing closing parenthesis"))
            invalid_actions.append((action.replace("(", ""), "Missing opening parenthesis"))

        # Extra spaces
        if "deactivate_reaction" in action or "knockout_species" in action:
            invalid_actions.append(
                (action.replace("(", " ("), "Extra space before opening parenthesis")
            )

        # Wrong parameter types
        if 'change_reaction_rate("' in action:
            rid = action.split('"')[1]
            invalid_actions.append(
                (
                    f'change_reaction_rate("{rid}", "1.5")',
                    "String instead of float for rate multiplier",
                )
            )
            invalid_actions.append(
                (
                    f'change_reaction_rate("{rid}", None)',
                    "None instead of float for rate multiplier",
                )
            )
            invalid_actions.append(
                (f'change_reaction_rate("{rid}", -0.5)', "Negative value for rate multiplier")
            )

        if 'change_initial_concentration("' in action:
            sid = action.split('"')[1]
            invalid_actions.append(
                (
                    f'change_initial_concentration("{sid}", "1.0")',
                    "String instead of float for rate multiplier",
                )
            )
            invalid_actions.append(
                (
                    f'change_initial_concentration("{sid}", None)',
                    "None instead of float for rate multiplier",
                )
            )
            invalid_actions.append(
                (
                    f'change_initial_concentration("{sid}", -1.0)',
                    "Negative value for rate multiplier",
                )
            )

        # Wrong number of parameters
        if 'deactivate_reaction("' in action:
            rid = action.split('"')[1]
            invalid_actions.append(
                (f'deactivate_reaction("{rid}", 1.0)', "Extra parameter for deactivate_reaction")
            )
            invalid_actions.append(
                (f"deactivate_reaction()", "Missing parameter for deactivate_reaction")
            )

        if 'change_reaction_rate("' in action:
            rid = action.split('"')[1]
            invalid_actions.append((f'change_reaction_rate("{rid}")', "Missing rate parameter"))
            invalid_actions.append(
                (f'change_reaction_rate("{rid}", 1.5, "extra")', "Extra parameter")
            )

        # Typos in function names
        if "deactivate_reaction" in action:
            invalid_actions.append(
                (
                    action.replace("deactivate_reaction", "nullify_reacction"),
                    "Typo in function name",
                )
            )

        if "knockout_species" in action:
            invalid_actions.append(
                (
                    action.replace("knockout_species", "nullifyspecies"),
                    "Missing underscore in function name",
                )
            )

        if "change_reaction_rate" in action:
            invalid_actions.append(
                (
                    action.replace("change_reaction_rate", "change_reactionrate"),
                    "Missing underscore in function name",
                )
            )

        if "change_initial_concentration" in action:
            invalid_actions.append(
                (
                    action.replace("change_initial_concentration", "change_initialconcentration"),
                    "Missing underscore in function name",
                )
            )

        # Empty strings for IDs
        if 'deactivate_reaction("' in action:
            invalid_actions.append(('deactivate_reaction("")', "Empty string as reaction ID"))

        if 'knockout_species("' in action:
            invalid_actions.append(('knockout_species("")', "Empty string as species ID"))

        # Mixed case or uppercase function names
        if "deactivate_reaction" in action:
            invalid_actions.append(
                (
                    action.replace("deactivate_reaction", "deactivate_reaction"),
                    "Capitalized function name",
                )
            )

    # Additional general invalid cases
    invalid_actions.extend(
        [
            ("", "Empty string"),
            ('unknown_action("id")', "Unknown action type"),
            ('nullify_both("rid", "sid")', "Non-existent function"),
            ("deactivate_reaction", "Missing parentheses and parameters"),
            ('deactivate_reaction"rid")', "Missing opening parenthesis"),
            ('deactivate_reaction("rid"', "Missing closing parenthesis"),
            ('"deactivate_reaction("rid")"', "Extra quotes around entire action"),
            (
                'deactivate_reaction("rid")knockout_species("sid")',
                "Multiple actions without separator",
            ),
            ("change_reaction_rate(, 1.5)", "Missing ID"),
            ('change_initial_concentration("sid",)', "Missing value after comma"),
            ('change_initial_concentration("sid" 1.5)', "Missing comma between parameters"),
        ]
    )

    return invalid_actions


def shuffle_parameters(model: libsbml.Model):
    n_params = model.getNumParameters()
    param_objects = []
    for i in range(n_params):
        param: libsbml.Parameter = model.getParameter(i)
        param_objects.append(param.clone())
    random.shuffle(param_objects)
    for i in range(n_params - 1, -1, -1):
        model.removeParameter(i)
    for param in param_objects:
        assert model.addParameter(param) == libsbml.LIBSBML_OPERATION_SUCCESS


def shuffle_reactions(model: libsbml.Model):
    n_reactions = model.getNumReactions()
    reaction_objects = []
    for i in range(n_reactions):
        reaction: libsbml.Reaction = model.getReaction(i)
        reaction_objects.append(reaction.clone())
    random.shuffle(reaction_objects)
    for i in range(n_reactions - 1, -1, -1):
        model.removeReaction(i)
    for reaction in reaction_objects:
        assert model.addReaction(reaction) == libsbml.LIBSBML_OPERATION_SUCCESS


def shuffle_species(model: libsbml.Model):
    n_species = model.getNumSpecies()
    species_objects = []
    for i in range(n_species):
        species: libsbml.Species = model.getSpecies(i)
        species_objects.append(species.clone())
    random.shuffle(species_objects)
    for i in range(n_species - 1, -1, -1):
        model.removeSpecies(i)
    for species in species_objects:
        assert model.addSpecies(species) == libsbml.LIBSBML_OPERATION_SUCCESS


def shuffle_rules(model: libsbml.Model):
    n_rules = model.getNumRules()
    rule_objects = []
    for i in range(n_rules):
        rule: libsbml.Rule = model.getRule(i)
        rule_objects.append(rule.clone())
    random.shuffle(rule_objects)
    for i in range(n_rules - 1, -1, -1):
        model.removeRule(i)
    for rule in rule_objects:
        assert model.addRule(rule) == libsbml.LIBSBML_OPERATION_SUCCESS


def shuffle_compartments(model: libsbml.Model):
    n_compartments = model.getNumCompartments()
    compartment_objects = []
    for i in range(n_compartments):
        compartment: libsbml.Compartment = model.getCompartment(i)
        compartment_objects.append(compartment.clone())
    random.shuffle(compartment_objects)
    for i in range(n_compartments - 1, -1, -1):
        model.removeCompartment(i)
    for compartment in compartment_objects:
        assert model.addCompartment(compartment) == libsbml.LIBSBML_OPERATION_SUCCESS


def shuffle_function_definitions(model: libsbml.Model):
    n_function_defs = model.getNumFunctionDefinitions()
    function_def_objects = []
    for i in range(n_function_defs):
        function_def: libsbml.FunctionDefinition = model.getFunctionDefinition(i)
        function_def_objects.append(function_def.clone())
    random.shuffle(function_def_objects)
    for i in range(n_function_defs - 1, -1, -1):
        model.removeFunctionDefinition(i)
    for function_def in function_def_objects:
        assert model.addFunctionDefinition(function_def) == libsbml.LIBSBML_OPERATION_SUCCESS


def shuffle_constraints(model: libsbml.Model):
    n_constraints = model.getNumConstraints()
    constraint_objects = []
    for i in range(n_constraints):
        constraint: libsbml.Constraint = model.getConstraint(i)
        constraint_objects.append(constraint.clone())
    random.shuffle(constraint_objects)
    for i in range(n_constraints - 1, -1, -1):
        model.removeConstraint(i)
    for constraint in constraint_objects:
        assert model.addConstraint(constraint) == libsbml.LIBSBML_OPERATION_SUCCESS


def shuffle_initial_assignments(model: libsbml.Model):
    n_initial_assignments = model.getNumInitialAssignments()
    initial_assignment_objects = []
    for i in range(n_initial_assignments):
        initial_assignment: libsbml.InitialAssignment = model.getInitialAssignment(i)
        initial_assignment_objects.append(initial_assignment.clone())
    random.shuffle(initial_assignment_objects)
    for i in range(n_initial_assignments - 1, -1, -1):
        model.removeInitialAssignment(i)
    for initial_assignment in initial_assignment_objects:
        assert model.addInitialAssignment(initial_assignment) == libsbml.LIBSBML_OPERATION_SUCCESS


def shuffle_unit_definitions(model: libsbml.Model):
    n_unit_definitions = model.getNumUnitDefinitions()
    unit_definition_objects = []
    for i in range(n_unit_definitions):
        unit_definition: libsbml.UnitDefinition = model.getUnitDefinition(i)
        unit_definition_objects.append(unit_definition.clone())
    random.shuffle(unit_definition_objects)
    for i in range(n_unit_definitions - 1, -1, -1):
        model.removeUnitDefinition(i)
    for unit_definition in unit_definition_objects:
        assert model.addUnitDefinition(unit_definition) == libsbml.LIBSBML_OPERATION_SUCCESS


def getAllIds(allElements: libsbml.SBaseList) -> List[str]:
    result = []
    if allElements is None or allElements.getSize() == 0:
        return result
    for i in range(0, allElements.getSize()):
        current: libsbml.SBase = allElements.get(i)
        if current.isSetId():
            result.append(current.getId())
    return result


def getAllNames(allElements: libsbml.SBaseList) -> List[str]:
    result = []
    if allElements is None or allElements.getSize() == 0:
        return result
    for i in range(0, allElements.getSize()):
        current: libsbml.SBase = allElements.get(i)
        if current.isSetName():
            result.append(current.getName())
    return result


def canonicalize_name(original_name):
    # Save original for cases where we can't properly parse
    original = original_name.strip()

    # Step 1: Remove compartment information in brackets and parentheses
    name = re.sub(
        r"\s*\[(cytoplasm|extracellular|mitochondrion|nucleus|vacuole|peroxisome|cell envelope|.*?membrane|.*?particle|.*?reticulum|Golgi|intracellular|.*?matrix|.*?fluid)\]",
        "",
        original,
    )
    name = re.sub(
        r"\s*\((cytoplasm|extracellular|mitochondrion|nucleus|vacuole|peroxisome|cell envelope|.*?membrane|.*?particle|.*?reticulum|Golgi|intracellular|.*?matrix|.*?fluid)\)",
        "",
        name,
    )
    name = re.sub(r"\s*\[.*?\]", "", name)  # Remove any remaining brackets

    # Step 2: Remove specific molecule IDs, chemical formulas, and concentrations
    name = re.sub(r"_c0\b|_e0\b|\[c0\]|\[e0\]|\[c\]|\[e\]", "", name)
    name = re.sub(r"N\d+O\d+P*R*\d*S*\d*", "", name)  # Remove formula patterns like N5O6P
    name = re.sub(r"C\d+H\d+.*?O\d*|O\d+|\(C\d+H\d+.*?\)", "", name)  # Remove chemical formulas
    name = re.sub(r"__91__.*?__93__", "", name)
    name = re.sub(r"\bM_|\bs_\d+", "", name)  # Remove M_ and s_digits prefixes
    name = re.sub(r"_\d+$|_\d+\b", "", name)  # Remove trailing _numbers

    # Step 3: Remove location and state indicators
    name = re.sub(r"\b(cytop|cyt|nuc|cyto|ext|mito|vac|er|golgi|int|peri|aw|bm)\b", "", name)
    name = re.sub(
        r"(_star|\*|\~.*?|\^[A-Za-z0-9]+|_[A-Za-z]+\d+$)", "", name
    )  # Remove state markers

    # Step 4: Extract fatty acid, lipid, and sugar patterns to a more generalized form
    # Handle lipid acyl chain notation like (1-16:0, 2-18:1) or n-C14-0
    name = re.sub(r"\s*\(\d+-\d+:\d+,\s*\d+-\d+:\d+(?:,\s*\d+-\d+:\d+)*\)", "", name)
    name = re.sub(r"\b(n-C\d+[-:]\d+)\b", "", name)
    name = re.sub(r"\b\d+:\d+\b", "", name)  # Remove fatty acid descriptors like 18:1

    # Step 5: Handle protein complexes with special notation
    if ":" in name and not re.search(r"tRNA\([A-Za-z]+\)", name):
        parts = name.split(":")
        if len(parts) > 2:
            name = parts[0] + " complex"
        else:
            name = parts[0] + "-" + parts[1]

    # Handle complex notation with slashes or special characters
    name = re.sub(r"([A-Za-z0-9]+)/([A-Za-z0-9]+)", r"\1 \2", name)  # Replace / with space

    # Step 6: Clean up some common specific patterns
    name = re.sub(r"mw[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}", "", name)
    name = re.sub(r"R[a-z0-9]{2}[A-Z]{2}|cam\s+[A-Z]{2}.*?\d+|Dp\s+[A-Za-z]+.*?", "", name)
    name = re.sub(r"Di\d{2}[A-Z]{2}|Da\d{2}[A-Z]{2}", "", name)
    name = re.sub(r"species_\d+", "", name)
    name = re.sub(r"GENE_\w+_\w+|PROTEIN_\w+_\w+|mRNA_\w+_\w+", "", name)
    name = re.sub(r"WTasyn\d+[a-zA-Z]*", "alpha-synuclein", name)  # Standardize synuclein variants

    # Step 7: Clean up structure notation
    name = re.sub(r"chy-\((GlcNAc|Man|Gal|Glc)\)(\d+)", r"\1 oligosaccharide", name)
    name = re.sub(r"\(GlcNAc\)(\d+)\(Man\)(\d+)", "GlcNAc-Man oligosaccharide", name)

    # Step 8: Standardize protein complex notation
    name = re.sub(r"(EGF|ErbB\d+|IL\d+|TNF|IFN)[-_]p?(.*?)[-_](\d+|[A-Z]+)", r"\1 \2", name)
    name = re.sub(
        r"p(ERK|MEK|STAT|RAF|RAS)(\d*)", r"\1", name
    )  # Standardize phosphorylated proteins
    name = re.sub(r"complex\s+br\s+\(.*?\)", "protein complex", name)
    name = re.sub(r"dnabound_(\w+)", r"\1", name)  # Remove DNA-bound prefix
    name = re.sub(r"Foxo\d+_Pa\d+_Pd\d+", "Foxo", name)  # Simplify Foxo variants

    # Step 9: Clean up punctuation, weird characters, extra spaces
    name = re.sub(r"_+", " ", name)  # Replace underscores with spaces
    name = re.sub(r"\\b|\\n|\\t", " ", name)  # Replace escape sequences with spaces
    name = re.sub(r"\s+", " ", name)  # Replace multiple spaces with single
    name = name.strip()  # Trim spaces from beginning and end

    # Step 10: Handle special abbreviations and terms
    name = re.sub(r"\bL-|\bD-", "", name)  # Remove L- and D- stereochemistry prefixes
    name = re.sub(r"@\s+\w+", "", name)  # Remove @ annotations

    # Step 11: If the result is empty, very short, or just numbers, return a more general form
    if not name or len(name) < 3 or re.match(r"^[0-9\s]+$", name):
        # Try to extract the main part of the original
        match = re.search(r"([A-Za-z]+\d*[A-Za-z]*)", original)
        if match:
            return match.group(1)
        return original

    return name
