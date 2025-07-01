from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class CodeResult:
    def __init__(
        self,
        success: bool,
        std_out: Optional[str] = None,
        data: Any = None,
        error: str | None = None,
    ):
        self.success = success
        self.data = data
        self.error = error
        self.std_out = std_out

    def to_string(self) -> str:
        """
        Convert the tool result to a clean string representation.
        No summaries or special formatting - just variable name and value.
        Numeric values are rounded to 2 decimal places.

        Returns:
            A string representation of the tool result
        """
        if self.success:
            result = f"Your code is successfully executed.\n"
            if self.std_out:
                result += "Below is the standard out from your code.\n"
                result += self.std_out + "\n"
            return result
        else:
            return f"Your code has bugs. This is the error message: {self.error}"

    def return_data(self, variable_name):
        if self.data and variable_name in self.data:
            return self.data[variable_name]


class CodeRequest:
    """
    Class to encapsulate a request to execute a tool.
    """

    def __init__(self, code: str, return_vars: Optional[List[str]] = None):
        """
        Initialize a tool request.

        Args:
            tool_name: Name of the tool to execute
            code: Python code to execute as a string
            return_vars: List of variable names to return from the execution
        """
        self.code = code
        self.return_vars = return_vars

    def to_dict(self) -> Dict:
        """
        Convert the tool request to a dictionary.

        Returns:
            Dictionary representation of the tool request
        """
        return {
            "code": self.code,
            "return_vars": self.return_vars,
        }

    def to_str(self) -> str:
        config = self.code + "\n"
        if self.return_vars:
            config += "Return variables: " + ", ".join(self.return_vars)
            config += "\n"
        return config


@dataclass(frozen=True)
class EvaluationResult:
    """Class to store evaluation results for an LLM-generated SBML model"""

    detailed_scores: dict

    def to_dict(self):
        return self.detailed_scores


@dataclass
class BenchmarkResult:
    """Complete results from a benchmark run"""

    model_name: str
    timestamp: str
    task_name: str
    final_score: Optional[EvaluationResult] = None
    observation_experiments_used: Optional[int] = 0
    observation_experiments_limit: Optional[int] = 0
    intervention_experiments_used: Optional[int] = 0
    intervention_experiments_limit: Optional[int] = 0
    iterations_completed: Optional[int] = 0
    experiment_history: Optional[str] = ""
    tool_history: Optional[str] = ""
    chat_history: Optional[List[str]] = None
    duration: Optional[float] = 0.0
    final_sbml: Optional[str] = ""
    score_history: Optional[list] = None

    def to_dict(self) -> Dict:
        """
        Convert the BenchmarkResult instance to a dictionary suitable for JSON serialization.

        Returns:
            Dict: Dictionary representation of the BenchmarkResult
        """
        result = {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "task_name": self.task_name,
            "observation_experiments_used": self.observation_experiments_used,
            "observation_experiments_limit": self.observation_experiments_limit,
            "intervention_experiments_used": self.intervention_experiments_used,
            "intervention_experiments_limit": self.intervention_experiments_limit,
            "chat_history": self.chat_history,
            "duration": self.duration,
            "score_history": self.score_history,
        }

        # Handle the EvaluationResult object if it exists
        if self.final_score:
            result["final_score"] = self.final_score.detailed_scores

        return result


@dataclass
class VariableStorage:
    variables_dict: dict = field(default_factory=dict)

    def add(self, variable_name, val):
        self.variables_dict[variable_name] = val

    def access(self, variable_name):
        return self.variables_dict[variable_name]

    def return_variables(self):
        return list(self.variables_dict.keys())

    def to_dict(self):
        return self.variables_dict


@dataclass
class LLMError:
    """An error message for the LLM to read"""

    message: str


@dataclass
class LLMMessage:
    """A message in the LLM conversation"""

    role: str  # "system", "user", or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RemoveReactionAction:
    """Action to remove a reaction from the SBML model"""

    reaction_id: str = field(repr=True, compare=True, hash=True)


@dataclass(frozen=True)
class RemoveSpeciesAction:
    """Action to remove a species from the SBML model"""

    species_id: str = field(repr=True, compare=True, hash=True)


@dataclass(frozen=True)
class RemoveKineticLawAction:
    """Action to remove a kinetic law from the SBML model"""

    reaction_id: str = field(repr=True, compare=True, hash=True)


CreateQuestionAction = RemoveReactionAction | RemoveSpeciesAction | RemoveKineticLawAction


@dataclass(frozen=True)
class NullifyReactionAction:
    """Action to nullify a reaction rate in the SBML model"""

    reaction_id: str = field(repr=True, compare=True, hash=True)


@dataclass(frozen=True)
class ModifyReactionAction:
    """Action to modify a reaction rate in the SBML model"""

    reaction_id: str = field(repr=True, compare=True, hash=True)
    multiply_factor: float = field(repr=True, compare=True, hash=True)


@dataclass(frozen=True)
class NullifySpeciesAction:
    """Action to nullify a species concentration in the SBML model"""

    species_id: str = field(repr=True, compare=True, hash=True)


@dataclass(frozen=True)
class ModifySpeciesAction:
    """Action to modify a species concentration in the SBML model"""

    species_id: str = field(repr=True, compare=True, hash=True)
    value: float = field(repr=True, compare=True, hash=True)


ExperimentAction = (
    NullifyReactionAction | ModifyReactionAction | NullifySpeciesAction | ModifySpeciesAction
)


@dataclass(frozen=True)
class ExperimentConstraint:
    """A constraint on which actions can be performed on the SBML reactions and species"""

    type_code: int = field(repr=True, compare=True, hash=True)
    id: str = field(repr=True, compare=True, hash=True)
    can_nullify: bool = field(repr=True, hash=False, compare=True, default=False)
    can_modify: bool = field(repr=True, hash=False, compare=True, default=False)

    def to_string(self) -> str:
        """
        Converts an ExperimentConstraint object to a readable string format.

        Args:
            constraint: An ExperimentConstraint object

        Returns:
            A string representation showing ID and capability statuses
        """
        return f"ID: {self.id}: Deactivatable: {self.can_nullify}. Modifiable: {self.can_modify}"


@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""

    experiment_action: list[str]
    observed_species: list[str]

    def to_str(self):
        return f"Experiment actions: {self.experiment_action}\n Observed species: {self.observed_species}"

    def to_dict(self):
        return {"experiment_action": self.experiment_action}


def display_dataframe_sample(df, n=50, decimal_places=3):
    """
    Returns a string representation of a DataFrame with n rows evenly distributed.
    Uses the "Time" column as row labels and hides the index.

    Parameters:
    df (pandas.DataFrame): The DataFrame to display (must contain a "Time" column)
    n (int): Number of rows to display
    decimal_places (int): Number of decimal places for floating point numbers

    Returns:
    str: A formatted string representation of the DataFrame
    """
    import pandas as pd

    # Make a copy to avoid modifying the original
    df_display = df.copy()

    # Format floating point numbers
    for col in df_display.select_dtypes(include=["float"]).columns:
        df_display[col] = df_display[col].apply(
            lambda x: "0"
            if (pd.notnull(x) and x == 0)
            else (
                f"{x:.{decimal_places}e}"
                if (pd.notnull(x) and abs(x) < 0.001)
                else (f"{x:.{decimal_places}f}" if pd.notnull(x) else x)
            )
        )

    # Get total number of rows
    total_rows = len(df_display)

    # Calculate indices to display
    indices = []
    if total_rows <= n:
        indices = list(range(total_rows))
    else:
        step = max(1, total_rows // (n - 1))  # Leave room for the last row
        for i in range(0, total_rows, step):
            indices.append(i)
            if len(indices) >= n - 1:
                break

        # Add the last row if it's not already included
        if (total_rows - 1) not in indices:
            indices.append(total_rows - 1)

    # Get the rows at the selected indices
    sample_df = df_display.iloc[indices]

    # Use the "Time" column as row labels
    if "Time" in sample_df.columns:
        row_labels = sample_df["Time"].copy()
        sample_df = sample_df.drop(columns=["Time"])

        # Convert to string representation without showing the index
        result = sample_df.to_string(index=False)

        # Insert the time column values as row labels
        lines = result.split("\n")
        header = lines[0]
        data_rows = lines[1:]

        # Add "Time" to the header row
        final_lines = [f"Time  {header}"]
        for i, row in enumerate(data_rows):
            final_lines.append(f"{row_labels.iloc[i]}  {row}")

        # Insert ellipsis between non-consecutive rows
        if total_rows > n:
            result_lines = [final_lines[0]]  # Header
            prev_idx = -1

            for i, idx in enumerate(indices):
                if i > 0 and idx > prev_idx + 1:
                    result_lines.append("...")
                result_lines.append(final_lines[i + 1])
                prev_idx = idx

            return "\n".join(result_lines)
        else:
            return "\n".join(final_lines)
    else:
        raise ValueError("DataFrame must contain a 'Time' column")


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


class ExperimentResult:
    """Results from a conducted experiment."""

    def __init__(
        self,
        time: List[float] | None = None,
        result: Dict[str, List[float]] | None = None,
        raw_result: Any = None,
        success: bool = True,
        error_message: str = "",
        expr_id: str | None = None,
    ):
        """
        Initialize experiment result.

        Args:
            time: List of time points
            result: Dictionary mapping variable names to lists of values over time
        """
        self.time = time
        self.result = result
        self.raw_result = raw_result
        self.success = success
        self.error_message = error_message
        self.expr_id = expr_id

    def set_id(self, id):
        self.expr_id = f"iteration_{id}"

    def to_string(self, precision: int = 2, max_rows: int = 50) -> str:
        """
        Convert experiment results to a simplified formatted table string.
        Shows only a subset of data with row/column counts.

        Args:
            precision: Number of decimal places to display
            max_rows: Maximum number of rows to display (beginning and end)
            max_cols: Maximum number of columns to display (including Time)

        Returns:
            Formatted table as a string with size information
        """
        if self.success:
            assert self.result is not None
            assert self.time is not None
            # Generate table using tabulate
            # Add experiment ID and data size information
            # result = f"Total size: {total_rows} rows × {total_cols} columns"
            df = create_dataframe(self.time, self.result)
            table = display_dataframe_sample(df, n=max_rows, decimal_places=precision)
            result = ""
            if self.expr_id:
                result = f"Experiment ID: {self.expr_id}\n"
                result += f"### Your requested experiment has finished. The details results are saved as a dataframe in experiment_history[{self.expr_id}]\n\n"
            num_rows, num_columns = df.shape
            result += f"Total size: {num_rows} rows × {num_columns} columns. \n\n"
            if self.expr_id:
                result += f"Below shows a subset of the table for your convenience. You should access experiment_history[{self.expr_id}] for the full data\n\n"
            else:
                result += f"Below shows a subset of the table for your convenience. You should access observed_data for the full data\n\n"
            return result + table
        else:
            return f"Your provided experiment configuration is not valid. This is the error message: {self.error_message}"

    def __eq__(self, other):
        """
        Compare if this ExperimentResult is equal to another object.

        Two ExperimentResult objects are considered equal if they have
        identical time lists and result dictionaries.

        Args:
            other: The object to compare with

        Returns:
            bool: True if equal, False otherwise
        """
        # Check if other is an ExperimentResult instance
        if not isinstance(other, ExperimentResult):
            return False

        if type(self.time) != type(other.time):
            return False

        # Check if time lists are equal
        if self.time is not None and other.time is not None:
            if len(self.time) != len(other.time):
                return False

            for t1, t2 in zip(self.time, other.time):
                if t1 != t2:
                    return False

        # Check if result dictionaries are equal
        if self.result is not None and other.result is not None:
            if set(self.result.keys()) != set(other.result.keys()):
                return False

            for key in self.result:
                if len(self.result[key]) != len(other.result[key]):
                    return False

                for v1, v2 in zip(self.result[key], other.result[key]):
                    if v1 != v2:
                        return False

        return True


class ModelProvider(ABC):
    """
    Abstract base class for model providers.
    Each LLM provider should implement this interface.
    """

    @abstractmethod
    def initialize(self, *args, **kwargs) -> None:
        """Initialize the model provider with necessary parameters"""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_response(self, user_message: str) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        """Get a response from the model and return response_text"""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_messages(self) -> List[Dict[str, str]]:
        """Get the full message history in a standard format"""
        raise NotImplementedError("Subclasses must implement this method")


class LLM(ModelProvider):
    """Base class that all LLM providers should inherit from"""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        system_prompt: str,
        temperature: float = 1.0,
        max_length=8192,
        *args,
        **kwargs,
    ):
        """
        Initialize with a model name and API key.

        Args:
            model_name: Name of the LLM to use (e.g., "claude-3-opus-20240229", "gpt-4", "gemini-pro")
            api_key: API key for the LLM service
            system_prompt: Initial system prompt for the conversation
            log_file: File to log conversations
            max_length: Maximum response length
        """
        self.model_name = model_name
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.temperature = temperature

        # Token usage tracking
        self.input_total_tokens = 0
        self.output_total_tokens = 0

        # Initialize the right provider
        self.initialize(*args, **kwargs)

    def get_message(self) -> dict:
        """
        Get the chat history organized by iteration ID.

        Returns:
            dict: A nested dictionary with iteration_id as the outer key and user/assistant messages as inner keys
        """
        messages = self.get_messages()
        chat_history = {}
        iteration_id = 0

        # Skip system message if present
        start_idx = 1 if messages and messages[0]["role"] == "system" else 0

        # Process messages in pairs (user, assistant)
        for i in range(start_idx, len(messages), 2):
            if i + 1 < len(messages):
                # Complete pair of user and assistant messages
                chat_history[f"Iteration {iteration_id}"] = {
                    "user": messages[i]["content"],
                    "assistant": messages[i + 1]["content"],
                }
                iteration_id += 1
            else:
                # Last user message without assistant response
                chat_history[f"Iteration {iteration_id}"] = {
                    "user": messages[i]["content"],
                    "assistant": None,
                }

        return chat_history

    def return_response(self, user_message: str | list) -> str:
        """
        Get a response from the LLM based on a user message and conversation history.

        Args:
            user_message: The message to send to the LLM

        Returns:
            LLM's response as a string
        """
        # Convert list to string if needed
        if isinstance(user_message, list):
            user_message = str(user_message)

        # Get response from provider
        response_text, usage_stats = self.get_response(user_message)
        return response_text
