"""Custom exceptions"""


class ParseExperimentActionError(Exception):
    """Exception raised when there is an error in parsing an action."""


class ApplyExperimentActionError(Exception):
    """Exception raised when there is an error in applying an action."""
