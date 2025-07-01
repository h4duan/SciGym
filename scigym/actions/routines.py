import tempfile

import pandas as pd
import roadrunner as rr

from scigym.data import SBML, Simulator


def create_simulate_function(sedml_file_path: str):
    def simulate(sbml_string: str) -> pd.DataFrame:
        """
        Simulates an SBML model and returns time series data for all observable species and parameters

        This function creates an SBML object from the provided string, then uses the
        Simulator class to run a time-course simulation. It returns a

        Args:
            sbml_string (str): A string containing the SBML model in XML format.

        Returns:
            pandas DataFrame: A DataFrame containing the simulation results, with time as one of the columns.
        """
        temp_log_file = tempfile.NamedTemporaryFile(suffix=".log")
        try:
            sbml = SBML(sbml_string, sedml_file_path)  # type: ignore
            simulation = Simulator(sbml)

            rr.Logger.setLevel(rr.Logger.LOG_ERROR)
            rr.Logger_disableConsoleLogging()
            rr.Logger.enableFileLogging(temp_log_file.name, rr.Logger.LOG_ERROR)
            data = simulation.run(
                observed_species=sbml.get_species_ids(),
                # observed_parameters=sbml.get_parameter_ids(),
                rm_concentration_brackets=True,
            )
            rr.Logger.setLevel(rr.Logger.LOG_CURRENT)

            assert data.result is not None
            data = {
                **data.result,
                "Time": data.time,
            }
            df = pd.DataFrame.from_dict(data)
            return df
        except Exception as e:
            error_message = (
                f"We could not run simulations on your SBML model. This is the error: {str(e)}"
            )
            with open(temp_log_file.name, "r") as f:
                log_contents = f.read()
                if log_contents:
                    error_message += f". Log file contents: {log_contents}"
            temp_log_file.close()
            raise Exception(error_message)

    return simulate
