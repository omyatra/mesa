from pathlib import Path
from abc import ABC, abstractmethod
from typing import Literal, Any


# "stalled": the plug-in has stopped trying to converge the run; it is
# scored like a complete run, and marked "stalled" in the evaluations file
RunStatus = Literal["running", "complete", "stalled", "timed-out", "crashed"]


class SimulationRun(ABC):
    parameters: dict[str, float]
    directory: Path
    run_number: int
    launch_time: float
    # the iteration the run was launched in; must be set on adopted runs
    # (see Simulation.adopt)
    iteration: int | None = None
    # optional: the scheduler's ID for the run's current job, if there is
    # one; shown in the heartbeat file
    job_id: str | None = None

    @abstractmethod
    def status(self) -> RunStatus:
        pass

    @abstractmethod
    def cleanup(self):
        pass

    @abstractmethod
    def cancel(self):
        pass

    @abstractmethod
    def get_results(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def __hash__(self):
        pass


class Simulation(ABC):

    @abstractmethod
    def launch(
        self,
        run_number: int,
        iteration: int,
        simulations_directory: Path,
        parameters: dict,
        **kwargs,
    ) -> SimulationRun:
        """
        Launches one run and returns it.

        The engine passes every argument by keyword. Implementations should
        accept **kwargs, so that the engine can pass more information in
        later versions without breaking existing plug-ins; arguments a
        plug-in doesn't use can be ignored.
        """
        pass

    def existing_run_numbers(self, simulations_directory: Path, **kwargs) -> set[int]:
        """
        Optional, for resuming after a driver restart: the run numbers that
        have a folder in the simulations directory, whether or not they have
        finished. The engine uses them to find runs to adopt, and so that new
        run numbers don't clash with existing folders. Default: none.
        """
        return set()

    def adopt(
        self, run_number: int, simulations_directory: Path, **kwargs
    ) -> SimulationRun | None:
        """
        Optional, for resuming after a driver restart: rebuilds the run with
        this number from what it saved when it was launched, or returns None
        if it can't (e.g. it was never submitted). The returned run must have
        its parameters and iteration set; the engine then polls it with
        status() like any other run. Called for each number from
        existing_run_numbers() that has no row in the evaluations file.
        Default: None.
        """
        return None

    # def create_case_directory(
    #     self,
    #     run_number: int,
    #     simulations_directory: Path,
    #     input_files: list[str],
    #     filename_map: dict[str, str] = None,
    # ):
    #     case_dir = simulations_directory / f"run_{run_number}"
    #     self.simulations_directory = simulations_directory
    #     filename_map = {} if filename_map is None else filename_map
    #     # create the case directory and copy all the reference files
    #     case_dir.mkdir()
    #     for file_name in input_files:
    #         f = self.simulations_directory / file_name
    #         if f.is_file():
    #             case_name = filename_map.get(file_name, file_name)
    #             subprocess.run(["cp", f, case_dir / case_name])
