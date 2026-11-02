from pathlib import Path
from numpy import nan
from pandas import DataFrame, read_hdf, concat
from time import sleep
from datetime import datetime
import json
import os
import socket
import logging

from mesa.simulations import Simulation, SimulationRun
from mesa.strategies import Strategy
from mesa.objectives import ObjectiveFunction
from mesa.input import parse_input_module


class Mesa:
    def __init__(
        self,
        parameters: dict[str, tuple[float, float] | float],
        simulation: Simulation,
        objective_function: ObjectiveFunction,
        strategy: Strategy,
        max_concurrent_runs: int,
        max_iterations: int,
        evaluations_filepath: Path,
        simulations_directory: Path,
        poll_interval: float = 1800.0,
    ):
        """
        max_iterations is the last iteration number to run. Iterations are
        numbered from 0 (with GPOptimizer, iteration 0 is the initial random
        batch), so up to max_iterations + 1 iterations are run.
        """
        self.parameters = parameters
        self.simulation = simulation
        self.objective_function = objective_function
        self.strategy = strategy
        self.max_concurrent_runs = max_concurrent_runs
        self.max_iterations = max_iterations
        # seconds between checks of the runs in flight
        self.poll_interval = poll_interval

        assert evaluations_filepath.is_file() or not evaluations_filepath.exists()
        assert simulations_directory.is_dir()
        self.evaluations_filepath = evaluations_filepath
        self.simulations_directory = simulations_directory
        # rewritten after each pass: whether the driver is alive and what it
        # is waiting on
        self.heartbeat_filepath = simulations_directory / "heartbeat.json"
        # the current iteration's points not yet launched, so that a restart
        # can carry on the iteration
        self.iteration_filepath = simulations_directory / "iteration.json"
        self.converged = False

        # split the parameters up into those which have free or fixed values
        self.free_parameter_keys = []
        self.optimization_bounds = {}
        self.fixed_parameters = {}
        for param, value in self.parameters.items():
            if isinstance(value, tuple):
                self.free_parameter_keys.append(param)
                self.optimization_bounds[param] = value
            else:
                self.fixed_parameters[param] = value

        parameter_keys = [k for k in self.parameters.keys()]
        objective_key = self.objective_function.name

        self.metadata_keys = ["run_number", "iteration"]
        # status of the finished run: "complete", "stalled", "crashed" or "timed-out"
        self.data_columns = [*self.metadata_keys, "status", objective_key, *parameter_keys]
        self.data_types = {
            **{k: "int64" for k in self.metadata_keys},
            "status": "object",
            **{objective_key: "float64"},
            **{k: "float64" for k in parameter_keys}
        }

        if not self.evaluations_filepath.is_file():
            self.__initialize_evaluations_data_file()

    def run(self):
        """
        Runs the optimisation. An error that is not handled for a single run
        (evaluation data file, optimiser, settings, anything unexpected) is
        logged with the runs still in flight, then raised: the driver stops,
        and the jobs are not cancelled, so they can be picked up again.

        The log goes to mesa.log in the simulations directory, and to the
        screen.

        heartbeat.json in the simulations directory is rewritten at start-up
        and after each pass (state "running"), and once more when the driver
        stops (state "stopped"). The driver refuses to start if it finds a
        "running" heartbeat younger than 2 x poll_interval: another driver
        may be working on the same case.

        iteration.json in the simulations directory holds the current
        iteration's points that have not been launched yet. On a restart,
        once the runs left in flight have finished, those points are
        launched under the same iteration number, so a stopped driver does
        not use up an iteration.
        """
        logger_setup(self.simulations_directory)
        self.__check_heartbeat()
        self.runs_in_flight: set[SimulationRun] = set()
        # the last status of each run in flight, for the heartbeat
        self.run_statuses: dict[SimulationRun, str] = {}
        self.current_iteration: int | None = None
        self.__write_heartbeat("running")
        stopped_by = "interrupted"  # e.g. Ctrl-C
        try:
            self.__run()
            stopped_by = "finished"
        except Exception as error:
            stopped_by = f"error: {error!r}"
            logging.exception("[ MESA error ] driver stopping on an unexpected error")
            if self.runs_in_flight:
                logging.error(
                    "runs still in flight (not cancelled): "
                    + ", ".join(
                        f"run {run.run_number}: {run.directory}"
                        for run in sorted(self.runs_in_flight, key=lambda r: r.run_number)
                    )
                )
            else:
                logging.error("no runs in flight")
            raise
        finally:
            self.__mark_heartbeat_stopped(stopped_by)

    def __run(self):
        # runs left in flight by an earlier driver are finished first
        self.__finish_adopted_runs()
        # then the points of that iteration it had not launched
        self.__resume_iteration()

        while not self.converged:
            df: DataFrame = read_hdf(self.evaluations_filepath)
            # get the current iteration number
            initial_run_number = self.__next_run_number(df)
            iteration = 0 if df.empty else df["iteration"].max() + 1

            if iteration > self.max_iterations:
                logging.info("maximum iterations reached without convergence")
                break
            logging.info(f"--- Starting iteration {iteration} ---")

            # get next set of points for this iteration
            new_free_params = self.strategy.propose_evaluations(
                evaluation_data=df,
                optimization_bounds=self.optimization_bounds,
                objective_name=self.objective_function.name
            )
            # join fixed / free parameters to get the full set
            new_points = [f | self.fixed_parameters for f in new_free_params]

            self.launch_iteration(
                iteration=iteration,
                initial_run_number=initial_run_number,
                pending_points=new_points,
            )

    def __next_run_number(self, df: DataFrame) -> int:
        # past every recorded run and every run folder
        run_folders = self.__existing_run_numbers()
        return max(
            0 if df.empty else int(df["run_number"].max()) + 1,
            max(run_folders, default=-1) + 1,
        )

    def __resume_iteration(self):
        """
        Launches the points an earlier driver proposed but had not launched
        when it stopped (from iteration.json), under their own iteration
        number. A missing file, or one with no points left, means there is
        nothing to resume; an unreadable file or one that doesn't match the
        evaluations file is logged and ignored, and the driver carries on
        with the next iteration.
        """
        if not self.iteration_filepath.exists():
            return
        try:
            saved = json.loads(self.iteration_filepath.read_text())
            iteration = int(saved["iteration"])
            points = [dict(point) for point in saved["pending_points"]]
        except Exception as error:
            logging.warning(
                f"can't read {self.iteration_filepath} ({error!r}); its points "
                f"are not resumed"
            )
            return
        if not points:
            return

        df: DataFrame = read_hdf(self.evaluations_filepath)
        last_iteration = -1 if df.empty else int(df["iteration"].max())
        if iteration not in (last_iteration, last_iteration + 1):
            logging.warning(
                f"{self.iteration_filepath} is for iteration {iteration}, but the "
                f"last recorded iteration is {last_iteration}; its points are "
                f"not resumed"
            )
            return
        if any(set(point) != set(self.parameters) for point in points):
            logging.warning(
                f"the parameters in {self.iteration_filepath} don't match the "
                f"input; its points are not resumed"
            )
            return

        logging.info(
            f"--- Resuming iteration {iteration}: {len(points)} points not "
            f"launched by an earlier driver ---"
        )
        self.launch_iteration(
            iteration=iteration,
            initial_run_number=self.__next_run_number(df),
            pending_points=points,
        )

    def __write_iteration_file(self, iteration: int, pending_points: list[dict]):
        # tmp file + rename, as for the heartbeat; numpy numbers are written
        # as plain numbers
        def plain(value):
            if hasattr(value, "item"):
                return value.item()
            raise TypeError(f"can't write {value!r} to {self.iteration_filepath}")

        tmp_filepath = self.iteration_filepath.with_name(
            self.iteration_filepath.name + ".tmp"
        )
        tmp_filepath.write_text(
            json.dumps(
                {"iteration": int(iteration), "pending_points": pending_points},
                indent=2,
                default=plain,
            )
            + "\n"
        )
        os.replace(tmp_filepath, self.iteration_filepath)

    def __existing_run_numbers(self) -> set[int]:
        return {
            int(n)
            for n in self.simulation.existing_run_numbers(
                simulations_directory=self.simulations_directory
            )
        }

    def __finish_adopted_runs(self):
        """
        Re-adopts the runs that have a folder but no row in the evaluations
        file (left in flight when an earlier driver stopped), and polls them
        with the normal status handling until all have finished and been
        recorded, each with its own iteration. A folder that can't be
        adopted is logged as needing attention and skipped.
        """
        df: DataFrame = read_hdf(self.evaluations_filepath)
        recorded = {int(n) for n in df["run_number"]}
        unrecorded = sorted(self.__existing_run_numbers() - recorded)
        if not unrecorded:
            return

        adopted: set[SimulationRun] = set()
        for run_number in unrecorded:
            try:
                run = self.simulation.adopt(
                    run_number=run_number,
                    simulations_directory=self.simulations_directory,
                )
            except Exception as error:
                logging.warning(
                    f"run {run_number}: adopting the run failed ({error!r})",
                    exc_info=True,
                )
                run = None
            if run is not None and getattr(run, "iteration", None) is None:
                logging.warning(
                    f"run {run_number}: the adopted run has no iteration number"
                )
                run = None
            if run is None:
                logging.warning(
                    f"run {run_number}: folder with no evaluation row can't be "
                    f"adopted, needs attention; skipped"
                )
                continue
            logging.info(
                f"run {run_number}: adopted from an earlier driver "
                f"(iteration {run.iteration}): {run.directory}"
            )
            adopted.add(run)

        if not adopted:
            return
        logging.info(f"--- Finishing {len(adopted)} adopted runs ---")
        # kept on self so that run() can log them on an error, and for the
        # heartbeat (no single iteration: each run has its own)
        self.runs_in_flight = adopted
        self.run_statuses = {run: "adopted" for run in adopted}
        self.current_iteration = None
        status_failures: dict[SimulationRun, int] = {}
        while adopted:
            for run in [run for run in adopted]:
                if self.__check_run(run, run.iteration, status_failures):
                    adopted.remove(run)
            self.__write_heartbeat("running")
            if adopted:
                sleep(self.poll_interval)

    def launch_iteration(
        self, iteration: int, initial_run_number: int, pending_points: list[dict]
    ):
        """
        Launches and manages all the simulations runs required to complete
        the current iteration.
        """
        # the runs in flight; kept on self so that run() can log them on an
        # error, and for the heartbeat
        current_runs: set[SimulationRun] = set()
        self.runs_in_flight = current_runs
        self.run_statuses = {}
        self.current_iteration = iteration
        completed_runs: set[SimulationRun] = set()
        # consecutive failed status checks, per run
        status_failures: dict[SimulationRun, int] = {}
        total_requested_runs = len(pending_points)
        pending_run_numbers = set(
            range(initial_run_number, initial_run_number + total_requested_runs)
        )
        # the points not launched yet, kept on disk for a restart
        self.__write_iteration_file(iteration, pending_points)

        # main monitoring loop for the simulation runs: each pass checks the
        # runs in flight, then fills the free slots, then waits
        while len(completed_runs) < total_requested_runs:
            # we can't modify the current_runs set while we're iterating over it,
            # so make a copy which we can iterate over
            current_runs_iterable = [run for run in current_runs]
            # loop through all currently running jobs to check if
            # they have finished or timed-out
            run: SimulationRun
            for run in current_runs_iterable:
                if self.__check_run(run, iteration, status_failures):
                    # the run is recorded, so we can stop tracking it
                    current_runs.remove(run)
                    completed_runs.add(run)

            # if we are not at the maximum allowed number of concurrent runs
            # then launch enough to bring us to the maximum
            available_runs = min(
                self.max_concurrent_runs - len(current_runs), len(pending_run_numbers)
            )
            if available_runs:
                runs_to_launch = [
                    pending_run_numbers.pop() for _ in range(available_runs)
                ]

                for run_number in runs_to_launch:
                    point = pending_points.pop()
                    logging.info(f"Run number {run_number} - New parameters:")
                    logging.info([point[k] for k in self.free_parameter_keys])

                    new_run = self.simulation.launch(
                        run_number=run_number,
                        iteration=iteration,
                        simulations_directory=self.simulations_directory,
                        parameters=point,
                    )
                    current_runs.add(new_run)
                    self.run_statuses[new_run] = "launched"
                    # after the launch: a driver stopped in between launches
                    # this point again on a restart, rather than losing it
                    self.__write_iteration_file(iteration, pending_points)

            self.__write_heartbeat("running")

            # wait before the next check, unless the iteration is done
            if len(completed_runs) < total_requested_runs:
                sleep(self.poll_interval)

    def __check_run(
        self,
        run: SimulationRun,
        iteration: int,
        status_failures: dict[SimulationRun, int],
    ) -> bool:
        """
        Checks one run in flight. If it has finished, records it in the
        evaluations file (with the given iteration) and returns True;
        otherwise returns False. status_failures counts consecutive failed
        status checks per run.
        """
        try:
            run_status = run.status()
        except Exception as error:
            # e.g. an output file read mid-write, or a scheduler or
            # file-system hiccup: treat the run as running, retry next poll
            failures = status_failures.get(run, 0) + 1
            status_failures[run] = failures
            self.run_statuses[run] = "status check failed"
            logging.warning(
                f"run {run.run_number}: status check failed ({error!r}), "
                f"retrying next poll",
                exc_info=failures == 1,  # traceback once per streak
            )
            if failures == 3:
                logging.warning(
                    f"run {run.run_number}: status check failing, needs attention"
                )
            return False
        status_failures.pop(run, None)
        self.run_statuses[run] = run_status

        if run_status in ("complete", "stalled"):
            # get the objective function value
            try:
                objective_values = self.objective_function.evaluate(
                    simulation_results=run.get_results()
                )
            except Exception as error:
                # a bad result from this run: count it as crashed and
                # carry on with the others (its folder is kept)
                logging.warning(
                    f"run {run.run_number}: scoring failed ({error!r}), "
                    f"counted as crashed; folder kept: {run.directory}",
                    exc_info=True,
                )
                self.__add_evaluation(run, iteration, "crashed")
                return True

            self.__add_evaluation(run, iteration, run_status, objective_values)
            # clean up the run directory
            run.cleanup()
            return True

        elif run_status == "crashed":
            logging.info("[ crash warning ]")
            logging.info(
                f">> run #{run.run_number} has crashed; folder kept: {run.directory}"
            )
            # currently no provision for re-starting runs, so it is finished
            self.__add_evaluation(run, iteration, "crashed")
            return True

        elif run_status == "timed-out":
            logging.info("[ time-out warning ]")
            logging.info(
                f">> iteration {run.run_number} has timed-out"
            )
            run.cancel()  # cancel the timed-out job
            # currently no provision for re-starting runs, so it is finished
            self.__add_evaluation(run, iteration, "timed-out")
            return True

        return False

    def __add_evaluation(
        self,
        run: SimulationRun,
        iteration: int,
        status: str,
        objective_values: dict[str, float] | None = None,
    ):
        # add a row for a finished run to the evaluations file; a run with no
        # score (crashed or timed-out) gets NaN as its objective value
        new_row = {"run_number": run.run_number, "iteration": iteration, "status": status}
        if objective_values is None:
            new_row[self.objective_function.name] = nan
        else:
            new_row.update(objective_values)
        new_row.update(run.parameters)

        # build a dataframe for the latest evaluation
        new_df = DataFrame(new_row, index=[run.run_number])
        new_df = new_df.astype(self.data_types)

        # update the evaluation data with the new dataframe
        df: DataFrame = read_hdf(self.evaluations_filepath)
        df = new_df if df.empty else concat([df, new_df])
        df.sort_index(inplace=True)
        self.__write_evaluations_data(df)

    def __heartbeat(self, state: str) -> dict:
        runs = sorted(self.runs_in_flight, key=lambda r: r.run_number)
        return {
            "state": state,
            "time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "iteration": (
                None if self.current_iteration is None else int(self.current_iteration)
            ),
            "runs_in_flight": [
                {
                    "run_number": int(run.run_number),
                    "job_id": run.job_id,
                    "status": self.run_statuses.get(run),
                }
                for run in runs
            ],
        }

    def __write_heartbeat(self, state: str, stopped_by: str | None = None):
        # tmp file + rename, so a reader never sees a partly written file
        heartbeat = self.__heartbeat(state)
        if stopped_by is not None:
            heartbeat["stopped_by"] = stopped_by
        tmp_filepath = self.heartbeat_filepath.with_name(
            self.heartbeat_filepath.name + ".tmp"
        )
        tmp_filepath.write_text(json.dumps(heartbeat, indent=2) + "\n")
        os.replace(tmp_filepath, self.heartbeat_filepath)

    def __read_heartbeat(self) -> dict | None:
        # None if there is no heartbeat file; raises if it can't be read
        if not self.heartbeat_filepath.exists():
            return None
        return json.loads(self.heartbeat_filepath.read_text())

    def __check_heartbeat(self):
        # refuse to start if another driver may be working on this case
        try:
            heartbeat = self.__read_heartbeat()
        except Exception as error:
            message = (
                f"[ MESA error ] can't read {self.heartbeat_filepath} ({error!r}); "
                f"if no other driver is running on this case, delete it and start again"
            )
            logging.error(message)
            raise RuntimeError(message) from error
        if heartbeat is None:
            return

        host, pid = heartbeat.get("host"), heartbeat.get("pid")
        last_seen = heartbeat.get("time")
        if heartbeat.get("state") == "stopped":
            logging.info(
                f"previous driver (host {host}, PID {pid}) stopped at {last_seen}: "
                f"{heartbeat.get('stopped_by')}"
            )
            return

        try:
            age = (
                datetime.now().astimezone() - datetime.fromisoformat(last_seen)
            ).total_seconds()
        except Exception:
            age = 0.0  # no usable time: assume it may be alive
        if age < 2 * self.poll_interval:
            message = (
                f"[ MESA error ] another driver may be running (host {host}, "
                f"PID {pid}, last seen {last_seen}); if not, delete "
                f"{self.heartbeat_filepath} and start again."
            )
            logging.error(message)
            raise RuntimeError(message)
        logging.warning(
            f"old heartbeat from a driver that did not stop cleanly (host {host}, "
            f"PID {pid}, last seen {last_seen}); taking over"
        )

    def __mark_heartbeat_stopped(self, stopped_by: str):
        # rewrite the heartbeat as stopped, unless it now belongs to another
        # driver; a failure here is logged, so it can't hide the original error
        try:
            heartbeat = self.__read_heartbeat()
            if heartbeat is not None and (
                heartbeat.get("host") != socket.gethostname()
                or heartbeat.get("pid") != os.getpid()
            ):
                logging.warning(
                    f"{self.heartbeat_filepath} now belongs to another driver "
                    f"(host {heartbeat.get('host')}, PID {heartbeat.get('pid')}); "
                    f"left as it is"
                )
                return
            self.__write_heartbeat("stopped", stopped_by=stopped_by)
        except Exception:
            logging.exception(f"could not mark {self.heartbeat_filepath} as stopped")

    def __initialize_evaluations_data_file(self):
        # create the empty dataframe to store the evaluation data and save it to HDF
        df = DataFrame(columns=self.data_columns)
        df = df.astype(self.data_types)
        self.__write_evaluations_data(df)

    def __write_evaluations_data(self, df: DataFrame):
        # write to a tmp file in the same folder, then rename it onto the
        # evaluations file: os.replace is atomic on the same file system, so
        # the file is always either the old or the new version, never a
        # partly written one
        tmp_filepath = self.evaluations_filepath.with_name(
            self.evaluations_filepath.name + ".tmp"
        )
        df.to_hdf(tmp_filepath, key="evaluations", mode="w")
        os.replace(tmp_filepath, self.evaluations_filepath)

    @classmethod
    def build_from_input_module(cls, input_module: Path):
        mesa_inputs = parse_input_module(input_module)
        return cls(
            parameters = mesa_inputs.parameters,
            simulation = mesa_inputs.simulation,
            objective_function = mesa_inputs.objective_function,
            strategy = mesa_inputs.strategy,
            max_concurrent_runs = mesa_inputs.max_concurrent_runs,
            max_iterations = mesa_inputs.max_iterations,
            evaluations_filepath = mesa_inputs.evaluations_filepath,
            simulations_directory = mesa_inputs.simulations_directory,
            poll_interval = mesa_inputs.poll_interval,
        )


def logger_setup(simulations_directory: Path):
    """
    Sends the log (INFO and above, from every logger) to mesa.log in the
    simulations directory, appended to across restarts, and to the screen,
    both with timestamps. Calling it again for the same folder adds nothing.
    """
    log_filepath = (Path(simulations_directory) / "mesa.log").resolve()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == log_filepath
        ):
            return  # already set up

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_filepath, mode="a", encoding="utf-8")
    # write to the screen as well
    screen_handler = logging.StreamHandler()
    for handler in (file_handler, screen_handler):
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
