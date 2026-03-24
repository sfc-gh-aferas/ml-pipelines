"""Task wrapper for Snowflake DAG task execution.

Provides utilities for resolving predecessor task return values and dynamically
executing Python modules within Snowflake DAG tasks. Used as the common entrypoint
for both ML Job (container) and direct function execution modes. Each task's
target module and dependencies are resolved at runtime from the DAG-level config.

Can also be run as a standalone script for ML Job entrypoints via:
    python task_wrapper.py --filename <module>
"""
import importlib
from argparse import ArgumentParser
import sys
import os
from snowflake.snowpark import Session
from snowflake.core.task.context import TaskContext
from typing import Union
import json

def _get_return_vals(task_context: TaskContext, return_from_tasks:list=[], script_args: bool = False) -> Union[list,dict]:
    """
    Retrieve and parse return values from predecessor tasks in a DAG.
    
    Extracts return values from specified predecessor tasks and formats them
    either as command-line arguments (for scripts) or as a dictionary (for functions).
    Handles empty or None results gracefully by skipping them.
    
    Args:
        task_context (TaskContext): Current task execution context
        return_from_tasks (list): List of predecessor task names to get values from
        script_args (bool): If True, format as CLI args (--key value); if False, return as dict
    
    Returns:
        Union[list, dict]: Either a list of CLI arguments ['--key1', 'val1', '--key2', 'val2']
                          or a merged dictionary {'key1': 'val1', 'key2': 'val2'}
    
    Note:
        TODO: Add validation for return value format and task existence
    """

    kw = [] if script_args else {}
    for task in return_from_tasks:
        result = task_context.get_predecessor_return_value(task).replace("'",'"')
        if result:
            try:
                val = json.loads(result)
            except json.JSONDecodeError:
                raise ValueError(f"Return value from task {task} is not a dictionary")
            if not isinstance(val, dict):
                raise ValueError(f"Return value from task {task} is not a dictionary")
            if script_args:
                kw += [i for k,v in val.items() for i in ("--"+str(k),str(v))]
            else:
                kw.update(val)         
    return kw

def task_func(session:Session) -> dict:
    """
    Execute the current DAG task's target module using config-driven resolution.
    
    Looks up the current task's name from the TaskContext, then reads the DAG-level
    config to determine which Python module to import and which predecessor tasks
    to pull return values from. Imports the target module dynamically and calls its
    main() function with the Snowflake session and any predecessor parameters.
    
    Args:
        session (Session): Active Snowflake session provided by the DAG runtime
    
    Returns:
        dict: Return value from the target module's main() function, also stored
              as the task's return value for downstream tasks via TaskContext
    """
    ctx = TaskContext(session)

    name = ctx.get_current_task_short_name()
    return_from_tasks = json.loads(ctx.get_task_graph_config()[f"{name}_dep"])

    # Get parameters as dictionary for **kwargs
    params = _get_return_vals(task_context=ctx, return_from_tasks=return_from_tasks)

    # Dynamically import and execute the module's main function
    mod_name = os.path.splitext(ctx.get_task_graph_config()[f"{name}_file"])[0]
    module = importlib.import_module(mod_name)

    results = module.main(session=session, **params)
    # Store results for downstream tasks
    results = results if results else ""
    ctx.set_return_value(results)

    return results


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--filename")
    args = parser.parse_args()
    
    session = Session.builder.getOrCreate()

    results = task_func(session)

    __return__ = results
