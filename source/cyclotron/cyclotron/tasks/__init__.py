"""Asimov-1 tasks."""

from isaaclab_tasks.utils import import_packages

# The MDP terms load with the task configs that use them, not when the tasks are registered.
import_packages(__name__, blacklist_pkgs=[".mdp"])
