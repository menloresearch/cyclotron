# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Get-up MDP terms.

Re-exports, in order: Isaac Lab ``envs.mdp`` and the walking task's terms (via ``locomotion.mdp``), the actuator API
(``set_effort_scale``, ``ankle_motor_torques``, ...), the reset/limp API (``reset_fallen_state``, ``set_limp``,
``update_category_probs``, ``GETUP_CATEGORIES``), and the get-up env terms (state, actions, assist, observations,
rewards, terminations, tracker, curricula).
"""

from isaac_asimov.tasks.locomotion.mdp import *  # noqa: F401, F403

# actuator API -- guarded: `tasks/` is auto-imported for every task, so a missing actuator symbol must not break imports
try:
    from isaac_asimov.assets.robots.getup_actuators import (  # noqa: F401
        ankle_motor_torques,
        set_effort_scale,
        set_gain_scale,
    )
except ImportError:  # pragma: no cover
    pass

# reset / limp API
from .limp import *  # noqa: F401, F403
from .resets import *  # noqa: F401, F403

# get-up env terms
from .state import *  # noqa: F401, F403
from .actions import *  # noqa: F401, F403
from .assist import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
from .tracker import *  # noqa: F401, F403
from .curriculums import *  # noqa: F401, F403
