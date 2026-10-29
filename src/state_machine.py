"""State machine module for AirDesk interaction state transitions.

Explicit state machine managing workspace interaction states:
- IDLE: No active gesture/interaction.
- DRAWING: Hand in POINT gesture, creating air stroke.
- SHAPE_CREATED: Transient state upon stroke recognition into object.
- SELECTED: Object pinched/selected by user.
- MOVING: Object currently being dragged across workspace.
- ROTATING: Object being rotated via hand orientation change.
- SCALING: Object being scaled via 2-hand distance change.
"""

from enum import Enum, auto
from typing import Optional, Tuple


class State(Enum):
    IDLE = auto()
    DRAWING = auto()
    SHAPE_CREATED = auto()
    SELECTED = auto()
    MOVING = auto()
    ROTATING = auto()
    SCALING = auto()

    def __str__(self) -> str:
        return self.name


class StateMachine:
    """Manages explicit application state transitions and active state query."""

    def __init__(self):
        self.current_state = State.IDLE
        self.active_object_id: Optional[int] = None
        self.drag_offset: Tuple[float, float] = (0.0, 0.0)
        self.initial_hand_angle: float = 0.0
        self.initial_object_rotation: float = 0.0
        self.initial_inter_hand_dist: float = 1.0
        self.initial_object_scale: float = 1.0

    def transition_to(self, new_state: State, object_id: Optional[int] = None) -> bool:
        """Attempt transition to new_state and update active context."""
        old_state = self.current_state
        self.current_state = new_state
        if object_id is not None:
            self.active_object_id = object_id
        elif new_state == State.IDLE:
            self.active_object_id = None

        if old_state != new_state:
            print(f"[StateMachine] State transition: {old_state.name} -> {new_state.name} (Active Obj: {self.active_object_id})")
            return True
        return False

    def is_idle(self) -> bool:
        return self.current_state == State.IDLE

    def is_drawing(self) -> bool:
        return self.current_state == State.DRAWING

    def is_selected(self) -> bool:
        return self.current_state in (State.SELECTED, State.MOVING, State.ROTATING, State.SCALING)

    def reset_to_idle(self):
        """Reset state machine context to IDLE."""
        self.transition_to(State.IDLE)
        self.drag_offset = (0.0, 0.0)
        self.initial_hand_angle = 0.0
        self.initial_object_rotation = 0.0
        self.initial_inter_hand_dist = 1.0
        self.initial_object_scale = 1.0
