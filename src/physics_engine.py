"""Physics engine module for AirDesk using Pymunk.

Provides 2D rigid body simulation, downward gravity, momentum transfer ("throwing"),
elastic wall/floor bouncing, and object-to-object collisions.
"""

from typing import Dict, List, Tuple, Optional
import time
import math
import pymunk

from object_manager import WorkspaceObject
from shape_recognizer import ShapeType

# Constants for physics simulation
DEFAULT_GRAVITY = (0.0, 600.0)    # Downward gravity (px/s^2)
DEFAULT_ELASTICITY = 0.70         # Bounciness coefficient [0..1]
DEFAULT_FRICTION = 0.50           # Surface friction
THROW_VELOCITY_SCALE = 1.25       # Multiplier for hand momentum throw velocity


class PhysicsEngine:
    """Integrates Pymunk 2D rigid body physics simulation with AirDesk workspace objects."""

    def __init__(self, viewport_size: Tuple[int, int] = (1280, 720), gravity: Tuple[float, float] = DEFAULT_GRAVITY):
        self.viewport_w, self.viewport_h = viewport_size
        self.enabled: bool = True

        # Initialize Pymunk Space
        self.space = pymunk.Space()
        self.space.gravity = gravity

        # Storage mapping obj.id -> pymunk Body and Shape
        self.bodies: Dict[int, pymunk.Body] = {}
        self.shapes: Dict[int, pymunk.Shape] = {}
        self.static_boundaries: List[pymunk.Shape] = []

        self.setup_boundaries()

    def setup_boundaries(self, margin: float = 10.0):
        """Create static boundary segments along frame edges (floor, walls, ceiling)."""
        w, h = self.viewport_w, self.viewport_h
        
        # Remove existing boundaries if any
        for shape in self.static_boundaries:
            if shape in self.space.shapes:
                self.space.remove(shape)
        self.static_boundaries.clear()

        # Define 4 boundary line segments
        segment_points = [
            ((margin, h - margin), (w - margin, h - margin)), # Floor
            ((margin, margin), (margin, h - margin)),         # Left Wall
            ((w - margin, margin), (w - margin, h - margin)), # Right Wall
            ((margin, margin), (w - margin, margin))          # Ceiling
        ]

        for p1, p2 in segment_points:
            segment = pymunk.Segment(self.space.static_body, p1, p2, 5.0)
            segment.elasticity = DEFAULT_ELASTICITY
            segment.friction = DEFAULT_FRICTION
            self.space.add(segment)
            self.static_boundaries.append(segment)

    def register_object(self, obj: WorkspaceObject):
        """Register or update a WorkspaceObject in the Pymunk physics space."""
        if obj.id in self.bodies:
            return  # Already registered

        mass = 1.0
        w, h = (obj.size[0] * obj.scale), (obj.size[1] * obj.scale)

        if obj.shape_type == ShapeType.CIRCLE:
            radius = max(w, h) / 2.0
            inertia = pymunk.moment_for_circle(mass, 0, radius)
            body = pymunk.Body(mass, inertia)
            shape = pymunk.Circle(body, radius)
        else: # RECTANGLE, TRIANGLE, LINE, UNRECOGNIZED
            inertia = pymunk.moment_for_box(mass, (w, h))
            body = pymunk.Body(mass, inertia)
            shape = pymunk.Poly.create_box(body, (w, h))

        body.position = obj.position
        body.angle = math.radians(obj.rotation)
        shape.elasticity = DEFAULT_ELASTICITY
        shape.friction = DEFAULT_FRICTION

        self.space.add(body, shape)
        self.bodies[obj.id] = body
        self.shapes[obj.id] = shape

    def unregister_object(self, obj_id: int):
        """Remove object from Pymunk physics space."""
        if obj_id in self.bodies:
            body = self.bodies.pop(obj_id)
            shape = self.shapes.pop(obj_id)
            if shape in self.space.shapes:
                self.space.remove(shape)
            if body in self.space.bodies:
                self.space.remove(body)

    def sync_held_object(self, obj: WorkspaceObject, pinch_pos: Tuple[float, float], velocity: Tuple[float, float]):
        """Update object position kinematically while held by hand PINCH."""
        if obj.id not in self.bodies:
            self.register_object(obj)

        body = self.bodies[obj.id]
        body.position = pinch_pos
        body.velocity = velocity
        body.body_type = pymunk.Body.KINEMATIC

    def release_object_with_throw(self, obj: WorkspaceObject, hand_velocity: Tuple[float, float]):
        """Convert object to DYNAMIC rigid body and impart hand velocity momentum for 'throwing'."""
        if obj.id not in self.bodies:
            self.register_object(obj)

        body = self.bodies[obj.id]
        body.body_type = pymunk.Body.DYNAMIC
        
        # Apply scaled hand throw velocity
        vx = hand_velocity[0] * THROW_VELOCITY_SCALE
        vy = hand_velocity[1] * THROW_VELOCITY_SCALE
        body.velocity = (vx, vy)
        obj.velocity = (vx, vy)
        print(f"[PhysicsEngine] Threw Object #{obj.id} with velocity ({int(vx)}, {int(vy)}) px/s")

    def step(self, dt: float, objects: List[WorkspaceObject]):
        """Advance Pymunk physics simulation and synchronize position/rotation back to objects."""
        if not self.enabled or dt <= 0:
            return

        # Ensure all active objects are registered
        active_ids = set()
        for obj in objects:
            active_ids.add(obj.id)
            if obj.id not in self.bodies:
                self.register_object(obj)

        # Unregister destroyed objects
        to_remove = [obj_id for obj_id in self.bodies if obj_id not in active_ids]
        for obj_id in to_remove:
            self.unregister_object(obj_id)

        # Step Pymunk simulation
        self.space.step(min(dt, 0.05))

        # Sync updated Pymunk positions & rotations back to WorkspaceObject instances
        for obj in objects:
            if not obj.selected and obj.id in self.bodies:
                body = self.bodies[obj.id]
                bx, by = float(body.position.x), float(body.position.y)
                bvx, bvy = float(body.velocity.x), float(body.velocity.y)

                # Guard against NaN/Inf from physics instability
                if not (math.isfinite(bx) and math.isfinite(by)):
                    # Reset to a safe position
                    bx = max(50.0, min(self.viewport_w - 50.0, obj.position[0] if math.isfinite(obj.position[0]) else self.viewport_w / 2.0))
                    by = max(50.0, min(self.viewport_h - 50.0, obj.position[1] if math.isfinite(obj.position[1]) else self.viewport_h / 2.0))
                    body.position = (bx, by)
                    body.velocity = (0.0, 0.0)
                    bvx, bvy = 0.0, 0.0

                obj.position = (bx, by)
                obj.rotation = math.degrees(body.angle) % 360.0
                if not math.isfinite(obj.rotation):
                    obj.rotation = 0.0
                    body.angle = 0.0
                obj.velocity = (bvx if math.isfinite(bvx) else 0.0,
                                bvy if math.isfinite(bvy) else 0.0)

    def toggle_physics(self) -> bool:
        """Toggle physics gravity simulation on/off."""

        self.enabled = not self.enabled
        print(f"[PhysicsEngine] Physics simulation {'ENABLED' if self.enabled else 'DISABLED'}")
        return self.enabled

    def clear_all(self):
        """Remove all rigid bodies from physics space."""
        for obj_id in list(self.bodies.keys()):
            self.unregister_object(obj_id)
