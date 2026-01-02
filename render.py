# render.py
import pygame
import numpy as np


class IceBlowRenderer:
    def __init__(self, window_size=600, fps=30):
        pygame.init()
        self.window_size = window_size
        self.screen = pygame.display.set_mode(
            (window_size, window_size)
        )
        pygame.display.set_caption("Ice Blow Environment")
        self.clock = pygame.time.Clock()
        self.fps = fps

    def _to_screen(self, pos, world_size):
        """
        Converts agent/goal position to screen coordinates.
        """
        if world_size > 1:
            # Discrete
            scale = self.window_size / world_size
            return (pos + 0.5) * scale
        else:
            # Continuous [0, 1]
            return pos * self.window_size

    def _draw_blow_axis(self, axis, coord, phase, world_size):
        """
        Draws the warning or active blow axis.
        """
        if axis is None or coord is None or phase == "idle":
            return

        # Color and transparency by phase
        if phase == "warning":
            color = (255, 215, 0, 80)   # yellow
            thickness = 4
        elif phase == "active":
            color = (255, 0, 0, 160)    # red
            thickness = 8
        else:
            return

        overlay = pygame.Surface(
            (self.window_size, self.window_size),
            pygame.SRCALPHA
        )

        if world_size > 1:
            # Discrete axis
            scale = self.window_size / world_size
            c = (coord + 0.5) * scale
        else:
            # Continuous axis
            c = coord * self.window_size

        if axis == 0:
            # x = coord → vertical line
            start = (c, 0)
            end = (c, self.window_size)
        else:
            # y = coord → horizontal line
            start = (0, c)
            end = (self.window_size, c)

        pygame.draw.line(
            overlay,
            color,
            start,
            end,
            thickness
        )

        self.screen.blit(overlay, (0, 0))

    def render(
        self,
        agent_pos,
        goal_pos,
        blow_phase,
        blow_axis,
        blow_coord,
        world_size,
    ):
        self.screen.fill((240, 240, 240))

        # Draw blow warning / active axis
        self._draw_blow_axis(
            blow_axis,
            blow_coord,
            blow_phase,
            world_size
        )

        # Convert positions
        agent_xy = self._to_screen(
            np.asarray(agent_pos),
            world_size
        )
        goal_xy = self._to_screen(
            np.asarray(goal_pos),
            world_size
        )

        # Draw goal
        pygame.draw.circle(
            self.screen,
            (0, 180, 0),
            goal_xy.astype(int),
            10
        )

        # Draw agent
        pygame.draw.circle(
            self.screen,
            (0, 0, 255),
            agent_xy.astype(int),
            10
        )

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()
