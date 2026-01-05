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
        if world_size > 1:
            scale = self.window_size / world_size
            return (pos + 0.5) * scale
        else:
            return pos * self.window_size

    def _draw_blow_bars(
        self,
        axis,
        centers,
        width,
        phase,
        world_size,
    ):
        if phase == "idle" or axis is None or centers is None:
            return

        if phase == "warning":
            color = (255, 215, 0, 80)   # yellow
        else:  # active
            color = (255, 0, 0, 160)   # red

        overlay = pygame.Surface(
            (self.window_size, self.window_size),
            pygame.SRCALPHA
        )

        if world_size > 1:
            # ---------- DISCRETE ----------
            cell = self.window_size / world_size

            for c in centers:
                low = max(0, c - width)
                high = min(world_size - 1, c + width)

                if axis == 0:
                    # vertical bar (x = const)
                    rect = pygame.Rect(
                        low * cell,
                        0,
                        (high - low + 1) * cell,
                        self.window_size,
                    )
                else:
                    # horizontal bar (y = const)
                    rect = pygame.Rect(
                        0,
                        low * cell,
                        self.window_size,
                        (high - low + 1) * cell,
                    )

                pygame.draw.rect(overlay, color, rect)

        else:
            # ---------- CONTINUOUS ----------
            for c in centers:
                low = max(0.0, c - width)
                high = min(1.0, c + width)

                if axis == 0:
                    # vertical strip
                    rect = pygame.Rect(
                        low * self.window_size,
                        0,
                        (high - low) * self.window_size,
                        self.window_size,
                    )
                else:
                    # horizontal strip
                    rect = pygame.Rect(
                        0,
                        low * self.window_size,
                        self.window_size,
                        (high - low) * self.window_size,
                    )

                pygame.draw.rect(overlay, color, rect)

        self.screen.blit(overlay, (0, 0))

    def render(
        self,
        agent_pos,
        goal_pos,
        blow_phase,
        blow_axis,
        blow_centers,
        blow_width,
        world_size,
    ):
        self.screen.fill((240, 240, 240))

        self._draw_blow_bars(
            axis=blow_axis,
            centers=blow_centers,
            width=blow_width,
            phase=blow_phase,
            world_size=world_size,
        )

        agent_xy = self._to_screen(
            np.asarray(agent_pos),
            world_size
        )
        goal_xy = self._to_screen(
            np.asarray(goal_pos),
            world_size
        )

        pygame.draw.circle(
            self.screen,
            (0, 180, 0),
            goal_xy.astype(int),
            10
        )

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
