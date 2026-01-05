import pygame


class IceBlowRenderer:
    def __init__(self, fps=30, window_size=600):
        """
        Assumes pygame.init() has already been called.
        """
        self.window_size = window_size
        self.fps = fps

        self.screen = pygame.display.set_mode(
            (window_size, window_size)
        )
        pygame.display.set_caption("Ice Blow")

        self.clock = pygame.time.Clock()

    # ------------------------------------------------------------------
    def step(self):
        self.clock.tick(self.fps)

    # ------------------------------------------------------------------
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
        """
        agent_pos, goal_pos: (x, y) in world coordinates
        """

        self.screen.fill((240, 240, 240))

        cell_size = self.window_size / world_size

        # --------------------------------------------------------------
        # Draw blow zones
        # --------------------------------------------------------------
        if blow_phase in ("warning", "active"):
            for center in blow_centers:
                for offset in range(-blow_width, blow_width + 1):
                    idx = center + offset
                    if idx < 0 or idx >= world_size:
                        continue

                    if blow_axis == "x":
                        rect = pygame.Rect(
                            idx * cell_size,
                            0,
                            cell_size,
                            self.window_size,
                        )
                    else:
                        rect = pygame.Rect(
                            0,
                            idx * cell_size,
                            self.window_size,
                            cell_size,
                        )

                    color = (255, 200, 0) if blow_phase == "warning" else (255, 0, 0)
                    pygame.draw.rect(self.screen, color, rect)

        # --------------------------------------------------------------
        # Draw goal
        # --------------------------------------------------------------
        gx, gy = goal_pos
        goal_rect = pygame.Rect(
            gx * cell_size,
            gy * cell_size,
            cell_size,
            cell_size,
        )
        pygame.draw.rect(self.screen, (0, 200, 0), goal_rect)

        # --------------------------------------------------------------
        # Draw agent
        # --------------------------------------------------------------
        ax, ay = agent_pos
        agent_rect = pygame.Rect(
            ax * cell_size,
            ay * cell_size,
            cell_size,
            cell_size,
        )
        pygame.draw.rect(self.screen, (0, 0, 255), agent_rect)

        pygame.display.flip()
