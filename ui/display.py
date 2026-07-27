"""240x240 rear-door display mock (ST7789 target resolution)."""

import pygame

from film_profiles.base import FilmProfile

DISPLAY_SIZE = 240


class RearDisplayUI:
    """Simulates the vintage film-box tab SPI display."""

    def __init__(self) -> None:
        pygame.init()
        self._surface = pygame.Surface((DISPLAY_SIZE, DISPLAY_SIZE))
        self._font_title = pygame.font.SysFont("Helvetica", 14, bold=True)
        self._font_small = pygame.font.SysFont("Helvetica", 11)
        self._font_mono = pygame.font.SysFont("Menlo", 12)

    def render(
        self,
        profile: FilmProfile,
        shot_count: int,
        battery_percent: int = 87,
    ) -> pygame.Surface:
        """Draw the film box UI and return the surface."""
        self._surface.fill((22, 22, 22))

        # Film box tab area
        box_rect = pygame.Rect(16, 24, 208, 140)
        pygame.draw.rect(self._surface, profile.box_color, box_rect, border_radius=4)
        pygame.draw.rect(self._surface, (240, 240, 240), box_rect, width=2, border_radius=4)

        # Tab notch at top
        tab = pygame.Rect(88, 12, 64, 18)
        pygame.draw.rect(self._surface, profile.box_color, tab, border_radius=3)
        pygame.draw.rect(self._surface, (240, 240, 240), tab, width=2, border_radius=3)

        title = self._font_title.render(profile.name.upper(), True, (255, 255, 255))
        title_shadow = self._font_title.render(profile.name.upper(), True, (0, 0, 0))
        self._surface.blit(title_shadow, (28, 52))
        self._surface.blit(title, (27, 51))

        stock_line = self._font_small.render("FILM SIMULATION", True, (255, 255, 255))
        self._surface.blit(stock_line, (27, 78))

        iso_text = self._estimate_iso_label(profile.key)
        iso = self._font_mono.render(iso_text, True, (255, 255, 255))
        self._surface.blit(iso, (27, 100))

        # Status bar
        status_y = 178
        pygame.draw.line(self._surface, (60, 60, 60), (16, status_y), (224, status_y), 1)

        shots = self._font_mono.render(f"SHOT {shot_count:04d}", True, (200, 200, 200))
        self._surface.blit(shots, (16, 192))

        battery = self._font_mono.render(f"BAT {battery_percent:3d}%", True, (200, 200, 200))
        self._surface.blit(battery, (148, 192))

        self._draw_battery_icon(200, 194, battery_percent)

        hint = self._font_small.render("[SPACE] capture  [1-6] profile  [Q] quit", True, (100, 100, 100))
        self._surface.blit(hint, (16, 216))

        return self._surface

    def _draw_battery_icon(self, x: int, y: int, percent: int) -> None:
        body = pygame.Rect(x, y, 28, 12)
        tip = pygame.Rect(x + 28, y + 3, 3, 6)
        pygame.draw.rect(self._surface, (120, 120, 120), body, width=1, border_radius=2)
        pygame.draw.rect(self._surface, (120, 120, 120), tip)

        fill_width = max(0, int(24 * percent / 100))
        if fill_width:
            fill = pygame.Rect(x + 2, y + 2, fill_width, 8)
            color = (80, 200, 80) if percent > 20 else (220, 80, 60)
            pygame.draw.rect(self._surface, color, fill, border_radius=1)

    @staticmethod
    def _estimate_iso_label(profile_key: str) -> str:
        iso_map = {
            "kodachrome64": "ISO 64",
            "portra400": "ISO 400",
            "superia400": "ISO 400",
            "trix400": "ISO 400",
            "hp5plus": "ISO 400",
            "cinestill800t": "ISO 800",
        }
        return iso_map.get(profile_key, "ISO ---")

    def quit(self) -> None:
        pygame.quit()
