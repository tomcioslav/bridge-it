#!/usr/bin/env python3
"""
GUI interface for playing Bridgit using pygame.

Supports human vs human and human vs AI modes.
Usage:
    python play.py [board_size]             # Human vs Human
    python play.py --ai [checkpoint_path]   # Human vs AI
"""

import argparse
import os

import pygame
import numpy as np
from bridgit import Bridgit, Player
from typing import Optional, Tuple


# Colors
BACKGROUND = (245, 245, 250)
HORIZONTAL_COLOR = (50, 200, 50)   # Green for player 1 (horizontal)
VERTICAL_COLOR = (220, 50, 50)     # Red for player 2 (vertical)
HORIZONTAL_DOT_COLOR = (100, 230, 100, 180)  # Lighter green for dots (with alpha)
VERTICAL_DOT_COLOR = (250, 100, 100, 180)    # Lighter red for dots (with alpha)
HOVER_ALPHA = 128                   # Transparency for hover preview
TEXT_COLOR = (40, 40, 60)
PANEL_BG = (255, 255, 255)
PANEL_BORDER_COLOR = (200, 200, 210)

# Dimensions
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
BOARD_MARGIN = 80
PANEL_WIDTH = 300
LINE_THICKNESS = 8
DOT_RADIUS = 4


class BridgitGUI:
    """GUI interface for Bridgit game using pygame."""

    def __init__(self, n: int = 5, ai_checkpoint: str | None = None):
        """Initialize the GUI.

        Args:
            n: Board size
            ai_checkpoint: Path to AI model checkpoint. If provided, AI plays as Player 2.
        """
        pygame.init()

        self.game = Bridgit(n)
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Bridgit Game")

        self.clock = pygame.time.Clock()
        self.running = True

        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.text_font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)

        # Calculate board dimensions
        board_width = WINDOW_WIDTH - PANEL_WIDTH - BOARD_MARGIN * 2
        board_height = WINDOW_HEIGHT - BOARD_MARGIN * 2

        # Cell size for the n×n grid
        self.cell_size = min(board_width / self.game.n, board_height / self.game.n)

        # Center the board
        self.board_offset_x = BOARD_MARGIN
        self.board_offset_y = BOARD_MARGIN + (board_height - self.cell_size * self.game.n) / 2

        self.hover_cell: Optional[Tuple[int, int]] = None

        # AI setup
        self.ai_player = None
        self.ai_thinking = False
        if ai_checkpoint is not None:
            self._setup_ai(ai_checkpoint)

    def _setup_ai(self, checkpoint_path: str):
        """Load AI model and create MCTS player."""
        from bridgit.ai.config import Config
        from bridgit.ai.game_wrapper import GameWrapper
        from bridgit.ai.neural_net import NeuralNetWrapper
        from bridgit.ai.mcts import MCTS

        config = Config(board_size=self.game.n)
        self.ai_game_wrapper = GameWrapper(config.board_size)
        self.ai_net = NeuralNetWrapper(config)

        if os.path.exists(checkpoint_path):
            self.ai_net.load_checkpoint(checkpoint_path)
            print(f"Loaded AI model from {checkpoint_path}")
        else:
            print(f"No checkpoint at {checkpoint_path}, using untrained model")

        self.ai_mcts = MCTS(self.ai_net, config)
        self.ai_player = Player.VERTICAL  # AI plays as Player 2

    def _get_ai_move(self) -> Tuple[int, int]:
        """Get AI's move using MCTS."""
        pi = self.ai_mcts.get_action_probs(self.game, temperature=0)
        action = int(np.argmax(pi))
        return self.ai_game_wrapper.action_to_move(action, self.game)

    def get_cell_rect(self, row: int, col: int) -> pygame.Rect:
        """Get the rectangle for a grid cell."""
        x = self.board_offset_x + col * self.cell_size
        y = self.board_offset_y + row * self.cell_size
        return pygame.Rect(x, y, self.cell_size, self.cell_size)

    def get_cell_from_mouse(self, mouse_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Get the grid cell from mouse position."""
        mx, my = mouse_pos

        # Check if within board area
        if (mx < self.board_offset_x or
            mx > self.board_offset_x + self.game.n * self.cell_size or
            my < self.board_offset_y or
            my > self.board_offset_y + self.game.n * self.cell_size):
            return None

        col = int((mx - self.board_offset_x) / self.cell_size)
        row = int((my - self.board_offset_y) / self.cell_size)

        if 0 <= row < self.game.n and 0 <= col < self.game.n:
            return (row, col)

        return None

    def draw_horizontal_line(self, row: int, col: int, color: Tuple[int, int, int], thickness: int = LINE_THICKNESS):
        """Draw a horizontal line in a grid cell."""
        rect = self.get_cell_rect(row, col)
        start = (rect.left + rect.width * 0.1, rect.centery)
        end = (rect.right - rect.width * 0.1, rect.centery)
        pygame.draw.line(self.screen, color, start, end, thickness)

    def draw_vertical_line(self, row: int, col: int, color: Tuple[int, int, int], thickness: int = LINE_THICKNESS):
        """Draw a vertical line in a grid cell."""
        rect = self.get_cell_rect(row, col)
        start = (rect.centerx, rect.top + rect.height * 0.1)
        end = (rect.centerx, rect.bottom - rect.height * 0.1)
        pygame.draw.line(self.screen, color, start, end, thickness)

    def draw_horizontal_line_transparent(self, row: int, col: int, color: Tuple[int, int, int, int], thickness: int = LINE_THICKNESS):
        """Draw a semi-transparent horizontal line in a grid cell."""
        rect = self.get_cell_rect(row, col)
        start = (rect.left + rect.width * 0.1, rect.centery)
        end = (rect.right - rect.width * 0.1, rect.centery)

        # Create a surface with alpha channel
        line_surface = pygame.Surface((int(rect.width), thickness + 10), pygame.SRCALPHA)
        line_start = (int(rect.width * 0.1), thickness // 2 + 5)
        line_end = (int(rect.width * 0.9), thickness // 2 + 5)
        pygame.draw.line(line_surface, color, line_start, line_end, thickness)

        # Blit to screen at the correct position
        self.screen.blit(line_surface, (rect.left, rect.centery - thickness // 2 - 5))

    def draw_vertical_line_transparent(self, row: int, col: int, color: Tuple[int, int, int, int], thickness: int = LINE_THICKNESS):
        """Draw a semi-transparent vertical line in a grid cell."""
        rect = self.get_cell_rect(row, col)
        start = (rect.centerx, rect.top + rect.height * 0.1)
        end = (rect.centerx, rect.bottom - rect.height * 0.1)

        # Create a surface with alpha channel
        line_surface = pygame.Surface((thickness + 10, int(rect.height)), pygame.SRCALPHA)
        line_start = (thickness // 2 + 5, int(rect.height * 0.1))
        line_end = (thickness // 2 + 5, int(rect.height * 0.9))
        pygame.draw.line(line_surface, color, line_start, line_end, thickness)

        # Blit to screen at the correct position
        self.screen.blit(line_surface, (rect.centerx - thickness // 2 - 5, rect.top))

    def draw_board(self):
        """Draw the game board."""
        n = self.game.n

        # Draw green dots on vertical grid lines (for horizontal connections)
        # These dots are at the midpoint of cells vertically
        for i in range(n + 1):  # Vertical grid lines
            for j in range(n):   # Rows
                x = self.board_offset_x + i * self.cell_size
                y = self.board_offset_y + j * self.cell_size + self.cell_size / 2
                pygame.draw.circle(self.screen, HORIZONTAL_COLOR[:3], (int(x), int(y)), DOT_RADIUS)

        # Draw red dots on horizontal grid lines (for vertical connections)
        # These dots are at the midpoint of cells horizontally
        for i in range(n + 1):  # Horizontal grid lines
            for j in range(n):   # Columns
                x = self.board_offset_x + j * self.cell_size + self.cell_size / 2
                y = self.board_offset_y + i * self.cell_size
                pygame.draw.circle(self.screen, VERTICAL_COLOR[:3], (int(x), int(y)), DOT_RADIUS)

        # Draw placed lines first
        for r in range(n):
            for c in range(n):
                val = self.game.grid[r, c]
                if val == -1:  # Horizontal player
                    self.draw_horizontal_line(r, c, HORIZONTAL_COLOR, LINE_THICKNESS)
                elif val == 1:  # Vertical player
                    self.draw_vertical_line(r, c, VERTICAL_COLOR, LINE_THICKNESS)

        # Draw hover preview on top (only for human's turn)
        if self.hover_cell and not self.game.game_over and not self.ai_thinking:
            # Don't show hover preview when it's AI's turn
            if self.ai_player is None or self.game.current_player != self.ai_player:
                row, col = self.hover_cell
                if self.game.is_valid_move(row, col):
                    # Draw semi-transparent preview line
                    if self.game.current_player == Player.HORIZONTAL:
                        color = HORIZONTAL_COLOR + (HOVER_ALPHA,)
                        self.draw_horizontal_line_transparent(row, col, color, LINE_THICKNESS)
                    else:
                        color = VERTICAL_COLOR + (HOVER_ALPHA,)
                        self.draw_vertical_line_transparent(row, col, color, LINE_THICKNESS)

    def draw_panel(self):
        """Draw the side panel with game info."""
        panel_x = WINDOW_WIDTH - PANEL_WIDTH

        # Panel background
        pygame.draw.rect(self.screen, PANEL_BG,
                        (panel_x, 0, PANEL_WIDTH, WINDOW_HEIGHT))
        pygame.draw.line(self.screen, PANEL_BORDER_COLOR,
                        (panel_x, 0), (panel_x, WINDOW_HEIGHT), 2)

        y_offset = 40

        # Title
        title = self.title_font.render("BRIDGIT", True, TEXT_COLOR)
        title_rect = title.get_rect(centerx=panel_x + PANEL_WIDTH // 2, top=y_offset)
        self.screen.blit(title, title_rect)
        y_offset += 80

        # Current player
        if not self.game.game_over:
            player_text = "Current Turn:"
            text = self.small_font.render(player_text, True, TEXT_COLOR)
            text_rect = text.get_rect(centerx=panel_x + PANEL_WIDTH // 2, top=y_offset)
            self.screen.blit(text, text_rect)
            y_offset += 40

            if self.game.current_player == Player.HORIZONTAL:
                player_name = "HORIZONTAL"
                player_color = HORIZONTAL_COLOR
            else:
                player_name = "VERTICAL"
                player_color = VERTICAL_COLOR

            # Show if it's AI's turn
            if self.ai_player and self.game.current_player == self.ai_player:
                player_name += " (AI)"

            text = self.text_font.render(player_name, True, player_color)
            text_rect = text.get_rect(centerx=panel_x + PANEL_WIDTH // 2, top=y_offset)
            self.screen.blit(text, text_rect)
            y_offset += 30

            # Show "Thinking..." when AI is computing
            if self.ai_thinking:
                thinking = self.small_font.render("Thinking...", True, TEXT_COLOR)
                thinking_rect = thinking.get_rect(centerx=panel_x + PANEL_WIDTH // 2, top=y_offset)
                self.screen.blit(thinking, thinking_rect)
            y_offset += 30

        # Player info boxes
        # Horizontal player
        box_y = y_offset
        pygame.draw.rect(self.screen, HORIZONTAL_COLOR,
                        (panel_x + 30, box_y, PANEL_WIDTH - 60, 120), 3)

        p1_label = "Player 1 (You)" if self.ai_player == Player.VERTICAL else "Player 1"
        text = self.text_font.render(p1_label, True, HORIZONTAL_COLOR)
        text_rect = text.get_rect(centerx=panel_x + PANEL_WIDTH // 2, top=box_y + 15)
        self.screen.blit(text, text_rect)

        # Draw example horizontal line
        line_y = box_y + 55
        pygame.draw.line(self.screen, HORIZONTAL_COLOR,
                        (panel_x + 60, line_y),
                        (panel_x + PANEL_WIDTH - 60, line_y), LINE_THICKNESS)

        text = self.small_font.render("Left → Right", True, TEXT_COLOR)
        text_rect = text.get_rect(centerx=panel_x + PANEL_WIDTH // 2, top=box_y + 80)
        self.screen.blit(text, text_rect)

        # Vertical player
        box_y += 140
        pygame.draw.rect(self.screen, VERTICAL_COLOR,
                        (panel_x + 30, box_y, PANEL_WIDTH - 60, 120), 3)

        p2_label = "Player 2 (AI)" if self.ai_player == Player.VERTICAL else "Player 2"
        text = self.text_font.render(p2_label, True, VERTICAL_COLOR)
        text_rect = text.get_rect(centerx=panel_x + PANEL_WIDTH // 2, top=box_y + 15)
        self.screen.blit(text, text_rect)

        # Draw example vertical line
        line_x = panel_x + PANEL_WIDTH // 2
        pygame.draw.line(self.screen, VERTICAL_COLOR,
                        (line_x, box_y + 40),
                        (line_x, box_y + 70), LINE_THICKNESS)

        text = self.small_font.render("Top → Bottom", True, TEXT_COLOR)
        text_rect = text.get_rect(centerx=panel_x + PANEL_WIDTH // 2, top=box_y + 80)
        self.screen.blit(text, text_rect)

        # Instructions
        y_offset = WINDOW_HEIGHT - 180
        instructions = [
            "Click on a grid cell",
            "to place your line",
            "",
            "Press R to restart",
            "Press Q to quit"
        ]

        for i, line in enumerate(instructions):
            text = self.small_font.render(line, True, TEXT_COLOR)
            text_rect = text.get_rect(centerx=panel_x + PANEL_WIDTH // 2,
                                     top=y_offset + i * 30)
            self.screen.blit(text, text_rect)

    def draw_win_screen(self):
        """Draw the win screen overlay."""
        # Semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        # Win message
        if self.game.winner == Player.HORIZONTAL:
            winner_text = "PLAYER 1 WINS!"
            winner_color = HORIZONTAL_COLOR
            subtitle = "Left-Right Connection Complete"
        else:
            winner_text = "PLAYER 2 WINS!"
            winner_color = VERTICAL_COLOR
            subtitle = "Top-Bottom Connection Complete"

        if self.ai_player:
            if self.game.winner == self.ai_player:
                subtitle += " — AI wins!"
            else:
                subtitle += " — You win!"

        # Draw win text
        win_text = self.title_font.render(winner_text, True, winner_color)
        win_rect = win_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 60))
        self.screen.blit(win_text, win_rect)

        subtitle_text = self.text_font.render(subtitle, True, (255, 255, 255))
        subtitle_rect = subtitle_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        self.screen.blit(subtitle_text, subtitle_rect)

        # Instructions
        restart_text = self.small_font.render("Press R to play again or Q to quit", True, (200, 200, 200))
        restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 60))
        self.screen.blit(restart_text, restart_rect)

    def handle_click(self, pos: Tuple[int, int]):
        """Handle mouse click."""
        if self.game.game_over:
            return

        # Ignore clicks during AI's turn
        if self.ai_player and self.game.current_player == self.ai_player:
            return

        cell = self.get_cell_from_mouse(pos)
        if cell:
            row, col = cell
            self.game.make_move(row, col)

    def restart_game(self):
        """Restart the game."""
        self.game = Bridgit(self.game.n)
        self.ai_thinking = False

    def run(self):
        """Main game loop."""
        while self.running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.running = False
                    elif event.key == pygame.K_r:
                        self.restart_game()

            # AI move
            if (self.ai_player and not self.game.game_over
                    and self.game.current_player == self.ai_player):
                self.ai_thinking = True
                # Draw "thinking" state before computing
                self.screen.fill(BACKGROUND)
                self.draw_board()
                self.draw_panel()
                pygame.display.flip()

                row, col = self._get_ai_move()
                self.game.make_move(row, col)
                self.ai_thinking = False

            # Update hover cell
            mouse_pos = pygame.mouse.get_pos()
            self.hover_cell = self.get_cell_from_mouse(mouse_pos)

            # Drawing
            self.screen.fill(BACKGROUND)
            self.draw_board()
            self.draw_panel()

            if self.game.game_over:
                self.draw_win_screen()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Play Bridgit")
    parser.add_argument("board_size", nargs="?", type=int, default=5,
                        help="Board size (default: 5)")
    parser.add_argument("--ai", nargs="?", const="checkpoints/best.pt",
                        default=None, metavar="CHECKPOINT",
                        help="Play against AI (optionally specify checkpoint path)")
    args = parser.parse_args()

    game = BridgitGUI(n=args.board_size, ai_checkpoint=args.ai)
    game.run()


if __name__ == "__main__":
    main()
