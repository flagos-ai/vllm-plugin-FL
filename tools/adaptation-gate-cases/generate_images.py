#!/usr/bin/env python3
"""Regenerate deterministic PNG fixtures for the adaptation gate."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH = 512
HEIGHT = 320
OUTPUT_DIR = Path(__file__).resolve().parent / "images"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

RED = (235, 48, 48)
GREEN = (34, 184, 84)
BLUE = (48, 99, 224)
YELLOW = (248, 210, 48)
BLACK = (28, 32, 40)
WHITE = (250, 250, 248)


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    return image, ImageDraw.Draw(image)


def save(image: Image.Image, index: int) -> None:
    image.save(OUTPUT_DIR / f"image_{index:02d}.png", format="PNG", optimize=True)


def draw_quadrants(index: int, colors: tuple[tuple[int, int, int], ...]) -> None:
    image, draw = canvas()
    boxes = (
        (0, 0, WIDTH // 2, HEIGHT // 2),
        (WIDTH // 2, 0, WIDTH, HEIGHT // 2),
        (0, HEIGHT // 2, WIDTH // 2, HEIGHT),
        (WIDTH // 2, HEIGHT // 2, WIDTH, HEIGHT),
    )
    for box, color in zip(boxes, colors):
        draw.rectangle(box, fill=color)
    save(image, index)


def draw_shape_order() -> None:
    image, draw = canvas()
    draw.ellipse((30, 95, 150, 215), fill=RED, outline=BLACK, width=5)
    draw.rectangle((200, 95, 320, 215), fill=BLUE, outline=BLACK, width=5)
    draw.polygon(((420, 85), (355, 220), (485, 220)), fill=GREEN, outline=BLACK)
    draw.line(
        ((420, 85), (355, 220), (485, 220), (420, 85)),
        fill=BLACK,
        width=5,
        joint="curve",
    )
    save(image, 3)


def draw_triangle_count() -> None:
    image, draw = canvas()
    triangles = (
        ((90, 35), (30, 150), (150, 150)),
        ((256, 35), (196, 150), (316, 150)),
        ((422, 35), (362, 150), (482, 150)),
    )
    for points, color in zip(triangles, (RED, GREEN, BLUE)):
        draw.polygon(points, fill=color, outline=BLACK)
        draw.line((*points, points[0]), fill=BLACK, width=5, joint="curve")
    draw.ellipse((120, 190, 220, 290), fill=YELLOW, outline=BLACK, width=5)
    draw.ellipse((292, 190, 392, 290), fill=RED, outline=BLACK, width=5)
    save(image, 4)


def draw_text_extraction() -> None:
    image = Image.new("RGB", (1024, 1024), WHITE)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, 180)

    def draw_centered(text: str, center_y: int, tracking: int = 0) -> None:
        bounds = [draw.textbbox((0, 0), char, font=font) for char in text]
        widths = [box[2] - box[0] for box in bounds]
        x = (image.width - sum(widths) - tracking * (len(text) - 1)) // 2
        for char, box, width in zip(text, bounds, widths):
            height = box[3] - box[1]
            draw.text(
                (x - box[0], center_y - height // 2 - box[1]),
                char,
                font=font,
                fill=BLACK,
            )
            x += width + tracking

    draw_centered("HELLO", 320)
    draw_centered("VLLM", 650, tracking=50)
    save(image, 5)


def draw_nested_shape() -> None:
    image, draw = canvas()
    draw.ellipse((96, 24, 416, 296), fill=RED, outline=BLACK, width=7)
    draw.rectangle((196, 100, 316, 220), fill=BLUE, outline=WHITE, width=7)
    save(image, 6)


def draw_shape_color() -> None:
    image, draw = canvas()
    draw.ellipse((28, 100, 148, 220), fill=RED, outline=BLACK, width=5)
    draw.rectangle((196, 100, 316, 220), fill=BLUE, outline=BLACK, width=5)
    points = ((420, 85), (355, 225), (485, 225))
    draw.polygon(points, fill=YELLOW, outline=BLACK)
    draw.line((*points, points[0]), fill=BLACK, width=5, joint="curve")
    save(image, 7)


def draw_blue_circle_count() -> None:
    image, draw = canvas()
    for box in (
        (35, 35, 145, 145),
        (201, 35, 311, 145),
        (367, 35, 477, 145),
        (118, 175, 228, 285),
    ):
        draw.ellipse(box, fill=BLUE, outline=BLACK, width=5)
    draw.rectangle((284, 175, 394, 285), fill=RED, outline=BLACK, width=5)
    draw.rectangle((414, 205, 484, 275), fill=RED, outline=BLACK, width=5)
    save(image, 8)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_quadrants(1, (RED, GREEN, BLUE, YELLOW))
    draw_quadrants(2, (BLUE, YELLOW, RED, GREEN))
    draw_shape_order()
    draw_triangle_count()
    draw_text_extraction()
    draw_nested_shape()
    draw_shape_color()
    draw_blue_circle_count()
    print(f"Generated 8 images in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
