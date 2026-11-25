"""Tiny uncompressed PNG writer so sample swatches need no image library."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, payload: bytes) -> bytes:
    check = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", check)


def write_solid_png(path: Path, rgb: tuple[int, int, int], size: int = 32) -> None:
    """Write an RGB PNG of a single color. ``size`` is width and height."""
    if size < 1:
        raise ValueError("size must be >= 1")
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise ValueError(f"rgb channels must be in 0..255, got {rgb}")
    row = b"\x00" + bytes(rgb) * size
    raw = row * size
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    png = PNG_SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, 9)) + _chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def read_png_size_and_first_pixel(path: Path) -> tuple[int, int, tuple[int, int, int]]:
    """Return ``(width, height, first RGB pixel)`` for PNGs this module writes."""
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"{path} is not a PNG")
    offset = len(PNG_SIGNATURE)
    width = height = None
    compressed = b""
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        tag = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if tag == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", payload[:10])
            if bit_depth != 8 or color_type != 2:
                raise ValueError("Only 8-bit RGB PNGs are supported by the reader")
        elif tag == b"IDAT":
            compressed += payload
        elif tag == b"IEND":
            break
    if width is None or height is None or not compressed:
        raise ValueError(f"{path} is missing IHDR or IDAT")
    raw = zlib.decompress(compressed)
    row_stride = 1 + width * 3
    if len(raw) < row_stride:
        raise ValueError(f"{path} decoded shorter than one row")
    pixel = raw[1:4]
    return width, height, (pixel[0], pixel[1], pixel[2])
