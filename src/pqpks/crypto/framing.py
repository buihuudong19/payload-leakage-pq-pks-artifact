"""Plaintext framing pt.

First image (flag SESSION_START):  pt = ver(1) || flags(1) || msg_len(4) || t0(8) || m || [sigma] || pad(0x00...)
Later image:                       pt = ver(1) || flags(1) || msg_len(4) || m || pad(0x00...)
|pt| is in B_t (the smallest bucket that fits, in the ladder of image type t). sigma is allowed only in the first image,
except in profile A5b (sigma in image 2): pack/unpack are then called with allow_sig_next=True for image 2 only.

`unpack` is a CANONICAL parser: every valid pt has exactly one reading. It rejects a wrong version, unknown flag
bits, sigma outside the first image, a length beyond pt, a sigma of the wrong length and non-zero padding.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..params import (FLAG_HAS_SIG, FLAG_SESSION_START, FRAME_HEADER_BYTES, FRAME_VERSION, KNOWN_FLAGS,
                      T0_BYTES)


@dataclass
class Frame:
    msg: bytes
    flags: int = 0
    sig: bytes | None = None
    t0: int | None = None        # required with flag SESSION_START


def encoded_len(frame: Frame) -> int:
    start = bool(frame.flags & FLAG_SESSION_START)
    return FRAME_HEADER_BYTES + (T0_BYTES if start else 0) + len(frame.msg) + len(frame.sig or b"")


def pack(frame: Frame, buckets: tuple[int, ...], allow_sig_next: bool = False) -> bytes:
    flags = frame.flags | (FLAG_HAS_SIG if frame.sig else 0)
    if flags & ~KNOWN_FLAGS:
        raise ValueError("invalid flags")
    start = bool(flags & FLAG_SESSION_START)
    if frame.sig and not start and not allow_sig_next:
        raise ValueError("sigma is allowed only in the first image of a session")
    if start and frame.t0 is None:
        raise ValueError("the first image of a session must carry t0")
    body = bytes([FRAME_VERSION, flags]) + len(frame.msg).to_bytes(4, "big")
    if start:
        body += int(frame.t0).to_bytes(T0_BYTES, "big")
    body += frame.msg + (frame.sig or b"")
    for b in sorted(buckets):
        if len(body) <= b:
            return body + b"\x00" * (b - len(body))
    raise ValueError(f"message too long: {len(body)} B > largest bucket {max(buckets)} B")


def unpack(pt: bytes, sig_bytes: int, allow_sig_next: bool = False) -> Frame | None:
    if len(pt) < FRAME_HEADER_BYTES or pt[0] != FRAME_VERSION:
        return None
    flags = pt[1]
    if flags & ~KNOWN_FLAGS:
        return None
    start = bool(flags & FLAG_SESSION_START)
    has_sig = bool(flags & FLAG_HAS_SIG)
    if has_sig and not start and not allow_sig_next:
        return None
    n = int.from_bytes(pt[2:6], "big")
    pos = FRAME_HEADER_BYTES
    t0 = None
    if start:
        if len(pt) < pos + T0_BYTES:
            return None
        t0 = int.from_bytes(pt[pos:pos + T0_BYTES], "big")
        pos += T0_BYTES
    end = pos + n
    sig_end = end + (sig_bytes if has_sig else 0)
    if sig_end > len(pt):
        return None
    if any(pt[sig_end:]):                      # the padding must be all zero
        return None
    return Frame(msg=pt[pos:end], flags=flags, sig=pt[end:sig_end] if has_sig else None, t0=t0)
