"""Post-processing: per-track kinetic parameters, as KymoButler's ``summary.csv`` reports them.

A port of ``getDerivedQuantities`` and ``pproc`` in KymoButlerPProc.wl (the functions the
KymoButler web form and the Mathematica package used), down to the floating-point arithmetic:

    getDerivedQuantities[trk_] := If[Length@trk > 1,
        v      = Mean[Flatten[Abs /@ Ratios /@ Differences@trk]]   (* mean |dx/dt| *)
        direct = Sign[trk[[-1, 2]] - trk[[1, 2]]]
        dist   = Total@Abs@Differences@trk[[;; , 2]]
        T      = Abs[trk[[-1, 1]] - trk[[1, 1]] + 1]
        , Missing[]]
    quant = Map[Round[N@getDerivedQuantities@#, 0.0001] &, trks]
    columns: direct, Round[xsz/tsz*v, .0001], Round[tsz*T, .0001], Round[xsz*dist, .0001],
             Round[xsz*dist/T/tsz, .0001]

Semantics worth knowing (all Mathematica's, reproduced on purpose):

* A track's points are used in their own order, which is not always increasing in time
  (a bidirectional track can step back a frame). ``v`` averages ``|dx/dt|`` over consecutive
  points, so a backward step counts with ``|dx|``; ``dist`` sums ``|dx|`` in point order.
* ``direct`` and ``T`` use the *first and last points*, not the earliest and latest frames:
  ``T = |t_last - t_first + 1|`` frames, ``direct = Sign[x_last - x_first]``.
* "Start2end velocity" is ``dist / T``: the path length over the duration, not the net
  displacement.
* Every value is first rounded to 0.0001 in pixel units (``Round[N@quant, 0.0001]``), then
  scaled and rounded again. ``Round[x, a]`` is ``a*Round[x/a]`` with ties to even, so values
  carry Mathematica's binary digits (``0.40280000000000005``), and the direction is a real
  (``1.``). Velocities are exact rationals until ``N`` (``mma_n``); the products with the
  pixel sizes are evaluated in the order Mathematica 14.3 was measured to use (integral pixel
  sizes are exact integers, as the web form's number fields give them). All of this was
  measured bit for bit on Mathematica 14.3, as ``pproc`` evaluates it: 180,000 ``Round``
  inputs, 60,000 products per column over 100 integer and decimal pixel sizes, 9,000
  rationals, and ``pproc``'s CSVs for 420 real and 8,000 synthetic tracks at 26 pixel
  sizes.

Where Mathematica has no defined value it produces garbage (``Missing[]`` expressions, or
``ComplexInfinity``); those cases raise a ``ValueError`` here instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Real
from typing import Iterable, Sequence

from kymobutler.tracking import Track

ROUND_TO = 0.0001  # Round[..., 0.0001] / Round[..., .0001] in KymoButlerPProc.wl

# Keys of the rows returned by ``postprocess``, in summary.csv column order.
STAT_KEYS = (
    "direction",
    "velocity_um_per_sec",
    "duration_sec",
    "distance_um",
    "start2end_velocity_um_per_sec",
)


@dataclass
class TrackStatistics:
    """``getDerivedQuantities`` of one track, in pixel units, before any rounding.

    ``mean_velocity`` is exact (a ``Fraction``); ``float()`` it for a machine number.
    """

    direction: int  # Sign[x_last - x_first]: +1, -1, or 0
    mean_velocity: Fraction  # Mean of |dx/dt| over consecutive points (pixels/frame)
    total_distance: int  # sum of |dx| over consecutive points (pixels)
    duration: int  # |t_last - t_first + 1| (frames)

    @property
    def start_to_end_velocity(self) -> Fraction:
        """``dist / T`` (pixels/frame), as the "Start2end velocity" column."""
        if self.duration == 0:
            raise ValueError("start-to-end velocity is undefined for a track of duration 0")
        return Fraction(self.total_distance, self.duration)


def mma_round(x: float, a: float = ROUND_TO) -> float:
    """Mathematica's ``Round[x, a]`` for a machine real: ``a * Round[x / a]``, ties to even,
    with ``x / a`` a machine division (measured on Mathematica 14.3 for 180,000 reals, many
    of them at ties; as ``pproc`` evaluates it. Auto-compiled code, e.g. ``Map`` over a
    packed array, computes ``Round[x * (1/a)]`` instead, which differs at 1 in 20)."""
    return round(float(x) / a) * a


def mma_n(q: Fraction | int) -> float:
    """``N`` of an exact rational: numerator and denominator converted to machine reals, then
    divided (measured on Mathematica 14.3; correctly rounded while both are below 2^53)."""
    q = Fraction(q)
    return float(q.numerator) / float(q.denominator)


def mma_number(x: Real) -> int | float:
    """A pixel size as Mathematica holds it: an exact integer when integral (the web form's
    number field turns "1" into the Integer 1), else a machine real."""
    if isinstance(x, bool) or not isinstance(x, Real):
        raise TypeError(f"pixel size must be a number, got {x!r}")
    if isinstance(x, Integral):
        return int(x)
    x = float(x)
    return int(x) if x.is_integer() else x


def _check_pixel_size(name: str, x: int | float) -> None:
    if not (math.isfinite(x) and x > 0):
        raise ValueError(f"{name} must be a positive number, got {x!r}")


def track_points(track: Track | Sequence) -> list[tuple[int, int]]:
    """A track's 0-based (time, space) pixel coordinates as Python ints, in the track's order.

    Raises:
        ValueError: for a coordinate that is not a whole number (KymoButler tracks are pixel
            positions; nothing here rounds them).
    """
    pts = track.points if isinstance(track, Track) else track
    out = []
    for p in pts:
        t, x = p
        if type(t) is not int or type(x) is not int:  # the fast path: Python ints
            if not all(_whole(v) for v in (t, x)):
                raise ValueError(f"track points must be whole pixel coordinates, got {tuple(p)}")
            t, x = int(t), int(x)
        out.append((t, x))
    return out


def _whole(v) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, Integral):
        return True
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and f.is_integer()


def get_derived_quantities(track: Track | Sequence) -> TrackStatistics | None:
    """``getDerivedQuantities`` (KymoButlerPProc.wl): kinetic parameters of one track.

    ``track`` is a ``Track`` or a sequence of (time, space) pixel coordinates, used in their
    order. Returns ``None`` for a track with fewer than two points (Mathematica's
    ``Missing[]``).

    Raises:
        ValueError: if two consecutive points lie on the same frame (``Ratios`` of ``{0, dx}``
            is ``ComplexInfinity`` in Mathematica). KymoButler tracks have one point per frame.
    """
    pts = track_points(track)
    if len(pts) < 2:
        return None
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])]
    for i, (dt, _) in enumerate(steps):
        if dt == 0:
            raise ValueError(
                f"track has two consecutive points on the same frame ({pts[i]} and "
                f"{pts[i + 1]}): its frame-to-frame velocity is undefined"
            )
    # Mean of |dx|/|dt|, exact: sum the |dx| of the steps with the same |dt| first (few distinct
    # |dt|, so few Fractions), which gives the same rational as summing the ratios one by one.
    dist_by_dt: dict[int, int] = {}
    for dt, dx in steps:
        dist_by_dt[abs(dt)] = dist_by_dt.get(abs(dt), 0) + abs(dx)
    v = sum((Fraction(s, dt) for dt, s in dist_by_dt.items()), Fraction(0)) / len(steps)
    dx_net = pts[-1][1] - pts[0][1]
    return TrackStatistics(
        direction=(dx_net > 0) - (dx_net < 0),
        mean_velocity=v,
        total_distance=sum(abs(dx) for _, dx in steps),
        duration=abs(pts[-1][0] - pts[0][0] + 1),
    )


def _inverse(x: int | float) -> Fraction | float:
    """``Power[x, -1]``: exact for an integer, a machine real otherwise."""
    return Fraction(1, x) if isinstance(x, int) else 1 / x


def _scaled_velocity(v: float, tsz: int | float, xsz: int | float) -> float:
    """``xsz/tsz*v`` = ``Times[xsz, Power[tsz, -1], v]``: the exact factors are combined and
    converted first, then the real factors are multiplied left to right (measured on
    Mathematica 14.3 for every combination of integer and real pixel sizes)."""
    exact, reals = Fraction(1), []
    for f in (xsz, _inverse(tsz)):
        if isinstance(f, (int, Fraction)):
            exact *= f
        else:
            reals.append(f)
    factors = ([mma_n(exact)] if exact != 1 else []) + reals + [v]
    out = factors[0]
    for f in factors[1:]:
        out *= f
    return out


def _start2end(dist: float, T: float, tsz: int | float, xsz: int | float) -> float:
    """``xsz*dist/T/tsz`` = ``Times[xsz, dist, Power[T, -1], Power[tsz, -1]]``, evaluated as
    ``xsz * ((dist * (1/T)) * (1/tsz))`` (measured on Mathematica 14.3)."""
    inv_tsz = _inverse(tsz)
    return float(xsz) * ((dist * (1 / T)) * (mma_n(inv_tsz) if isinstance(tsz, int) else inv_tsz))


def _row(stats: TrackStatistics, tsz: int | float, xsz: int | float) -> dict:
    # quant = Round[N@getDerivedQuantities@trk, 0.0001]
    direct = mma_round(float(stats.direction))
    v = mma_round(mma_n(stats.mean_velocity))
    dist = mma_round(float(stats.total_distance))
    T = mma_round(float(stats.duration))
    if T == 0:
        raise ValueError(
            "track ends one frame before it starts (duration T = t_last - t_first + 1 = 0): "
            "its start-to-end velocity is undefined"
        )
    return {
        "direction": direct,
        "velocity_um_per_sec": mma_round(_scaled_velocity(v, tsz, xsz)),
        "duration_sec": mma_round(tsz * T),
        "distance_um": mma_round(xsz * dist),
        "start2end_velocity_um_per_sec": mma_round(_start2end(dist, T, tsz, xsz)),
    }


def postprocess(
    tracks: Iterable[Track | Sequence],
    pixel_size_time: float = 1.0,
    pixel_size_space: float = 1.0,
) -> list[dict]:
    """Per-track statistics of ``pproc`` (KymoButlerPProc.wl), one row per track, in order.

    Args:
        tracks: Tracks (or sequences of (time, space) pixel coordinates) in output order;
            for unidirectional analyses the anterograde tracks, then the retrograde ones.
        pixel_size_time: Seconds per frame (``tsz``).
        pixel_size_space: Micrometers per pixel (``xsz``).

    Returns:
        One dict per track with the keys of ``STAT_KEYS`` (the summary.csv columns
        "Direction", "Av frame2frame velocity [um/sec]", "track duration [sec]",
        "track total displacement [um]", "Start2end velocity [um/sec]"), holding the
        values Mathematica computes (machine reals; the direction is 1., -1. or 0.).

    Raises:
        ValueError: for a track with fewer than two points (Mathematica writes ``Missing[]``
            expressions into the summary; KymoButler never outputs such a track when
            ``min_frames >= 1``), a track whose statistics are undefined (see
            ``get_derived_quantities``), or a pixel size that is not a positive number.
    """
    tsz, xsz = mma_number(pixel_size_time), mma_number(pixel_size_space)
    _check_pixel_size("pixel_size_time", tsz)
    _check_pixel_size("pixel_size_space", xsz)
    rows = []
    for k, track in enumerate(tracks, 1):
        try:
            stats = get_derived_quantities(track)
            if stats is None:
                raise ValueError(
                    "it has fewer than two points, so it has no velocity or duration "
                    "(use min_frames >= 1)"
                )
            rows.append(_row(stats, tsz, xsz))
        except ValueError as exc:
            raise ValueError(f"track #{k}: {exc}") from None
    return rows
