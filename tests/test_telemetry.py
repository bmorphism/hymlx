"""Tests for HyMLX telemetry system."""

import pytest
import time


def test_telemetry_import():
    """Test telemetry module loads."""
    from hymlx import telemetry
    assert hasattr(telemetry, "get_context")
    assert hasattr(telemetry, "TritType")
    assert hasattr(telemetry, "Event")


def test_event_ring_buffer():
    """Test ring buffer operations."""
    from hymlx.telemetry import EventRing, Event
    
    ring = EventRing(capacity=10)
    
    # Emit events
    for i in range(5):
        event = Event(
            name=f"test_{i}",
            trit=i % 3 - 1,  # -1, 0, 1, -1, 0
            timestamp=time.time()
        )
        ring.emit(event)
    
    assert len(ring) == 5
    
    # Peek doesn't consume
    peeked = ring.peek(3)
    assert len(peeked) == 3
    assert len(ring) == 5
    
    # Drain consumes
    drained = ring.drain(2)
    assert len(drained) == 2
    assert len(ring) == 3
    
    # Drain all
    remaining = ring.drain()
    assert len(remaining) == 3
    assert len(ring) == 0


def test_gf3_trit_types():
    """Test GF(3) trit classification."""
    from hymlx.telemetry import TritType
    
    assert TritType.PLUS == 1
    assert TritType.ERGODIC == 0
    assert TritType.MINUS == -1
    
    # GF(3) conservation
    assert (TritType.PLUS + TritType.MINUS + TritType.ERGODIC) % 3 == 0


def test_context_emit_and_balance():
    """Test context emission and GF(3) balance tracking."""
    from hymlx.telemetry import reset_context, get_context, TritType
    
    reset_context()
    ctx = get_context()
    
    # Emit balanced triad
    ctx.emit("gen", TritType.PLUS)
    ctx.emit("coord", TritType.ERGODIC)
    ctx.emit("val", TritType.MINUS)
    
    # Balance should be 0 after balanced emissions
    assert ctx.gf3_balance() == 0
    
    # Unbalanced
    ctx.emit("extra", TritType.PLUS)
    assert ctx.gf3_balance() == 1


def test_emit_helpers():
    """Test typed emit helpers."""
    from hymlx.telemetry import (
        reset_context, get_context,
        emit_plus, emit_ergodic, emit_minus
    )
    
    reset_context()
    ctx = get_context()
    
    emit_plus("generate")
    emit_ergodic("sync")
    emit_minus("validate")
    
    events = ctx.ring.drain()
    assert len(events) == 3
    assert events[0][1].trit == 1
    assert events[1][1].trit == 0
    assert events[2][1].trit == -1


def test_subscription():
    """Test event subscription for TUI."""
    from hymlx.telemetry import reset_context, get_context
    
    reset_context()
    ctx = get_context()
    
    received = []
    unsubscribe = ctx.subscribe(lambda seq, event: received.append((seq, event.name)))
    
    ctx.emit("test1", 0)
    ctx.emit("test2", 1)
    
    assert len(received) == 2
    assert received[0] == (0, "test1")
    assert received[1] == (1, "test2")
    
    # Unsubscribe
    unsubscribe()
    ctx.emit("test3", -1)
    assert len(received) == 2  # No new events


def test_jsonl_export():
    """Test JSON Lines export for replay."""
    import json
    from hymlx.telemetry import reset_context, get_context
    
    reset_context()
    ctx = get_context()
    
    ctx.emit("event1", 1, {"key": "value"})
    ctx.emit("event2", 0)
    
    jsonl = ctx.export_jsonl()
    lines = jsonl.strip().split("\n")
    
    assert len(lines) == 2
    
    data = json.loads(lines[0])
    assert data["name"] == "event1"
    assert data["trit"] == 1
    assert data["payload"] == {"key": "value"}


def test_turn_tracker():
    """Test model turn tracking."""
    from hymlx.telemetry import reset_context, get_context, TurnTracker
    
    reset_context()
    ctx = get_context()
    tracker = TurnTracker(ctx)
    tracker.attach()
    
    # First turn
    turn_id = tracker.start_turn()
    ctx.emit("inference", 1)
    ctx.emit("output", 1)
    tracker.end_turn()
    
    # Second turn (child of first)
    tracker.start_turn(parent=turn_id)
    ctx.emit("refinement", 0)
    tracker.end_turn()
    
    tracker.detach()
    
    export = tracker.export()
    assert len(export) == 2
    assert export[1]["parent"] == turn_id


def test_toad_protocol():
    """Test TUI protocol formatting."""
    from hymlx.telemetry import ToadProtocol, Event, get_context, reset_context
    
    reset_context()
    ctx = get_context()
    toad = ToadProtocol(ctx)
    
    event = Event(name="test:event", trit=1, timestamp=time.time())
    
    # With ANSI
    formatted = toad.format_event(event, ansi=True)
    assert "[+]" in formatted
    assert "test:event" in formatted
    
    # Without ANSI
    formatted = toad.format_event(event, ansi=False)
    assert formatted == "[+] test:event"


def test_traced_transforms():
    """Test traced transform variants."""
    import mlx.core as mx
    from hymlx.transforms import traced_jit, traced_grad
    from hymlx.telemetry import reset_context, get_context
    
    reset_context()
    ctx = get_context()
    
    @traced_jit
    def add_one(x):
        return x + 1
    
    result = add_one(mx.array([1.0, 2.0, 3.0]))
    mx.eval(result)
    
    events = ctx.ring.drain()
    names = [e[1].name for e in events]
    
    assert any("jit:enter" in n for n in names)
    assert any("jit:exit" in n for n in names)


def test_load_and_replay():
    """Test event loading and replay."""
    import json
    from hymlx.telemetry import load_jsonl, Event
    
    jsonl = '\n'.join([
        json.dumps({"name": "e1", "trit": 1, "timestamp": 1000.0}),
        json.dumps({"name": "e2", "trit": 0, "timestamp": 1001.0}),
    ])
    
    events = load_jsonl(jsonl)
    assert len(events) == 2
    assert events[0].name == "e1"
    assert events[1].trit == 0
