"""Tests for the Agent Studio event bus."""

from __future__ import annotations

import asyncio

from qmt_agent_trader.web.event_bus import AgentEvent, AgentEventType, EventBus


def test_publish_subscribe_and_history() -> None:
    asyncio.run(_test_publish_subscribe_and_history())


async def _test_publish_subscribe_and_history() -> None:
    bus = EventBus()
    queue = await bus.subscribe("run_1")
    event = AgentEvent(
        run_id="run_1",
        event_type=AgentEventType.PROGRESS,
        title="Progress",
        message="Halfway",
    )

    await bus.publish(event)

    assert await queue.get() == event
    assert bus.get_history("run_1") == [event]


def test_run_subscriptions_are_isolated() -> None:
    asyncio.run(_test_run_subscriptions_are_isolated())


async def _test_run_subscriptions_are_isolated() -> None:
    bus = EventBus()
    run_1 = await bus.subscribe("run_1")
    run_2 = await bus.subscribe("run_2")

    await bus.publish(
        AgentEvent(
            run_id="run_1",
            event_type=AgentEventType.RUN_STARTED,
            title="Started",
            message="run 1",
        )
    )

    assert run_1.qsize() == 1
    assert run_2.qsize() == 0


def test_global_subscription_receives_all_runs() -> None:
    asyncio.run(_test_global_subscription_receives_all_runs())


async def _test_global_subscription_receives_all_runs() -> None:
    bus = EventBus()
    global_queue = await bus.subscribe("*")

    await bus.publish(
        AgentEvent(
            run_id="run_any",
            event_type=AgentEventType.RUN_COMPLETED,
            title="Done",
            message="complete",
        )
    )

    assert (await global_queue.get()).run_id == "run_any"
