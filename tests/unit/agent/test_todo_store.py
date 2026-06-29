from __future__ import annotations

import pytest

from qmt_agent_trader.agent.todos import TodoListStore, TodoStatus


def test_todo_store_hashes_session_id_for_path_safety(tmp_path) -> None:
    store = TodoListStore(tmp_path / "todos")

    path = store.path_for_session("../escape/session")

    assert path.parent == tmp_path / "todos"
    assert path.name.endswith(".json")
    assert ".." not in path.name
    assert "/" not in path.name


def test_todo_store_persists_items_across_instances(tmp_path) -> None:
    store = TodoListStore(tmp_path / "todos")
    record = store.replace_items(
        "chat_1",
        [{"title": "确认数据覆盖"}, {"title": "运行回测", "notes": "低波动策略"}],
        goal="研究低波动策略",
    )

    reloaded = TodoListStore(tmp_path / "todos").get("chat_1")

    assert reloaded.session_id == "chat_1"
    assert reloaded.goal == "研究低波动策略"
    assert [item.title for item in reloaded.items] == ["确认数据覆盖", "运行回测"]
    assert reloaded.items[0].item_id == record.items[0].item_id


def test_todo_store_allows_only_one_in_progress_item(tmp_path) -> None:
    store = TodoListStore(tmp_path / "todos")
    record = store.replace_items(
        "chat_1",
        [{"title": "第一步"}, {"title": "第二步"}],
    )

    first = record.items[0].item_id
    second = record.items[1].item_id
    store.update_item("chat_1", first, status=TodoStatus.IN_PROGRESS)
    updated = store.update_item("chat_1", second, status=TodoStatus.IN_PROGRESS)

    statuses = {item.item_id: item.status for item in updated.items}
    assert statuses[first] == TodoStatus.PENDING
    assert statuses[second] == TodoStatus.IN_PROGRESS
    assert updated.active_item is not None
    assert updated.active_item.item_id == second


def test_todo_store_rejects_too_many_items(tmp_path) -> None:
    store = TodoListStore(tmp_path / "todos")

    with pytest.raises(ValueError, match="at most 50"):
        store.replace_items(
            "chat_1",
            [{"title": f"任务 {index}"} for index in range(51)],
        )


def test_todo_store_rejects_long_titles(tmp_path) -> None:
    store = TodoListStore(tmp_path / "todos")

    with pytest.raises(ValueError, match="title must be 1-200"):
        store.replace_items("chat_1", [{"title": "x" * 201}])
