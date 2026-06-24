"""Chat page — natural conversation with real LLM orchestration via SSE."""

from __future__ import annotations

import json

from nicegui import ui

from qmt_agent_trader.web.ui.layout import shell

SUGGESTED_PROMPTS = [
    "帮我发现几个适合A股个股和ETF的低波动高胜率因子，并自动跑初步验证。",
    "列出当前数据湖中所有可用的因子，并验证 momentum_20d 因子。",
    "基于最近有效的候选因子，写一个日频轮动策略并回测。",
    "看看最近失败的实验，判断是不是缺少某个工具。",
    "解释一下上一个回测为什么收益高但回撤也大。",
]


def register() -> None:
    @ui.page("/")
    def chat_page() -> None:
        shell("Chat")

        ui.label("QMT Agent Studio").classes("text-2xl font-semibold")
        ui.label(
            "Ask anything about quantitative research — Agent routes and executes automatically."
        ).classes("text-sm text-gray-500 mb-4")

        with ui.column().classes("w-full gap-4"):
            # ── Transcript (markdown area for agent output) ──
            transcript = ui.markdown(
                "**🤖 Assistant:** Ready for research. Type your question and press Send."
            ).classes("min-h-[300px] p-4 border rounded-lg bg-gray-50")

            # ── Agent Plan card ──
            plan_card = ui.card().classes("w-full bg-blue-50 p-4")
            plan_card.visible = False

            # ── Live tool-call progress card ──
            progress_card = ui.card().classes("w-full bg-green-50 p-4")
            progress_card.visible = False

            # ── Composer ──
            with ui.row().classes("w-full items-end gap-2"):
                message = (
                    ui.textarea("Type your research question...")
                    .classes("grow")
                    .props("autogrow rows=2")
                )
                with ui.column().classes("gap-1"):
                    ui.button(
                        "Send",
                        on_click=lambda: _send(transcript, message, plan_card, progress_card),
                    ).props("color=primary")

            # ── Advanced settings ──
            with ui.expansion("Advanced", icon="tune").classes("w-full"):
                with ui.row().classes("gap-4"):
                    ui.select(
                        ["auto", "stock", "etf", "stock_etf"],
                        value="auto",
                        label="Universe",
                    ).classes("w-40")
                    with ui.row().classes("gap-2"):
                        ui.input("Start Date", value="").classes("w-36").props("placeholder=auto")
                        ui.input("End Date", value="").classes("w-36").props("placeholder=auto")
                    ui.select(
                        ["balanced", "fast", "thorough"],
                        value="balanced",
                        label="Budget",
                    ).classes("w-36")

        # ── Suggested prompts ──
        with ui.expansion("Suggested prompts", icon="lightbulb").classes("w-full mt-4"):
            for p in SUGGESTED_PROMPTS:
                ui.chip(p, on_click=lambda _, text=p: _fill_prompt(message, text))


def _fill_prompt(message_input: ui.textarea, text: str) -> None:
    message_input.value = text


async def _send(
    transcript: ui.markdown,
    message_input: ui.textarea,
    plan_card: ui.card,
    progress_card: ui.card,
) -> None:
    content = (message_input.value or "").strip()
    if not content:
        ui.notify("Enter a message first.", type="warning")
        return

    # Clear previous state
    message_input.value = ""
    transcript_lines = [f"**🧑 You:** {content}", ""]
    plan_card.visible = False
    progress_card.visible = False

    # ── Step 1: Get routing ──
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            r = await client.post(
                "http://127.0.0.1:7860/api/chat/sessions",
                json={"title": content[:60]},
                timeout=5,
            )
            r.raise_for_status()
            session = r.json()
            sid = session["session_id"]
    except Exception as exc:
        transcript_lines.append("**⚠️ Could not reach Agent Studio server.**")
        transcript_lines.append(f"Error: {exc}")
        transcript_lines.append("Make sure `qmt-agent web` is running on port 7860.")
        transcript.set_content("\n".join(transcript_lines))
        return

    # ── Step 2: Run real SSE orchestration ──
    progress_card.clear()
    with progress_card:
        ui.spinner(size="sm")
        ui.label("Running LLM orchestration...").classes("text-sm")
    progress_card.visible = True

    try:
        import httpx

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            async with client.stream(
                "POST",
                f"http://127.0.0.1:7860/api/chat/sessions/{sid}/execute",
                json={"message": content},
            ) as response:
                if response.status_code != 200:
                    transcript_lines.append(
                        f"**❌ Server error (HTTP {response.status_code}).**"
                    )
                    transcript.set_content("\n".join(transcript_lines))
                    progress_card.visible = False
                    return

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    # SSE format: "event: <type>" or "data: <json>"
                    if line.startswith("event: "):
                        continue  # handled via data payload
                    if not line.startswith("data: "):
                        continue

                    data_str = line[len("data: "):]
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type", "")
                    emsg = event.get("message", "")

                    if etype == "routing":
                        # Show agent plan
                        intent = event.get("intent", "?")
                        conf = event.get("confidence", 0)
                        plan_card.clear()
                        with plan_card:
                            ui.markdown(
                                f"### 🤖 Agent Plan\n\n"
                                f"| | |\n|---|---|\n"
                                f"| **Intent** | `{intent}` |\n"
                                f"| **Confidence** | {conf:.0%} |\n"
                            )
                        plan_card.visible = True

                    elif etype == "run_started":
                        transcript_lines.append(
                            f"**🚀 Running** — `{event.get('data', {}).get('experiment_id', '?')}`"
                        )
                        progress_card.clear()
                        with progress_card:
                            ui.spinner(size="sm")
                            ui.label(emsg).classes("text-sm")
                        progress_card.visible = True

                    elif etype in ("progress", "tool_done"):
                        tool_name = event.get("data", {}).get("tool_name", "")
                        idx = event.get("data", {}).get("index", 0)
                        total = event.get("data", {}).get("total", 0)
                        if tool_name:
                            line = f"**🔧 [{idx}/{total}]** `{tool_name}`"
                        else:
                            line = f"**🔄** {emsg}"
                        transcript_lines.append(line)
                        if emsg:
                            progress_card.clear()
                            with progress_card:
                                ui.spinner(size="sm")
                                ui.label(emsg).classes("text-sm")
                            progress_card.visible = True

                    elif etype == "llm_message":
                        transcript_lines.append("")
                        transcript_lines.append("**🤖 Assistant:**")
                        transcript_lines.append(emsg)

                    elif etype == "done":
                        progress_card.visible = False
                        plan_card.visible = False
                        tool_count = event.get("data", {}).get("tool_calls_count", 0)
                        transcript_lines.append("")
                        transcript_lines.append(
                            f"**✅ Done** — {tool_count} tool call(s) completed."
                        )

                    elif etype == "error":
                        progress_card.visible = False
                        transcript_lines.append(f"**❌ Error:** {emsg}")

                    # Update display
                    transcript.set_content("\n".join(transcript_lines))

    except httpx.ConnectError:
        transcript_lines.append(
            "**⚠️ Cannot connect to Agent Studio SSE endpoint.**\n"
            "Make sure `qmt-agent web` is running."
        )
        progress_card.visible = False
        transcript.set_content("\n".join(transcript_lines))
    except Exception as exc:
        transcript_lines.append(f"**❌ SSE error:** {exc}")
        progress_card.visible = False
        transcript.set_content("\n".join(transcript_lines))
