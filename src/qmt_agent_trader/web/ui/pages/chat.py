"""Chat page — natural conversation, no forced mode selection."""

from __future__ import annotations

from nicegui import ui

from qmt_agent_trader.web.ui.layout import shell

SUGGESTED_PROMPTS = [
    "帮我发现几个适合A股个股和ETF的低波动高胜率因子，并自动跑初步验证。",
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
            "Ask anything about quantitative research — the Agent will route automatically."
        ).classes("text-sm text-gray-500 mb-4")

        with ui.column().classes("w-full gap-4"):
            # ── Transcript ──
            transcript = ui.markdown(
                "**Assistant:** Ready for research. Type your question below."
            ).classes("min-h-[300px] p-4 border rounded-lg bg-gray-50")

            # ── Agent Plan card (initially hidden) ──
            plan_card = ui.card().classes("w-full bg-blue-50 p-4")
            plan_card.visible = False

            # ── Composer: natural language + Advanced (collapsed) ──
            with ui.row().classes("w-full items-end gap-2"):
                message = (
                    ui.textarea("Type your research question...")
                    .classes("grow")
                    .props("autogrow rows=2")
                )
                with ui.column().classes("gap-1"):
                    ui.button(
                        "Send",
                        on_click=lambda: _send(transcript, message, plan_card),
                    ).props("color=primary")

            # ── Advanced settings (collapsed by default) ──
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
    transcript: ui.markdown, message_input: ui.textarea, plan_card: ui.card
) -> None:
    content = (message_input.value or "").strip()
    if not content:
        ui.notify("Enter a message first.", type="warning")
        return

    used_api = False

    # Try API roundtrip (requires server running)
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

            r2 = await client.post(
                f"http://127.0.0.1:7860/api/chat/sessions/{session['session_id']}/messages",
                json={"content": content},
                timeout=5,
            )
            r2.raise_for_status()
            result = r2.json()

            rd = result.get("routing_decision", {})
            assistant_content = result.get("message", {}).get("content", "")
            used_api = True
    except Exception:
        # Fallback: local router directly
        pass

    if not used_api:
        from qmt_agent_trader.agent.router import agent_router

        decision_obj = agent_router.route(content)
        rd = {
            "intent": decision_obj.intent.value,
            "confidence": decision_obj.confidence,
            "rationale": decision_obj.rationale,
            "required_tools": decision_obj.required_tools,
            "proposed_workflow": decision_obj.proposed_workflow,
        }
        assistant_content = (
            f"**Intent:** {decision_obj.intent.value} "
            f"(confidence: {decision_obj.confidence:.0%})\n\n"
            f"{decision_obj.rationale}\n\n"
            "Full LLM orchestration is stubbed in v1."
        )

    # ── Show Agent Plan ──
    intent = rd.get("intent", "GENERAL_RESEARCH")
    confidence = rd.get("confidence", 0.0)
    rationale = rd.get("rationale", "")
    tools = rd.get("required_tools", [])
    workflow = rd.get("proposed_workflow")

    plan_html = (
        f"### 🤖 Agent Plan\n\n"
        f"| | |\n|---|---|\n"
        f"| **Intent** | `{intent}` |\n"
        f"| **Confidence** | {confidence:.0%} |\n"
    )
    if workflow:
        plan_html += f"| **Workflow** | `{workflow}` |\n"
    plan_html += f"\n**Rationale:** {rationale}\n\n"
    if tools:
        plan_html += (
            f"**Tools:** {', '.join(tools[:8])}"
            f"{'…' if len(tools) > 8 else ''}\n"
        )

    plan_card.clear()
    with plan_card:
        ui.markdown(plan_html)
    plan_card.visible = True

    # ── Update transcript ──
    transcript.set_content(
        f"**You:** {content}\n\n**Assistant:** {assistant_content}"
    )
    message_input.value = ""
