from decimal import Decimal

from core.action_policy import ActionCategory, ActionPolicy
from core.agent import Agent, Genome, ToolSpec, TOOL_REQUEST_PREFIX
from core.wallet import Wallet


def make_agent(llm_decide):
    def hold():
        return "held"

    tools = {
        "hold": ToolSpec(name="hold", category=ActionCategory.MARKET_DATA_READ, cost=Decimal("0.05"), run=hold),
    }
    wallet = Wallet(100)
    genome = Genome(enabled_tools=("hold",))
    return Agent("root", wallet, ActionPolicy(), genome, tools, llm_decide=llm_decide), wallet


def test_normal_decision_runs_tool_and_charges_cost():
    agent, wallet = make_agent(lambda strategy_prompt, options, temperature: "hold")
    result = agent.step()
    assert result == "held"
    assert wallet.balance == Decimal("99.95")


def test_tool_request_is_recorded_without_charging_or_running_anything():
    agent, wallet = make_agent(
        lambda strategy_prompt, options, temperature: f"{TOOL_REQUEST_PREFIX} needs a way to send email"
    )
    result = agent.step()
    assert result is None
    assert agent.tool_requests == ["needs a way to send email"]
    assert wallet.balance == Decimal("100.00")  # nothing charged - no tool actually ran


def test_multiple_tool_requests_accumulate():
    agent, _ = make_agent(lambda strategy_prompt, options, temperature: f"{TOOL_REQUEST_PREFIX} thing A")
    agent.step()
    agent.step()
    assert agent.tool_requests == ["thing A", "thing A"]
