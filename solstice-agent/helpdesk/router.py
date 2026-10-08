"""Routes a user message to the right tool."""

from .tools import TOOLS
import re
ROUTER_PROMPT = """You are the routing layer for a customer-support agent.

Available tools:
- search_kb: search the Solstice help-center knowledge base for policies, plans, setup guides, and troubleshooting
- lookup_ticket: look up an existing support ticket by ticket ID
- list_user_tickets: list the support tickets filed by an email address
- invoice_status: check the status of an invoice or a recent charge

User message: {message}

Respond with the name of the single best tool."""


def route(llm, message):
    m = message.lower()
    # fast paths — skip the LLM call for obvious cases
    #Before
    # if "ticket" in m:
    #     return "lookup_ticket"
    # if "invoice" in m or "charge" in m or "billing" in m:
    #     return "invoice_status"

    #After
    # if re.search(r"INV-\d+", m):
    #     return "invoice_status"
    # if re.search(r"TKT-\d+", m):
    #     return "lookup_ticket"

    # Fallback: ask the LLM to choose the tool
    name = llm.complete(ROUTER_PROMPT.format(message=message)).strip()
    if name not in TOOLS:
        name = "search_kb"

    # Guard: data-lookup tools are useless without a concrete identifier.
    # Fall back to search_kb if the required identifier is missing.
    if name == "invoice_status" and not re.search(r"INV-\d+", message, re.I):
        name = "search_kb"
    if name == "lookup_ticket" and not re.search(r"TKT-\d+", message, re.I):
        name = "search_kb"
    if name == "list_user_tickets" and not re.search(r"[\w.+-]+@[\w.-]+", message):
        name = "search_kb"

    return name
