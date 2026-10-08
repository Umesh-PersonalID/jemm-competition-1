"""The Solstice support agent: route -> gather context -> answer."""

from . import config
from .llm import LLMClient, LLMError
from .memory import Memory
from .router import route
from .tools import TOOLS

SYSTEM = (
    "You are the Solstice support agent. Answer the customer's question "
    "using the context provided. Be concise and friendly."
    "If the context does not contain enough information to answer, say: "
    "'I don't have that information — please contact support at help@solstice.app.'"
)


class Agent:
    def __init__(self, llm=None):
        self.llm = llm or LLMClient()
        self.memory = Memory()

    def handle(self, message):
        self.memory.add("user", message)

        tool_name = route(self.llm, message)
        context = TOOLS[tool_name](self.llm, message)

        prompt = (
            SYSTEM
            + "\n\nConversation:\n" + self.memory.render()
            + "\n\nContext:\n" + context
            + "\n\nQuestion: " + message
            + "\nAnswer:"
        )
        # if len(prompt) > config.MAX_PROMPT_CHARS:
        #     prompt = prompt[: config.MAX_PROMPT_CHARS]

        QUESTION_SUFFIX = "\n\nContext:\n" + context + "\n\nQuestion: " + message + "\nAnswer:"
        budget = config.MAX_PROMPT_CHARS - len(SYSTEM) - len(QUESTION_SUFFIX)
        history = self.memory.render()
        if len(history) > budget:
            history = history[-budget:]   # keep the most recent turns
        prompt = SYSTEM + "\n\nConversation:\n" + history + QUESTION_SUFFIX

        try:
            answer = self.llm.complete(prompt, temperature=0.2)
        except LLMError:
            answer = "Sorry — something went wrong on our end. Please try again."

        self.memory.add("assistant", answer)
        self.memory.maybe_summarize(self.llm)
        return answer
