from __future__ import annotations
import asyncio
from groq import Groq
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, TextLog

class GroqApp(App):
    """Simple terminal UI for interacting with the Groq API."""

    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield TextLog(id="log", highlight=False)
        yield Input(placeholder="Type your prompt and press Enter", id="prompt")
        yield Footer()

    async def on_mount(self) -> None:
        self.client = Groq()
        self.model = "llama3-groq-70b-8192-tool-use-preview"
        self.query_one(Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value
        event.input.value = ""
        log = self.query_one(TextLog)
        log.write(f"> {prompt}")
        response = await self.call_groq(prompt)
        log.write(response)
        log.write("")

    async def call_groq(self, prompt: str) -> str:
        return await asyncio.to_thread(self._call_groq_sync, prompt)

    def _call_groq_sync(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return resp.choices[0].message.content.strip()

if __name__ == "__main__":
    GroqApp().run()
