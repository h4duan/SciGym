import os
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from scigym.api import LLM


class GPT(LLM):
    """Implementation for OpenAI GPT models"""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: str = os.getenv("OPENAI_API_KEY", ""),
        system_prompt: str = "",
        temperature: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            system_prompt=system_prompt,
            temperature=temperature,
            *args,
            **kwargs,
        )

    def initialize(self):
        self.client = OpenAI(api_key=self.api_key)
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        """Get the full message history"""
        return self.messages

    def get_response(self, user_message: str) -> tuple:
        """Get a response from OpenAI API"""
        # Add user message to history
        self.add_message("user", user_message)

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            max_tokens=self.max_length,
            temperature=self.temperature,
        )

        # Get response text
        response_text = response.choices[0].message.content
        assert isinstance(response_text, str) and len(response_text) > 0

        # Add assistant response to history
        self.add_message("assistant", response_text)

        return response_text, {}
