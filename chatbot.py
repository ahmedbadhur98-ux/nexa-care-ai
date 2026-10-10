"""AutoCare AI - Intelligent Car Service Advisor Chatbot.

This module provides a command-line chat session with an AI car service advisor
grounded in the business data, services, prices, and operating instructions from
`car_service_context.json`.

It reuses the OpenAI Responses API configuration and model settings from
`chat_llm.ipynb`, supports multi-turn session memory, and securely loads the API
key from `.env` without printing or exposing it.
"""

import json
import os
import sys
from pathlib import Path
from dotenv import dotenv_values, load_dotenv
from openai import OpenAI

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
CONTEXT_PATH = PROJECT_ROOT / "car_service_context.json"

# Model configuration matching chat_llm.ipynb
DEFAULT_MODEL = "gpt-5.6"
TEXT_CONFIG = {"verbosity": "low"}


def load_environment(env_file: Path = ENV_PATH) -> None:
    """Securely load environment variables from the project .env file.

    Validates that the file exists and that OPENAI_API_KEY is present without
    ever printing or logging the key value.
    """
    if not env_file.is_file():
        raise FileNotFoundError(
            f".env is missing at expected path: {env_file}. "
            "Please create it in the project root."
        )

    # Validate that OPENAI_API_KEY is defined and non-empty
    raw_values = dotenv_values(env_file)
    api_key_val = (raw_values.get("OPENAI_API_KEY") or "").strip()
    if not api_key_val:
        raise RuntimeError("OPENAI_API_KEY is missing or empty in the project .env file.")

    # Load into system environment for the OpenAI SDK
    load_dotenv(dotenv_path=env_file, override=True)

    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        raise RuntimeError("OPENAI_API_KEY could not be loaded into the environment.")


def load_car_service_context(json_file: Path = CONTEXT_PATH) -> dict:
    """Load and validate the business context JSON file."""
    if not json_file.is_file():
        raise FileNotFoundError(f"Context file not found: {json_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def build_system_instructions(context: dict) -> str:
    """Construct structured advisor instructions grounded in the business context."""
    business_name = context.get("business_name", "AutoCare AI")
    location = context.get("location", "Chennai")
    business_type = context.get("business_type", "Car Service Center")
    working_hours = context.get("working_hours", "9 AM to 7 PM")
    services = context.get("services", [])
    supported_cars = context.get("supported_cars", [])
    raw_instructions = context.get("instructions", [])

    # Format services with prices
    services_formatted = "\n".join(
        f"- {s.get('name', 'Service')}: INR {s.get('price', 'N/A')}"
        for s in services
    )

    # Format supported vehicle makes
    cars_formatted = ", ".join(supported_cars) if supported_cars else "All major consumer cars"

    # Format rules
    rules_formatted = "\n".join(f"- {rule}" for rule in raw_instructions)

    prompt = f"""You are the official AI Car Service Advisor for {business_name}, a premier {business_type} located in {location}.

### Center Profile:
- Business Name: {business_name}
- Type: {business_type}
- Location: {location}
- Working Hours: {working_hours}
- Supported Car Brands: {cars_formatted}

### Official Services & Pricing:
{services_formatted}

### Core Operating Rules:
{rules_formatted}

### Role & Diagnostic Behavior:
1. Persona: Friendly, polite, knowledgeable, and professional car service advisor.
2. Automotive Diagnostics & Reasoning:
   - When customers describe car issues (e.g., strange noises, vibration, warning lights, poor cooling, starting troubles), ask clarifying diagnostic questions (car make/model, symptoms, when it occurs) to understand the root cause.
   - Use genuine automotive knowledge to explain plausible causes in clear, accessible terms.
   - Suggest relevant services from our official offerings (e.g., Brake Inspection for squealing/grinding brakes, AC Service for cooling issues, General Service for periodic maintenance).
3. Grounding & Strict Business Boundaries:
   - Only quote the official service prices listed above.
   - Never invent business facts, additional package prices, discounts, or policies not in your context.
   - If a customer's car brand is not among our supported brands ({cars_formatted}), politely inform them.
   - If asked for an unlisted service, explain that it is not part of our standard catalog and invite them to visit or contact the center during working hours ({working_hours}) for an assessment.
4. Booking Policy:
   - Strictly DO NOT confirm bookings or appointments.
   - Inform the customer that appointments cannot be booked directly through this chat, and advise them to visit the center in {location} or reach out during working hours ({working_hours}).
5. Safety Priority:
   - For severe or safety-critical issues (e.g., brake failure, loss of steering control, burning odor, engine overheating, fuel leaks), clearly advise caution and urge immediate professional inspection or roadside assistance.
"""
    return prompt.strip()


class CarServiceAdvisorChatbot:
    """Manages the conversation state and interactions with the OpenAI API."""

    def __init__(
        self,
        client: OpenAI,
        instructions: str,
        model: str = DEFAULT_MODEL,
        text_config: dict = None,
    ):
        self.client = client
        self.instructions = instructions
        self.model = model
        self.text_config = text_config or TEXT_CONFIG
        self.previous_response_id: str | None = None
        self.turn_count = 0

    def reset_conversation(self) -> None:
        """Clear conversation history to start a fresh interaction."""
        self.previous_response_id = None
        self.turn_count = 0

    def ask(self, user_message: str) -> str:
        """Send a message to the advisor and return the response."""
        call_kwargs = {
            "model": self.model,
            "input": user_message,
            "instructions": self.instructions,
            "text": self.text_config,
        }

        # Maintain session memory using previous_response_id
        if self.previous_response_id is not None:
            call_kwargs["previous_response_id"] = self.previous_response_id

        response = self.client.responses.create(**call_kwargs)

        # Track the response ID for conversation continuity
        self.previous_response_id = response.id
        self.turn_count += 1
        return response.output_text


def run_cli() -> None:
    """Run the interactive terminal chat loop."""
    print("=" * 64)
    print("  AutoCare AI - Intelligent Car Service Advisor")
    print("=" * 64)
    print("Initializing advisor...")

    # Load environment securely
    try:
        load_environment(ENV_PATH)
    except Exception as e:
        print(f"\n[Configuration Error] {e}")
        sys.exit(1)

    # Load business context
    try:
        context = load_car_service_context(CONTEXT_PATH)
    except Exception as e:
        print(f"\n[Data Error] Could not load context: {e}")
        sys.exit(1)

    # Initialize OpenAI client and advisor system prompt
    client = OpenAI()
    instructions = build_system_instructions(context)
    bot = CarServiceAdvisorChatbot(client=client, instructions=instructions)

    business_name = context.get("business_name", "AutoCare AI")
    location = context.get("location", "Chennai")
    hours = context.get("working_hours", "9 AM to 7 PM")

    print(f"Connected to {business_name} ({location})")
    print(f"Operating Hours: {hours}")
    print("-" * 64)
    print("Commands:")
    print("  - Type 'exit' or 'quit' to end the session.")
    print("  - Type 'reset' to start a fresh conversation.")
    print("-" * 64)
    print("Advisor is ready. Ask any question about your car or our services!\n")

    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChat ended.")
            break

        if not user_message:
            print("Please type a message, or enter 'exit' to stop.")
            continue

        if user_message.lower() in {"exit", "quit"}:
            print("Chat ended. Thank you for contacting AutoCare AI!")
            break

        if user_message.lower() == "reset":
            bot.reset_conversation()
            print("[Conversation history cleared. Started a fresh session.]\n")
            continue

        try:
            reply = bot.ask(user_message)
            print(f"\nAdvisor: {reply}\n")
        except Exception:
            print(
                "\nRequest failed. Check network access, API access, account billing "
                "or credits, model access, and your .env configuration.\n"
            )


if __name__ == "__main__":
    run_cli()
