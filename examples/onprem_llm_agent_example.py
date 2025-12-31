"""
Complete Example: Using On-Prem LLM with an Agent

This example shows how to configure and use an on-premise LLM provider
with a full agent implementation. This is useful for companies deploying
their own LLM infrastructure.

The example demonstrates:
1. Configuring an on-prem LLM provider (OpenAI-compatible API)
2. Setting up routing to use the on-prem model
3. Creating an agent that uses the on-prem LLM
4. Running the agent with the custom configuration
"""

import logging
from typing import Dict, Any, Optional

from curio_agent_sdk import (
    BaseAgent,
    AgentConfig,
    InMemoryPersistence,
    LLMService,
)
from curio_agent_sdk.llm.routing import LLMRoutingConfig
from curio_agent_sdk.llm.providers.openai import OpenAIProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleOnPremAgent(BaseAgent):
    """
    A simple agent configured to use an on-prem LLM provider.
    """
    
    def __init__(
        self,
        agent_id: str,
        onprem_base_url: str,
        onprem_api_key: str = "",
        onprem_model: str = "llama-2-70b-chat",
        config: Optional[AgentConfig] = None,
        persistence: Optional[Any] = None,
        llm_service: Optional[LLMService] = None,
    ):
        """
        Initialize agent with on-prem LLM configuration.
        
        Args:
            agent_id: Unique identifier for the agent
            onprem_base_url: Base URL for your on-prem LLM endpoint
            onprem_api_key: API key for authentication (empty if not required)
            onprem_model: Model name on your on-prem server
            config: Optional AgentConfig
            persistence: Optional persistence instance
            llm_service: Optional LLMService (will be created if not provided)
        """
        # Create LLM service with on-prem configuration if not provided
        if llm_service is None:
            llm_service = self._create_onprem_llm_service(
                onprem_base_url, onprem_api_key, onprem_model
            )
        
        super().__init__(
            agent_id=agent_id,
            config=config,
            persistence=persistence or InMemoryPersistence(),
            llm_service=llm_service,
            plan_tier="tier2",
            critique_tier="tier2",
            synthesis_tier="tier1",
            action_tier="tier1",
        )
        self.agent_name = "SimpleOnPremAgent"
        self.description = "Agent using on-prem LLM provider"
        self.max_iterations = 5
        self.initialize_tools()
    
    def _create_onprem_llm_service(
        self, base_url: str, api_key: str, model: str
    ) -> LLMService:
        """
        Create LLM service configured for on-prem provider.
        
        Args:
            base_url: Base URL for on-prem endpoint
            api_key: API key for authentication
            model: Model name
            
        Returns:
            Configured LLMService instance
        """
        # Create routing configuration
        routing_config = LLMRoutingConfig()
        
        # Register the on-prem provider
        routing_config.register_custom_provider(
            provider_name="onprem-llm",
            api_key=api_key,
            default_model=model,
            base_url=base_url,
            key_name="onprem-key",
        )
        
        # Configure tiers to use on-prem model
        # You can mix on-prem and cloud models if desired
        custom_tiers = {
            "tier1": [f"onprem-llm:{model}"],
            "tier2": [f"onprem-llm:{model}"],
            "tier3": [f"onprem-llm:{model}"],
        }
        
        # Recreate routing with custom tiers
        routing_config = LLMRoutingConfig(custom_tiers=custom_tiers)
        routing_config.register_custom_provider(
            provider_name="onprem-llm",
            api_key=api_key,
            default_model=model,
            base_url=base_url,
        )
        
        # Create LLM service
        # Use OpenAIProvider since most on-prem servers use OpenAI-compatible API
        persistence = InMemoryPersistence()
        llm_service = LLMService(
            persistence=persistence,
            routing_config=routing_config,
            custom_providers={"onprem-llm": OpenAIProvider},
        )
        
        logger.info(f"Created LLM service with on-prem provider: {base_url}")
        return llm_service
    
    def get_agent_instructions(self) -> str:
        """Define the agent's role and guidelines."""
        return """
You are a helpful assistant powered by an on-premise LLM.
You have access to tools for calculations, text manipulation, and answering questions.

## GUIDELINES
- Use the available tools to accomplish the objective
- Be efficient - don't use more tools than necessary
- If you can answer directly without tools, do so
- Always provide clear, concise responses
"""
    
    def initialize_tools(self) -> None:
        """Register tools for this agent."""
        self.register_tool("calculate", self.calculate_tool)
        self.register_tool("respond", self.respond_tool)
    
    def calculate_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        name: calculate
        description: Perform a mathematical calculation
        parameters:
            expression: A mathematical expression to evaluate
        required_parameters:
            - expression
        """
        try:
            expression = args.get("expression", "")
            allowed_chars = set("0123456789+-*/().% ")
            if not all(c in allowed_chars for c in expression):
                return {
                    "status": "error",
                    "result": "Invalid characters in expression"
                }
            result = eval(expression)
            return {"status": "ok", "result": f"Result: {result}"}
        except Exception as e:
            return {"status": "error", "result": f"Calculation error: {str(e)}"}
    
    def respond_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        name: respond
        description: Provide a response to the user
        parameters:
            message: The response message
        required_parameters:
            - message
        """
        message = args.get("message", "")
        return {"status": "ok", "result": f"Response: {message}"}


def main():
    """Main example function."""
    print("=" * 70)
    print("On-Prem LLM Agent Example")
    print("=" * 70)
    print()
    
    # Configuration for your on-prem LLM
    # Update these values to match your deployment
    ONPREM_BASE_URL = "https://llm.yourcompany.com/v1"  # Your on-prem endpoint
    ONPREM_API_KEY = "sk-your-api-key"  # Your API key (or "" if not required)
    ONPREM_MODEL = "llama-2-70b-chat"  # Your model name
    
    print("Configuration:")
    print(f"  Base URL: {ONPREM_BASE_URL}")
    print(f"  Model: {ONPREM_MODEL}")
    print(f"  API Key: {'***' if ONPREM_API_KEY else '(not required)'}")
    print()
    
    # Create agent with on-prem configuration
    print("Creating agent with on-prem LLM configuration...")
    agent = SimpleOnPremAgent(
        agent_id="onprem-agent-1",
        onprem_base_url=ONPREM_BASE_URL,
        onprem_api_key=ONPREM_API_KEY,
        onprem_model=ONPREM_MODEL,
    )
    
    print("Agent created successfully!")
    print()
    
    # Show agent status
    print("Agent Status:")
    status = agent.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    print()
    
    # Test the agent (this will fail if endpoint is not available)
    print("Testing agent with a simple task...")
    print("(Note: This will fail if the on-prem endpoint is not accessible)")
    print()
    
    try:
        result = agent.run(
            objective="Calculate 25 * 4 and tell me the result",
            additional_context={"user": "test_user"},
        )
        
        print(f"Result Status: {result.status}")
        print(f"Iterations: {result.total_iterations}")
        print(f"Summary: {result.synthesis_summary}")
        
    except Exception as e:
        print(f"Error running agent: {e}")
        print()
        print("This is expected if:")
        print("  1. The on-prem endpoint is not accessible")
        print("  2. The endpoint URL is incorrect")
        print("  3. Authentication failed")
        print()
        print("To fix:")
        print("  1. Update ONPREM_BASE_URL to your actual endpoint")
        print("  2. Update ONPREM_API_KEY if authentication is required")
        print("  3. Update ONPREM_MODEL to match your deployed model name")
    
    print()
    print("=" * 70)
    print("Example Complete!")
    print("=" * 70)
    print()
    print("Next Steps:")
    print("1. Update the configuration variables at the top of main()")
    print("2. Ensure your on-prem LLM endpoint is accessible")
    print("3. Verify the API format matches OpenAI-compatible format")
    print("4. If using a custom API format, create a custom provider class")
    print("   (see custom_onprem_provider.py for examples)")


if __name__ == "__main__":
    main()

