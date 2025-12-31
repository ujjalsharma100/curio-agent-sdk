"""
Base Agent class for the Curio Agent SDK.

This module provides the BaseAgent abstract class that all agents should inherit from.
It implements the common patterns found across all agent types including:
- Plan-Critique-Synthesize agentic loop
- Object identifier map for context optimization
- Tool registry for action execution
- Event logging for observability
- Database persistence integration

Example:
    from curio_agent_sdk import BaseAgent, AgentConfig

    class MyCustomAgent(BaseAgent):
        def __init__(self, agent_id: str, config: AgentConfig):
            super().__init__(agent_id, config)
            self.agent_name = "MyCustomAgent"
            self.max_iterations = 5
            self.initialize_tools()

        def get_agent_instructions(self) -> str:
            # Only define your agent's role and guidelines
            # Objective, tools, history are automatically included!
            return '''
            You are a helpful assistant specialized in data analysis.

            ## GUIDELINES
            - Be concise and accurate
            - Always explain your reasoning
            '''

        def initialize_tools(self):
            self.register_tool("analyze", self.analyze_method)

    # Usage
    agent = MyCustomAgent("my-agent-id", config)
    result = agent.run("Analyze the sales data", {"user_prefs": {...}})
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Callable, Optional, Type
from datetime import datetime
import json
import uuid
import logging

from curio_agent_sdk.core.object_identifier_map import ObjectIdentifierMap
from curio_agent_sdk.core.tool_registry import ToolRegistry
from curio_agent_sdk.core.models import (
    AgentRun,
    AgentRunEvent,
    AgentRunResult,
    PlanResult,
    PlannedAction,
    CritiqueResult,
    SynthesisResult,
    EventType,
    AgentRunStatus,
)

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Curio Agent SDK.

    This class provides the core infrastructure for building autonomous agents,
    including the plan-critique-synthesize loop, object management for context
    optimization, tool registration, and observability through event logging.

    The prompt is automatically assembled with:
    - Agent instructions (your custom role/guidelines/preferences/etc.)
    - Objective
    - Additional context (if provided)
    - Available tools
    - Execution history

    Subclasses must implement:
        - get_agent_instructions(): Define agent role, guidelines, and any custom sections
        - initialize_tools(): Register tools the agent can use

    Optional overrides:
        - plan(objective, context, history): Custom planning logic
        - critique(history, objective, context): Custom critique logic
        - synthesis(history, objective, context): Custom synthesis logic
        - execute_action(action_name, args): Custom action execution

    Attributes:
        agent_id: Unique identifier for this agent instance
        agent_name: Human-readable name for the agent
        description: Description of what the agent does
        max_iterations: Maximum iterations in the agentic loop
        plan_tier: Model tier for planning step (default: tier3)
        critique_tier: Model tier for critique step (default: tier3)
        synthesis_tier: Model tier for synthesis step (default: tier1)
        action_tier: Model tier for action execution (default: tier2)
        object_map: ObjectIdentifierMap for context optimization
        tool_registry: ToolRegistry for managing tools
        event_log: In-memory list of events
        current_run_id: ID of the current run (if running)

    Tier Configuration:
        You can customize which model tier is used for each step:

        class MyAgent(BaseAgent):
            def __init__(self, agent_id, config):
                super().__init__(
                    agent_id,
                    config=config,
                    plan_tier="tier3",      # Best model for planning
                    critique_tier="tier2",  # Balanced for critique
                    synthesis_tier="tier1", # Fast/cheap for synthesis
                )
    """

    def __init__(
        self,
        agent_id: str,
        config: Optional[Any] = None,
        persistence: Optional[Any] = None,
        llm_service: Optional[Any] = None,
        # Tier configuration for different steps
        plan_tier: str = "tier3",
        critique_tier: str = "tier3",
        synthesis_tier: str = "tier1",
        action_tier: str = "tier2",
    ):
        """
        Initialize the BaseAgent.

        Args:
            agent_id: Unique identifier for this agent instance
            config: Optional AgentConfig for configuration
            persistence: Optional BasePersistence for database operations
            llm_service: Optional LLMService for LLM calls
            plan_tier: Model tier for planning step (default: tier3 - best quality)
            critique_tier: Model tier for critique step (default: tier3 - best quality)
            synthesis_tier: Model tier for synthesis step (default: tier1 - fast/cheap)
            action_tier: Model tier for action execution LLM calls (default: tier2 - balanced)
        """
        self.agent_id = agent_id
        self.agent_name = "BaseAgent"
        self.description = ""
        self.max_iterations = 7

        # Tier configuration for different steps
        self.plan_tier = plan_tier
        self.critique_tier = critique_tier
        self.synthesis_tier = synthesis_tier
        self.action_tier = action_tier

        # Configuration
        self.config = config

        # Services
        self.persistence = persistence
        self.llm_service = llm_service

        # Object identifier map for context optimization
        self.object_map = ObjectIdentifierMap(
            on_store=self._on_object_stored,
            on_not_found=self._on_object_not_found,
        )

        # Tool registry
        self.tool_registry = ToolRegistry()

        # Event log (in-memory)
        self.event_log: List[Dict[str, Any]] = []

        # Current run state
        self.current_run_id: Optional[str] = None
        self._current_run: Optional[AgentRun] = None

    # ==================== Abstract Methods ====================

    @abstractmethod
    def get_agent_instructions(self) -> str:
        """
        Define the agent's role, persona, guidelines, and any custom sections.

        This is the ONLY prompt section you need to define. The SDK automatically
        adds: objective, additional context, tools, and execution history.

        Include any custom sections here like user preferences, conversation
        history, stored objects display, etc. You have full control over this
        section and can pull data from persistence/db services as needed.

        Returns:
            String containing agent instructions and any custom sections

        Example:
            def get_agent_instructions(self) -> str:
                # Pull user preferences from your service
                prefs = self.user_service.get_preferences(self.user_id)

                return f'''
                You are Curio, a helpful knowledge curator assistant.
                Your role is to help users stay updated with relevant information.

                ## GUIDELINES
                - Be proactive and helpful
                - Provide concise, accurate information
                - Always cite your sources

                ## USER PREFERENCES
                {json.dumps(prefs, indent=2)}

                ## STORED ARTICLES
                {self.format_objects_for_prompt("Article")}
                '''
        """
        pass

    @abstractmethod
    def initialize_tools(self) -> None:
        """
        Initialize and register tools for this agent.

        This method must be implemented by subclasses to register all
        tools the agent can use.

        Example:
            def initialize_tools(self):
                self.register_tool("search", self.search_method)
                self.register_tool("analyze", self.analyze_method)
        """
        pass

    # ==================== Prompt Building ====================

    def _build_full_prompt(
        self,
        objective: str,
        additional_context: Dict[str, Any],
        execution_history: List[Dict[str, Any]],
    ) -> str:
        """
        Build the complete prompt with all standard sections.

        This method automatically assembles:
        1. Agent instructions (from get_agent_instructions)
        2. Objective
        3. Additional context (if provided)
        4. Available tools
        5. Execution history

        Args:
            objective: The goal/objective for this run
            additional_context: Additional context from run()
            execution_history: What has been done so far

        Returns:
            Complete prompt string
        """
        sections = []

        # 1. Agent instructions (user-defined - includes any custom sections they want)
        agent_instructions = self.get_agent_instructions()
        if agent_instructions:
            sections.append(agent_instructions.strip())

        # 2. Objective
        sections.append(f"## OBJECTIVE\n{objective}")

        # 3. Additional context (if provided)
        if additional_context:
            sections.append(f"## ADDITIONAL CONTEXT\n{json.dumps(additional_context, indent=2)}")

        # 4. Available tools
        tools_desc = self.get_tools_description()
        if tools_desc:
            sections.append(f"## AVAILABLE TOOLS/ACTIONS\n{tools_desc}")

        # 5. Execution history
        if execution_history:
            sections.append(f"## EXECUTION HISTORY\n{json.dumps(execution_history, indent=2)}")
        else:
            sections.append("## EXECUTION HISTORY\nNo actions executed yet.")

        return "\n\n".join(sections)

    # ==================== Legacy Compatibility ====================

    def get_system_prompt(
        self,
        objective: str,
        additional_context: Dict[str, Any],
        execution_history: List[Dict[str, Any]],
    ) -> str:
        """
        Generate the complete system prompt for the LLM.

        This method is kept for backward compatibility but now delegates
        to _build_full_prompt() which automatically includes all standard sections.

        For new agents, you only need to implement get_agent_instructions()
        and optionally get_custom_prompt_sections().

        Args:
            objective: The goal/objective for this run
            additional_context: Any additional context provided
            execution_history: What has been done so far

        Returns:
            The complete system prompt string
        """
        return self._build_full_prompt(objective, additional_context, execution_history)

    # ==================== Tool Registration ====================

    def register_tool(
        self,
        name: str,
        function: Callable,
        description: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Register a tool with the agent.

        Args:
            name: Unique name for the tool
            function: The callable to execute
            description: Optional description (uses docstring if not provided)
            **kwargs: Additional arguments passed to ToolRegistry.register
        """
        self.tool_registry.register(name, function, description, **kwargs)

    def get_tools_description(self) -> str:
        """
        Get formatted tool descriptions for use in prompts.

        Returns:
            String containing all tool descriptions
        """
        return self.tool_registry.get_descriptions_text()

    # ==================== Object Management ====================

    def store_object(
        self,
        obj: Any,
        object_type: str,
        key: Optional[str] = None,
    ) -> str:
        """
        Store an object and return its identifier.

        Args:
            obj: The object to store
            object_type: Type/category of the object
            key: Optional deduplication key

        Returns:
            The generated identifier (e.g., "Article1")
        """
        return self.object_map.store(obj, object_type, key)

    def get_object(self, identifier: str) -> Optional[Any]:
        """
        Retrieve an object by its identifier.

        Args:
            identifier: The object identifier

        Returns:
            The stored object or None
        """
        return self.object_map.get(identifier)

    def clear_objects(self, object_type: Optional[str] = None) -> int:
        """
        Clear stored objects.

        Args:
            object_type: If provided, only clear this type

        Returns:
            Number of objects cleared
        """
        return self.object_map.clear(object_type)

    def format_objects_for_prompt(
        self,
        object_type: Optional[str] = None,
        formatter: Optional[Callable[[str, Any], str]] = None,
    ) -> str:
        """
        Format stored objects for inclusion in prompts.

        Convenience method for use in get_stored_objects_section().

        Args:
            object_type: Filter by type (optional)
            formatter: Custom formatter function (identifier, obj) -> str

        Returns:
            Formatted string of stored objects
        """
        return self.object_map.format_for_prompt(object_type, formatter)

    # ==================== Event Logging ====================

    def _log(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Log an event to the event log and persist to database.

        Args:
            event_type: Type of event
            data: Event data dictionary
        """
        timestamp = datetime.now()

        # Create event for in-memory log
        event = {
            "event_type": event_type,
            "timestamp": timestamp.isoformat(),
            "agent_id": self.agent_id,
            "data": data,
        }
        self.event_log.append(event)

        # Persist to database if we have persistence and a current run
        if self.persistence and self.current_run_id:
            try:
                db_event = AgentRunEvent(
                    agent_id=self.agent_id,
                    run_id=self.current_run_id,
                    agent_name=self.agent_name,
                    timestamp=timestamp,
                    event_type=event_type,
                    data=json.dumps(data) if data else None,
                )
                self.persistence.log_agent_run_event(db_event)
            except Exception as e:
                logger.error(f"Failed to persist event to database: {e}")

    def _on_object_stored(self, identifier: str, object_type: str, map_size: int) -> None:
        """Callback when an object is stored."""
        self._log(EventType.OBJECT_STORED.value, {
            "identifier": identifier,
            "object_type": object_type,
            "map_size": map_size,
        })

    def _on_object_not_found(self, identifier: str, available: List[str]) -> None:
        """Callback when an object is not found."""
        self._log(EventType.OBJECT_NOT_FOUND.value, {
            "identifier": identifier,
            "available_identifiers": available,
        })

    # ==================== Agentic Loop Methods ====================

    def plan(
        self,
        objective: str,
        additional_context: Dict[str, Any],
        execution_history: List[Dict[str, Any]],
    ) -> PlanResult:
        """
        Plan the next actions to accomplish the objective.

        Args:
            objective: The goal/objective
            additional_context: Additional context
            execution_history: What has been done so far

        Returns:
            PlanResult containing list of planned actions
        """
        try:
            # Build planning prompt
            planning_prompt = self._build_full_prompt(
                objective, additional_context, execution_history
            )
            planning_prompt += "\n\n" + self._get_planning_instructions()

            # Log planning started
            self._log(EventType.PLANNING_STARTED.value, {
                "prompt_length": len(planning_prompt),
            })

            # Call LLM
            response = self._call_llm(planning_prompt, tier=self.plan_tier)

            # Parse response
            sanitized = self._sanitize_json(response)
            result_dict = json.loads(sanitized)

            plan_result = PlanResult.from_dict(result_dict)

            # Log planning completed
            self._log(EventType.PLANNING_COMPLETED.value, {
                "action_count": len(plan_result.plan),
                "notes": plan_result.notes[:200] if plan_result.notes else "",
            })

            return plan_result

        except Exception as e:
            logger.error(f"Error planning: {str(e)}")
            return PlanResult(
                plan=[],
                notes="Error planning",
                debug_info=str(e),
            )

    def critique(
        self,
        execution_history: List[Dict[str, Any]],
        objective: str,
        additional_context: Dict[str, Any],
    ) -> CritiqueResult:
        """
        Critique the execution so far and decide whether to continue.

        Args:
            execution_history: What has been done so far
            objective: The original objective
            additional_context: Additional context

        Returns:
            CritiqueResult with status ("done" or "continue")
        """
        try:
            # Build critique prompt
            critique_prompt = self._build_full_prompt(
                objective, additional_context, execution_history
            )
            critique_prompt += "\n\n" + self._get_critique_instructions()

            # Log critique started
            self._log(EventType.CRITIQUE_STARTED.value, {
                "prompt_length": len(critique_prompt),
            })

            # Call LLM
            response = self._call_llm(critique_prompt, tier=self.critique_tier)

            # Parse response
            sanitized = self._sanitize_json(response)
            result_dict = json.loads(sanitized)

            critique_result = CritiqueResult.from_dict(result_dict)

            # Log critique completed
            self._log(EventType.CRITIQUE_COMPLETED.value, {
                "status": critique_result.status,
                "summary": critique_result.critique_summary[:200] if critique_result.critique_summary else "",
            })

            return critique_result

        except Exception as e:
            logger.error(f"Error in critique: {str(e)}")
            return CritiqueResult(
                status="done",
                critique_summary="Error occurred during critique",
                recommendations=str(e),
            )

    def synthesis(
        self,
        execution_history: List[Dict[str, Any]],
        objective: str,
        additional_context: Dict[str, Any],
    ) -> SynthesisResult:
        """
        Synthesize the results of the agent run.

        Args:
            execution_history: Complete execution history
            objective: The original objective
            additional_context: Additional context

        Returns:
            SynthesisResult with summary
        """
        try:
            # Build synthesis prompt
            synthesis_prompt = self._build_full_prompt(
                objective, additional_context, execution_history
            )
            synthesis_prompt += "\n\n" + self._get_synthesis_instructions()

            # Log synthesis started
            self._log(EventType.SYNTHESIS_STARTED.value, {
                "prompt_length": len(synthesis_prompt),
            })

            # Call LLM
            synthesis_summary = self._call_llm(synthesis_prompt, tier=self.synthesis_tier)

            # Log synthesis completed
            self._log(EventType.SYNTHESIS_COMPLETED.value, {
                "summary_length": len(synthesis_summary),
            })

            return SynthesisResult(synthesis_summary=synthesis_summary)

        except Exception as e:
            logger.error(f"Error in synthesis: {str(e)}")
            return SynthesisResult(
                synthesis_summary=f"Error occurred during synthesis: {str(e)}"
            )

    def execute_action(
        self,
        action_name: str,
        action_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a single action/tool.

        Args:
            action_name: Name of the action to execute
            action_args: Arguments for the action

        Returns:
            Result dictionary with "status" and "result" keys
        """
        return self.tool_registry.execute(action_name, action_args)

    # ==================== Main Run Method ====================

    def run(
        self,
        objective: str,
        additional_context: Optional[Dict[str, Any]] = None,
        max_iterations: Optional[int] = None,
    ) -> AgentRunResult:
        """
        Run the agent with the given objective.

        This is the main entry point for agent execution. It implements
        the plan-execute-critique loop until the objective is accomplished
        or max iterations are reached.

        Args:
            objective: The goal/objective for this run
            additional_context: Optional additional context (passed to get_custom_prompt_sections)
            max_iterations: Override the default max iterations

        Returns:
            AgentRunResult with status, synthesis, and history
        """
        if max_iterations is None:
            max_iterations = self.max_iterations

        if additional_context is None:
            additional_context = {}

        # Generate unique run ID
        run_id = str(uuid.uuid4())
        self.current_run_id = run_id

        # Create agent run record
        self._current_run = AgentRun(
            agent_id=self.agent_id,
            run_id=run_id,
            agent_name=self.agent_name,
            objective=objective,
            additional_context=json.dumps(additional_context) if additional_context else None,
            started_at=datetime.now(),
            total_iterations=0,
            status=AgentRunStatus.RUNNING.value,
            execution_history=json.dumps([]),
        )

        # Persist run start
        if self.persistence:
            try:
                self.persistence.create_agent_run(self._current_run)
            except Exception as e:
                logger.error(f"Failed to create agent run record: {e}")

        # Log run start
        self._log(EventType.RUN_STARTED.value, {
            "objective": objective,
            "additional_context": additional_context,
            "max_iterations": max_iterations,
            "run_id": run_id,
        })

        execution_history = []
        iteration_count = 0

        try:
            while iteration_count < max_iterations:
                iteration_count += 1

                # Log iteration start
                self._log(EventType.ITERATION_STARTED.value, {
                    "iteration": iteration_count,
                    "run_id": run_id,
                })

                # Plan actions
                plan_result = self.plan(objective, additional_context, execution_history)

                execution_history.append({
                    "step": "planned_actions",
                    "plan": plan_result.to_dict(),
                })

                # Check for empty plan
                if not plan_result.plan:
                    logger.info(f"No actions planned for iteration {iteration_count}")
                    break

                # Execute each action
                for action in plan_result.plan:
                    action_name = action.action
                    action_args = action.args

                    # Log action start
                    self._log(EventType.ACTION_EXECUTION_STARTED.value, {
                        "action_name": action_name,
                        "action_args": action_args,
                        "iteration": iteration_count,
                    })

                    # Execute
                    observation = self.execute_action(action_name, action_args)

                    # Log action completion
                    self._log(EventType.ACTION_EXECUTION_COMPLETED.value, {
                        "action_name": action_name,
                        "observation": observation,
                        "iteration": iteration_count,
                    })

                    # Add to history
                    execution_history.append({
                        "step": "action_executed",
                        "action": action_name,
                        "args": action_args,
                        "observation": observation,
                    })

                # Critique the execution
                critique_result = self.critique(
                    execution_history, objective, additional_context
                )

                # Log critique result
                self._log(EventType.CRITIQUE_RESULT.value, {
                    "critique": critique_result.to_dict(),
                    "iteration": iteration_count,
                })

                execution_history.append({
                    "step": "critique_result",
                    "critique": critique_result.critique_summary,
                    "recommendations": critique_result.recommendations,
                    "status": critique_result.status,
                })

                # Check if we should continue
                if not critique_result.should_continue():
                    break

                # Log iteration completion
                self._log(EventType.ITERATION_COMPLETED.value, {
                    "iteration": iteration_count,
                })

            # Synthesize results
            synthesis_result = self.synthesis(
                execution_history, objective, additional_context
            )

            execution_history.append({
                "step": "synthesis_result",
                "synthesis_summary": synthesis_result.synthesis_summary,
                "status": "done",
            })

            # Update run record
            if self._current_run:
                self._current_run.finished_at = datetime.now()
                self._current_run.total_iterations = iteration_count
                self._current_run.final_synthesis_output = synthesis_result.synthesis_summary
                self._current_run.execution_history = json.dumps(execution_history)
                self._current_run.status = AgentRunStatus.COMPLETED.value

                if self.persistence:
                    try:
                        self.persistence.update_agent_run(run_id, self._current_run)
                    except Exception as e:
                        logger.error(f"Failed to update agent run record: {e}")

            # Log run completion
            self._log(EventType.RUN_COMPLETED.value, {
                "status": "done",
                "total_iterations": iteration_count,
                "run_id": run_id,
            })

            return AgentRunResult(
                status="done",
                synthesis_summary=synthesis_result.synthesis_summary,
                total_iterations=iteration_count,
                run_id=run_id,
                execution_history=execution_history,
            )

        except Exception as e:
            # Update run record with error
            if self._current_run:
                self._current_run.finished_at = datetime.now()
                self._current_run.total_iterations = iteration_count
                self._current_run.execution_history = json.dumps(execution_history)
                self._current_run.status = AgentRunStatus.ERROR.value
                self._current_run.error_message = str(e)

                if self.persistence:
                    try:
                        self.persistence.update_agent_run(run_id, self._current_run)
                    except Exception as db_error:
                        logger.error(f"Failed to update agent run with error: {db_error}")

            # Log error
            self._log(EventType.RUN_ERROR.value, {
                "error": str(e),
                "iteration": iteration_count,
                "run_id": run_id,
            })

            return AgentRunResult(
                status="error",
                error=str(e),
                total_iterations=iteration_count,
                run_id=run_id,
                execution_history=execution_history,
            )

        finally:
            # Clear current run state
            self.current_run_id = None
            self._current_run = None

    # ==================== Helper Methods ====================

    def _call_llm(self, prompt: str, tier: str = "tier3", **kwargs) -> str:
        """
        Call the LLM service.

        Args:
            prompt: The prompt to send
            tier: The tier to use for routing
            **kwargs: Additional arguments

        Returns:
            The LLM response content
        """
        if self.llm_service:
            response = self.llm_service.call_llm(
                prompt,
                tier=tier,
                run_id=self.current_run_id,
                agent_id=self.agent_id,
                **kwargs,
            )
            if hasattr(response, 'content'):
                return response.content
            if hasattr(response, 'error') and response.error:
                return f"Error: {response.error}"
            return str(response)
        else:
            raise RuntimeError("No LLM service configured. Please provide llm_service to the agent.")

    def _sanitize_json(self, text: str) -> str:
        """
        Sanitize JSON response from LLM.

        Handles common issues like markdown code blocks.

        Args:
            text: Raw LLM response

        Returns:
            Cleaned JSON string
        """
        text = text.strip()

        # Remove markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        return text.strip()

    def _get_planning_instructions(self) -> str:
        """Get the planning-specific instructions to append to the prompt."""
        return """## PLANNING MODE
You are now planning the next actions.

### INSTRUCTIONS
- Based on the given information, create a plan of actions to accomplish the objective.
- Return a JSON list of actions.
- Actions and arguments should be chosen from the available tools/actions.
- The response should be of the following format:
{
    "plan": [
        {
            "action": "<action_name>",
            "args": {"argument1": "value1", "argument2": "value2"}
        }
    ],
    "notes": "<Notes on the plan>",
    "debugInfo": "<Your advice on improvements>"
}
- Set plan to empty list if no action is required and mention it in the notes.
- Return only the JSON string with the following keys: plan, notes, debugInfo.
- Do not return any other text or information."""

    def _get_critique_instructions(self) -> str:
        """Get the critique-specific instructions to append to the prompt."""
        return """## CRITIQUE MODE
You are critiquing what has been executed and done so far to determine whether to continue.

### INSTRUCTIONS
- Critique the execution history and determine to keep continuing or not.
- The response should be of the following format:
{
   "status": "done" or "continue",
   "critique_summary": "<Summary of your evaluation>",
   "recommendations": "<Recommendations for next steps>"
}
- Return a JSON response with the following keys: status, critique_summary, recommendations.
- Do not return any other text or information."""

    def _get_synthesis_instructions(self) -> str:
        """Get the synthesis-specific instructions to append to the prompt."""
        return """## SYNTHESIS MODE
The agentic run is complete. Synthesize a summary of what has been done.

### INSTRUCTIONS
- Synthesize the summary of what has been done during the run.
- The response should be the synthesis summary only.
- Do not return any other text or information."""

    # ==================== Status and History ====================

    def get_event_log(self) -> List[Dict[str, Any]]:
        """Get the complete in-memory event log."""
        return self.event_log

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "description": self.description,
            "max_iterations": self.max_iterations,
            "tool_count": len(self.tool_registry),
            "object_count": len(self.object_map),
            "event_count": len(self.event_log),
            "available_tools": self.tool_registry.get_names(),
            "last_event": self.event_log[-1] if self.event_log else None,
            "current_run_id": self.current_run_id,
        }

    def get_run_history(self, limit: int = 10) -> List[AgentRun]:
        """Get recent agent run history from database."""
        if self.persistence:
            try:
                return self.persistence.get_agent_runs(
                    agent_id=self.agent_id, limit=limit
                )
            except Exception as e:
                logger.error(f"Failed to get run history: {e}")
        return []

    def get_run_events(
        self,
        run_id: str,
        event_type: Optional[str] = None,
    ) -> List[AgentRunEvent]:
        """Get events for a specific run."""
        if self.persistence:
            try:
                return self.persistence.get_agent_run_events(run_id, event_type)
            except Exception as e:
                logger.error(f"Failed to get run events: {e}")
        return []

    def get_run_stats(self) -> Dict[str, Any]:
        """Get agent run statistics."""
        if self.persistence:
            try:
                return self.persistence.get_agent_run_stats(agent_id=self.agent_id)
            except Exception as e:
                logger.error(f"Failed to get run stats: {e}")
        return {}
