"""
Content Curator Agent Example

This example demonstrates a more complex agent that curates and analyzes content,
similar to the agents in the Curio project.

The SDK automatically handles: objective, additional context, tools, and execution history.
You just define your agent's role, guidelines, and any custom sections in get_agent_instructions().

Custom Tier Configuration:
This example demonstrates custom tier configuration optimized for content curation:
- plan_tier: tier3 (high quality for complex content discovery planning)
- critique_tier: tier3 (high quality for evaluating content relevance)
- synthesis_tier: tier2 (balanced for summarizing curated content)
- action_tier: tier2 (balanced for content fetching/analysis tools)

This configuration prioritizes quality for planning and critique (where accuracy matters most)
while using balanced models for synthesis and actions (where speed/cost optimization helps).
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from curio_agent_sdk import (
    BaseAgent,
    AgentConfig,
    InMemoryPersistence,
    LLMService,
)


@dataclass
class Article:
    """Simple article data model."""
    title: str
    content: str
    source: str
    url: str
    published: str


class ContentCuratorAgent(BaseAgent):
    """
    An agent that curates and analyzes content for users.

    This demonstrates:
    - Object identifier maps for storing articles
    - Tool composition for fetching and analyzing content
    - Including dynamic content (stored objects) in agent instructions
    """

    def __init__(
        self,
        agent_id: str,
        config: Optional[AgentConfig] = None,
        persistence: Optional[Any] = None,
        llm_service: Optional[LLMService] = None,
    ):
        # Configure custom tiers for different phases
        # For content curation, we need better quality for analysis but can optimize elsewhere
        # - plan_tier: tier3 (high quality - complex planning for content discovery)
        # - critique_tier: tier3 (high quality - need good critique for content relevance)
        # - synthesis_tier: tier2 (balanced - summarizing curated content)
        # - action_tier: tier2 (balanced - tool calls for fetching/analyzing content)
        super().__init__(
            agent_id=agent_id,
            config=config,
            persistence=persistence,
            llm_service=llm_service,
            plan_tier="tier3",      # High quality for complex content discovery planning
            critique_tier="tier3",  # High quality for evaluating content relevance
            synthesis_tier="tier2", # Balanced for summarizing curated content
            action_tier="tier2",    # Balanced for content fetching/analysis tools
        )
        self.agent_name = "ContentCuratorAgent"
        self.description = "Curates and analyzes content for users"
        self.max_iterations = 10
        self.initialize_tools()

    def get_agent_instructions(self) -> str:
        """
        Define the agent's role, guidelines, and any custom sections.

        The SDK automatically adds: objective, additional context, tools, and execution history.
        Include any dynamic content here (stored objects, preferences from DB, etc.)
        """
        # Get stored articles and analyses for the prompt
        articles_section = self._format_articles_section()
        analyses_section = self._format_analyses_section()

        return f"""
You are a content curator assistant that helps users find and analyze relevant content.

## YOUR ROLE
- Fetch content based on user interests and objectives
- Analyze content for relevance and quality
- Provide concise summaries and recommendations
- Store and reference content using identifiers (e.g., "Article1", "Analysis1")

## WORKFLOW
1. Fetch articles using the fetch_articles tool
2. Analyze each article using the analyze_article tool
3. Send relevant content to the user using send_content_to_user
4. Use respond_to_user for general messages

## GUIDELINES
- Always analyze articles before recommending them
- Use identifiers to reference stored content (saves context space!)
- Consider user preferences when analyzing relevance
- Be concise in summaries - users are busy
- If no relevant content found, let the user know

## STORED ARTICLES
{articles_section}

## ANALYSES
{analyses_section}
"""

    def _format_articles_section(self) -> str:
        """Format stored articles for the prompt."""
        identifiers = self.object_map.get_identifiers_by_type("Article")
        if not identifiers:
            return "No articles stored yet."

        lines = []
        for identifier in identifiers:
            article = self.object_map.get(identifier)
            if article:
                lines.append(f"- {identifier}: '{article.title}' from {article.source}")

        return "\n".join(lines)

    def _format_analyses_section(self) -> str:
        """Format stored analyses for the prompt."""
        identifiers = self.object_map.get_identifiers_by_type("Analysis")
        if not identifiers:
            return "No analyses yet."

        lines = []
        for identifier in identifiers:
            analysis = self.object_map.get(identifier)
            if analysis:
                article_id = analysis.get("article_id", "Unknown")
                relevance = analysis.get("relevance_score", "N/A")
                lines.append(f"- {identifier}: Analysis of {article_id} (relevance: {relevance})")

        return "\n".join(lines)

    def initialize_tools(self) -> None:
        """Register tools for content curation."""
        self.register_tool("fetch_articles", self.fetch_articles)
        self.register_tool("analyze_article", self.analyze_article)
        self.register_tool("get_article_details", self.get_article_details)
        self.register_tool("send_content_to_user", self.send_content_to_user)
        self.register_tool("respond_to_user", self.respond_to_user)

    # ==================== Tool Implementations ====================

    def fetch_articles(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        name: fetch_articles
        description: Fetch articles based on a topic or search query. Returns identifiers for the fetched articles.
        parameters:
            topic: The topic or search query to fetch articles about
            max_results: Maximum number of articles to fetch (default: 5)
        required_parameters:
            - topic
        response_format:
            List of article identifiers that were fetched and stored
        examples:
            >>> fetch_articles({"topic": "artificial intelligence", "max_results": 3})
            >>> fetch_articles({"topic": "machine learning news"})
        """
        topic = args.get("topic", "")
        max_results = args.get("max_results", 5)

        # Simulate fetching articles (in real implementation, would call RSS/API)
        sample_articles = self._simulate_article_fetch(topic, max_results)

        # Store each article and collect identifiers
        identifiers = []
        for article in sample_articles:
            # Use URL as deduplication key
            identifier = self.store_object(article, "Article", key=article.url)
            identifiers.append(identifier)

        return {
            "status": "ok",
            "result": {
                "message": f"Fetched {len(identifiers)} articles about '{topic}'",
                "article_identifiers": identifiers,
            }
        }

    def _simulate_article_fetch(self, topic: str, max_results: int) -> List[Article]:
        """Simulate fetching articles (replace with real implementation)."""
        # In real implementation, this would call RSS feeds, APIs, etc.
        now = datetime.now().isoformat()
        return [
            Article(
                title=f"Latest Developments in {topic.title()} - Article {i+1}",
                content=f"This is a detailed article about {topic}. It covers various aspects including recent developments, key players, and future predictions. The article provides in-depth analysis and expert opinions on the subject matter.",
                source="TechNews",
                url=f"https://example.com/{topic.replace(' ', '-')}-{i+1}",
                published=now,
            )
            for i in range(min(max_results, 3))
        ]

    def analyze_article(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        name: analyze_article
        description: Analyze an article for relevance to user interests. Creates an analysis with relevance score, summary, and recommendation.
        parameters:
            article_id: The article identifier (e.g., "Article1")
            additional_context: Optional additional context for analysis
        required_parameters:
            - article_id
        response_format:
            Analysis identifier and summary of the analysis
        examples:
            >>> analyze_article({"article_id": "Article1"})
            >>> analyze_article({"article_id": "Article2", "additional_context": "Focus on practical applications"})
        """
        article_id = args.get("article_id", "")
        additional_context = args.get("additional_context", "")

        # Get the article
        article = self.object_map.get(article_id)
        if not article:
            return {
                "status": "error",
                "result": f"Article '{article_id}' not found. Available: {self.object_map.get_identifiers_by_type('Article')}"
            }

        # In real implementation, would use LLM to analyze
        # Here we simulate analysis
        analysis = {
            "article_id": article_id,
            "title": article.title,
            "relevance_score": 0.85,  # Simulated score
            "summary": f"This article provides a comprehensive overview of {article.title.lower()}. Key points include recent developments and industry impact.",
            "recommendation": "Recommend",
            "key_topics": ["innovation", "technology", "trends"],
            "analyzed_at": datetime.now().isoformat(),
        }

        # Store analysis
        analysis_id = self.store_object(analysis, "Analysis")

        return {
            "status": "ok",
            "result": {
                "analysis_id": analysis_id,
                "article_id": article_id,
                "relevance_score": analysis["relevance_score"],
                "recommendation": analysis["recommendation"],
                "summary_preview": analysis["summary"][:100] + "...",
            }
        }

    def get_article_details(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        name: get_article_details
        description: Get full details of a stored article
        parameters:
            article_id: The article identifier (e.g., "Article1")
        required_parameters:
            - article_id
        response_format:
            Full article details
        examples:
            >>> get_article_details({"article_id": "Article1"})
        """
        article_id = args.get("article_id", "")
        article = self.object_map.get(article_id)

        if not article:
            return {
                "status": "error",
                "result": f"Article '{article_id}' not found"
            }

        return {
            "status": "ok",
            "result": {
                "identifier": article_id,
                "title": article.title,
                "content": article.content[:500] + "..." if len(article.content) > 500 else article.content,
                "source": article.source,
                "url": article.url,
                "published": article.published,
            }
        }

    def send_content_to_user(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        name: send_content_to_user
        description: Send a content recommendation to the user based on an analysis
        parameters:
            analysis_id: The analysis identifier (e.g., "Analysis1")
            custom_message: Optional custom message to include
        required_parameters:
            - analysis_id
        response_format:
            Confirmation that the content was sent
        examples:
            >>> send_content_to_user({"analysis_id": "Analysis1"})
            >>> send_content_to_user({"analysis_id": "Analysis1", "custom_message": "You might find this interesting!"})
        """
        analysis_id = args.get("analysis_id", "")
        custom_message = args.get("custom_message", "")

        analysis = self.object_map.get(analysis_id)
        if not analysis:
            return {
                "status": "error",
                "result": f"Analysis '{analysis_id}' not found"
            }

        # Get the original article
        article_id = analysis.get("article_id", "")
        article = self.object_map.get(article_id)

        if not article:
            return {
                "status": "error",
                "result": f"Original article '{article_id}' not found"
            }

        # Format content message
        message = f"""
📰 **{article.title}**

{analysis.get('summary', '')}

🔗 Read more: {article.url}
📊 Relevance: {analysis.get('relevance_score', 'N/A')}

{custom_message}
"""

        # In real implementation, would send to user via notification system
        self._log("content_sent_to_user", {
            "analysis_id": analysis_id,
            "article_id": article_id,
            "title": article.title,
        })

        return {
            "status": "ok",
            "result": f"Content sent to user: {article.title}"
        }

    def respond_to_user(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        name: respond_to_user
        description: Send a general response message to the user
        parameters:
            message: The message to send
        required_parameters:
            - message
        response_format:
            Confirmation that the message was sent
        examples:
            >>> respond_to_user({"message": "I found 3 relevant articles for you!"})
        """
        message = args.get("message", "")

        # In real implementation, would send to user
        self._log("message_sent_to_user", {
            "message": message[:200],
        })

        return {
            "status": "ok",
            "result": f"Message sent: {message[:50]}..."
        }


def main():
    """Run the content curator example."""
    # Create persistence
    persistence = InMemoryPersistence()

    # Try to load config
    try:
        config = AgentConfig.from_env()
        llm_service = config.get_llm_service()
        print("Loaded config from environment")
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        print("Running without LLM service")
        config = None
        llm_service = None

    # Create agent
    agent = ContentCuratorAgent(
        agent_id="curator-1",
        config=config,
        persistence=persistence,
        llm_service=llm_service,
    )

    print(f"\nAgent: {agent.agent_name}")
    print(f"Description: {agent.description}")
    print(f"Tools: {agent.tool_registry.get_names()}")

    # Run if we have LLM
    if llm_service:
        print("\n" + "="*50)
        print("Running agent...")
        print("="*50 + "\n")

        result = agent.run(
            objective="Find and analyze recent AI articles, then send the most relevant ones to me",
            additional_context={
                "preferences": {
                    "topics": ["machine learning", "AI agents", "LLMs"],
                    "preferred_sources": ["TechNews", "AI Weekly"],
                }
            }
        )

        print(f"\nResult: {result.status}")
        print(f"Iterations: {result.total_iterations}")
        print(f"\nSummary:\n{result.synthesis_summary}")

        # Show what was stored
        print("\n" + "="*50)
        print("Object Map Contents:")
        print("="*50)
        print(f"Articles: {agent.object_map.get_identifiers_by_type('Article')}")
        print(f"Analyses: {agent.object_map.get_identifiers_by_type('Analysis')}")
    else:
        print("\nSkipping run (no LLM service)")
        print("Set API keys in environment to test full functionality")


if __name__ == "__main__":
    main()
