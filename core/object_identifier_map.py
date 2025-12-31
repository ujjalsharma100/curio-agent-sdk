"""
Object Identifier Map for Context Window Optimization.

This module provides the ObjectIdentifierMap class, which is a key innovation
for reducing LLM context window usage. Instead of passing full objects to the LLM,
we store them locally and reference them with short identifiers.

Example:
    # Without ObjectIdentifierMap (500+ tokens per object):
    prompt = f"Here are the articles: {json.dumps(articles)}"

    # With ObjectIdentifierMap (2-3 tokens per reference):
    object_map = ObjectIdentifierMap()
    for article in articles:
        identifier = object_map.store(article, "Article")
    prompt = f"Available articles: Article1, Article2, Article3..."

Benefits:
    - Dramatically reduces token usage
    - Maintains object references during agentic loops
    - Clean interface for LLM to reference objects
    - Easy resolution back to actual objects for execution
"""

from typing import Dict, Any, Optional, List, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class StoredObject:
    """
    Metadata wrapper for stored objects.

    Attributes:
        obj: The actual stored object
        object_type: Type/category of the object
        identifier: The generated identifier
        stored_at: Timestamp when the object was stored
        access_count: Number of times the object has been accessed
        metadata: Additional metadata about the object
    """
    obj: Any
    object_type: str
    identifier: str
    stored_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def access(self) -> Any:
        """Access the object and increment access count."""
        self.access_count += 1
        return self.obj


class ObjectIdentifierMap:
    """
    A map that stores objects with short identifiers for context window optimization.

    This class maintains a mapping between short identifiers (e.g., "Article1", "Paper2")
    and full objects. This allows LLMs to reference objects using short identifiers
    instead of passing full object data in every prompt, dramatically reducing token usage.

    Key Features:
        - Automatic identifier generation per object type
        - Duplicate detection via custom key functions
        - Access tracking for optimization
        - Event callbacks for logging/observability

    Example:
        >>> object_map = ObjectIdentifierMap()
        >>>
        >>> # Store objects
        >>> id1 = object_map.store({"title": "AI News", "content": "..."}, "NewsArticle")
        >>> print(id1)  # "NewsArticle1"
        >>>
        >>> # Retrieve objects
        >>> article = object_map.get("NewsArticle1")
        >>> print(article["title"])  # "AI News"
        >>>
        >>> # Use in prompts
        >>> prompt = f"Available articles: {', '.join(object_map.get_identifiers_by_type('NewsArticle'))}"

    Attributes:
        objects: Internal storage mapping identifiers to StoredObject wrappers
        counters: Per-type counters for identifier generation
        key_to_identifier: Optional mapping from custom keys to identifiers (for deduplication)
        on_store: Optional callback invoked when an object is stored
        on_access: Optional callback invoked when an object is accessed
        on_not_found: Optional callback invoked when an object is not found
    """

    def __init__(
        self,
        on_store: Optional[Callable[[str, str, int], None]] = None,
        on_access: Optional[Callable[[str, Any], None]] = None,
        on_not_found: Optional[Callable[[str, List[str]], None]] = None,
    ):
        """
        Initialize the ObjectIdentifierMap.

        Args:
            on_store: Callback(identifier, object_type, map_size) called when storing
            on_access: Callback(identifier, object) called when accessing
            on_not_found: Callback(identifier, available_identifiers) called when not found
        """
        self.objects: Dict[str, StoredObject] = {}
        self.counters: Dict[str, int] = {}
        self.key_to_identifier: Dict[str, str] = {}

        self.on_store = on_store
        self.on_access = on_access
        self.on_not_found = on_not_found

    def _generate_identifier(self, object_type: str) -> str:
        """
        Generate a unique identifier for an object type.

        The identifier is formed by concatenating the object_type with an
        incrementing counter (e.g., "Article1", "Article2", etc.).

        Args:
            object_type: The type/category of the object

        Returns:
            A unique identifier string
        """
        if object_type not in self.counters:
            self.counters[object_type] = 0

        self.counters[object_type] += 1
        return f"{object_type}{self.counters[object_type]}"

    def store(
        self,
        obj: Any,
        object_type: str,
        key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store an object and return its identifier.

        If a key is provided and an object with that key already exists,
        the existing identifier is returned instead of creating a new one.

        Args:
            obj: The object to store
            object_type: Type/category of the object (e.g., "NewsArticle", "Paper")
            key: Optional unique key for deduplication (e.g., URL, ID)
            metadata: Optional additional metadata to store with the object

        Returns:
            The identifier for the stored object

        Example:
            >>> # Without deduplication key
            >>> id1 = object_map.store(article, "Article")
            >>>
            >>> # With deduplication key (URL-based)
            >>> id2 = object_map.store(article, "Article", key=article.url)
            >>> id3 = object_map.store(same_article, "Article", key=article.url)
            >>> assert id2 == id3  # Same identifier returned
        """
        # Check for existing object with same key
        if key and key in self.key_to_identifier:
            existing_id = self.key_to_identifier[key]
            logger.debug(f"Object with key '{key}' already exists as '{existing_id}'")
            return existing_id

        # Generate new identifier
        identifier = self._generate_identifier(object_type)

        # Create stored object wrapper
        stored = StoredObject(
            obj=obj,
            object_type=object_type,
            identifier=identifier,
            metadata=metadata or {},
        )

        # Store the object
        self.objects[identifier] = stored

        # Store key mapping if provided
        if key:
            self.key_to_identifier[key] = identifier

        # Invoke callback
        if self.on_store:
            self.on_store(identifier, object_type, len(self.objects))

        logger.debug(f"Stored object as '{identifier}' (type: {object_type}, map_size: {len(self.objects)})")
        return identifier

    def get(self, identifier: str) -> Optional[Any]:
        """
        Retrieve an object by its identifier.

        Args:
            identifier: The identifier of the object to retrieve

        Returns:
            The stored object, or None if not found

        Example:
            >>> article = object_map.get("Article1")
            >>> if article:
            ...     print(article["title"])
        """
        stored = self.objects.get(identifier)

        if stored is None:
            if self.on_not_found:
                self.on_not_found(identifier, list(self.objects.keys()))
            logger.warning(f"Object '{identifier}' not found. Available: {list(self.objects.keys())}")
            return None

        obj = stored.access()

        if self.on_access:
            self.on_access(identifier, obj)

        return obj

    def get_by_key(self, key: str) -> Optional[Any]:
        """
        Retrieve an object by its deduplication key.

        Args:
            key: The key used when storing the object

        Returns:
            The stored object, or None if not found
        """
        identifier = self.key_to_identifier.get(key)
        if identifier:
            return self.get(identifier)
        return None

    def get_identifier_by_key(self, key: str) -> Optional[str]:
        """
        Get the identifier for a given deduplication key.

        Args:
            key: The key used when storing the object

        Returns:
            The identifier, or None if not found
        """
        return self.key_to_identifier.get(key)

    def remove(self, identifier: str) -> bool:
        """
        Remove an object by its identifier.

        Args:
            identifier: The identifier of the object to remove

        Returns:
            True if the object was removed, False if not found
        """
        if identifier not in self.objects:
            return False

        stored = self.objects.pop(identifier)

        # Remove from key mapping if present
        for key, mapped_id in list(self.key_to_identifier.items()):
            if mapped_id == identifier:
                del self.key_to_identifier[key]
                break

        logger.debug(f"Removed object '{identifier}'")
        return True

    def clear(self, object_type: Optional[str] = None) -> int:
        """
        Clear objects from the map.

        Args:
            object_type: If provided, only clear objects of this type.
                        If None, clear all objects.

        Returns:
            Number of objects cleared
        """
        if object_type is None:
            count = len(self.objects)
            self.objects.clear()
            self.counters.clear()
            self.key_to_identifier.clear()
            logger.debug(f"Cleared all {count} objects")
            return count

        # Clear only specific type
        to_remove = [
            identifier for identifier, stored in self.objects.items()
            if stored.object_type == object_type
        ]

        for identifier in to_remove:
            self.remove(identifier)

        # Reset counter for this type
        if object_type in self.counters:
            self.counters[object_type] = 0

        logger.debug(f"Cleared {len(to_remove)} objects of type '{object_type}'")
        return len(to_remove)

    def get_identifiers_by_type(self, object_type: str) -> List[str]:
        """
        Get all identifiers for a specific object type.

        Args:
            object_type: The type of objects to retrieve identifiers for

        Returns:
            List of identifiers for the specified type
        """
        return [
            identifier for identifier, stored in self.objects.items()
            if stored.object_type == object_type
        ]

    def get_all_identifiers(self) -> List[str]:
        """
        Get all identifiers in the map.

        Returns:
            List of all identifiers
        """
        return list(self.objects.keys())

    def get_types(self) -> List[str]:
        """
        Get all object types currently in the map.

        Returns:
            List of unique object types
        """
        return list(set(stored.object_type for stored in self.objects.values()))

    def size(self, object_type: Optional[str] = None) -> int:
        """
        Get the number of objects in the map.

        Args:
            object_type: If provided, count only objects of this type

        Returns:
            Number of objects
        """
        if object_type is None:
            return len(self.objects)

        return sum(
            1 for stored in self.objects.values()
            if stored.object_type == object_type
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the object map.

        Returns:
            Dictionary with statistics including:
            - total_objects: Total number of stored objects
            - types: Count per object type
            - total_accesses: Total access count across all objects
            - most_accessed: The most frequently accessed identifier
        """
        type_counts: Dict[str, int] = {}
        total_accesses = 0
        most_accessed_id = None
        most_accessed_count = 0

        for identifier, stored in self.objects.items():
            # Count by type
            type_counts[stored.object_type] = type_counts.get(stored.object_type, 0) + 1

            # Track accesses
            total_accesses += stored.access_count
            if stored.access_count > most_accessed_count:
                most_accessed_count = stored.access_count
                most_accessed_id = identifier

        return {
            "total_objects": len(self.objects),
            "types": type_counts,
            "total_accesses": total_accesses,
            "most_accessed": most_accessed_id,
            "most_accessed_count": most_accessed_count,
        }

    def format_for_prompt(
        self,
        object_type: Optional[str] = None,
        formatter: Optional[Callable[[str, Any], str]] = None,
        separator: str = "\n",
    ) -> str:
        """
        Format stored objects for inclusion in an LLM prompt.

        Args:
            object_type: If provided, only include objects of this type
            formatter: Custom function to format each object.
                      Signature: (identifier, object) -> formatted_string
                      If None, uses default "- {identifier}: {brief_repr}"
            separator: String to join formatted objects

        Returns:
            Formatted string suitable for inclusion in prompts

        Example:
            >>> # Default formatting
            >>> prompt_section = object_map.format_for_prompt("Article")
            >>> # "- Article1: {'title': 'AI News'...}"
            >>> # "- Article2: {'title': 'ML Update'...}"
            >>>
            >>> # Custom formatting
            >>> def custom_fmt(id, obj):
            ...     return f"- {id}: '{obj['title']}' ({obj['source']})"
            >>> prompt_section = object_map.format_for_prompt("Article", formatter=custom_fmt)
        """
        identifiers = (
            self.get_identifiers_by_type(object_type)
            if object_type
            else self.get_all_identifiers()
        )

        if not identifiers:
            return ""

        if formatter is None:
            def default_formatter(identifier: str, obj: Any) -> str:
                # Create a brief representation
                if isinstance(obj, dict):
                    brief = {k: v for k, v in list(obj.items())[:3]}
                    return f"- {identifier}: {brief}"
                else:
                    brief_repr = str(obj)[:100]
                    return f"- {identifier}: {brief_repr}"
            formatter = default_formatter

        lines = []
        for identifier in identifiers:
            stored = self.objects.get(identifier)
            if stored:
                lines.append(formatter(identifier, stored.obj))

        return separator.join(lines)

    def __len__(self) -> int:
        """Return the number of stored objects."""
        return len(self.objects)

    def __contains__(self, identifier: str) -> bool:
        """Check if an identifier exists in the map."""
        return identifier in self.objects

    def __iter__(self):
        """Iterate over identifiers."""
        return iter(self.objects.keys())
