"""
Unit tests for curio_agent_sdk.persistence.base — BasePersistence ABC.

Covers: abstract class cannot be instantiated, audit hook methods exist.
"""

import pytest

from curio_agent_sdk.persistence.base import BasePersistence


@pytest.mark.unit
class TestBasePersistence:
    def test_base_persistence_is_abstract(self):
        """BasePersistence cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePersistence()

    def test_base_persistence_audit_hooks(self):
        """Audit hook methods exist and are callable (default no-op)."""
        # Use a minimal concrete subclass to test the default audit behavior
        class MinimalPersistence(BasePersistence):
            def create_agent_run(self, run):
                pass

            def update_agent_run(self, run_id, run):
                pass

            def get_agent_run(self, run_id):
                return None

            def get_agent_runs(self, agent_id=None, limit=10, offset=0):
                return []

            def delete_agent_run(self, run_id):
                return False

            def log_agent_run_event(self, event):
                pass

            def get_agent_run_events(self, run_id, event_type=None):
                return []

            def log_llm_usage(self, usage):
                pass

            def get_llm_usage(self, agent_id=None, run_id=None, limit=100):
                return []

            def get_agent_run_stats(self, agent_id=None):
                return {}

        p = MinimalPersistence()
        # Default log_audit_event is no-op
        p.log_audit_event({"action": "test"})
        # Default get_audit_events returns []
        assert p.get_audit_events() == []
