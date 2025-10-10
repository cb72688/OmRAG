"""
Tests for Episode Protocol Buffer schema.
These tests validate the structure, serialization, and functionality
of the Episode protobuf messages.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timezone
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf import json_format
import json
import math

# Add the generated proto directory to the path
SCHEMA_DIR = Path(__file__).parent.parent
PROTO_DIR = SCHEMA_DIR / "python"

print(f"\n=== Test Setup Debug ===")
print(f"Schema dir: {SCHEMA_DIR}")
print(f"Proto dir: {PROTO_DIR}")
print(f"Proto dir exists: {PROTO_DIR.exists()}")

if PROTO_DIR.exists():
    print(f"Proto dir contents: {list(PROTO_DIR.iterdir())}")
else:
    print("Proto directory does not exist!")
    print(f"Run: cd {SCHEMA_DIR} && ./generate.sh")

# Add to path
if str(PROTO_DIR) not in sys.path:
    sys.path.insert(0, str(PROTO_DIR))

print(f"Python path includes proto dir: {str(PROTO_DIR) in sys.path}")

# Import the generated protobuf classes
episode_pb2 = None
import_error = None

try:
    from .python import episode_pb2
    print(f"✓ Successfully imported episode_pb2 from {episode_pb2.__file__}")
except ImportError as e:
    import_error = e
    print(f"✗ Failed to import episode_pb2: {e}")
    print(f"Current sys.path: {sys.path[:3]}...")

print("=== End Test Setup Debug ===\n")

# Skip all tests if import failed
if episode_pb2 is None:
    pytest.skip(
        f"Protocol buffer files not generated or not importable. "
        f"Error: {import_error}. "
        f"Run 'cd {SCHEMA_DIR} && ./generate.sh' to generate the files.",
        allow_module_level=True
    )


def approx_equal(a, b, rel_tol=1e-5, abs_tol=1e-7):
    """Check if two floats are approximately equal (handles float32 precision)."""
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def approx_equal_list(list_a, list_b, rel_tol=1e-5, abs_tol=1e-7):
    """Check if two lists of floats are approximately equal."""
    if len(list_a) != len(list_b):
        return False
    return all(approx_equal(a, b, rel_tol, abs_tol) for a, b in zip(list_a, list_b))


class TestEpisodeStructure:
    """Test the structure and basic functionality of Episode messages."""

    def test_episode_creation(self):
        """Test that an Episode can be created with all required fields."""
        episode = episode_pb2.Episode()
        episode.id = "ep_12345"

        # Set timestamps
        now = datetime.now(timezone.utc)
        episode.created_at.FromDatetime(now)
        episode.updated_at.FromDatetime(now)

        # Set problem
        episode.problem.description = "Write a function to calculate fibonacci"
        episode.problem.category = "coding"
        episode.problem.difficulty = 5
        episode.problem.domain = "algorithms"

        assert episode.id == "ep_12345"
        assert episode.problem.description == "Write a function to calculate fibonacci"
        assert episode.problem.difficulty == 5

    def test_problem_with_context(self):
        """Test Problem message with context map."""
        problem = episode_pb2.Problem()
        problem.description = "Test problem"
        problem.context["language"] = "python"
        problem.context["environment"] = "production"
        problem.context["user_level"] = "intermediate"

        assert len(problem.context) == 3
        assert problem.context["language"] == "python"
        assert problem.context["environment"] == "production"

    def test_plan_with_subtasks(self):
        """Test Plan message with multiple subtasks."""
        plan = episode_pb2.Plan()
        plan.plan_id = "plan_001"
        plan.strategy = "Break down into smaller steps"
        plan.estimated_duration = 120.5
        plan.confidence = 0.85

        # Add subtasks
        subtask1 = plan.subtasks.add()
        subtask1.id = "st_1"
        subtask1.description = "Parse input"
        subtask1.tool = "text_parser"
        subtask1.parameters["format"] = "json"

        subtask2 = plan.subtasks.add()
        subtask2.id = "st_2"
        subtask2.description = "Process data"
        subtask2.tool = "data_processor"
        subtask2.dependencies.append("st_1")

        assert len(plan.subtasks) == 2
        assert plan.subtasks[0].id == "st_1"
        assert plan.subtasks[1].dependencies[0] == "st_1"

    def test_trajectory_with_steps(self):
        """Test Trajectory with execution steps."""
        trajectory = episode_pb2.Trajectory()
        trajectory.total_duration = 45.2
        trajectory.tool_call_count = 3
        trajectory.retry_count = 1

        # Add execution steps
        step1 = trajectory.steps.add()
        step1.step_number = 1
        step1.subtask_id = "st_1"
        step1.tool = "text_parser"
        step1.input = '{"text": "sample"}'
        step1.output = '{"parsed": true}'
        step1.status = episode_pb2.STEP_STATUS_SUCCESS
        step1.duration = 2.3

        step2 = trajectory.steps.add()
        step2.step_number = 2
        step2.subtask_id = "st_2"
        step2.tool = "data_processor"
        step2.status = episode_pb2.STEP_STATUS_FAILED
        step2.error_message = "Invalid input format"
        step2.duration = 1.5

        assert len(trajectory.steps) == 2
        assert trajectory.steps[0].status == episode_pb2.STEP_STATUS_SUCCESS
        assert trajectory.steps[1].status == episode_pb2.STEP_STATUS_FAILED

    def test_outcome_with_metrics(self):
        """Test Outcome message with metrics."""
        outcome = episode_pb2.Outcome()
        outcome.success = True
        outcome.result = "Successfully completed task"
        outcome.reward = 0.95
        outcome.quality_score = 0.88
        outcome.user_satisfaction = 0.9

        # Add metrics
        outcome.metrics["latency_ms"] = 450.2
        outcome.metrics["tokens_used"] = 1250.0
        outcome.metrics["memory_mb"] = 128.5

        assert outcome.success is True
        # Use approximate comparison for float32 values
        assert approx_equal(outcome.reward, 0.95)
        assert approx_equal(outcome.quality_score, 0.88)
        assert approx_equal(outcome.user_satisfaction, 0.9)
        assert len(outcome.metrics) == 3
        assert approx_equal(outcome.metrics["latency_ms"], 450.2)

    def test_episode_metadata(self):
        """Test EpisodeMetadata with various fields."""
        metadata = episode_pb2.EpisodeMetadata()
        metadata.agent_version = "v1.2.3"
        metadata.planner_model = "gpt-4"
        metadata.session_id = "session_abc123"
        metadata.user_id = "user_xyz"
        metadata.tags.extend(["python", "algorithms", "fibonacci"])
        metadata.retrieval_count = 5
        metadata.success_influence_count = 3
        metadata.consolidated = False
        metadata.archived = False
        metadata.custom_fields["priority"] = "high"
        metadata.custom_fields["team"] = "research"

        assert metadata.agent_version == "v1.2.3"
        assert len(metadata.tags) == 3
        assert "algorithms" in metadata.tags
        assert metadata.custom_fields["priority"] == "high"


class TestEpisodeSerialization:
    """Test serialization and deserialization of Episode messages."""

    def test_serialize_deserialize_episode(self):
        """Test that an Episode can be serialized and deserialized."""
        # Create original episode
        original = episode_pb2.Episode()
        original.id = "ep_serialize_test"
        original.problem.description = "Test problem"
        original.problem.category = "test"
        original.outcome.success = True
        original.outcome.reward = 0.8

        # Serialize
        serialized = original.SerializeToString()
        assert isinstance(serialized, bytes)

        # Deserialize
        deserialized = episode_pb2.Episode()
        deserialized.ParseFromString(serialized)

        assert deserialized.id == original.id
        assert deserialized.problem.description == original.problem.description
        assert approx_equal(deserialized.outcome.reward,
                            original.outcome.reward)

    def test_json_conversion(self):
        """Test conversion to and from JSON."""
        episode = episode_pb2.Episode()
        episode.id = "ep_json_test"
        episode.problem.description = "JSON test problem"
        episode.outcome.success = True

        # Convert to JSON
        json_str = json_format.MessageToJson(episode)
        json_dict = json.loads(json_str)

        assert json_dict["id"] == "ep_json_test"
        assert json_dict["problem"]["description"] == "JSON test problem"

        # Convert back from JSON
        reconstructed = json_format.Parse(json_str, episode_pb2.Episode())
        assert reconstructed.id == episode.id
        assert reconstructed.problem.description == episode.problem.description

    def test_embedding_serialization(self):
        """Test that embeddings (repeated floats) serialize correctly."""
        episode = episode_pb2.Episode()
        episode.id = "ep_embedding_test"

        # Add embedding vector
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 100  # 500-dim vector
        episode.embedding.extend(embedding)

        # Serialize and deserialize
        serialized = episode.SerializeToString()
        deserialized = episode_pb2.Episode()
        deserialized.ParseFromString(serialized)

        assert len(deserialized.embedding) == len(embedding)
        # Use approximate comparison for float32 values
        assert approx_equal_list(list(deserialized.embedding), embedding)


class TestSearchRequests:
    """Test search request and response messages."""

    def test_search_request_with_embedding(self):
        """Test creating a search request with embedding query."""
        request = episode_pb2.SearchEpisodesRequest()

        # Create embedding query
        embedding = episode_pb2.QueryEmbedding()
        embedding.values.extend([0.1, 0.2, 0.3, 0.4])
        request.embedding_query.CopyFrom(embedding)

        request.top_k = 5
        request.min_similarity = 0.7
        request.collection = "main_episodes"

        assert len(request.embedding_query.values) == 4
        assert request.top_k == 5
        assert request.HasField("embedding_query")
        assert request.WhichOneof("query") == "embedding_query"

    def test_search_request_with_text(self):
        """Test creating a search request with text query."""
        request = episode_pb2.SearchEpisodesRequest()
        request.text_query = "How to implement binary search?"
        request.top_k = 10
        request.min_similarity = 0.6

        assert request.text_query == "How to implement binary search?"
        assert request.HasField("text_query")
        assert request.WhichOneof("query") == "text_query"

    def test_search_request_oneof_behavior(self):
        """Test that oneof properly clears the other field."""
        request = episode_pb2.SearchEpisodesRequest()

        # Set text query
        request.text_query = "test query"
        assert request.WhichOneof("query") == "text_query"

        # Set embedding query (should clear text_query)
        embedding = episode_pb2.QueryEmbedding()
        embedding.values.extend([0.1, 0.2])
        request.embedding_query.CopyFrom(embedding)

        assert request.WhichOneof("query") == "embedding_query"
        assert request.text_query == ""  # Should be cleared

    def test_episode_filter(self):
        """Test EpisodeFilter configuration."""
        filter_msg = episode_pb2.EpisodeFilter()
        filter_msg.success_only = True
        filter_msg.has_success_filter = True
        filter_msg.categories.extend(["coding", "algorithms"])
        filter_msg.domains.extend(["python", "javascript"])
        filter_msg.tags.extend(["sorting", "searching"])
        filter_msg.min_reward = 0.7
        filter_msg.has_min_reward = True
        filter_msg.min_quality = 0.8
        filter_msg.has_min_quality = True
        filter_msg.exclude_archived = True

        # Set date range
        now = datetime.now(timezone.utc)
        filter_msg.created_after.FromDatetime(now)

        assert filter_msg.success_only is True
        assert filter_msg.has_success_filter is True
        assert len(filter_msg.categories) == 2
        assert len(filter_msg.tags) == 2
        # Use approximate comparison for float32 values
        assert approx_equal(filter_msg.min_reward, 0.7)
        assert approx_equal(filter_msg.min_quality, 0.8)
        assert filter_msg.has_min_reward is True

    def test_search_response(self):
        """Test SearchEpisodesResponse structure."""
        response = episode_pb2.SearchEpisodesResponse()
        response.total_matches = 15
        response.search_duration_ms = 23.5

        # Add results
        for i in range(5):
            result = response.results.add()
            result.episode.id = f"ep_{i}"
            result.episode.problem.description = f"Problem {i}"
            result.similarity = 0.9 - (i * 0.05)
            result.rank = i + 1

        assert len(response.results) == 5
        assert response.results[0].similarity > response.results[1].similarity
        assert response.results[0].rank == 1


class TestOneofBehavior:
    """Test oneof field behavior specifically."""

    def test_oneof_field_detection(self):
        """Test detecting which oneof field is set."""
        request = episode_pb2.SearchEpisodesRequest()

        # Initially, no field is set
        assert request.WhichOneof("query") is None

        # Set text query
        request.text_query = "test"
        assert request.WhichOneof("query") == "text_query"

        # Clear and set embedding
        request.ClearField("query")
        assert request.WhichOneof("query") is None

        embedding = episode_pb2.QueryEmbedding()
        embedding.values.append(0.1)
        request.embedding_query.CopyFrom(embedding)
        assert request.WhichOneof("query") == "embedding_query"

    def test_oneof_mutual_exclusion(self):
        """Test that setting one oneof field clears the other."""
        request = episode_pb2.SearchEpisodesRequest()

        # Set both and verify only last one is set
        request.text_query = "first"

        embedding = episode_pb2.QueryEmbedding()
        embedding.values.extend([0.1, 0.2])
        request.embedding_query.CopyFrom(embedding)

        # Only embedding_query should be set
        assert request.WhichOneof("query") == "embedding_query"
        assert len(request.embedding_query.values) == 2
        assert request.text_query == ""

    def test_oneof_serialization(self):
        """Test that oneof fields serialize correctly."""
        request1 = episode_pb2.SearchEpisodesRequest()
        request1.text_query = "serialization test"
        request1.top_k = 10

        # Serialize
        data = request1.SerializeToString()

        # Deserialize
        request2 = episode_pb2.SearchEpisodesRequest()
        request2.ParseFromString(data)

        assert request2.WhichOneof("query") == "text_query"
        assert request2.text_query == "serialization test"
        assert request2.top_k == 10


class TestBatchOperations:
    """Test batch operation messages."""

    def test_batch_store_request(self):
        """Test batch store request with multiple episodes."""
        request = episode_pb2.BatchStoreEpisodesRequest()
        request.auto_embed = True
        request.collection = "batch_test"

        # Add multiple episodes
        for i in range(10):
            episode = request.episodes.add()
            episode.id = f"ep_batch_{i}"
            episode.problem.description = f"Batch problem {i}"
            episode.outcome.success = i % 2 == 0

        assert len(request.episodes) == 10
        assert request.auto_embed is True

    def test_batch_store_response(self):
        """Test batch store response structure."""
        response = episode_pb2.BatchStoreEpisodesResponse()
        response.episode_ids.extend([f"ep_{i}" for i in range(10)])
        response.success_count = 8
        response.failure_count = 2
        response.error_messages.extend([
            "Episode 3 failed: invalid format",
            "Episode 7 failed: duplicate ID"
        ])

        assert len(response.episode_ids) == 10
        assert response.success_count == 8
        assert len(response.error_messages) == 2


class TestStatistics:
    """Test statistics messages."""

    def test_episode_statistics(self):
        """Test EpisodeStatistics structure."""
        stats = episode_pb2.EpisodeStatistics()
        stats.total_episodes = 1000
        stats.successful_episodes = 850
        stats.failed_episodes = 150
        stats.average_reward = 0.78
        stats.average_quality = 0.82
        stats.average_duration = 45.3

        # Category breakdown
        stats.episodes_by_category["coding"] = 400
        stats.episodes_by_category["research"] = 300
        stats.episodes_by_category["analysis"] = 300

        # Domain breakdown
        stats.episodes_by_domain["python"] = 500
        stats.episodes_by_domain["javascript"] = 300
        stats.episodes_by_domain["general"] = 200

        assert stats.total_episodes == 1000
        assert stats.successful_episodes + stats.failed_episodes == stats.total_episodes
        assert len(stats.episodes_by_category) == 3
        assert len(stats.episodes_by_domain) == 3


class TestDataIntegrity:
    """Test data integrity and validation scenarios."""

    def test_required_fields_can_be_empty(self):
        """Test that messages can be created with minimal fields (proto3 allows this)."""
        episode = episode_pb2.Episode()
        # In proto3, all fields are optional, but we should handle empty messages
        assert episode.id == ""
        assert episode.problem.description == ""

    def test_repeated_fields_are_lists(self):
        """Test that repeated fields behave like lists."""
        plan = episode_pb2.Plan()

        # Test append
        subtask = plan.subtasks.add()
        subtask.id = "st_1"

        assert len(plan.subtasks) == 1
        assert isinstance(plan.subtasks, type(
            plan.subtasks))  # RepeatedComposite

        # Test iteration
        for st in plan.subtasks:
            assert st.id == "st_1"

    def test_map_fields_are_dicts(self):
        """Test that map fields behave like dictionaries."""
        problem = episode_pb2.Problem()
        problem.context["key1"] = "value1"
        problem.context["key2"] = "value2"

        assert "key1" in problem.context
        assert problem.context["key1"] == "value1"
        assert len(problem.context) == 2

        # Test iteration
        keys = list(problem.context.keys())
        assert "key1" in keys
        assert "key2" in keys

    def test_optional_fields(self):
        """Test optional field handling."""
        outcome = episode_pb2.Outcome()

        # user_satisfaction is optional
        assert not outcome.HasField("user_satisfaction")

        outcome.user_satisfaction = 0.9
        assert outcome.HasField("user_satisfaction")
        assert approx_equal(outcome.user_satisfaction, 0.9)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_large_embedding_vector(self):
        """Test handling of large embedding vectors."""
        episode = episode_pb2.Episode()
        # OpenAI embedding size
        large_embedding = [float(i) / 1000 for i in range(1536)]
        episode.embedding.extend(large_embedding)

        serialized = episode.SerializeToString()
        deserialized = episode_pb2.Episode()
        deserialized.ParseFromString(serialized)

        assert len(deserialized.embedding) == 1536
        # Just check a few values for approximate equality
        assert approx_equal(deserialized.embedding[0], large_embedding[0])
        assert approx_equal(deserialized.embedding[100], large_embedding[100])
        assert approx_equal(deserialized.embedding[-1], large_embedding[-1])

    def test_empty_collections(self):
        """Test handling of empty repeated fields."""
        plan = episode_pb2.Plan()
        assert len(plan.subtasks) == 0

        trajectory = episode_pb2.Trajectory()
        assert len(trajectory.steps) == 0

    def test_unicode_strings(self):
        """Test handling of Unicode strings."""
        problem = episode_pb2.Problem()
        problem.description = "Calculate π to 10 decimal places 计算π"
        problem.domain = "mathematics 数学"

        serialized = problem.SerializeToString()
        deserialized = episode_pb2.Problem()
        deserialized.ParseFromString(serialized)

        assert deserialized.description == "Calculate π to 10 decimal places 计算π"
        assert deserialized.domain == "mathematics 数学"

    def test_boundary_values(self):
        """Test boundary values for numeric fields."""
        outcome = episode_pb2.Outcome()
        outcome.reward = -1.0  # Minimum
        assert approx_equal(outcome.reward, -1.0)

        outcome.reward = 1.0  # Maximum
        assert approx_equal(outcome.reward, 1.0)

        outcome.quality_score = 0.0  # Minimum
        assert approx_equal(outcome.quality_score, 0.0)

        outcome.quality_score = 1.0  # Maximum
        assert approx_equal(outcome.quality_score, 1.0)


@pytest.fixture
def sample_episode():
    """Fixture providing a complete sample episode."""
    episode = episode_pb2.Episode()
    episode.id = "ep_sample_001"

    # Set timestamps
    now = datetime.now(timezone.utc)
    episode.created_at.FromDatetime(now)
    episode.updated_at.FromDatetime(now)

    # Problem
    episode.problem.description = "Implement quicksort algorithm"
    episode.problem.category = "coding"
    episode.problem.difficulty = 6
    episode.problem.domain = "algorithms"
    episode.problem.context["language"] = "python"

    # Plan
    episode.plan.plan_id = "plan_001"
    episode.plan.strategy = "Divide and conquer approach"
    episode.plan.confidence = 0.9

    # Subtasks
    st1 = episode.plan.subtasks.add()
    st1.id = "st_1"
    st1.description = "Choose pivot element"
    st1.tool = "code_generator"

    st2 = episode.plan.subtasks.add()
    st2.id = "st_2"
    st2.description = "Partition array"
    st2.tool = "code_generator"
    st2.dependencies.append("st_1")

    # Trajectory
    step1 = episode.trajectory.steps.add()
    step1.step_number = 1
    step1.subtask_id = "st_1"
    step1.status = episode_pb2.STEP_STATUS_SUCCESS
    step1.duration = 2.5

    # Outcome
    episode.outcome.success = True
    episode.outcome.reward = 0.85
    episode.outcome.quality_score = 0.9

    # Metadata
    episode.metadata.agent_version = "v1.0.0"
    episode.metadata.planner_model = "gpt-4"
    episode.metadata.tags.extend(["sorting", "algorithms", "quicksort"])

    return episode


class TestCompleteWorkflow:
    """Test complete workflows using the sample episode."""

    def test_full_episode_lifecycle(self, sample_episode):
        """Test complete episode creation, serialization, and retrieval."""
        # Verify episode is complete
        assert sample_episode.id != ""
        assert sample_episode.problem.description != ""
        assert len(sample_episode.plan.subtasks) > 0
        assert sample_episode.outcome.success is True

        # Serialize
        serialized = sample_episode.SerializeToString()
        assert len(serialized) > 0

        # Deserialize
        retrieved = episode_pb2.Episode()
        retrieved.ParseFromString(serialized)

        # Verify all key fields
        assert retrieved.id == sample_episode.id
        assert retrieved.problem.description == sample_episode.problem.description
        assert len(retrieved.plan.subtasks) == len(
            sample_episode.plan.subtasks)
        assert approx_equal(retrieved.outcome.reward,
                            sample_episode.outcome.reward)

    def test_search_workflow(self, sample_episode):
        """Test search request/response workflow."""
        # Create search request
        request = episode_pb2.SearchEpisodesRequest()
        request.text_query = "quicksort implementation"
        request.top_k = 5
        request.min_similarity = 0.7
        request.filter.success_only = True
        request.filter.categories.append("coding")

        # Simulate response
        response = episode_pb2.SearchEpisodesResponse()
        result = response.results.add()
        result.episode.CopyFrom(sample_episode)
        result.similarity = 0.95
        result.rank = 1

        assert len(response.results) == 1
        assert response.results[0].episode.id == sample_episode.id
        assert approx_equal(response.results[0].similarity, 0.95)


# Add a simple test to verify import worked
def test_import_successful():
    """Verify that the episode_pb2 module was imported successfully."""
    assert episode_pb2 is not None
    assert hasattr(episode_pb2, 'Episode')
    assert hasattr(episode_pb2, 'Problem')
    assert hasattr(episode_pb2, 'Plan')
