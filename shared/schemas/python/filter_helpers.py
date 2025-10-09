#!/usr/bin/python
# shared/schemas/python/filter_helpers.py
"""
    Helper functions for working with EpisodeFilter messages.  Handles the has_* pattern for optional fields in proto3
"""

import episode_pb2
from typing import Optional, List
from datetime import datetime
from google.protobuf.timestamp_pb2 import Timestamp


def create_episode_filter(
    success_only: Optional[bool] = None,
    categories: Optional[bool] = None,
    domains: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    min_reward: Optional[float] = None,
    min_quality: Optional[float] = None,
    created_after: Optional[datetime] = None,
    created_before: Optional[datetime] = None,
    session_id: Optional[str] = None,
    exclude_archived: bool = True,
) -> episode_pb2.EpisodeFilter:
    """
        Create an EpisodeFilter with proper handling of optional fields.

        Args:
            success_only: If set, filter for successful episodes only
            categories: List of categories to filter by
            domains: List of domains to filter by
            tags: List of tags to filter by
            min_reward: Minimum reward threshold
            min_quality: Minimum quality threshold
            created_after: Filter episodes created after this date
            created_before: Filter episodes created before this date
            session_id: Filter by session ID
            exclude_archived: Exclude archived episodes

        Returns:
            EpipsodeFilter message
    """
    filter_msg = episode_pb2.EpisodeFilter()

    # Handle success_only
    if success_only is not None:
        filter_msg.success_only = success_only
        filter_msg.has_success_filter = True

    # Handle list fields
    if categories:
        filter_msg.categories.extend(categories)
    if domains:
        filter_msg.domains.extend(domains)
    if tags:
        filter_msg.tags.extend(tags)

    # Handle min_reward
    if min_reward is not None:
        filter_msg.min_reward = min_reward
        filter_msg.has_min_reward = True

    # Handle min_quality
    if min_quality is not None:
        filter_msg.min_quality = min_quality
        filter_msg.has_min_quality = True

    # Handle date range
    if created_after:
        filter_msg.created_after.FromDatetime(created_after)
    if created_before:
        filter_msg.created_before.FromDatetime(created_before)

    # Handle session_id
    if session_id:
        filter_msg.session_id = session_id

    filter_msg.exclude_archived = exclude_archived

    return filter_msg


def has_filter_value(filter_msg: episode_pb2.EpisodeFilter, field_name: str) -> bool:
    """
        Check if a filter field has been set
        Args:
            filter_msg: The filter message
            field_name: Name of the fields to check
        Returns:
            True if the field has been set
    """
    if field_name == "success_only":
        return filter_msg.has_success_filter
    elif field_name == "min_reward":
        return filter_msg.has_min_reward
    elif field_name == "min_quality":
        return filter_msg.has_min_quality
    elif field_name == "created_after":
        return filter_msg.HasField("created_after")
    elif field_name == "created_before":
        return filter_msg.HasField("created_before")
    elif field_name == "session_id":
        return bool(filter_msg.session_id)
    elif field_name == "categories":
        return len(filter_msg.categories) > 0
    elif field_name == "domains":
        return len(filter_msg.domains) > 0
    elif field_name == "tags":
        return len(filter_msg.tags) > 0
    else:
        return False
