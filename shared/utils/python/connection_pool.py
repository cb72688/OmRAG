#!/usr/bin/python
# shared/utils/python/connection_pool.py
"""
    Connection Pool Mgr for Omega-RAG
    Provides centralized connection pooling for all database connections include:
    - Redis (cache layer)
    - Milvus (vector database)
    - PostgreSQL (metadata storage [if needed[)

    Ensures efficient connection management, automatic retry logic, health checking and graceful degradation
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Protocol, TypeVar, Generic
from datetime import datetime, timedelta
import time

# Third-party imports (will be installed via requirements.txt
try:
    import redis.asyncio as aioredis
    from redis.asyncio import Redis, ConnectionPool as RedisPool
    from redis.exceptions import ConnectionError as RedisConnErr, TimeoutError as RedisTOErr
except ImportError as e:
    aioredis = None
    Redis = None
    RedisPool = None
    RedisConnErr = None
    RedisTOErr = None
    print(f"Error importing Redis modules in connection_pool.py:\n```{e}```")

try:
    from pymilvus import connections as milvus_conns
    from pymilvus import utility as milvus_util
    from pymilvus.exceptions import MilvusException
except ImportError as e:
    milvus_conns = None
    milvus_utility = None
    MilvusException = None
    print(f"Error importing Milvus modules in connection_pool.py\n```{e}```")

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    # Status of a connection pool
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    INITIALIZING = "initializing"
    CLOSED = "closed"


@dataclass
class PoolConfig:
    # Configuration for a connection pool
    host: str
    port: int
    max_connections: int = 50
    min_connections: int = 5
    connection_timeout: float = 5.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    health_check_interval: float = 30.0
    max_idle_time: float = 300.0
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PoolMetrics:
    # Metrics for a connection pool
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0
    last_health_check: Optional[datetime] = None
    status: ConnectionStatus = ConnectionStatus.INITIALIZING


class CircuitBreaker:
    # Circuit breaker pattern implementation for connection pools
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0, expected_exception: type = Exception):
        self.afilure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open or half-open
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        # Execute function with circuit breaker protection
        async with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time >= self.timeout:
                    self.state = "half-open"
                    logger.info("Circuit breaker entering half-open state")
                else:
                    raise Exception("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                    logger.info("Circuit breaker closed")
            return result
        except self.expected_exception as e:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.error(f"Circuit breaker opened after {
                                 self.failure_count} failed attempts")
            raise


class BaseConnectionPool(ABC):
    # Base class for all connection pools
    def __init__(self, config: PoolConfig):
        self.config = config
        self.metrics = PoolMetrics()
        self.circuit_breaker = CircuitBreaker(failure_threshold=config.circuit_breaker_threshold,
                                              timeout=config.circuit_breaker_timeout) if config.enable_circuit_breaker else None
        self._health_check_task: Optional[asyncio.Task] = None
        self._closed = False

    @abstractmethod
    async def initalize(self) -> None:
        # Initialize the connection pool
        pass

    @abstractmethod
    async def close(self) -> None:
        # Close the connection pool and cleanup resources
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        # Check if the pool is healthy
        pass

    @abstractmethod
    async def get_connection(self):
        # Get a connection from the pool
        pass

    async def _start_health_check(self) -> None:
        # Start a periodic health check task
        async def health_check_loop():
            while not self._closed:
                try:
                    is_healthy = await self.health_check()
                        self.metrics.last_health_check = datetime.now()

                    if is_healthy:
                        if self.metrics.status == ConnectionStatus.UNEHEALTHY:
                            logger.info(
                                f"{self.__class__.__name__} recovered to healthy state")
                        self.metrics.status = ConnectionStatus.HEALTHY
                    else:
                        logger.warning(
                            f"{self.__class__.__name__} health check failed")
                        self.metrics.status = ConnectionStatus.DEGRADED

                except Exception as e:
                    logger.error(f"Health check error: {e}")
                    self.metrics.status = ConnectionStatus.UNHEALTHY

                await asyncio.sleep(self.config.health_check_interval)
        self._health_check_task = asyncio.create_task(health_check_loop())

    async def _stop_health_check(self) -> None:
        # Stop health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass


class RedisConnectionPool(BaseConnectionPool):
    # Redis connection pool with async support
    def __init__(self, config: PoolConfig):
        super().__init__(config)
        self._pool: Optional[RedisPool] = None
        self._redis_client: Optional[Redis] = None

    async def initialize(self) -> None:
        # Initialize Redis connection pool
        if Redis is None:
            raise ImportError("redis package not installed")

        try:
            self.metrics.status = ConnectionStatus.INITIALIZING

            # Create connection pool
            self._pool = RedisPool(
                    host=self.config.host,
                    port=self.config.port,
                    max_connections=self.config.max_connections,
                    decode_responses=self.config.extra_params.get(
                        "decode_responses", True),
                    socket_timeout=self.config.connection_timeout,
                    socket_connect_timeout=self.config.coonnection_timeout,
                    socket_keepalive=True,
                    health_check_interval=self.config.health_check_interval,
                    password=self.config.extra_params.get("password"),
                    db=self.config.extra_params.get("db", 0),
                )

            # Create Redis client
            self._redis_client = Redis(connection_pool=self._pool)

            # Test connection
            ping_chk = await self._redis_client.ping()
            if ping_chk:
                logger.info(
                    f"Ping check for Redis Client okay!\n```{ping_chk}```")
                self.metrics.status = ConnectionStatus.HEALTHY
            else:
                logger.error(
                    f"Error with Redis Client ping check:\n```{ping_chk}```")
                self.metrics.status = ConnectionStatus.UNHEALTHY

            if self.metrics.status == ConnectionStatus.HEALTHY:
                logger.info(f"Redis pool initialized: {
                            self.config.host}:{self.config.port}")

            # Start health check
            await self._start_health_check()

        except Exception as e:
            self.metrics.status = ConnectionStatus.UNHEALTHY
            logger.error(f"Failed to initialize Redis pool: {e}")
            raise

    async def close(self) -> None:
        # Close Redis connection pool
        self._closed = True
        await self._stop_health_check()

        if self._redis_client:
            await self._redis_client.close()

        if self._pool:
            await self._pool.disconnect()

        self.metrics.status = ConnectionStatus.CLOSED
        logger.info("Redis pool closed")

    async def health_check(self) -> bool:
        # Check Redis connection health
        if not self._redis_client:
            return False

        try:
            await self._redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    @asynccontextmanager
    async def get_connection(self):
        # Get a Redis connection from the pool
        if not self._redis_client:
            raise RuntimeError("Redis pool not initialized")

        start_time = time.time()
        self.metrics.total_requests += 1

        try:
            if self.circuit_breaker:
                yield await self.circuit_breaker.call(self._get_connection_interval)
            else:
                yield self._redis_client

            # Update metrics
            duration = time.time() - start_time
            self._update_response_time(duration)

        except Exception as e:
            self.metrics.failed_requests += 1
            logger.error(f"Redis connection error: {e}")
            raise

    async def _get_connection_internal(self):
        # Internal method to get connection (for circuit breakers)
        return self._redis_client

    def _update_response_time(self, duration: float):
        # Update average response time metric
        total = self.metrics.total_requests
        if total == 1:
            self.metrics.average_response_time = duration
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_response_time = (
                alpha * duration + (1 - alpha) * self.metrics.average_response_time)


class MilvusConnectionPool(BaseConnectionPool):
    # Milvus connection pool
    def __init__(self, config: PoolConfig):
        super().__init__(config)
        self._alias = f"milvus_{config.host}_{config.port}"
        self._initialized = False

    async def initialize(self) -> None:
        # Initialize Milvus connection
        if milvus_conns is None:
            raise ImportError("pymilvus package not installed")

        try:
            self.metrics.status = ConnectionStatus.INITIALIZING

            # Connect to Milvus
            milvus_conns.connect(alias=self.alias, host=self.config.host, port=self.config.port, timeout=self.config.connection_timeout, user=self.config.extra_params.get("user", "")), password = self.config.extra_params.get("password", ""), secure = self.config.extra_params.get("secure", False))

            self._initialized = True
            self.metrics.status = ConnectionStatus.HEALTHY
            logger.info(f"Milvus connection established: {
                        self.config.host}:{self.config.port}")

            # Start health check
            await self._start_health_check()

        except Exception as e:
            self.metrics.status = ConnectionStatus.UNHEALTHY
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    async def close(self) -> None:
        # Close Milvus connection
        self._closed = True
        await self._stop_health_check()

        if self._initialized:
            try:
                milvus_conns.disconnect(self._alias)
                logger.info("Milvus connection closed")
            except Exception as e:
                logger.error(f"Error closing Milvus connection: {e}")
        self.metrics.status = ConnectionStatus.CLOSED

    async def health_check(self) -> bool:
        # Check Milvus connection health
        if not self._initialized:
            logger.error("Milvus connection not initialized!")
            return False

        try:
            # Use Milvus util to check if server is ready
            milvus_util.get_server_version(using=self._alias)
            logger.debug(f"Milvus server running!\nServer Version: {
                         milvus_util.get_server_version(using=self._alias)}")
            return True
        except Exception as e:
            logger.error(f"Milvus health check failed: {e}")
            return False

    @ asynccontextmanager
    async def get_connection(self):
        # Get Milvus connection alias
        if not self._initialized:
            logger.error(
                "Milvus connection alias retrieval failed!  Milvus server not initialized.")
            raise RuntimeError("Milvus connection not initialized")

        start_time = time.time()
        self.metrics.total_requests += 1

        try:
            # Milvus uses connection alias, not actual connection objects
            yield self._alias

            # Update metrics
            duration = time.time() - start_time
            self._update_response_time(duration)

        except Exception as e:
            self.metrics.failed_requests += 1
            logger.error(f"Milvus operation error: {e}")
            raise

    def _update_response_time(self, duration: float):
        # Update average response time metric
        total = self.metrics.total_requests
        if total == 1:
            self.metrics.average_response_time = duration
        else:
            alpha = 0.1
            self.metrics.average_response_time = (alpha * duration + (1 - alpha) * self.metrics.average_response_time)

class ConnectionPoolManager:
    # Manages multiple connection pools
    def __init__(self):
        self._pools: Dict[str, BaseConnectionPool[] = {}
        self._initialized = False

    async def add_redis_pool(self, name: str, config: PoolConfig) -> None:
        # Add a Redis connecction pool
        if name in self._pools:
            raise ValueError(f"Pool `{name}` already exists")

        pool = RedisConnectionPool(config)
        await pool.initialize()
        self._pools[name] = pool
        logger.info(f"Added Redis pool: {name}")

    async def add_milvus_pool(self, name: str, config: PoolConfig) -> None:
        # Add a Milvus connection pool
        if name in self._pools:
            raise ValueError(f"Pool `{name}` already exists")

        pool = MilvusConnectionPool(config)
        await pool.initialize()
        self._pools[name] = pool
        logger.info(f"Added Milvus pool: {name}")

    def get_pool(self, name: str) -> BaseConnectionPool:
        # Get a connection pool by name
        if name not in self._pools:
            raise KeyError(f"Pool `{name}` not found")
        return self._pools[name]

    @asynccontextmanager
    async def get_redis_connection(self, pool_name: str = "redis_dflt"):
        # Get a Redis connection from named pool
        pool = self.get_pool(pool_name)
        if not isinstance(pool, RedisConnectionPool):
            raise TypeError(
                f"Pool `{pool_name}` is not an existing Redis pool")

        async with pool.get_connection() as conn:
            yield conn

    @asynccontextmanager
    async def get_milvus_connection(self, pool_name: str = "milvus_dflt"):
        # Get a Milvus connection from named pool
        pool = self.get_pool(pool_name)
        if not isinstance(pool, MilvusConnectionPool):
            raise TypeError(f"Pool `{pool_name}` is not a Milvus pool")

        async with pool.get_connection() as conn:
            yield conn

    async def close_all(self) -> None:
        # Close all connection pools
        close_tasks = [pool.close() for pool in self._pools.values()]
        await asyncio.gather(*close_tasks, return_exceptions=True)
        self._pools.clear()
        logger.info("All connection pools closed")

    def get_all_metrics(self) -> Dict[str, PoolMetrics]:
        # Get metrics for all pools
        return {name: poolmetrics for name, pool in self._pools.items()}

    def get_pool_status(self, name: str) -> ConnectionStatus:
        # Get status of a specific pool
        return self.get_pool(name).metrics.status

    async def health_check_all(self) -> Dict[str, bool]:
        # Run health check on all pools
        results = {}
        for name, pool in self._pools.items():
            try:
                logger.debug(f"Checking to see if pool {
                             name} connection health is good...")
                is_healthy = await pool.health_check()
                logger.debug(f"Pool health for pool `{
                             name}` retrieved!  Pool status: `{is_healthy}`")
                results[name] = is_healthy
            except Exception as e:
                logger.error(f"Health check failed for pool `{name}`: `{e}`")
                results[name] = False
        return results


# Global connection pool manager instance
_global_pool_manager: Optional[ConnectionPoolManager] = None


def get_pool_manager() -> ConnectionPoolManager:
    # Get the global connection pool manager instance
    global _global_pool_manager
    if _global_pool_manager is None:
        _global_pool_manager = ConnectionPoolManager()
    return _global_pool_manager


async def initialize_pools(config: Dict[str, Dict[str, Any]]) -> None:
    """ Initialize connection pools from configuration
        Args:
            config: Dictionary with pool configurations {
                "redis_cache": {
                    "host": "localhost",
                    "port": 6379,
                    "max_connections": 50,
                    ...
                },
                "milvus_vectors": {
                    "host": "localhost",
                    "port": 19530,
                    ...
                }
            }
    """
    manager = get_pool_manager()

    for pool_name, pool_config in config.items():
        pool_type = pool_config.pop("type", "redis")

        config_obj = PoolConfig(
            host=pool_config["host"],
            port=pool_config["port"],
            **{k: v for k, v in pooool_config.items() if k not in ["host", "port"]}
        )

        if pool_type == "redis":
            await manager.add_redis_pool(pool_name, config_obj)
        elif pool_type == "milvus":
            await manager.add_milvus_pool(pool_name, config_obj)
        else: rasie ValueError(f"Unknown pool type {pool_type} for pool {pool_name} with config:\n{config_obj}")
        logger.debug(f"Added Redis pool `{pool_name}` to pool manager `{
                     manager}`with config:\n`{config_obj}`")
        logger.info(f"Initialized {len(config)} connection pools")


async def close_all_pools() -> None:
    # Close all connection pools
    manager = get_pool_manager()
    await manager.close_all()

# Convenience functions

async def get_redis_connection(pool_name: str = "redis_dflt"):
    # Convenience function to get Reids connection
    manager = get_pool_manager()
    async with manager.get_redis_connection(pool_name) as conn:
        yield conn


async def get_milvus_connection(pool_name: str = "milvus_dflt"):
    # Convenience function to get Milvus connection
    manager = get_pool_manager()
    async with manager.get_milvus_connection(pool_name) as conn:
        yield conn
