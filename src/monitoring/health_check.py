"""
Health check system for production monitoring
"""
import asyncio
import psutil
import aiohttp
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
from enum import Enum

from src.monitoring.logger import logger
from src.config.settings import settings
from src.data.storage.redis_client import redis_client
from src.monitoring.metrics import metrics_collector


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth:
    """Health status of a single component"""

    def __init__(self, name: str, status: HealthStatus, message: str = "", details: Dict = None):
        self.name = name
        self.status = status
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class HealthChecker:
    """Comprehensive health checking system"""

    def __init__(self):
        self.checks = []
        self.last_check = None
        self.check_interval = 30  # seconds

    async def check_redis(self) -> ComponentHealth:
        """Check Redis connectivity and performance"""
        try:
            start = asyncio.get_event_loop().time()

            # Test connection
            await redis_client.set("health_check", "OK", expire=10)
            result = await redis_client.get("health_check")

            latency = (asyncio.get_event_loop().time() - start) * 1000

            if result != "OK":
                return ComponentHealth(
                    "redis",
                    HealthStatus.UNHEALTHY,
                    "Redis read/write test failed"
                )

            # Check latency
            if latency > 100:  # >100ms is concerning
                return ComponentHealth(
                    "redis",
                    HealthStatus.DEGRADED,
                    f"High latency: {latency:.2f}ms",
                    {"latency_ms": latency}
                )

            return ComponentHealth(
                "redis",
                HealthStatus.HEALTHY,
                f"Latency: {latency:.2f}ms",
                {"latency_ms": latency}
            )

        except Exception as e:
            return ComponentHealth(
                "redis",
                HealthStatus.UNHEALTHY,
                f"Connection failed: {str(e)}"
            )

    async def check_database(self) -> ComponentHealth:
        """Check database connectivity"""
        try:
            from src.data.storage.database import db_manager

            async with db_manager.acquire() as conn:
                result = await conn.fetchval("SELECT 1")

                if result == 1:
                    # Check table existence
                    tables = await conn.fetch("""
                        SELECT tablename FROM pg_tables 
                        WHERE schemaname = 'public'
                    """)

                    expected_tables = {'klines', 'trades', 'positions', 'signals'}
                    existing_tables = {t['tablename'] for t in tables}

                    if not expected_tables.issubset(existing_tables):
                        missing = expected_tables - existing_tables
                        return ComponentHealth(
                            "database",
                            HealthStatus.DEGRADED,
                            f"Missing tables: {missing}"
                        )

                    return ComponentHealth(
                        "database",
                        HealthStatus.HEALTHY,
                        "All tables present"
                    )

        except Exception as e:
            return ComponentHealth(
                "database",
                HealthStatus.UNHEALTHY,
                f"Connection failed: {str(e)}"
            )

    async def check_binance_api(self) -> ComponentHealth:
        """Check Binance API connectivity"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.binance.com/api/v3/ping" if not settings.api.TESTNET \
                    else "https://testnet.binance.vision/api/v3/ping"

                start = asyncio.get_event_loop().time()
                async with session.get(url, timeout=5) as response:
                    latency = (asyncio.get_event_loop().time() - start) * 1000

                    if response.status == 200:
                        if latency > 500:  # >500ms is concerning for API
                            return ComponentHealth(
                                "binance_api",
                                HealthStatus.DEGRADED,
                                f"High latency: {latency:.2f}ms",
                                {"latency_ms": latency}
                            )

                        return ComponentHealth(
                            "binance_api",
                            HealthStatus.HEALTHY,
                            f"Latency: {latency:.2f}ms",
                            {"latency_ms": latency}
                        )
                    else:
                        return ComponentHealth(
                            "binance_api",
                            HealthStatus.UNHEALTHY,
                            f"HTTP {response.status}"
                        )

        except asyncio.TimeoutError:
            return ComponentHealth(
                "binance_api",
                HealthStatus.UNHEALTHY,
                "Request timeout"
            )
        except Exception as e:
            return ComponentHealth(
                "binance_api",
                HealthStatus.UNHEALTHY,
                f"Connection failed: {str(e)}"
            )

    async def check_system_resources(self) -> ComponentHealth:
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            issues = []
            status = HealthStatus.HEALTHY

            # CPU check
            if cpu_percent > 90:
                issues.append(f"CPU critical: {cpu_percent:.1f}%")
                status = HealthStatus.UNHEALTHY
            elif cpu_percent > 70:
                issues.append(f"CPU high: {cpu_percent:.1f}%")
                status = HealthStatus.DEGRADED if status == HealthStatus.HEALTHY else status

            # Memory check
            if memory.percent > 90:
                issues.append(f"Memory critical: {memory.percent:.1f}%")
                status = HealthStatus.UNHEALTHY
            elif memory.percent > 80:
                issues.append(f"Memory high: {memory.percent:.1f}%")
                status = HealthStatus.DEGRADED if status == HealthStatus.HEALTHY else status

            # Disk check
            if disk.percent > 90:
                issues.append(f"Disk critical: {disk.percent:.1f}%")
                status = HealthStatus.UNHEALTHY
            elif disk.percent > 80:
                issues.append(f"Disk high: {disk.percent:.1f}%")
                status = HealthStatus.DEGRADED if status == HealthStatus.HEALTHY else status

            message = "; ".join(issues) if issues else "All resources normal"

            return ComponentHealth(
                "system",
                status,
                message,
                {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                    "memory_available_gb": memory.available / (1024 ** 3),
                    "disk_free_gb": disk.free / (1024 ** 3)
                }
            )

        except Exception as e:
            return ComponentHealth(
                "system",
                HealthStatus.UNHEALTHY,
                f"Check failed: {str(e)}"
            )

    async def check_data_freshness(self) -> ComponentHealth:
        """Check if market data is fresh"""
        try:
            # Check latest price update for main symbols
            stale_symbols = []

            for symbol in settings.trading.SYMBOLS[:3]:
                cache_key = f"market:{symbol}"
                data = await redis_client.get(cache_key)

                if data and isinstance(data, dict):
                    timestamp = data.get('timestamp')
                    if timestamp:
                        last_update = datetime.fromisoformat(timestamp)
                        age = (datetime.utcnow() - last_update).total_seconds()

                        if age > 300:  # >5 minutes is stale
                            stale_symbols.append(f"{symbol}({age:.0f}s)")

            if stale_symbols:
                if len(stale_symbols) == len(settings.trading.SYMBOLS[:3]):
                    return ComponentHealth(
                        "data_freshness",
                        HealthStatus.UNHEALTHY,
                        f"All data stale: {', '.join(stale_symbols)}"
                    )
                else:
                    return ComponentHealth(
                        "data_freshness",
                        HealthStatus.DEGRADED,
                        f"Some data stale: {', '.join(stale_symbols)}"
                    )

            return ComponentHealth(
                "data_freshness",
                HealthStatus.HEALTHY,
                "All data fresh"
            )

        except Exception as e:
            return ComponentHealth(
                "data_freshness",
                HealthStatus.UNHEALTHY,
                f"Check failed: {str(e)}"
            )

    async def check_ml_models(self) -> ComponentHealth:
        """Check ML models availability"""
        try:
            from src.ml.models.ml_engine import ml_engine

            if ml_engine.is_trained:
                # Test prediction
                import pandas as pd
                test_df = pd.DataFrame({
                    'close': [50000],
                    'rsi': [50],
                    'macd': [0],
                    'bb_percent': [0.5],
                    'volume_ratio': [1.0]
                })

                result = await ml_engine.predict(test_df)

                if result and 'prediction_label' in result:
                    return ComponentHealth(
                        "ml_models",
                        HealthStatus.HEALTHY,
                        "Models operational"
                    )
                else:
                    return ComponentHealth(
                        "ml_models",
                        HealthStatus.DEGRADED,
                        "Prediction test failed"
                    )
            else:
                return ComponentHealth(
                    "ml_models",
                    HealthStatus.DEGRADED,
                    "Models not trained"
                )

        except Exception as e:
            return ComponentHealth(
                "ml_models",
                HealthStatus.UNHEALTHY,
                f"Check failed: {str(e)}"
            )

    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        checks = await asyncio.gather(
            self.check_redis(),
            self.check_database(),
            self.check_binance_api(),
            self.check_system_resources(),
            self.check_data_freshness(),
            self.check_ml_models(),
            return_exceptions=True
        )

        results = {}
        overall_status = HealthStatus.HEALTHY

        for check in checks:
            if isinstance(check, Exception):
                logger.logger.error(f"Health check failed: {check}")
                continue

            if isinstance(check, ComponentHealth):
                results[check.name] = {
                    "status": check.status.value,
                    "message": check.message,
                    "details": check.details,
                    "timestamp": check.timestamp.isoformat()
                }

                # Update overall status
                if check.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif check.status == HealthStatus.DEGRADED and overall_status != HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.DEGRADED

        self.last_check = datetime.utcnow()

        return {
            "status": overall_status.value,
            "timestamp": self.last_check.isoformat(),
            "components": results
        }

    async def start_monitoring(self):
        """Start continuous health monitoring"""
        while True:
            try:
                health_status = await self.run_all_checks()

                # Update metrics
                for component, status in health_status['components'].items():
                    if status['status'] == 'unhealthy':
                        metrics_collector.record_error('health_check', component)

                # Log if unhealthy
                if health_status['status'] != 'healthy':
                    logger.logger.warning(
                        f"Health check: {health_status['status']}",
                        components=health_status['components']
                    )

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.check_interval)


# Global health checker instance
health_checker = HealthChecker()


async def check_health() -> bool:
    """Simple health check for Docker healthcheck"""
    try:
        result = await health_checker.run_all_checks()
        return result['status'] != 'unhealthy'
    except:
        return False


# HTTP endpoint for health checks
async def health_endpoint(request):
    """HTTP endpoint for health status"""
    result = await health_checker.run_all_checks()

    # Return appropriate HTTP status code
    if result['status'] == 'healthy':
        status_code = 200
    elif result['status'] == 'degraded':
        status_code = 200  # Still operational but degraded
    else:
        status_code = 503  # Service unavailable

    return status_code, result