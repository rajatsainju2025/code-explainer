"""GraphQL API interface for code explanation service."""

import logging
from typing import Any, Dict, List, Optional, Union
import asyncio
from datetime import datetime

try:
    import strawberry
    from strawberry.fastapi import GraphQLRouter
    from strawberry.types import Info
    HAS_STRAWBERRY = True
except ImportError:
    HAS_STRAWBERRY = False
    strawberry = None
    GraphQLRouter = None
    Info = None

logger = logging.getLogger(__name__)


if HAS_STRAWBERRY:
    
    @strawberry.type
    class ExplanationResult:
        """GraphQL type for explanation results."""
        explanation: str
        strategy: str
        execution_time_ms: float
        confidence_score: Optional[float] = None
        cached: bool = False
        metadata: Optional[str] = None  # JSON string
        
    @strawberry.type
    class SecurityValidation:
        """GraphQL type for security validation results."""
        is_safe: bool
        issues: List[str]
        recommendations: List[str]
        risk_level: str
        
    @strawberry.type
    class CodeAnalysis:
        """GraphQL type for code analysis results."""
        complexity_score: int
        function_count: int
        class_count: int
        line_count: int
        has_imports: bool
        has_loops: bool
        ast_valid: bool
        
    @strawberry.type
    class BatchExplanationResult:
        """GraphQL type for batch explanation results."""
        total_processed: int
        successful: int
        failed: int
        results: List[ExplanationResult]
        total_time_ms: float
        
    @strawberry.input
    class ExplanationInput:
        """GraphQL input type for explanation requests."""
        code: str
        strategy: Optional[str] = "enhanced_rag"
        include_security_check: bool = True
        include_analysis: bool = False
        cache_enabled: bool = True
        
    @strawberry.input
    class BatchExplanationInput:
        """GraphQL input type for batch explanation requests."""
        codes: List[str]
        strategy: Optional[str] = "enhanced_rag"
        batch_size: Optional[int] = 10
        include_security_check: bool = True
        
    @strawberry.type
    class MetricsSummary:
        """GraphQL type for metrics summary."""
        total_explanations: int
        average_response_time_ms: float
        cache_hit_rate: float
        error_rate: float
        active_alerts: int
        
    @strawberry.type
    class SystemHealth:
        """GraphQL type for system health status."""
        status: str
        uptime_seconds: float
        memory_usage_mb: float
        cpu_usage_percent: float
        disk_usage_percent: float
        
    @strawberry.type
    class Query:
        """GraphQL Query type."""
        
        async def explain_code(
            self,
            info: Info,
            input: ExplanationInput
        ) -> ExplanationResult:
            """Explain a single code snippet."""
            from code_explainer import CodeExplainer
            from code_explainer.monitoring import get_metrics
            from code_explainer.security import CodeSecurityValidator
            import time
            import json
            
            start_time = time.time()
            metrics = get_metrics()
            
            try:
                # Security validation if requested
                if input.include_security_check:
                    validator = CodeSecurityValidator()
                    security_result = validator.validate_code(input.code)
                    if not security_result["is_safe"]:
                        metrics.increment_counter("security_violations")
                        raise Exception(f"Security validation failed: {security_result['issues']}")
                
                # Explain code
                explainer = CodeExplainer()
                explanation = explainer.explain(input.code, input.strategy)
                
                execution_time = (time.time() - start_time) * 1000
                
                # Record metrics
                metrics.record_event("explanation_duration", execution_time)
                metrics.increment_counter("explanations_total")
                
                return ExplanationResult(
                    explanation=explanation,
                    strategy=input.strategy,
                    execution_time_ms=execution_time,
                    cached=False,  # TODO: Check cache status
                    metadata=json.dumps({
                        "timestamp": datetime.now().isoformat(),
                        "code_length": len(input.code),
                        "strategy": input.strategy
                    })
                )
                
            except Exception as e:
                metrics.increment_counter("explanation_errors")
                logger.error(f"GraphQL explanation error: {e}")
                raise
        
        async def validate_security(
            self,
            info: Info,
            code: str
        ) -> SecurityValidation:
            """Validate code security."""
            from code_explainer.security import CodeSecurityValidator
            
            validator = CodeSecurityValidator()
            result = validator.validate_code(code)
            
            risk_levels = {0: "low", 1: "medium", 2: "high", 3: "critical"}
            risk_level = risk_levels.get(len(result.get("issues", [])), "unknown")
            
            return SecurityValidation(
                is_safe=result["is_safe"],
                issues=result.get("issues", []),
                recommendations=result.get("recommendations", []),
                risk_level=risk_level
            )
        
        async def analyze_code(
            self,
            info: Info,
            code: str
        ) -> CodeAnalysis:
            """Analyze code structure and complexity."""
            import ast
            from collections import defaultdict
            
            try:
                tree = ast.parse(code)
                
                # Count different node types
                node_counts = defaultdict(int)
                for node in ast.walk(tree):
                    node_counts[type(node).__name__] += 1
                
                return CodeAnalysis(
                    complexity_score=sum(node_counts.values()),
                    function_count=node_counts.get('FunctionDef', 0),
                    class_count=node_counts.get('ClassDef', 0),
                    line_count=len(code.splitlines()),
                    has_imports=(node_counts.get('Import', 0) + node_counts.get('ImportFrom', 0)) > 0,
                    has_loops=(node_counts.get('For', 0) + node_counts.get('While', 0)) > 0,
                    ast_valid=True
                )
                
            except SyntaxError:
                return CodeAnalysis(
                    complexity_score=0,
                    function_count=0,
                    class_count=0,
                    line_count=len(code.splitlines()),
                    has_imports=False,
                    has_loops=False,
                    ast_valid=False
                )
        
        async def get_metrics(self, info: Info) -> MetricsSummary:
            """Get system metrics summary."""
            from code_explainer.monitoring import get_metrics
            
            metrics = get_metrics()
            summary = metrics.get_metrics_summary(time_window=300)  # Last 5 minutes
            alerts = metrics.get_current_alerts()
            
            # Calculate aggregate metrics
            total_explanations = summary.get("explanations_total", {}).get("count", 0)
            avg_response_time = summary.get("explanation_duration", {}).get("avg", 0)
            cache_hits = summary.get("cache_hits", {}).get("count", 0)
            cache_misses = summary.get("cache_misses", {}).get("count", 0)
            total_requests = cache_hits + cache_misses
            cache_hit_rate = cache_hits / total_requests if total_requests > 0 else 0
            
            error_count = summary.get("explanation_errors", {}).get("count", 0)
            error_rate = error_count / total_explanations if total_explanations > 0 else 0
            
            return MetricsSummary(
                total_explanations=total_explanations,
                average_response_time_ms=avg_response_time,
                cache_hit_rate=cache_hit_rate,
                error_rate=error_rate,
                active_alerts=len(alerts)
            )
        
        async def system_health(self, info: Info) -> SystemHealth:
            """Get system health status."""
            import psutil
            import time
            
            # Get system metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('/')
            
            # Calculate uptime (simplified)
            uptime = time.time() - getattr(system_health, '_start_time', time.time())
            
            return SystemHealth(
                status="healthy" if cpu_percent < 80 and memory.percent < 80 else "degraded",
                uptime_seconds=uptime,
                memory_usage_mb=memory.used / (1024 * 1024),
                cpu_usage_percent=cpu_percent,
                disk_usage_percent=disk.percent
            )
    
    @strawberry.type
    class Mutation:
        """GraphQL Mutation type."""
        
        async def batch_explain(
            self,
            info: Info,
            input: BatchExplanationInput
        ) -> BatchExplanationResult:
            """Explain multiple code snippets in batch."""
            from code_explainer.async_processor import AsyncCodeExplainer, BatchProcessor
            from code_explainer import CodeExplainer
            import time
            
            start_time = time.time()
            
            # Initialize async explainer
            base_explainer = CodeExplainer()
            async_explainer = AsyncCodeExplainer(base_explainer, max_workers=4)
            
            try:
                # Process batch
                explanations = await async_explainer.batch_explain(
                    input.codes,
                    input.strategy,
                    input.batch_size
                )
                
                # Create results
                results = []
                successful = 0
                failed = 0
                
                for i, (code, explanation) in enumerate(zip(input.codes, explanations)):
                    if explanation.startswith("Error:"):
                        failed += 1
                    else:
                        successful += 1
                    
                    results.append(ExplanationResult(
                        explanation=explanation,
                        strategy=input.strategy,
                        execution_time_ms=0,  # Individual timing not available
                        cached=False
                    ))
                
                total_time = (time.time() - start_time) * 1000
                
                return BatchExplanationResult(
                    total_processed=len(input.codes),
                    successful=successful,
                    failed=failed,
                    results=results,
                    total_time_ms=total_time
                )
                
            except Exception as e:
                logger.error(f"Batch explanation error: {e}")
                raise
        
        async def clear_cache(self, info: Info) -> bool:
            """Clear all caches."""
            try:
                from code_explainer.cache import get_cache_manager
                
                cache_manager = get_cache_manager()
                cache_manager.clear_all()
                
                logger.info("Cache cleared via GraphQL")
                return True
                
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
                return False
        
        async def trigger_model_reload(self, info: Info) -> bool:
            """Trigger model reload for updates."""
            try:
                # This would trigger a model reload in a production system
                logger.info("Model reload triggered via GraphQL")
                return True
                
            except Exception as e:
                logger.error(f"Error reloading model: {e}")
                return False
    
    # Create GraphQL schema
    schema = strawberry.Schema(query=Query, mutation=Mutation)
    
    def create_graphql_router() -> GraphQLRouter:
        """Create GraphQL router for FastAPI integration."""
        return GraphQLRouter(schema, path="/graphql")

else:
    # Fallback when strawberry is not available
    def create_graphql_router():
        """Fallback when GraphQL dependencies are not available."""
        logger.warning("Strawberry GraphQL not installed. GraphQL interface not available.")
        return None


class GraphQLConfig:
    """Configuration for GraphQL API."""
    
    def __init__(
        self,
        enable_introspection: bool = True,
        enable_playground: bool = True,
        max_query_depth: int = 10,
        query_cache_size: int = 100
    ):
        """Initialize GraphQL configuration.
        
        Args:
            enable_introspection: Enable GraphQL introspection
            enable_playground: Enable GraphQL playground
            max_query_depth: Maximum query depth allowed
            query_cache_size: Size of query cache
        """
        self.enable_introspection = enable_introspection
        self.enable_playground = enable_playground
        self.max_query_depth = max_query_depth
        self.query_cache_size = query_cache_size


def main():
    """Example GraphQL server setup."""
    if not HAS_STRAWBERRY:
        print("Strawberry GraphQL not installed. Run: pip install strawberry-graphql")
        return
    
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="Code Explainer GraphQL API")
    
    # Add GraphQL router
    graphql_router = create_graphql_router()
    if graphql_router:
        app.include_router(graphql_router, prefix="/api")
    
    @app.get("/")
    async def root():
        return {"message": "Code Explainer GraphQL API", "graphql": "/api/graphql"}
    
    print("Starting GraphQL server on http://localhost:8000")
    print("GraphQL Playground: http://localhost:8000/api/graphql")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
