# FastAPI Advanced Patterns: Building Production-Ready APIs

## 1. Authentication and Authorization

### 1.1 JWT Token Authentication

JSON Web Tokens (JWT) provide a stateless authentication mechanism. The server
generates a token containing encoded claims (user ID, roles, expiration time)
and signs it with a secret key. The client includes this token in subsequent
requests via the Authorization header.

Implementation considerations:
- Always use short-lived access tokens (15-30 minutes) paired with refresh tokens
- Store refresh tokens in HTTP-only cookies to prevent XSS attacks
- Implement token rotation: when a refresh token is used, issue a new one
- Use asymmetric keys (RS256) in distributed systems so services can verify
  tokens without sharing the signing key

### 1.2 Role-Based Access Control (RBAC)

RBAC maps users to roles, and roles to permissions. FastAPI's dependency injection
system provides an elegant way to implement RBAC:

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import SecurityScopes

async def check_permissions(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme),
):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
    )
    token_data = decode_token(token)
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
    return token_data
```

## 2. Database Patterns

### 2.1 Repository Pattern

The repository pattern abstracts database operations behind a clean interface.
This decouples business logic from data access, making the code more testable
and allowing easy switching between different storage backends (PostgreSQL,
MongoDB, in-memory stores for testing).

### 2.2 Connection Pooling

Database connections are expensive to create. Connection pooling maintains a
pool of reusable connections. SQLAlchemy's async engine with asyncpg provides
excellent connection pooling for PostgreSQL:

- Set pool_size to match expected concurrent connections
- Configure max_overflow for burst capacity
- Use pool_timeout to prevent connection starvation
- Enable pool_pre_ping to detect stale connections

### 2.3 Database Migrations

Alembic provides robust schema migration support. Best practices:
- Always generate migrations from model changes (autogenerate)
- Review generated migrations before applying
- Test migrations in CI/CD pipeline
- Support both upgrade and downgrade paths
- Use batch operations for SQLite compatibility

## 3. Caching Strategies

### 3.1 Redis Caching

Redis provides an excellent caching layer for API responses. Consider:
- Cache-aside pattern: check cache first, fall back to database
- Time-based expiration: set TTL based on data volatility
- Cache invalidation: update/delete cache on data mutations
- Serialization: use msgpack for faster serialization than JSON

### 3.2 Request Deduplication

When multiple identical requests arrive simultaneously, only execute the
database query once and share the result. This is particularly important
for GraphQL APIs where the same resolver might be called multiple times
in a single query.

## 4. Error Handling and Observability

### 4.1 Structured Error Responses

Consistent error responses improve API usability:

```json
{
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid input data",
        "details": [
            {"field": "email", "message": "Invalid email format"},
            {"field": "age", "message": "Must be a positive integer"}
        ],
        "request_id": "req_abc123"
    }
}
```

### 4.2 Distributed Tracing

OpenTelemetry provides vendor-neutral distributed tracing. Each request
receives a trace ID that propagates through microservices, enabling
end-to-end visibility into request processing:

- Instrument HTTP clients and database queries automatically
- Add custom spans for business-critical operations
- Export traces to Jaeger, Zipkin, or cloud-native solutions
- Use baggage items to propagate request context

### 4.3 Health Checks

Implement comprehensive health checks:
- Liveness probe: is the process running?
- Readiness probe: can the service handle requests?
- Check database connectivity, cache availability, and external dependencies
- Include response time metrics and version information

## 5. Rate Limiting and Throttling

Rate limiting protects APIs from abuse and ensures fair resource allocation.
Common algorithms:
- Fixed window: simple but can allow burst at window boundaries
- Sliding window: smoother rate limiting, more memory intensive
- Token bucket: allows controlled bursts while maintaining average rate
- Leaky bucket: processes requests at a constant rate

Implementation with Redis:
```python
async def rate_limit(key: str, limit: int, window: int) -> bool:
    current = await redis.incr(key)
    if current == 1:
        await redis.expire(key, window)
    return current <= limit
```

## 6. Testing Strategies

### 6.1 API Testing Pyramid

- Unit tests: test individual functions and validators
- Integration tests: test endpoints with real database
- Contract tests: verify API schema compatibility
- Load tests: measure performance under expected traffic

### 6.2 Test Fixtures

Use pytest fixtures for consistent test data:
- Factory functions for creating test objects
- Database transactions that rollback after each test
- Mock external services to isolate tests
- Use testcontainers for realistic integration tests
