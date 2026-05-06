"""
api/middleware/auth.py — JWT authentication dependency.
api/routes/auth.py — Login, register, token refresh endpoints.
"""
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

import jwt as _jwt
import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext

# Use PyJWT (already installed) — python-jose API-compatible subset
JWTError = _jwt.PyJWTError


class _JWTCompat:
    """Thin shim so existing jwt.encode / jwt.decode calls work unchanged."""
    @staticmethod
    def encode(payload: dict, key: str, algorithm: str = "HS256") -> str:
        return _jwt.encode(payload, key, algorithm=algorithm)

    @staticmethod
    def decode(token: str, key: str, algorithms: list) -> dict:
        return _jwt.decode(token, key, algorithms=algorithms)


jwt = _JWTCompat()
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from config import get_settings
from db.models import User
from db.session import get_db

settings = get_settings()
logger = structlog.get_logger(__name__)


# ─── JWT Middleware (used by main.py) ──────────────────────────────────────────
class JWTAuthMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that attaches the decoded JWT payload to request.state.
    Routes that require authentication use the get_current_user dependency
    instead — this middleware is for request-level context only.
    """
    EXEMPT_PATHS = {"/health", "/metrics", "/openapi.json", "/docs", "/redoc"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.EXEMPT_PATHS or request.url.path.startswith("/api/v1/auth"):
            return await call_next(request)
        token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if token:
            try:
                payload = jwt.decode(token, settings.app_secret_key, algorithms=["HS256"])
                request.state.user_id = payload.get("sub")
            except JWTError:
                pass
        return await call_next(request)

# ─── Auth configuration ───────────────────────────────────────────────────────
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8   # 8 hours
REFRESH_TOKEN_EXPIRE_DAYS = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.api_prefix}/auth/login")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def create_token(subject: str, expires_delta: timedelta) -> str:
    expire = datetime.utcnow() + expires_delta
    payload = {"sub": subject, "exp": expire, "iat": datetime.utcnow()}
    return jwt.encode(payload, settings.app_secret_key, algorithm=ALGORITHM)


# ─── FastAPI dependency: current user ─────────────────────────────────────────
async def get_current_user(
    token: str = Depends(oauth2_scheme),
) -> User:
    """
    Validate the Bearer token first (raises 401 immediately if absent/invalid),
    then look up the user in the DB.  Keeping DB out of the dependency signature
    ensures FastAPI resolves token validation before opening a session — which
    also means unauthenticated requests never touch the DB.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.app_secret_key, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Token is valid — now open a session to fetch the user row
    from db.session import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise credentials_exception
    return user


async def get_current_superuser(user: User = Depends(get_current_user)) -> User:
    if not user.is_superuser:
        raise HTTPException(status_code=403, detail="Superuser access required")
    return user


# ─── Auth router ──────────────────────────────────────────────────────────────
router = APIRouter()


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class UserOut(BaseModel):
    id: str
    email: str
    full_name: Optional[str]
    is_active: bool
    is_superuser: bool


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


@router.post("/register", response_model=UserOut, status_code=201)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    """Register a new user account."""
    existing = await db.execute(select(User).where(User.email == user_data.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        full_name=user_data.full_name,
        is_active=True,
        is_superuser=False,
    )
    db.add(user)
    await db.commit()
    logger.info("User registered", email=user_data.email)

    return UserOut(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """Authenticate and return JWT access + refresh tokens."""
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Account is disabled")

    access_token = create_token(str(user.id), timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    refresh_token = create_token(str(user.id), timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))

    logger.info("User logged in", user_id=str(user.id))
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_token: str,
    db: AsyncSession = Depends(get_db),
):
    """Exchange a refresh token for a new access token."""
    try:
        payload = jwt.decode(refresh_token, settings.app_secret_key, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found")

    new_access = create_token(str(user.id), timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    new_refresh = create_token(str(user.id), timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))
    return TokenResponse(access_token=new_access, refresh_token=new_refresh)


@router.get("/me", response_model=UserOut)
async def get_me(user: User = Depends(get_current_user)):
    return UserOut(
        id=str(user.id),
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
    )
