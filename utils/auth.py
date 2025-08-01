from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from config.settings import get_settings

settings = get_settings()
security = HTTPBearer()

async def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Verify Bearer token authentication
    """
    if credentials.scheme != "Bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme. Expected Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if credentials.credentials != settings.bearer_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return credentials.credentials