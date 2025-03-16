from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
import datetime
import jwt


auth_router = APIRouter()

JWT_SECRET = 'secret_key'
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRATION_MINUTES = 120

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class LoginModel(BaseModel):
    username: str
    password: str


@auth_router.post("/token")
async def login(request: LoginModel):
    username = request.username
    password = request.password

    if username == "chris" and password == "password":
        logger.info("Login successful for user: %s", request.username)

        payload = {
            "user": "user",
            "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=TOKEN_EXPIRATION_MINUTES),
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        logger.debug("Token generated for user %s: %s", request.username, token)

        return {'token': token}
    else:
        logger.warning("Login failed for user: %s", request.username)
        raise HTTPException(status_code=401, detail="Invalid username or password")
