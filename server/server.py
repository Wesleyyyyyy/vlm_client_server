import os
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi import FastAPI
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import jose.jwt as jwt
import datetime
import logging

from server.api.endpoints import auth_router
from server.api.endpoints import data_router


app = FastAPI()

# Include endpoints
app.include_router(auth_router)
app.include_router(data_router)
