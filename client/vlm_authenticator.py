import datetime
import requests
from fastapi import FastAPI, HTTPException
from jose import ExpiredSignatureError
from jose import jwt as jwt
from yarl import URL
from token_saver import BaseTokenSaver, KeyringTokenSaver
import logging

EDNA_API_URL = 'http://127.0.0.1:8000'
# EDNA_API_URL = os.environ.get("EDNA_API_URL") or "https://api.edna.cps.cit.tum.de"
login_url = 'http://127.0.0.1:8000/token'
_logger = logging.getLogger(__name__)

# User credentials
username = "edgar"
password = "password"


class EdnaAuthenticator:
    def __init__(self, endpoint_url: str, token_saver: BaseTokenSaver):
        self._endpoint_url = URL(endpoint_url)
        self._token_saver = token_saver

    def login(self):
        """Perform authorization and save access token for later use."""
        self.get_token()

    def logout(self) -> None:
        """Revoke refresh token and drop access token."""
        # Check if logged in
        if self._token_saver.read_access_token() is None:
            return
        # Delete saved tokens
        self._token_saver.delete_entry()

    def get_token(self):
        access_token = self._token_saver.read_access_token()
        if access_token is None or not self._is_token_valid(access_token):
            if access_token is None:
                access_token = self._login()

            self._token_saver.save_token(access_token, "refresh_token")

            return access_token

    def _login(self):
        try:
            # Send POST request to retrieve token
            response = requests.post(login_url, json={"username": username, "password": password})
            response.raise_for_status()  # Raise HTTPError for bad responses
            # Attempt to parse JSON response
            token = response.json().get('token')
            if not token:
                raise HTTPException(status_code=401, detail="Failed to retrieve token")
            print(f"Successful to retrieve token")
            print(f"Local token: {token}")
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Request exception occurred: {e}")
        return token

    def _is_token_valid(self, token):
        if token is None:
            return False
        try:
            self._get_jwt_expiration_time(token)
        except ExpiredSignatureError:
            _logger.debug("Token expired!")
            return False
        return True

    def _get_jwt_expiration_time(self, token):
        token_data = jwt.decode(token, "", options={"verify_signature": False})
        exp_date = datetime.datetime.fromtimestamp(token_data["exp"])
        _logger.debug("Token expires at %s", str(exp_date))
        return exp_date


def get_authenticator():
    authenticator = EdnaAuthenticator(
        endpoint_url=EDNA_API_URL, token_saver=KeyringTokenSaver(EDNA_API_URL),
    )
    return authenticator
