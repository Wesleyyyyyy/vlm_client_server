from abc import ABCMeta, abstractmethod
from typing import Optional
import keyring


class BaseTokenSaver(metaclass=ABCMeta):
    @abstractmethod
    def save_token(self, access_token, refresh_token):
        pass

    @abstractmethod
    def read_access_token(self):
        pass

    @abstractmethod
    def read_refresh_token(self):
        pass

    @abstractmethod
    def delete_entry(self):
        pass


class BaseEntrySaver(BaseTokenSaver, metaclass=ABCMeta):
    def __init__(self, endpoint_url: str):
        self._host_name = endpoint_url

    def save_token(self, access_token: str, refresh_token: str) -> None:
        self._save(self._access_host_name, access_token)
        self._save(self._refresh_host_name, refresh_token)

    def read_access_token(self) -> Optional[str]:
        return self._get(self._access_host_name)

    def read_refresh_token(self) -> Optional[str]:
        return self._get(self._refresh_host_name)

    def delete_entry(self) -> None:
        self._delete(self._refresh_host_name)
        self._delete(self._access_host_name)

    @abstractmethod
    def _save(self, machine, token):
        pass

    @abstractmethod
    def _get(self, machine):
        pass

    @abstractmethod
    def _delete(self, machine):
        pass

    @property
    def _access_host_name(self) -> str:
        return f"{self._host_name}_access"

    @property
    def _refresh_host_name(self) -> str:
        return f"{self._host_name}_refresh"


class KeyringTokenSaver(BaseEntrySaver):
    def _save(self, machine, token):
        keyring.set_password(machine, "apitoken", token)

    def _get(self, machine):
        return keyring.get_password(machine, "apitoken")

    def _delete(self, machine):
        keyring.delete_password(machine, "apitoken")

