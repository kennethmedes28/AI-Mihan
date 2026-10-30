"""One database connection shared by the agents and the API, opened on first use."""

import threading

import psycopg

from src.services.fetch_postgre_database import get_connection


class SharedConnection:
    """One database connection for the agents, opened on first use and reopened if it dropped.

    The agents only read evacuation centers and write reports, so one
    connection is enough and cannot leak the way a new one per question would.
    Calling the object returns the open connection.
    """

    def __init__(self, open_connection=get_connection):
        self._open_connection = open_connection
        self._connection: psycopg.Connection | None = None
        self._lock = threading.Lock()

    def __call__(self) -> psycopg.Connection:
        with self._lock:
            if self._connection is None or self._connection.closed or self._connection.broken:
                self._connection = self._open_connection()
            return self._connection

    def close(self) -> None:
        with self._lock:
            if self._connection is not None and not self._connection.closed:
                self._connection.close()
            self._connection = None
