"""
    Modulo to handle database connection.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.config_db import DB_SERVER_NAME, DB_USER_NAME, DB_USER_PASSWORD, DB_NAME, DB_PORT

class DBConnectionHandler:
    """Sqlalchemy database connection"""

    def __init__(self) -> None:
        self.__connection_string = f"postgresql+psycopg2://{
            DB_USER_NAME
        }:{
            DB_USER_PASSWORD
        }@{
            DB_SERVER_NAME
        }:{
            DB_PORT
        }/{
            DB_NAME
        }"
        self.__engine = self.__create_database_engine()
        self.__session = None

    def __create_database_engine(self):
        engine = create_engine(self.__connection_string, pool_size= 5, max_overflow=0)
        return engine

    def get_engine(self):
        """Return database engine"""
        return self.__engine

    def __enter__(self):
        seession_make = sessionmaker(bind=self.__engine)
        self.__session = seession_make()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__session.close()

    def get_session(self):
        """Return database session"""
        return self.__session
