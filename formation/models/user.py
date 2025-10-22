from sqlalchemy.orm import declarative_base, mapped_column, Mapped
from sqlalchemy import Column, Integer, String

from .base import Base


class User(Base):
    __tablename__ = "users"

    login = Column(String, index=True)
    password = Column(String)

    def __repr__(self) -> str:
        return f"{self.login!r}"