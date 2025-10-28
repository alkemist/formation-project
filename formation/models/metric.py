from typing import Optional

from sqlalchemy.orm import declarative_base, mapped_column, Mapped, relationship, backref
from sqlalchemy import Column, Integer, String, ForeignKey, JSON

from .base import Base


class Metric(Base):
    __tablename__ = "metrics"

    configuration_id: Mapped[int] = mapped_column(Integer, ForeignKey("configurations.id", ondelete="CASCADE"))
    configuration: Mapped["Configuration"] = relationship(back_populates="metrics")

    code = Column(String, index=True)
    value = Column(String)

    def __repr__(self) -> str:
        return f"{self.id!r}"