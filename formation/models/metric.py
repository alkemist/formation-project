from typing import Optional

from sqlalchemy.orm import declarative_base, mapped_column, Mapped, relationship, backref
from sqlalchemy import Column, Integer, String, ForeignKey, JSON

from .base import Base


class Metric(Base):
    __tablename__ = "metrics"

    batch_id: Mapped[int] = mapped_column(ForeignKey("batches.id"))
    batch: Mapped["Batch"] = relationship(back_populates="metrics")

    code = Column(String, index=True)
    value = Column(String)

    params: Mapped[Optional[JSON]] = mapped_column(type_=JSON)

    def __repr__(self) -> str:
        return f"{self.id!r}"