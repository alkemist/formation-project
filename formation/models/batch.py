from typing import List

from sqlalchemy import Column, Integer
from sqlalchemy.orm import Mapped, relationship
import pandas as pd

from .base import Base


class Batch(Base):
    __tablename__ = "batches"

    configurations: Mapped[List["Configuration"]] = relationship(back_populates="batch", cascade="all, delete")
    layers: Mapped[List["Layer"]] = relationship(back_populates="batch", cascade="all, delete")
    points_count = Column(Integer)

    def get_layers(self) -> pd.DataFrame:
        return pd.DataFrame(
            [[p.id, p.layers_count] for p in self.layers],
            columns=["id", "level"],
        )

    def __repr__(self) -> str:
        return f"Lot {self.id!r} avec {self.points_count} points"