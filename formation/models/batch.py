from typing import List

from sqlalchemy.orm import Mapped, relationship
import pandas as pd

from .base import Base


class Batch(Base):
    __tablename__ = "batches"

    layers: Mapped[List["Layer"]] = relationship(back_populates="batch", cascade="all, delete")
    metrics: Mapped[List["Metric"]] = relationship(back_populates="batch", cascade="all, delete")

    def get_layers(self) -> pd.DataFrame:
        return pd.DataFrame(
            [[p.id, p.level] for p in self.layers],
            columns=["id", "level"],
        )

    def __repr__(self) -> str:
        return f"Lot {self.id!r}"