from typing import List

from sqlalchemy.orm import declarative_base, mapped_column, Mapped, relationship, backref
from sqlalchemy import Column, Integer, String, ForeignKey
import pandas as pd

from .base import Base


class Layer(Base):
    __tablename__ = "layers"

    batch_id: Mapped[int] = mapped_column(Integer, ForeignKey("batches.id", ondelete="CASCADE"))
    batch: Mapped["Batch"] = relationship(back_populates="layers")
    points: Mapped[List["Point"]] = relationship(back_populates="layer", cascade="all, delete")

    level = Column(Integer, index=True)

    def get_points(self) -> pd.DataFrame:
        return pd.DataFrame(
            [[p.id, p.code, p.lat, p.lng, p.x, p.y] for p in self.points],
            columns=["id", "code", "lat", "lng", "x", "y"],
        )

    def __repr__(self) -> str:
        return f"{self.batch!r} - Couche {self.id!r}"