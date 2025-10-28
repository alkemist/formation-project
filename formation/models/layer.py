from typing import List

from sqlalchemy.orm import declarative_base, mapped_column, Mapped, relationship, backref
from sqlalchemy import Column, Integer, String, ForeignKey, select, or_
import pandas as pd

from .base import Base


class Layer(Base):
    __tablename__ = "layers"

    batch_id: Mapped[int] = mapped_column(Integer, ForeignKey("batches.id", ondelete="CASCADE"))
    batch: Mapped["Batch"] = relationship(back_populates="layers")

    configuration_id: Mapped[int] = mapped_column(Integer, ForeignKey("configurations.id", ondelete="CASCADE"), nullable=True)
    configuration: Mapped["Configuration"] = relationship(back_populates="layers")
    
    points: Mapped[List["Point"]] = relationship(back_populates="layer", cascade="all, delete")

    level = Column(Integer, index=True)

    def get_points(self) -> pd.DataFrame:
        return pd.DataFrame(
            [[p.id, p.code, p.lat, p.lng, p.x, p.y, p.cluster, p.code_next, p.lat_next, p.lng_next] for p in self.points],
            columns=["id", "code", "lat", "lng", "x", "y", "cluster", 'code_next', 'lat_next', 'lng_next'],
        )

    def __repr__(self) -> str:
        return f"{(self.batch if self.configuration is None else self.configuration)!r} - Couche {self.level!r}"