from typing import Optional, List

from sqlalchemy.orm import declarative_base, mapped_column, Mapped, relationship, backref
from sqlalchemy import Column, Integer, String, ForeignKey, JSON

from .base import Base


class Configuration(Base):
    __tablename__ = "configurations"

    batch_id: Mapped[int] = mapped_column(Integer, ForeignKey("batches.id", ondelete="CASCADE"))
    batch: Mapped["Batch"] = relationship(back_populates="configurations")

    layers: Mapped[List["Layer"]] = relationship(back_populates="configuration", cascade="all, delete")
    metrics: Mapped[List["Metric"]] = relationship(back_populates="configuration", cascade="all, delete")

    clustering_type = Column(String)
    clustering_length = Column(Integer)
    clustering_params: Mapped[Optional[JSON]] = mapped_column(type_=JSON)

    distance_min = Column(Integer)

    def __repr__(self) -> str:
        str_repr = f"{self.batch!r} - Configuration {self.clustering_type} de taille {self.clustering_length}"
        if self.clustering_params != {}:
            str_repr = f"{str_repr} et avec comme paramètres {self.clustering_params!r}"

        return str_repr