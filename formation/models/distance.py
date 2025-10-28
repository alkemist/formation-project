from sqlalchemy.orm import declarative_base, mapped_column, Mapped, relationship, backref
from sqlalchemy import Column, Integer, String, ForeignKey, Float, UniqueConstraint

from .base import Base


class Distance(Base):
    __tablename__ = "distances"

    lat_in = Column(Float, index=True)
    lng_in = Column(Float, index=True)
    lat_out = Column(Float, index=True)
    lng_out = Column(Float, index=True)
    distance = Column(Float)

    __table_args__ = (
        UniqueConstraint('lat_in', 'lng_in', 'lat_out', 'lng_out', name='_unique_distance', sqlite_on_conflict="IGNORE"),
    )

    def __repr__(self) -> str:
        return f"{self.id!r}"