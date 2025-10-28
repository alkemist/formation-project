from sqlalchemy.orm import declarative_base, mapped_column, Mapped, relationship, backref
from sqlalchemy import Column, Integer, String, ForeignKey, Float

from .base import Base


class Point(Base):
    __tablename__ = "points"

    layer_id: Mapped[int] = mapped_column(Integer, ForeignKey("layers.id", ondelete="CASCADE"))
    layer: Mapped["Layer"] = relationship(back_populates="points")

    code = Column(String, index=True)
    cluster = Column(String)
    lat = Column(Float)
    lng = Column(Float)
    x = Column(Integer)
    y = Column(Integer)

    code_next = Column(String, nullable=True)
    lat_next = Column(Float, nullable=True)
    lng_next = Column(Float, nullable=True)

    distance = Column(Integer, nullable=True)

    def __repr__(self) -> str:
        return f"{self.id!r}"