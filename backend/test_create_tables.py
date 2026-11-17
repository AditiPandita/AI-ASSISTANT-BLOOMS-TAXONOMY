
from app.database.connection import Base, engine

# Import models so SQLAlchemy registers all tables
from app.database import models

try:
    Base.metadata.create_all(bind=engine)

    print("Database tables created successfully!")

    for table_name in Base.metadata.tables:
        print(f"Created table: {table_name}")

except Exception as error:
    print("Table creation failed!")
    print(error)
