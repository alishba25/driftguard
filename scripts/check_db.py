"""Check that Python can reach the database in your .env file."""
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])
with engine.connect() as conn:
    version = conn.execute(text("SELECT version()")).scalar()
print("Connected!", version.split(",")[0])