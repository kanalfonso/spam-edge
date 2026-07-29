"""One-time run that generates the chathistory_db on Postgres"""

# pre-req: must run command `CREATE DATABASE chathistory_db;` on pgadmin4 before executing code

from langgraph.checkpoint.postgres import PostgresSaver
from sqlalchemy import URL

url_object = URL.create(
    "postgresql",
    username="postgres",
    password="postgres",
    host="localhost",
    database="chathistory_db",
    port=5432
)

DB_URI = url_object.render_as_string(hide_password=False)


with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # Creates necessary internal checkpoint tables if they don't exist
    checkpointer.setup()
    print("PostgresSaver checkpointer tables initialized successfully!")
