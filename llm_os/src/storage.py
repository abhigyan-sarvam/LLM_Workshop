import json
from sqlite3 import OperationalError
from datetime import datetime
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, ConfigDict
from abc import ABC, abstractmethod
try:
    from sqlalchemy.dialects import postgresql, mysql, sqlite
    from sqlalchemy.engine import create_engine, Engine
    from sqlalchemy.engine.row import Row
    from sqlalchemy.inspection import inspect
    from sqlalchemy.orm import Session, sessionmaker
    from sqlalchemy.schema import MetaData, Table, Column
    from sqlalchemy.sql.expression import text, select
    from sqlalchemy.types import DateTime, String
except ImportError:
    raise ImportError("`sqlalchemy` not installed")

from src.utils import logger, current_datetime

class AssistantRun(BaseModel):
    """Assistant Run that is stored in the database"""

    # Assistant name
    name: Optional[str] = None
    # Run UUID
    run_id: str
    # Run name
    run_name: Optional[str] = None
    # ID of the user participating in this run
    user_id: Optional[str] = None
    # LLM data (name, model, etc.)
    llm: Optional[Dict[str, Any]] = None
    # Assistant Memory
    memory: Optional[Dict[str, Any]] = None
    # Metadata associated with this assistant
    assistant_data: Optional[Dict[str, Any]] = None
    # Metadata associated with this run
    run_data: Optional[Dict[str, Any]] = None
    # Metadata associated the user participating in this run
    user_data: Optional[Dict[str, Any]] = None
    # Metadata associated with the assistant tasks
    task_data: Optional[Dict[str, Any]] = None
    # The timestamp of when this run was created
    created_at: Optional[datetime] = None
    # The timestamp of when this run was last updated
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

    def serializable_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(exclude={"created_at", "updated_at"})
        _dict["created_at"] = self.created_at.isoformat() if self.created_at else None
        _dict["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
        return _dict

    def assistant_dict(self) -> Dict[str, Any]:
        _dict = self.model_dump(exclude={"created_at", "updated_at", "task_data"})
        _dict["created_at"] = self.created_at.isoformat() if self.created_at else None
        _dict["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
        return _dict

class AssistantStorage(ABC):
    @abstractmethod
    def create(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self, run_id: str) -> Optional[AssistantRun]:
        raise NotImplementedError

    @abstractmethod
    def get_all_run_ids(self, user_id: Optional[str] = None) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_all_runs(self, user_id: Optional[str] = None) -> List[AssistantRun]:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, row: AssistantRun) -> Optional[AssistantRun]:
        raise NotImplementedError

    @abstractmethod
    def delete(self) -> None:
        raise NotImplementedError

class PgAssistantStorage(AssistantStorage):
    def __init__(
        self,
        table_name: str,
        schema: Optional[str] = "ai",
        db_url: Optional[str] = None,
        db_engine: Optional[Engine] = None,
    ):
        """
        This class provides assistant storage using a postgres table.

        The following order is used to determine the database connection:
            1. Use the db_engine if provided
            2. Use the db_url

        :param table_name: The name of the table to store assistant runs.
        :param schema: The schema to store the table in.
        :param db_url: The database URL to connect to.
        :param db_engine: The database engine to use.
        """
        _engine: Optional[Engine] = db_engine
        if _engine is None and db_url is not None:
            _engine = create_engine(db_url)

        if _engine is None:
            raise ValueError("Must provide either db_url or db_engine")

        # Database attributes
        self.table_name: str = table_name
        self.schema: Optional[str] = schema
        self.db_url: Optional[str] = db_url
        self.db_engine: Engine = _engine
        self.metadata: MetaData = MetaData(schema=self.schema)

        # Database session
        self.Session: sessionmaker[Session] = sessionmaker(bind=self.db_engine)

        # Database table for storage
        self.table: Table = self.get_table()

    def get_table(self) -> Table:
        return Table(
            self.table_name,
            self.metadata,
            # Primary key for this run
            Column("run_id", String, primary_key=True),
            # Assistant name
            Column("name", String),
            # Run name
            Column("run_name", String),
            # ID of the user participating in this run
            Column("user_id", String),
            # -*- LLM data (name, model, etc.)
            Column("llm", postgresql.JSONB),
            # -*- Assistant memory
            Column("memory", postgresql.JSONB),
            # Metadata associated with this assistant
            Column("assistant_data", postgresql.JSONB),
            # Metadata associated with this run
            Column("run_data", postgresql.JSONB),
            # Metadata associated the user participating in this run
            Column("user_data", postgresql.JSONB),
            # Metadata associated with the assistant tasks
            Column("task_data", postgresql.JSONB),
            # The timestamp of when this run was created.
            Column("created_at", DateTime(timezone=True), server_default=text("now()")),
            # The timestamp of when this run was last updated.
            Column("updated_at", DateTime(timezone=True), onupdate=text("now()")),
            extend_existing=True,
        )

    def table_exists(self) -> bool:
        logger.debug(f"Checking if table exists: {self.table.name}")
        try:
            return inspect(self.db_engine).has_table(self.table.name, schema=self.schema)
        except Exception as e:
            logger.error(e)
            return False

    def create(self) -> None:
        if not self.table_exists():
            if self.schema is not None:
                with self.Session() as sess, sess.begin():
                    logger.debug(f"Creating schema: {self.schema}")
                    sess.execute(text(f"create schema if not exists {self.schema};"))
            logger.debug(f"Creating table: {self.table_name}")
            self.table.create(self.db_engine)

    def _read(self, session: Session, run_id: str) -> Optional[Row[Any]]:
        stmt = select(self.table).where(self.table.c.run_id == run_id)
        try:
            return session.execute(stmt).first()
        except Exception:
            # Create table if it does not exist
            self.create()
        return None

    def read(self, run_id: str) -> Optional[AssistantRun]:
        with self.Session() as sess, sess.begin():
            existing_row: Optional[Row[Any]] = self._read(session=sess, run_id=run_id)
            return AssistantRun.model_validate(existing_row) if existing_row is not None else None

    def get_all_run_ids(self, user_id: Optional[str] = None) -> List[str]:
        run_ids: List[str] = []
        try:
            with self.Session() as sess, sess.begin():
                # get all run_ids for this user
                stmt = select(self.table)
                if user_id is not None:
                    stmt = stmt.where(self.table.c.user_id == user_id)
                # order by created_at desc
                stmt = stmt.order_by(self.table.c.created_at.desc())
                # execute query
                rows = sess.execute(stmt).fetchall()
                for row in rows:
                    if row is not None and row.run_id is not None:
                        run_ids.append(row.run_id)
        except Exception:
            logger.debug(f"Table does not exist: {self.table.name}")
        return run_ids

    def get_all_runs(self, user_id: Optional[str] = None) -> List[AssistantRun]:
        runs: List[AssistantRun] = []
        try:
            with self.Session() as sess, sess.begin():
                # get all runs for this user
                stmt = select(self.table)
                if user_id is not None:
                    stmt = stmt.where(self.table.c.user_id == user_id)
                # order by created_at desc
                stmt = stmt.order_by(self.table.c.created_at.desc())
                # execute query
                rows = sess.execute(stmt).fetchall()
                for row in rows:
                    if row.run_id is not None:
                        runs.append(AssistantRun.model_validate(row))
        except Exception:
            logger.debug(f"Table does not exist: {self.table.name}")
        return runs

    def upsert(self, row: AssistantRun) -> Optional[AssistantRun]:
        """
        Create a new assistant run if it does not exist, otherwise update the existing assistant.
        """

        with self.Session() as sess, sess.begin():
            # Create an insert statement
            stmt = postgresql.insert(self.table).values(
                run_id=row.run_id,
                name=row.name,
                run_name=row.run_name,
                user_id=row.user_id,
                llm=row.llm,
                memory=row.memory,
                assistant_data=row.assistant_data,
                run_data=row.run_data,
                user_data=row.user_data,
                task_data=row.task_data,
            )

            # Define the upsert if the run_id already exists
            # See: https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#postgresql-insert-on-conflict
            stmt = stmt.on_conflict_do_update(
                index_elements=["run_id"],
                set_=dict(
                    name=row.name,
                    run_name=row.run_name,
                    user_id=row.user_id,
                    llm=row.llm,
                    memory=row.memory,
                    assistant_data=row.assistant_data,
                    run_data=row.run_data,
                    user_data=row.user_data,
                    task_data=row.task_data,
                ),  # The updated value for each column
            )

            try:
                sess.execute(stmt)
            except Exception:
                # Create table and try again
                self.create()
                sess.execute(stmt)
        return self.read(run_id=row.run_id)

    def delete(self) -> None:
        if self.table_exists():
            logger.debug(f"Deleting table: {self.table_name}")
            self.table.drop(self.db_engine)

class S2AssistantStorage(AssistantStorage):
    def __init__(
        self,
        table_name: str,
        schema: Optional[str] = "ai",
        db_url: Optional[str] = None,
        db_engine: Optional[Engine] = None,
    ):
        """
        This class provides assistant storage using a singlestore table.

        The following order is used to determine the database connection:
            1. Use the db_engine if provided
            2. Use the db_url

        :param table_name: The name of the table to store assistant runs.
        :param schema: The schema to store the table in.
        :param db_url: The database URL to connect to.
        :param db_engine: The database engine to use.
        """
        _engine: Optional[Engine] = db_engine
        if _engine is None and db_url is not None:
            _engine = create_engine(db_url, connect_args={"charset": "utf8mb4"})

        if _engine is None:
            raise ValueError("Must provide either db_url or db_engine")

        # Database attributes
        self.table_name: str = table_name
        self.schema: Optional[str] = schema
        self.db_url: Optional[str] = db_url
        self.db_engine: Engine = _engine
        self.metadata: MetaData = MetaData(schema=self.schema)

        # Database session
        self.Session: sessionmaker[Session] = sessionmaker(bind=self.db_engine)

        # Database table for storage
        self.table: Table = self.get_table()

    def get_table(self) -> Table:
        return Table(
            self.table_name,
            self.metadata,
            # Primary key for this run
            Column("run_id", mysql.TEXT, primary_key=True),
            # Assistant name
            Column("name", mysql.TEXT),
            # Run name
            Column("run_name", mysql.TEXT),
            # ID of the user participating in this run
            Column("user_id", mysql.TEXT),
            # -*- LLM data (name, model, etc.)
            Column("llm", mysql.JSON),
            # -*- Assistant memory
            Column("memory", mysql.JSON),
            # Metadata associated with this assistant
            Column("assistant_data", mysql.JSON),
            # Metadata associated with this run
            Column("run_data", mysql.JSON),
            # Metadata associated with the user participating in this run
            Column("user_data", mysql.JSON),
            # Metadata associated with the assistant tasks
            Column("task_data", mysql.JSON),
            # The timestamp of when this run was created.
            Column("created_at", DateTime(timezone=True), server_default=text("now()")),
            # The timestamp of when this run was last updated.
            Column("updated_at", DateTime(timezone=True), onupdate=text("now()")),
            extend_existing=True,
        )

    def table_exists(self) -> bool:
        logger.debug(f"Checking if table exists: {self.table.name}")
        try:
            return inspect(self.db_engine).has_table(self.table.name, schema=self.schema)
        except Exception as e:
            logger.error(e)
            return False

    def create(self) -> None:
        if not self.table_exists():
            logger.info(f"\nCreating table: {self.table_name}\n")
            self.table.create(self.db_engine)

    def _read(self, session: Session, run_id: str) -> Optional[Row[Any]]:
        stmt = select(self.table).where(self.table.c.run_id == run_id)
        try:
            return session.execute(stmt).first()
        except Exception as e:
            logger.debug(e)
            # Create table if it does not exist
            self.create()
        return None

    def read(self, run_id: str) -> Optional[AssistantRun]:
        with self.Session.begin() as sess:
            existing_row: Optional[Row[Any]] = self._read(session=sess, run_id=run_id)
            return AssistantRun.model_validate(existing_row) if existing_row is not None else None

    def get_all_run_ids(self, user_id: Optional[str] = None) -> List[str]:
        run_ids: List[str] = []
        try:
            with self.Session.begin() as sess:
                # get all run_ids for this user
                stmt = select(self.table.c.run_id)
                if user_id is not None:
                    stmt = stmt.where(self.table.c.user_id == user_id)
                # order by created_at desc
                stmt = stmt.order_by(self.table.c.created_at.desc())
                # execute query
                rows = sess.execute(stmt).fetchall()
                for row in rows:
                    if row is not None and row.run_id is not None:
                        run_ids.append(row.run_id)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
        return run_ids

    def get_all_runs(self, user_id: Optional[str] = None) -> List[AssistantRun]:
        runs: List[AssistantRun] = []
        try:
            with self.Session.begin() as sess:
                # get all runs for this user
                stmt = select(self.table)
                if user_id is not None:
                    stmt = stmt.where(self.table.c.user_id == user_id)
                # order by created_at desc
                stmt = stmt.order_by(self.table.c.created_at.desc())
                # execute query
                rows = sess.execute(stmt).fetchall()
                for row in rows:
                    if row.run_id is not None:
                        runs.append(AssistantRun.model_validate(row))
        except Exception:
            logger.debug(f"Table does not exist: {self.table.name}")
        return runs

    def upsert(self, row: AssistantRun) -> Optional[AssistantRun]:
        """
        Create a new assistant run if it does not exist, otherwise update the existing assistant.
        """

        with self.Session.begin() as sess:
            # Create an insert statement using SingleStore's ON DUPLICATE KEY UPDATE syntax
            upsert_sql = text(
                f"""
            INSERT INTO {self.schema}.{self.table_name}
            (run_id, name, run_name, user_id, llm, memory, assistant_data, run_data, user_data, task_data)
            VALUES
            (:run_id, :name, :run_name, :user_id, :llm, :memory, :assistant_data, :run_data, :user_data, :task_data)
            ON DUPLICATE KEY UPDATE
                name = VALUES(name),
                run_name = VALUES(run_name),
                user_id = VALUES(user_id),
                llm = VALUES(llm),
                memory = VALUES(memory),
                assistant_data = VALUES(assistant_data),
                run_data = VALUES(run_data),
                user_data = VALUES(user_data),
                task_data = VALUES(task_data);
            """
            )

            try:
                sess.execute(
                    upsert_sql,
                    {
                        "run_id": row.run_id,
                        "name": row.name,
                        "run_name": row.run_name,
                        "user_id": row.user_id,
                        "llm": json.dumps(row.llm, ensure_ascii=False) if row.llm is not None else None,
                        "memory": json.dumps(row.memory, ensure_ascii=False) if row.memory is not None else None,
                        "assistant_data": json.dumps(row.assistant_data, ensure_ascii=False)
                        if row.assistant_data is not None
                        else None,
                        "run_data": json.dumps(row.run_data, ensure_ascii=False) if row.run_data is not None else None,
                        "user_data": json.dumps(row.user_data, ensure_ascii=False)
                        if row.user_data is not None
                        else None,
                        "task_data": json.dumps(row.task_data, ensure_ascii=False)
                        if row.task_data is not None
                        else None,
                    },
                )
            except Exception:
                # Create table and try again
                self.create()
                sess.execute(
                    upsert_sql,
                    {
                        "run_id": row.run_id,
                        "name": row.name,
                        "run_name": row.run_name,
                        "user_id": row.user_id,
                        "llm": json.dumps(row.llm) if row.llm is not None else None,
                        "memory": json.dumps(row.memory, ensure_ascii=False) if row.memory is not None else None,
                        "assistant_data": json.dumps(row.assistant_data, ensure_ascii=False)
                        if row.assistant_data is not None
                        else None,
                        "run_data": json.dumps(row.run_data, ensure_ascii=False) if row.run_data is not None else None,
                        "user_data": json.dumps(row.user_data, ensure_ascii=False)
                        if row.user_data is not None
                        else None,
                        "task_data": json.dumps(row.task_data, ensure_ascii=False)
                        if row.task_data is not None
                        else None,
                    },
                )
        return self.read(run_id=row.run_id)

    def delete(self) -> None:
        if self.table_exists():
            logger.info(f"Deleting table: {self.table_name}")
            self.table.drop(self.db_engine)

class SqlAssistantStorage(AssistantStorage):
    def __init__(
        self,
        table_name: str,
        db_url: Optional[str] = None,
        db_file: Optional[str] = None,
        db_engine: Optional[Engine] = None,
    ):
        """
        This class provides assistant storage using a sqlite database.

        The following order is used to determine the database connection:
            1. Use the db_engine if provided
            2. Use the db_url
            3. Use the db_file
            4. Create a new in-memory database

        :param table_name: The name of the table to store assistant runs.
        :param db_url: The database URL to connect to.
        :param db_file: The database file to connect to.
        :param db_engine: The database engine to use.
        """
        _engine: Optional[Engine] = db_engine
        if _engine is None and db_url is not None:
            _engine = create_engine(db_url)
        elif _engine is None and db_file is not None:
            _engine = create_engine(f"sqlite:///{db_file}")
        else:
            _engine = create_engine("sqlite://")

        if _engine is None:
            raise ValueError("Must provide either db_url, db_file or db_engine")

        # Database attributes
        self.table_name: str = table_name
        self.db_url: Optional[str] = db_url
        self.db_engine: Engine = _engine
        self.metadata: MetaData = MetaData()

        # Database session
        self.Session: sessionmaker[Session] = sessionmaker(bind=self.db_engine)

        # Database table for storage
        self.table: Table = self.get_table()

    def get_table(self) -> Table:
        return Table(
            self.table_name,
            self.metadata,
            # Database ID/Primary key for this run
            Column("run_id", String, primary_key=True),
            # Assistant name
            Column("name", String),
            # Run name
            Column("run_name", String),
            # ID of the user participating in this run
            Column("user_id", String),
            # -*- LLM data (name, model, etc.)
            Column("llm", sqlite.JSON),
            # -*- Assistant memory
            Column("memory", sqlite.JSON),
            # Metadata associated with this assistant
            Column("assistant_data", sqlite.JSON),
            # Metadata associated with this run
            Column("run_data", sqlite.JSON),
            # Metadata associated the user participating in this run
            Column("user_data", sqlite.JSON),
            # Metadata associated with the assistant tasks
            Column("task_data", sqlite.JSON),
            # The timestamp of when this run was created.
            Column("created_at", sqlite.DATETIME, default=current_datetime()),
            # The timestamp of when this run was last updated.
            Column("updated_at", sqlite.DATETIME, onupdate=current_datetime()),
            extend_existing=True,
            sqlite_autoincrement=True,
        )

    def table_exists(self) -> bool:
        logger.debug(f"Checking if table exists: {self.table.name}")
        try:
            return inspect(self.db_engine).has_table(self.table.name)
        except Exception as e:
            logger.error(e)
            return False

    def create(self) -> None:
        if not self.table_exists():
            logger.debug(f"Creating table: {self.table.name}")
            self.table.create(self.db_engine)

    def _read(self, session: Session, run_id: str) -> Optional[Row[Any]]:
        stmt = select(self.table).where(self.table.c.run_id == run_id)
        try:
            return session.execute(stmt).first()
        except OperationalError:
            # Create table if it does not exist
            self.create()
        except Exception as e:
            logger.warning(e)
        return None

    def read(self, run_id: str) -> Optional[AssistantRun]:
        with self.Session() as sess:
            existing_row: Optional[Row[Any]] = self._read(session=sess, run_id=run_id)
            return AssistantRun.model_validate(existing_row) if existing_row is not None else None

    def get_all_run_ids(self, user_id: Optional[str] = None) -> List[str]:
        run_ids: List[str] = []
        try:
            with self.Session() as sess:
                # get all run_ids for this user
                stmt = select(self.table)
                if user_id is not None:
                    stmt = stmt.where(self.table.c.user_id == user_id)
                # order by created_at desc
                stmt = stmt.order_by(self.table.c.created_at.desc())
                # execute query
                rows = sess.execute(stmt).fetchall()
                for row in rows:
                    if row is not None and row.run_id is not None:
                        run_ids.append(row.run_id)
        except OperationalError:
            logger.debug(f"Table does not exist: {self.table.name}")
            pass
        return run_ids

    def get_all_runs(self, user_id: Optional[str] = None) -> List[AssistantRun]:
        conversations: List[AssistantRun] = []
        try:
            with self.Session() as sess:
                # get all runs for this user
                stmt = select(self.table)
                if user_id is not None:
                    stmt = stmt.where(self.table.c.user_id == user_id)
                # order by created_at desc
                stmt = stmt.order_by(self.table.c.created_at.desc())
                # execute query
                rows = sess.execute(stmt).fetchall()
                for row in rows:
                    if row.run_id is not None:
                        conversations.append(AssistantRun.model_validate(row))
        except OperationalError:
            logger.debug(f"Table does not exist: {self.table.name}")
            pass
        return conversations

    def upsert(self, row: AssistantRun) -> Optional[AssistantRun]:
        """
        Create a new assistant run if it does not exist, otherwise update the existing conversation.
        """
        with self.Session() as sess:
            # Create an insert statement
            stmt = sqlite.insert(self.table).values(
                run_id=row.run_id,
                name=row.name,
                run_name=row.run_name,
                user_id=row.user_id,
                llm=row.llm,
                memory=row.memory,
                assistant_data=row.assistant_data,
                run_data=row.run_data,
                user_data=row.user_data,
                task_data=row.task_data,
            )

            # Define the upsert if the run_id already exists
            # See: https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#insert-on-conflict-upsert
            stmt = stmt.on_conflict_do_update(
                index_elements=["run_id"],
                set_=dict(
                    name=row.name,
                    run_name=row.run_name,
                    user_id=row.user_id,
                    llm=row.llm,
                    memory=row.memory,
                    assistant_data=row.assistant_data,
                    run_data=row.run_data,
                    user_data=row.user_data,
                    task_data=row.task_data,
                ),  # The updated value for each column
            )

            try:
                sess.execute(stmt)
                sess.commit()  # Make sure to commit the changes to the database
                return self.read(run_id=row.run_id)
            except OperationalError as oe:
                logger.debug(f"OperationalError occurred: {oe}")
                self.create()  # This will only create the table if it doesn't exist
                try:
                    sess.execute(stmt)
                    sess.commit()
                    return self.read(run_id=row.run_id)
                except Exception as e:
                    logger.warning(f"Error during upsert: {e}")
                    sess.rollback()  # Rollback the session in case of any error
            except Exception as e:
                logger.warning(f"Error during upsert: {e}")
                sess.rollback()
        return None

    def delete(self) -> None:
        if self.table_exists():
            logger.debug(f"Deleting table: {self.table_name}")
            self.table.drop(self.db_engine)
