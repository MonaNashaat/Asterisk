from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from builtins import *

import os
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Sets connection string
Asterisk_conn_string = os.environ['AsteriskDB'] if 'AsteriskDB' in os.environ and os.environ['AsteriskDB'] != '' \
    else 'sqlite:///' + os.getcwd() + os.sep + 'Asterisk.db'


# Sets global variable indicating whether we are using Postgres
Asterisk_postgres = Asterisk_conn_string.startswith('postgres')


# Automatically turns on foreign key enforcement for SQLite
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if Asterisk_conn_string.startswith('sqlite'):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# Defines procedure for setting up a sessionmaker
def new_sessionmaker():
    
    # Turning on autocommit for Postgres, see http://oddbird.net/2014/06/14/sqlalchemy-postgres-autocommit/
    # Otherwise any e.g. query starts a transaction, locking tables... very bad for e.g. multiple notebooks
    # open, multiple processes, etc.
    if Asterisk_postgres:
        Asterisk_engine = create_engine(Asterisk_conn_string, isolation_level="AUTOCOMMIT")
    else:
        Asterisk_engine = create_engine(Asterisk_conn_string)

    # New sessionmaker
    AsteriskSession = sessionmaker(bind=Asterisk_engine)
    return AsteriskSession


# We initialize the engine within the models module because models' schema can depend on
# which data types are supported by the engine
AsteriskSession = new_sessionmaker()
Asterisk_engine = AsteriskSession.kw['bind']

AsteriskBase = declarative_base(name='AsteriskBase', cls=object)
