# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#
"""
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import threading
from functools import wraps
from types import FunctionType
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.pool import NullPool

from tvb.basic.profile import TvbProfile
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.storage.exceptions import NestedTransactionUnsupported, InvalidTransactionAccess


###
### INITIALIZATION AREA
###

LOGGER = get_logger(__name__)


def build_db_engine():
    if TvbProfile.current.db.SELECTED_DB == 'postgres':
        if TvbProfile.current.db.MAX_CONNECTIONS == 0:
            # Disable psycopg pooling if MAX_CONNECTIONS flag is set to 0. In this case we will use an external pooling tool.
            DB_ENGINE = create_engine(TvbProfile.current.db.DB_URL, poolclass=NullPool)
        else:
            ### Control the pool size for PostgreSQL, otherwise we might end with multiple
            ### concurrent Python processes failing because of too many opened connections.
            DB_ENGINE = create_engine(TvbProfile.current.db.DB_URL, pool_recycle=5, max_overflow=1,
                                      pool_size=TvbProfile.current.db.MAX_CONNECTIONS)
    else:
        ### SqlLite does not support pool-size
        DB_ENGINE = create_engine(TvbProfile.current.db.DB_URL, pool_recycle=5)

        def __have_journal_in_memory(con, con_record):
            con.execute("PRAGMA journal_mode = MEMORY")
            con.execute("PRAGMA synchronous = OFF")
            con.execute("PRAGMA temp_store = MEMORY")
            con.execute("PRAGMA cache_size = 500000")

        def __have_journal_WAL(con, con_record):
            con.execute("PRAGMA journal_mode=WAL")

        if getattr(TvbProfile.current, "TRADE_CRASH_SAFETY_FOR_SPEED", False):
            # use for speed, but without crash safety; use only in development
            LOGGER.warning("TRADE_CRASH_SAFETY_FOR_SPEED is on")
            event.listen(DB_ENGINE, 'connect', __have_journal_in_memory)
        else:
            event.listen(DB_ENGINE, 'connect', __have_journal_WAL)

    return DB_ENGINE


SA_SESSIONMAKER = sessionmaker(bind=build_db_engine(), expire_on_commit=False)

# expire_on_commit – Defaults to True. When True, all instances will be fully expired after each commit(),
#           so that all attribute/object access subsequent to a completed transaction will need to load
#           from the most recent database state.


def singleton(cls):
    """
    Class decorator that makes sure only one instance of that class is ever returned.
    """
    instances = {}


    def getinstance(*args, **kwargs):
        """
        Called when a new instance is about to be created.
        """
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]


    return getinstance



def MetaClassFactory(decorator_functions, new_attributes):
    """
    A meta-class factory that creates a meta-class which makes sure a list of decorators
    are applied to all it's classes and also adds a dictionary of attributes.
    
    :param decorator_functions: a list of functions. These will be applied as decorators to
        all methods from the class that uses the returned meta-class.
    :param new_attributes: a dictionary of attribute_name & attribute_value pairs that will
        be added to the class that uses the returned meta-class
    """

    class MetaClass(type):
        """
        New MetaClass.
        """

        def __new__(mcs, classname, bases, class_dict):
            """
            Called when a new class gets instantiated.
            """
            new_class_dict = {}
            for attr_name, attribute in class_dict.items():
                if (type(attribute) == FunctionType and not (attribute.__name__.startswith('__')
                                                             and attribute.__name__.endswith('__'))):
                    for function in decorator_functions:
                        attribute = function(attribute)

                new_class_dict[attr_name] = attribute
                new_class_dict.update(new_attributes)
            return type.__new__(mcs, classname, bases, new_class_dict)

    return MetaClass



class SessionsStack(object):
    """
    Helper class that holds a stack of SqlAlchemy's session object and a counter that
    keeps track of how many transactions are opened.
    """


    def __init__(self):
        """
        In the empty state just add a list that will hold all the sessions and 
        initialize the transactions counter to 0.
        """
        self.sessions_stack = []
        self.open_transactions = 0


    def close_session(self):
        """
        Method called by all '`add_session` decorated methods. First check if there
        are any changes that needed to be committed but weren't. Then either close the
        session if it's not part of a transaction, or just expunge all objects otherwise.
        """
        top_session = self.sessions_stack.pop()
        if top_session.dirty or top_session.deleted or top_session.new:
            top_session.commit()
        if self.open_transactions == 0:
            # We are not part of a transaction. Just close the session.
            top_session.close()
        else:
            # We are part of a transaction. Just expunge the objects, the transaction will handle the close.
            top_session.expunge_all()
        del top_session


    def open_session(self):
        """
        Create a new session. If we are part of a transaction we bind it to the parent
        session, otherwise just create a new session.
        """
        if self.open_transactions == 0:
            new_session = SA_SESSIONMAKER()
        else:
            new_session = SA_SESSIONMAKER(bind=self.sessions_stack[-1].connection())
        self.sessions_stack.append(new_session)


    @property
    def current_session(self):
        """
        Property just for ease of access. Current session will always be top of stack.
        """
        return self.sessions_stack[-1]


    def start_transaction(self):
        """
        Start a new transaction. If this is top level transaction just created new session.
        Otherwise depending if we support nested or not, either raise exception or create a session
        bound to parent one.
        """
        if self.open_transactions == 0:
            # New transaction, just create a new session.
            transaction = SA_SESSIONMAKER()
        else:
            # We are part of a nested transaction.
            if TvbProfile.current.db.ALLOW_NESTED_TRANSACTIONS:
                transaction = SA_SESSIONMAKER(bind=self.sessions_stack[-1].connection())
            else:
                raise NestedTransactionUnsupported("We do not support nested transaction in TVB.")
        self.sessions_stack.append(transaction)
        self.open_transactions += 1


    def rollback_transaction(self):
        """
        RollBack a transaction. 
        If we are part of nested transaction - rollback everything up to top parent transaction.
        """
        if not self.open_transactions:
            raise InvalidTransactionAccess("You are trying to close a transaction that was not started.")
        for transaction_idx in range(self.open_transactions):
            transaction = self.sessions_stack[-(1 + transaction_idx)]
            transaction.rollback()


    def close_transaction(self):
        """
        Close a transaction. Make sure to commit beforehand so all changes are written to database. Then
        depending on if we are top level or not either close or expunge session.
        """
        if not self.open_transactions:
            raise InvalidTransactionAccess("You are trying to close a transaction that was not started.")
        self.open_transactions -= 1
        top_transaction_session = self.sessions_stack.pop()
        top_transaction_session.commit()
        if self.open_transactions == 0:
            top_transaction_session.close()
        else:
            top_transaction_session.expunge_all()
        del top_transaction_session



@singleton
class SessionMaker(object):
    """
    This is our custom SessionMaker class, aggregating SessionsStack class.
    It has the purpose of obtaining a new SessionsStack for each thread.
    When calling self.session._something_ our mechanism comes in place and checks having a new stack for every threadID.
    """


    def __init__(self):
        """
        Initialize a dictionary with thread : session pairs to make sure we are thread-safe.
        """
        self.handled_sessions = {threading.current_thread(): SessionsStack()}


    def __getattr__(self, name):
        """
        __getattr__ is only called if `name` was not found in standard lookup (e.g. class or super-class attributes)
        In that case just delegate to the corresponding SQLAlchemy session.
        """
        current_thread = threading.current_thread()
        if current_thread not in self.handled_sessions:
            # This if first session for a new threads. Just create a new one.
            self.handled_sessions[current_thread] = SessionsStack()
        for thread in list(self.handled_sessions):
            # If thread finished we just delete entry to avoid dangling unused objects.
            if not thread.is_alive():
                try:
                    del self.handled_sessions[thread]
                except Exception:
                    ### Ignore this error because a concurrent thread might have removed this meanwhile.
                    pass
        delegate_method = getattr(self.handled_sessions[current_thread].current_session, name)
        return delegate_method


    def open_session(self):
        """
        Open a new session for the current thread.
        """
        current_thread = threading.current_thread()
        if current_thread not in self.handled_sessions:
            self.handled_sessions[current_thread] = SessionsStack()
        self.handled_sessions[current_thread].open_session()


    def close_session(self):
        """
        Close the session for the current thread.
        """
        current_thread = threading.current_thread()
        self.handled_sessions[current_thread].close_session()


    def rollback_transaction(self):
        """
        Rollback a transaction for the current thread.
        """
        current_thread = threading.current_thread()
        self.handled_sessions[current_thread].rollback_transaction()


    def start_transaction(self):
        """
        Start a new transaction for the current thread.
        """
        current_thread = threading.current_thread()
        if current_thread not in self.handled_sessions:
            self.handled_sessions[current_thread] = SessionsStack()
        self.handled_sessions[current_thread].start_transaction()


    def close_transaction(self):
        """
        Close a transaction for the current thread.
        """
        current_thread = threading.current_thread()
        self.handled_sessions[current_thread].close_transaction()


###
### PUBLIC EXPOSED ENTITIES FOR USAGE: 2 decorators and 1 meta-class-factory.
### 

def transactional(func):
    """
    Decorator that makes sure that all DAO calls that will result from the decorated
    method will be encapsulated in a transaction that will be rolled back if any 
    unexpected exceptions appear.
    This is intended to be used on service layer methods.
    """

    @wraps(func)
    def dec(*args, **kwargs):
        """
        Decorate methods.
        """
        session_maker = SessionMaker()
        session_maker.start_transaction()
        try:
            result = func(*args, **kwargs)
        except Exception:
            session_maker.rollback_transaction()
            raise
        finally:
            session_maker.close_transaction()
        return result


    return dec



def add_session(func):
    """
    Decorator that handles session related precautions before/after method call.
    Before each new method a session is created that will later on be closed/rolled back as necessary.
    This is intended to be used on all DAO methods
    """

    @wraps(func)
    def dec(*args, **kwargs):
        """
        Decorate by populating self.session
        """
        args[0].session.open_session()

        try:
            result = func(*args, **kwargs)

        except NoResultFound:
            raise

        except Exception:
            LOGGER.exception("Could not commit session...")
            args[0].session.rollback()
            raise

        finally:
            args[0].session.close_session()

        return result


    return dec


### All Classes having this meta-class will have automatically populated:
### - Attribute self.session
### - Annotation add_session over every method in that class.
SESSION_META_CLASS = MetaClassFactory([add_session], {'session': SessionMaker()})



