from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Dict, Optional, List
import pandas as pd
import snowflake.connector
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from dotenv import load_dotenv
from .base import SourceConnector

# Load environment variables from .env file
load_dotenv()


class SnowflakeConnector:
    """Connector for Snowflake tables.
    
    Supports both password and key pair authentication (with MFA).
    
    Reads connection parameters from environment variables:
    - SNOWFLAKE_ACCOUNT (required)
    - SNOWFLAKE_USER (required)
    - SNOWFLAKE_WAREHOUSE (required)
    - SNOWFLAKE_ROLE (optional)
    
    For password authentication:
    - SNOWFLAKE_PASSWORD (required if no private key)
    
    For key pair authentication (MFA):
    - SNOWFLAKE_PRIVATE_KEY_PATH (path to private key file) OR
    - SNOWFLAKE_PRIVATE_KEY (private key content as string)
    - SNOWFLAKE_PRIVATE_KEY_PASSPHRASE (optional, if private key is encrypted)
    """
    
    def __init__(self, database: str, schema: str, tables: Optional[List[str]] = None):
        """Initialize Snowflake connector.
        
        Args:
            database: Database name
            schema: Schema name
            tables: Optional list of table names to limit scope (None = all tables)
        """
        self.database = database.upper()
        self.schema = schema.upper()
        self.tables = [t.upper() for t in tables] if tables else None
        
        # Get connection params from env
        self.account = os.getenv("SNOWFLAKE_ACCOUNT")
        self.user = os.getenv("SNOWFLAKE_USER")
        self.password = os.getenv("SNOWFLAKE_PASSWORD")
        self.role = os.getenv("SNOWFLAKE_ROLE")
        self.warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
        
        # Key pair authentication
        self.private_key_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
        self.private_key = os.getenv("SNOWFLAKE_PRIVATE_KEY")
        self.private_key_passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
        
        # Validate required params
        if not all([self.account, self.user, self.warehouse]):
            raise ValueError(
                "Missing required Snowflake env vars: SNOWFLAKE_ACCOUNT, "
                "SNOWFLAKE_USER, SNOWFLAKE_WAREHOUSE"
            )
        
        # Validate authentication method
        has_password = bool(self.password)
        has_private_key = bool(self.private_key_path or self.private_key)
        
        if not has_password and not has_private_key:
            raise ValueError(
                "Must provide either SNOWFLAKE_PASSWORD or "
                "SNOWFLAKE_PRIVATE_KEY_PATH/SNOWFLAKE_PRIVATE_KEY for authentication"
            )
        
        self._conn: Optional[snowflake.connector.SnowflakeConnection] = None
    
    def _load_private_key(self) -> Optional[bytes]:
        """Load private key from file or environment variable.
        
        Returns:
            Private key bytes, or None if not available
        """
        if self.private_key_path:
            # Load from file
            key_path = Path(self.private_key_path).expanduser().resolve()
            if not key_path.exists():
                raise FileNotFoundError(f"Private key file not found: {key_path}")
            
            with open(key_path, "rb") as key_file:
                private_key_data = key_file.read()
        elif self.private_key:
            # Load from environment variable (may be base64 encoded or raw PEM)
            private_key_data = self.private_key.encode("utf-8")
        else:
            return None
        
        # Parse private key
        passphrase = self.private_key_passphrase.encode("utf-8") if self.private_key_passphrase else None
        
        try:
            # Try PEM format first
            p_key = serialization.load_pem_private_key(
                private_key_data,
                password=passphrase,
                backend=default_backend()
            )
        except ValueError:
            # Try DER format if PEM fails
            try:
                p_key = serialization.load_der_private_key(
                    private_key_data,
                    password=passphrase,
                    backend=default_backend()
                )
            except ValueError:
                # If it's base64 encoded, try decoding
                import base64
                try:
                    decoded_key = base64.b64decode(private_key_data)
                    p_key = serialization.load_pem_private_key(
                        decoded_key,
                        password=passphrase,
                        backend=default_backend()
                    )
                except Exception:
                    raise ValueError("Failed to parse private key. Ensure it's in PEM or DER format.")
        
        # Serialize to PKCS8 DER format for Snowflake (Snowflake connector expects DER format)
        der = p_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        return der
    
    def _get_connection(self) -> snowflake.connector.SnowflakeConnection:
        """Get or create a Snowflake connection.
        
        Uses key pair authentication if available, otherwise falls back to password.
        """
        if self._conn is None:
            # Build connection parameters
            connect_params = {
                "account": self.account,
                "user": self.user,
                "warehouse": self.warehouse,
                "database": self.database,
                "schema": self.schema,
            }
            
            # Add role if specified
            if self.role:
                connect_params["role"] = self.role
            
            # Use key pair authentication if available, otherwise password
            if self.private_key_path or self.private_key:
                private_key_pem = self._load_private_key()
                connect_params["private_key"] = private_key_pem
            else:
                connect_params["password"] = self.password
            
            self._conn = snowflake.connector.connect(**connect_params)
        
        return self._conn
    
    def list_tables(self) -> Iterable[str]:
        """List available tables in the schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if self.tables:
            # Return only specified tables that exist
            existing = []
            for table in self.tables:
                try:
                    cursor.execute(f"SELECT 1 FROM {self.database}.{self.schema}.{table} LIMIT 1")
                    existing.append(table)
                except Exception:
                    continue
            return existing
        
        # Query INFORMATION_SCHEMA for all tables
        cursor.execute(
            f"""
            SELECT TABLE_NAME 
            FROM {self.database}.INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = '{self.schema}' 
            AND TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
            """
        )
        return [row[0] for row in cursor.fetchall()]
    
    def get_schema(self, table: str) -> Dict[str, str]:
        """Get column schema for a table."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            f"""
            SELECT COLUMN_NAME, DATA_TYPE 
            FROM {self.database}.INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = '{self.schema}' 
            AND TABLE_NAME = '{table.upper()}'
            ORDER BY ORDINAL_POSITION
            """
        )
        return {row[0]: row[1] for row in cursor.fetchall()}
    
    def get_table_stats(self, table: str) -> Dict[str, Any]:
        """Get table statistics from INFORMATION_SCHEMA."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                f"""
                SELECT ROW_COUNT, BYTES, LAST_ALTERED
                FROM {self.database}.INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = '{self.schema}' 
                AND TABLE_NAME = '{table.upper()}'
                """
            )
            result = cursor.fetchone()
            if result:
                return {
                    "row_count": result[0],
                    "bytes": result[1],
                    "last_altered": result[2],
                }
        except Exception:
            pass
        
        return {}
    
    def get_column_cardinality(self, table: str, column: str) -> Optional[int]:
        """Get approximate cardinality for a column."""
        conn = self._get_connection()
        cursor = conn.cursor()
        full_table = f"{self.database}.{self.schema}.{table.upper()}"
        
        try:
            cursor.execute(
                f"SELECT APPROX_COUNT_DISTINCT({column}) FROM {full_table}"
            )
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception:
            return None
    
    def sample(
        self,
        table: str,
        limit: int,
        method: str = "random",
        stratify_by: Optional[str] = None,
    ) -> pd.DataFrame:
        """Sample rows from a Snowflake table."""
        conn = self._get_connection()
        
        full_table_name = f"{self.database}.{self.schema}.{table.upper()}"
        
        if method == "head":
            query = f"SELECT * FROM {full_table_name} LIMIT {limit}"
        elif method == "random":
            # Use TABLESAMPLE for random sampling
            query = f"SELECT * FROM {full_table_name} TABLESAMPLE SYSTEM ({min(100, limit * 100 // 1000)}%) LIMIT {limit}"
        else:
            # Fallback to head for unsupported methods
            query = f"SELECT * FROM {full_table_name} LIMIT {limit}"
        
        return pd.read_sql(query, conn)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._conn:
            self._conn.close()
            self._conn = None

