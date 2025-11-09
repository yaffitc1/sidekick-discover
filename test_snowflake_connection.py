#!/usr/bin/env python3
"""Test Snowflake connection and display connection info."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from discovery.connectors.snowflake import SnowflakeConnector

# Try to find .env file in common locations
env_paths = [
    Path.cwd() / ".env",  # Current directory
    Path(__file__).parent / ".env",  # Script directory
    Path.home() / ".env",  # Home directory
]

env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        print(f"📁 Loading .env from: {env_path}")
        load_dotenv(env_path)
        env_loaded = True
        break

# Fallback to default load_dotenv() behavior (searches current dir and parents)
if not env_loaded:
    print("📁 Searching for .env file...")
    load_dotenv()

def test_connection():
    """Test Snowflake connection with a simple query."""
    print("Testing Snowflake connection...")
    print("-" * 50)
    
    # Check required environment variables
    required_vars = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_WAREHOUSE"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("\nPlease set these in your .env file or environment:")
        print("  - SNOWFLAKE_ACCOUNT")
        print("  - SNOWFLAKE_USER")
        print("  - SNOWFLAKE_WAREHOUSE")
        print("  - SNOWFLAKE_PASSWORD (or SNOWFLAKE_PRIVATE_KEY_PATH)")
        print("  - SNOWFLAKE_ROLE (optional)")
        return False
    
    # Check authentication method
    has_password = bool(os.getenv("SNOWFLAKE_PASSWORD"))
    has_private_key = bool(os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH") or os.getenv("SNOWFLAKE_PRIVATE_KEY"))
    
    if not has_password and not has_private_key:
        print("❌ Missing authentication credentials!")
        print("\nPlease provide one of:")
        print("  - SNOWFLAKE_PASSWORD (for password authentication)")
        print("  - SNOWFLAKE_PRIVATE_KEY_PATH (for key pair authentication)")
        print("  - SNOWFLAKE_PRIVATE_KEY (for key pair authentication)")
        return False
    
    # Display connection info (without password)
    print("Connection parameters:")
    print(f"  Account: {os.getenv('SNOWFLAKE_ACCOUNT')}")
    print(f"  User: {os.getenv('SNOWFLAKE_USER')}")
    print(f"  Warehouse: {os.getenv('SNOWFLAKE_WAREHOUSE')}")
    print(f"  Role: {os.getenv('SNOWFLAKE_ROLE', 'default')}")
    
    auth_method = "Private Key" if has_private_key else "Password"
    print(f"  Authentication: {auth_method}")
    print()
    
    # Get database and schema from args or use defaults
    database = sys.argv[1] if len(sys.argv) > 1 else os.getenv("SNOWFLAKE_DATABASE", "SNOWFLAKE_SAMPLE_DATA")
    schema = sys.argv[2] if len(sys.argv) > 2 else os.getenv("SNOWFLAKE_SCHEMA", "TPCH_SF1")
    
    print(f"Testing connection to: {database}.{schema}")
    print("-" * 50)
    
    try:
        # Create connector
        connector = SnowflakeConnector(database=database, schema=schema)
        print("✅ Connector initialized successfully")
        
        # Test connection
        conn = connector._get_connection()
        print("✅ Connection established successfully")
        
        # Test query - get current database/schema
        cursor = conn.cursor()
        cursor.execute("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_WAREHOUSE(), CURRENT_ROLE()")
        result = cursor.fetchone()
        
        if result:
            print("\n✅ Connection verified!")
            print(f"  Current Database: {result[0]}")
            print(f"  Current Schema: {result[1]}")
            print(f"  Current Warehouse: {result[2]}")
            print(f"  Current Role: {result[3]}")
        
        # Test listing tables
        print("\n📋 Listing tables...")
        tables = list(connector.list_tables())
        if tables:
            print(f"✅ Found {len(tables)} table(s):")
            for table in tables[:10]:  # Show first 10
                print(f"  - {table}")
            if len(tables) > 10:
                print(f"  ... and {len(tables) - 10} more")
        else:
            print("⚠️  No tables found in this schema")
        
        # Test getting schema for first table if available
        if tables:
            first_table = tables[0]
            print(f"\n📊 Testing schema retrieval for table: {first_table}")
            schema_info = connector.get_schema(first_table)
            if schema_info:
                print(f"✅ Retrieved schema with {len(schema_info)} columns:")
                for col_name, col_type in list(schema_info.items())[:5]:
                    print(f"  - {col_name}: {col_type}")
                if len(schema_info) > 5:
                    print(f"  ... and {len(schema_info) - 5} more columns")
        
        # Close connection
        conn.close()
        print("\n✅ Connection test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Connection test failed!")
        print(f"Error: {type(e).__name__}: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
