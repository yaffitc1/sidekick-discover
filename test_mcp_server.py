#!/usr/bin/env python3
"""Test script to verify the MCP server is working correctly."""

import json
import subprocess
import sys

def test_mcp_server():
    """Test that the MCP server responds correctly."""
    print("Testing MCP server...")
    print("=" * 50)
    
    # Initialize message
    init_message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0"
            }
        }
    }
    
    print("\n1. Sending initialize message...")
    try:
        result = subprocess.run(
            ["python", "-m", "discovery.mcp"],
            input=json.dumps(init_message) + "\n",
            text=True,
            capture_output=True,
            timeout=5
        )
        
        if result.returncode != 0:
            print(f"✗ Server returned error code: {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
        
        if result.stdout:
            response = json.loads(result.stdout.strip())
            if response.get("result") and response["result"].get("serverInfo"):
                server_name = response["result"]["serverInfo"].get("name")
                print(f"✓ Server responded: {server_name}")
                print(f"  Response: {json.dumps(response, indent=2)}")
                return True
            else:
                print(f"✗ Unexpected response: {result.stdout}")
                return False
        else:
            print("✗ No response from server")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Server timed out (this might be normal if waiting for more input)")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ Failed to parse JSON response: {e}")
        if result.stdout:
            print(f"  Output: {result.stdout}")
        return False
    except Exception as e:
        print(f"✗ Error testing server: {e}")
        return False

def main():
    """Run the test."""
    success = test_mcp_server()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ MCP server is working correctly!")
        print("\nNote: When run directly, the server appears silent because")
        print("it waits for JSON-RPC messages via stdio. This is expected behavior.")
        print("\nTo use in Cursor, add it to your MCP configuration.")
        sys.exit(0)
    else:
        print("✗ MCP server test failed")
        print("\nTroubleshooting:")
        print("1. Ensure dependencies are installed: pip install -r requirements.txt")
        print("2. Check that python -m discovery.mcp runs without errors")
        print("3. Verify MCP SDK is installed correctly")
        sys.exit(1)

if __name__ == "__main__":
    main()

