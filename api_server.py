#!/usr/bin/env python
"""
Entry point script to start the BCEA Classification API server.
"""
import os
import sys
from api.main import start

if __name__ == "__main__":
    # Start the API server
    start()