#!/usr/bin/env python3
"""
Helper script for managing Streamlit secrets.

Streamlit Cloud does NOT have a public API for updating secrets programmatically.
This script helps with:
1. Creating/updating local secrets for development
2. Validating your DATABASE_URL format
3. Printing instructions for Streamlit Cloud updates

Usage:
    python scripts/update_secrets.py --validate "postgresql://..."
    python scripts/update_secrets.py --create-local
    python scripts/update_secrets.py --show-instructions
"""

import argparse
import os
import sys
from urllib.parse import urlparse
from pathlib import Path


def validate_database_url(url: str) -> tuple[bool, str]:
    """Validate a PostgreSQL DATABASE_URL format."""
    try:
        parsed = urlparse(url)

        errors = []
        if parsed.scheme not in ('postgresql', 'postgres'):
            errors.append(f"Scheme should be 'postgresql' or 'postgres', got '{parsed.scheme}'")

        if not parsed.username:
            errors.append("Missing username")

        if not parsed.password:
            errors.append("Missing password")

        if not parsed.hostname:
            errors.append("Missing hostname")

        if not parsed.port:
            errors.append("Missing port number")

        if not parsed.path or parsed.path == '/':
            errors.append("Missing database name")

        if errors:
            return False, "Validation errors:\n  - " + "\n  - ".join(errors)

        # Mask password for display
        masked_url = f"{parsed.scheme}://{parsed.username}:****@{parsed.hostname}:{parsed.port}{parsed.path}"
        return True, f"Valid DATABASE_URL format:\n  {masked_url}"

    except Exception as e:
        return False, f"Failed to parse URL: {e}"


def create_local_secrets(database_url: str = None, admin_password: str = None):
    """Create or update local .streamlit/secrets.toml file."""
    secrets_dir = Path(__file__).parent.parent / '.streamlit'
    secrets_file = secrets_dir / 'secrets.toml'

    secrets_dir.mkdir(exist_ok=True)

    content_lines = ['# Streamlit Secrets - DO NOT COMMIT THIS FILE!', '']

    if database_url:
        content_lines.append(f'DATABASE_URL = "{database_url}"')
    else:
        content_lines.append('# DATABASE_URL = "postgresql://postgres:password@host:port/database"')

    content_lines.append('')

    if admin_password:
        content_lines.append(f'ADMIN_PASSWORD = "{admin_password}"')
    else:
        content_lines.append('# ADMIN_PASSWORD = "your-admin-password"')

    content = '\n'.join(content_lines) + '\n'

    secrets_file.write_text(content)
    print(f"Created/updated: {secrets_file}")
    print("\nRemember: This file is gitignored and for LOCAL development only!")


def show_streamlit_cloud_instructions():
    """Print instructions for updating Streamlit Cloud secrets."""
    print("""
================================================================================
HOW TO UPDATE STREAMLIT CLOUD SECRETS
================================================================================

Streamlit Cloud does NOT have a public API or CLI for managing secrets.
You must update secrets through the web dashboard:

STEPS:
------
1. Go to https://share.streamlit.io

2. Log in with your GitHub account

3. Find your app in the dashboard

4. Click the "Settings" icon (gear/cog) next to your app
   OR click on your app → "Manage app" → "Settings"

5. Navigate to the "Secrets" tab

6. Update your secrets in TOML format:

   DATABASE_URL = "postgresql://postgres:YOUR_PASSWORD@host.railway.app:PORT/railway"
   ADMIN_PASSWORD = "your-admin-password"

7. Click "Save"

8. Your app will automatically restart with the new secrets

================================================================================
TIPS:
================================================================================
- Use the PUBLIC URL from Railway (not the internal one)
- Railway format: postgresql://postgres:PASSWORD@HOST.railway.app:PORT/railway
- The password should be URL-encoded if it contains special characters
- After saving, check your app's Admin page to verify the connection

================================================================================
""")


def main():
    parser = argparse.ArgumentParser(
        description='Helper for managing Streamlit secrets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/update_secrets.py --validate "postgresql://user:pass@host:5432/db"
  python scripts/update_secrets.py --create-local --database-url "postgresql://..."
  python scripts/update_secrets.py --show-instructions
        """
    )

    parser.add_argument('--validate', metavar='URL',
                        help='Validate a DATABASE_URL format')
    parser.add_argument('--create-local', action='store_true',
                        help='Create local .streamlit/secrets.toml')
    parser.add_argument('--database-url', metavar='URL',
                        help='DATABASE_URL to use when creating local secrets')
    parser.add_argument('--admin-password', metavar='PASSWORD',
                        help='ADMIN_PASSWORD to use when creating local secrets')
    parser.add_argument('--show-instructions', action='store_true',
                        help='Show instructions for updating Streamlit Cloud secrets')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    if args.validate:
        valid, message = validate_database_url(args.validate)
        print(message)
        sys.exit(0 if valid else 1)

    if args.show_instructions:
        show_streamlit_cloud_instructions()
        sys.exit(0)

    if args.create_local:
        create_local_secrets(args.database_url, args.admin_password)
        sys.exit(0)


if __name__ == '__main__':
    main()
