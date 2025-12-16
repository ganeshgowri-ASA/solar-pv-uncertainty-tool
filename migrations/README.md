# Database Migrations

This folder contains SQL migration files for schema changes.

## Naming Convention

- UP migrations: `XXX_description_UP.sql` (e.g., `001_initial_schema_UP.sql`)
- DOWN migrations: `XXX_description_DOWN.sql` (e.g., `001_initial_schema_DOWN.sql`)

## Running Migrations

1. Go to Admin page: `/Admin`
2. Navigate to "Run Migrations" tab
3. Select the migration to run
4. Preview and confirm execution

## Best Practices

1. Always create both UP and DOWN migrations
2. Test locally before deploying
3. Never modify a migration that has been run in production
4. Use sequential numbering (001, 002, 003...)

## Note

Base tables are created automatically by SQLAlchemy ORM using `Base.metadata.create_all()`.
Use migration files for:
- Custom indexes not defined in models
- Custom constraints
- Data migrations
- Schema alterations not suitable for ORM
