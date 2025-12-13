-- Migration: 007_add_missing_service_requests_columns_UP.sql
-- Description: Add expected_sample_quantity and receipt_id columns to service_requests table
-- Date: 2025-12-13

-- Add expected_sample_quantity column
ALTER TABLE service_requests
ADD COLUMN IF NOT EXISTS expected_sample_quantity INTEGER;

-- Add receipt_id column with foreign key to sample_receipts
ALTER TABLE service_requests
ADD COLUMN IF NOT EXISTS receipt_id INTEGER REFERENCES sample_receipts(id);

-- Create index on receipt_id for better join performance
CREATE INDEX IF NOT EXISTS idx_service_requests_receipt_id
ON service_requests(receipt_id);

-- Add comment for documentation
COMMENT ON COLUMN service_requests.expected_sample_quantity IS 'Expected number of samples for this service request';
COMMENT ON COLUMN service_requests.receipt_id IS 'Foreign key reference to sample_receipts table';
