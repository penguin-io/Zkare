-- Initial database schema for ZK Proof Service
-- Creates tables for tracking proofs, verifications, and audit logs

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table for storing proof metadata
CREATE TABLE proofs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    proof_id VARCHAR(255) UNIQUE NOT NULL,
    inference_type TEXT NOT NULL,
    user_id VARCHAR(255),
    generation_time_ms BIGINT,
    proof_size_bytes INTEGER,
    confidence_score INTEGER,
    risk_category TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    is_cached BOOLEAN DEFAULT FALSE,
    metadata JSONB
);

-- Table for storing verification attempts
CREATE TABLE verifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    proof_id VARCHAR(255) NOT NULL,
    verification_time_ms BIGINT NOT NULL,
    is_valid BOOLEAN,
    verified_by VARCHAR(255),
    client_ip INET,
    user_agent TEXT,
    verified_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    FOREIGN KEY (proof_id) REFERENCES proofs(proof_id) ON DELETE CASCADE
);

-- Table for rate limiting tracking
CREATE TABLE rate_limits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_identifier VARCHAR(255) NOT NULL,
    request_count INTEGER NOT NULL DEFAULT 1,
    window_start TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    window_end TIMESTAMP WITH TIME ZONE DEFAULT NOW() + INTERVAL '1 minute',
    blocked_until TIMESTAMP WITH TIME ZONE,

    UNIQUE(user_identifier, window_start)
);

-- Table for audit logs
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    user_id VARCHAR(255),
    proof_id VARCHAR(255),
    event_data JSONB,
    client_ip INET,
    user_agent TEXT,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for system metrics
CREATE TABLE system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    metric_unit VARCHAR(50),
    tags JSONB,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_proofs_user_id ON proofs(user_id);
CREATE INDEX idx_proofs_created_at ON proofs(created_at);
CREATE INDEX idx_proofs_inference_type ON proofs(inference_type);
CREATE INDEX idx_proofs_expires_at ON proofs(expires_at);

CREATE INDEX idx_verifications_proof_id ON verifications(proof_id);
CREATE INDEX idx_verifications_verified_at ON verifications(verified_at);
CREATE INDEX idx_verifications_is_valid ON verifications(is_valid);

CREATE INDEX idx_rate_limits_user_identifier ON rate_limits(user_identifier);
CREATE INDEX idx_rate_limits_window_start ON rate_limits(window_start);
CREATE INDEX idx_rate_limits_window_end ON rate_limits(window_end);

CREATE INDEX idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at);

CREATE INDEX idx_system_metrics_metric_name ON system_metrics(metric_name);
CREATE INDEX idx_system_metrics_recorded_at ON system_metrics(recorded_at);

-- Views for analytics
CREATE VIEW proof_statistics AS
SELECT
    DATE_TRUNC('hour', created_at) as hour,
    inference_type,
    COUNT(*) as total_proofs,
    AVG(generation_time_ms) as avg_generation_time,
    AVG(confidence_score) as avg_confidence,
    COUNT(CASE WHEN is_cached THEN 1 END) as cached_proofs
FROM proofs
GROUP BY DATE_TRUNC('hour', created_at), inference_type;

CREATE VIEW verification_statistics AS
SELECT
    DATE_TRUNC('hour', verified_at) as hour,
    COUNT(*) as total_verifications,
    AVG(verification_time_ms) as avg_verification_time,
    COUNT(CASE WHEN is_valid THEN 1 END) as valid_verifications,
    COUNT(CASE WHEN NOT is_valid THEN 1 END) as invalid_verifications
FROM verifications
GROUP BY DATE_TRUNC('hour', verified_at);

-- Function to clean up expired proofs
CREATE OR REPLACE FUNCTION cleanup_expired_proofs()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM proofs
    WHERE expires_at < NOW() - INTERVAL '7 days';

    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    -- Log cleanup operation
    INSERT INTO audit_logs (event_type, event_data)
    VALUES ('CLEANUP_EXPIRED_PROOFS', jsonb_build_object('deleted_count', deleted_count));

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate cache hit rate
CREATE OR REPLACE FUNCTION get_cache_hit_rate(time_window INTERVAL DEFAULT INTERVAL '1 hour')
RETURNS DECIMAL(5,4) AS $$
DECLARE
    total_requests INTEGER;
    cached_requests INTEGER;
BEGIN
    SELECT COUNT(*) INTO total_requests
    FROM proofs
    WHERE created_at >= NOW() - time_window;

    SELECT COUNT(*) INTO cached_requests
    FROM proofs
    WHERE created_at >= NOW() - time_window
    AND is_cached = TRUE;

    IF total_requests = 0 THEN
        RETURN 0.0000;
    END IF;

    RETURN (cached_requests::DECIMAL / total_requests::DECIMAL);
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically set expires_at based on created_at
CREATE OR REPLACE FUNCTION set_proof_expiry()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.expires_at IS NULL THEN
        NEW.expires_at := NEW.created_at + INTERVAL '24 hours';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_set_proof_expiry
    BEFORE INSERT ON proofs
    FOR EACH ROW
    EXECUTE FUNCTION set_proof_expiry();

-- Insert initial system configuration
INSERT INTO system_metrics (metric_name, metric_value, metric_unit, tags)
VALUES
    ('service_startup', 1, 'count', '{"version": "0.1.0", "component": "zkproof-service"}'),
    ('database_schema_version', 1, 'version', '{"migration": "001_initial_schema"}');

-- Create a user for the application (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'zkproof_app') THEN
        CREATE ROLE zkproof_app LOGIN PASSWORD 'zkproof_app_password';
    END IF;
END
$$;

-- Grant necessary permissions
GRANT CONNECT ON DATABASE zkp_llm TO zkproof_app;
GRANT USAGE ON SCHEMA public TO zkproof_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO zkproof_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO zkproof_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO zkproof_app;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO zkproof_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO zkproof_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO zkproof_app;
