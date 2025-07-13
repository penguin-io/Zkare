-- Database initialization script for PenguLLM
-- Creates the initial database schema and user accounts

-- Create database if it doesn't exist
SELECT 'CREATE DATABASE zkp_llm'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'zkp_llm');

-- Connect to the database
\c zkp_llm;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS public;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS metrics;

-- Create enum types
CREATE TYPE user_role AS ENUM ('admin', 'user', 'system');
CREATE TYPE request_status AS ENUM ('pending', 'processing', 'completed', 'failed');
CREATE TYPE advice_domain AS ENUM ('financial', 'healthcare', 'career', 'education', 'lifestyle', 'general');
CREATE TYPE risk_category AS ENUM ('conservative', 'steady_growth', 'balanced', 'aggressive_investment');

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    password_hash VARCHAR(255),
    role user_role DEFAULT 'user',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    metadata JSONB
);

-- Create sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    client_ip INET,
    user_agent TEXT
);

-- Create advice requests table
CREATE TABLE IF NOT EXISTS advice_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    session_id VARCHAR(255),
    query TEXT NOT NULL,
    domain advice_domain,
    verified_traits JSONB,
    unverifiable_traits JSONB,
    proof_id VARCHAR(255),
    proof_verified BOOLEAN DEFAULT false,
    advice_text TEXT,
    explanation TEXT,
    confidence_score DECIMAL(3,2),
    response_time_ms INTEGER,
    status request_status DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB
);

-- Create chat conversations table
CREATE TABLE IF NOT EXISTS chat_conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    session_id VARCHAR(255) NOT NULL,
    title VARCHAR(255),
    domain advice_domain,
    verified_traits JSONB,
    message_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- Create chat messages table
CREATE TABLE IF NOT EXISTS chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES chat_conversations(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tokens_used INTEGER,
    metadata JSONB
);

-- Create proof verifications table
CREATE TABLE IF NOT EXISTS proof_verifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    proof_id VARCHAR(255) NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    verification_result BOOLEAN NOT NULL,
    verification_time_ms INTEGER,
    verified_traits JSONB,
    error_message TEXT,
    verified_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- Create system metrics table
CREATE TABLE IF NOT EXISTS metrics.system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    metric_unit VARCHAR(50),
    service_name VARCHAR(100),
    tags JSONB,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create audit log table
CREATE TABLE IF NOT EXISTS audit.audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(255) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    old_values JSONB,
    new_values JSONB,
    client_ip INET,
    user_agent TEXT,
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create rate limiting table
CREATE TABLE IF NOT EXISTS rate_limits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    identifier VARCHAR(255) NOT NULL,
    resource VARCHAR(100) NOT NULL,
    window_start TIMESTAMP WITH TIME ZONE NOT NULL,
    request_count INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(identifier, resource, window_start)
);

-- Create performance logs table
CREATE TABLE IF NOT EXISTS performance_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id UUID,
    service_name VARCHAR(100) NOT NULL,
    endpoint VARCHAR(255),
    method VARCHAR(10),
    response_time_ms INTEGER NOT NULL,
    status_code INTEGER,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- Create indexes for performance

-- Users table indexes
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

-- Sessions table indexes
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires ON user_sessions(expires_at);

-- Advice requests table indexes
CREATE INDEX IF NOT EXISTS idx_advice_requests_user_id ON advice_requests(user_id);
CREATE INDEX IF NOT EXISTS idx_advice_requests_session_id ON advice_requests(session_id);
CREATE INDEX IF NOT EXISTS idx_advice_requests_domain ON advice_requests(domain);
CREATE INDEX IF NOT EXISTS idx_advice_requests_status ON advice_requests(status);
CREATE INDEX IF NOT EXISTS idx_advice_requests_created_at ON advice_requests(created_at);
CREATE INDEX IF NOT EXISTS idx_advice_requests_proof_id ON advice_requests(proof_id);

-- Chat tables indexes
CREATE INDEX IF NOT EXISTS idx_chat_conversations_user_id ON chat_conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_conversations_session_id ON chat_conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_conversations_created_at ON chat_conversations(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id ON chat_messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_timestamp ON chat_messages(timestamp);

-- Proof verifications indexes
CREATE INDEX IF NOT EXISTS idx_proof_verifications_proof_id ON proof_verifications(proof_id);
CREATE INDEX IF NOT EXISTS idx_proof_verifications_user_id ON proof_verifications(user_id);
CREATE INDEX IF NOT EXISTS idx_proof_verifications_verified_at ON proof_verifications(verified_at);

-- Metrics and audit indexes
CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON metrics.system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_recorded_at ON metrics.system_metrics(recorded_at);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit.audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit.audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit.audit_logs(created_at);

-- Rate limiting indexes
CREATE INDEX IF NOT EXISTS idx_rate_limits_identifier ON rate_limits(identifier);
CREATE INDEX IF NOT EXISTS idx_rate_limits_window_start ON rate_limits(window_start);

-- Performance logs indexes
CREATE INDEX IF NOT EXISTS idx_performance_logs_service ON performance_logs(service_name);
CREATE INDEX IF NOT EXISTS idx_performance_logs_timestamp ON performance_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_logs_user_id ON performance_logs(user_id);

-- Create views for analytics

-- Advice request analytics view
CREATE OR REPLACE VIEW advice_analytics AS
SELECT
    DATE_TRUNC('hour', created_at) as hour,
    domain,
    COUNT(*) as total_requests,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_requests,
    COUNT(CASE WHEN proof_verified = true THEN 1 END) as verified_requests,
    AVG(response_time_ms) as avg_response_time,
    AVG(confidence_score) as avg_confidence_score
FROM advice_requests
GROUP BY DATE_TRUNC('hour', created_at), domain;

-- User activity analytics view
CREATE OR REPLACE VIEW user_activity_analytics AS
SELECT
    DATE_TRUNC('day', created_at) as day,
    COUNT(DISTINCT user_id) as active_users,
    COUNT(*) as total_requests,
    AVG(response_time_ms) as avg_response_time
FROM advice_requests
WHERE user_id IS NOT NULL
GROUP BY DATE_TRUNC('day', created_at);

-- System performance view
CREATE OR REPLACE VIEW system_performance AS
SELECT
    DATE_TRUNC('minute', timestamp) as minute,
    service_name,
    endpoint,
    COUNT(*) as request_count,
    AVG(response_time_ms) as avg_response_time,
    MIN(response_time_ms) as min_response_time,
    MAX(response_time_ms) as max_response_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time
FROM performance_logs
GROUP BY DATE_TRUNC('minute', timestamp), service_name, endpoint;

-- Create functions

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Function to log audit events
CREATE OR REPLACE FUNCTION log_audit_event(
    p_user_id UUID,
    p_action VARCHAR(255),
    p_resource_type VARCHAR(100),
    p_resource_id VARCHAR(255),
    p_old_values JSONB DEFAULT NULL,
    p_new_values JSONB DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    INSERT INTO audit.audit_logs (
        user_id, action, resource_type, resource_id, old_values, new_values
    ) VALUES (
        p_user_id, p_action, p_resource_type, p_resource_id, p_old_values, p_new_values
    );
END;
$$ LANGUAGE plpgsql;

-- Function to clean up expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM user_sessions WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;

    PERFORM log_audit_event(
        NULL,
        'CLEANUP_EXPIRED_SESSIONS',
        'user_sessions',
        NULL,
        NULL,
        jsonb_build_object('deleted_count', deleted_count)
    );

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get user statistics
CREATE OR REPLACE FUNCTION get_user_stats(p_user_id UUID)
RETURNS TABLE(
    total_requests INTEGER,
    completed_requests INTEGER,
    verified_requests INTEGER,
    avg_confidence DECIMAL,
    last_request_date TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::INTEGER as total_requests,
        COUNT(CASE WHEN status = 'completed' THEN 1 END)::INTEGER as completed_requests,
        COUNT(CASE WHEN proof_verified = true THEN 1 END)::INTEGER as verified_requests,
        AVG(confidence_score) as avg_confidence,
        MAX(created_at) as last_request_date
    FROM advice_requests
    WHERE user_id = p_user_id;
END;
$$ LANGUAGE plpgsql;

-- Create triggers

-- Trigger to update updated_at on users table
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger to update updated_at on chat_conversations table
CREATE TRIGGER update_chat_conversations_updated_at
    BEFORE UPDATE ON chat_conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create application users and roles

-- Create application role
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'zkp_app_role') THEN
        CREATE ROLE zkp_app_role;
    END IF;
END
$$;

-- Create application user for Entity 1
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'zkp_entity1') THEN
        CREATE USER zkp_entity1 WITH PASSWORD 'zkp_entity1_secure_password';
    END IF;
END
$$;

-- Create application user for Entity 2
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'zkp_entity2') THEN
        CREATE USER zkp_entity2 WITH PASSWORD 'zkp_entity2_secure_password';
    END IF;
END
$$;

-- Grant permissions

-- Grant basic database access
GRANT CONNECT ON DATABASE zkp_llm TO zkp_app_role;
GRANT USAGE ON SCHEMA public TO zkp_app_role;
GRANT USAGE ON SCHEMA audit TO zkp_app_role;
GRANT USAGE ON SCHEMA metrics TO zkp_app_role;

-- Grant table permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO zkp_app_role;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA audit TO zkp_app_role;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA metrics TO zkp_app_role;

-- Grant sequence permissions
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO zkp_app_role;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA audit TO zkp_app_role;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA metrics TO zkp_app_role;

-- Grant function permissions
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO zkp_app_role;

-- Assign role to users
GRANT zkp_app_role TO zkp_entity1;
GRANT zkp_app_role TO zkp_entity2;
GRANT zkp_app_role TO zkp_user;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO zkp_app_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA audit GRANT SELECT, INSERT ON TABLES TO zkp_app_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA metrics GRANT SELECT, INSERT ON TABLES TO zkp_app_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO zkp_app_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA audit GRANT USAGE, SELECT ON SEQUENCES TO zkp_app_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA metrics GRANT USAGE, SELECT ON SEQUENCES TO zkp_app_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO zkp_app_role;

-- Insert initial data

-- Insert system user
INSERT INTO users (id, username, email, role, is_active) VALUES
    ('00000000-0000-0000-0000-000000000001', 'system', 'system@pengu-llm.com', 'system', true)
ON CONFLICT (username) DO NOTHING;

-- Insert admin user (password: admin123 - change in production!)
INSERT INTO users (id, username, email, password_hash, role, is_active) VALUES
    ('00000000-0000-0000-0000-000000000002', 'admin', 'admin@pengu-llm.com',
     crypt('admin123', gen_salt('bf')), 'admin', true)
ON CONFLICT (username) DO NOTHING;

-- Insert initial metrics
INSERT INTO metrics.system_metrics (metric_name, metric_value, metric_unit, service_name, tags) VALUES
    ('database_schema_version', 1.0, 'version', 'database', '{"component": "schema"}'),
    ('initialization_timestamp', EXTRACT(epoch FROM NOW()), 'timestamp', 'database', '{"event": "init"}')
ON CONFLICT DO NOTHING;

-- Log initialization
SELECT log_audit_event(
    '00000000-0000-0000-0000-000000000001',
    'DATABASE_INITIALIZED',
    'database',
    'zkp_llm',
    NULL,
    jsonb_build_object(
        'schema_version', '1.0',
        'timestamp', NOW(),
        'tables_created', (
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        )
    )
);

-- Display completion message
SELECT 'Database initialization completed successfully!' as status,
       NOW() as completed_at,
       (SELECT COUNT(*) FROM information_schema.tables
        WHERE table_schema IN ('public', 'audit', 'metrics')
        AND table_type = 'BASE TABLE') as tables_created;
