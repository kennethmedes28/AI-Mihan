-- Community flood reports. Run once on the Supabase database, before 001_reports_geo.sql.
-- The API (POST /reports) and the report agent both write here; every row starts unverified.
CREATE TABLE IF NOT EXISTS reports (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    hazard      text NOT NULL,
    city        text NOT NULL,
    barangay    text,
    body        text NOT NULL,
    status      text NOT NULL DEFAULT 'unverified' CHECK (status IN ('unverified', 'verified', 'rejected')),
    trust_level text NOT NULL DEFAULT 'community',
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS reports_created_at_idx ON reports (created_at DESC);

-- Only the backend's direct Postgres connection may read or write (no policies on purpose).
ALTER TABLE reports ENABLE ROW LEVEL SECURITY;
