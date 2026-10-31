-- Supply and assistance requests (POST /requests). Run once on the Supabase database.
-- Rows are unverified until a person (barangay or DRRMO staff) reviews them. There is no
-- public read endpoint: the contact phone number must not be listed to everyone.
CREATE TABLE IF NOT EXISTS requests (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    kind          text NOT NULL,                       -- supply or assistance
    city          text NOT NULL,
    barangay      text,
    details       text NOT NULL,
    people_count  integer,
    contact_phone text,
    lat           double precision,
    lng           double precision,
    status        text NOT NULL DEFAULT 'unverified' CHECK (status IN ('unverified', 'verified', 'assigned', 'closed')),
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS requests_created_at_idx ON requests (created_at DESC);

-- Only the backend's direct Postgres connection may read or write (no policies on purpose).
ALTER TABLE requests ENABLE ROW LEVEL SECURITY;
