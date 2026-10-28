-- Lets community flood reports carry a map position, so the route agent can avoid them.
-- Run once on the Supabase database.
ALTER TABLE reports ADD COLUMN IF NOT EXISTS lat double precision;
ALTER TABLE reports ADD COLUMN IF NOT EXISTS lng double precision;
