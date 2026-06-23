-- ============================================================
-- MeTernak — Skema Supabase (jalankan di SQL Editor > Run)
-- ============================================================

create table if not exists cows (
    cattle_id     text primary key,
    farmer_name   text,
    farm_address  text,
    cattle_age    text,
    created_at    timestamptz default now()
);

create table if not exists esp32_readings (
    id             bigint generated always as identity primary key,
    cattle_id      text,
    resistance_ohm integer,
    created_at     timestamptz default now()
);
create index if not exists idx_esp32_cattle_id on esp32_readings(cattle_id);

create table if not exists tracking_logs (
    id             bigint generated always as identity primary key,
    cattle_id      text references cows(cattle_id) on delete cascade,
    farmer_name    text,
    mucus_type     integer,
    mucus_color    text,
    confidence     numeric,
    temperature    numeric,
    resistance_ohm integer,
    p_day1         numeric,
    p_day2         numeric,
    p_day3         numeric,
    p_kuning       numeric,
    predicted      text,
    created_at     timestamptz default now()
);
create index if not exists idx_tracking_cattle_id on tracking_logs(cattle_id);

-- RLS — backend pakai service_role key (bypass RLS otomatis), jadi cukup enable saja
alter table cows enable row level security;
alter table esp32_readings enable row level security;
alter table tracking_logs enable row level security;
