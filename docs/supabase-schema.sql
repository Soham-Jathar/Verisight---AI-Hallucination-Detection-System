-- Run this once in the Supabase SQL Editor.
-- Each signed-in user can only read and change their own saved conversations.

create table if not exists public.conversations (
  id uuid primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  title text not null default 'New conversation',
  messages jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.conversations enable row level security;

create policy "Users can read their own conversations"
on public.conversations for select to authenticated
using ((select auth.uid()) = user_id);

create policy "Users can create their own conversations"
on public.conversations for insert to authenticated
with check ((select auth.uid()) = user_id);

create policy "Users can update their own conversations"
on public.conversations for update to authenticated
using ((select auth.uid()) = user_id)
with check ((select auth.uid()) = user_id);

create policy "Users can delete their own conversations"
on public.conversations for delete to authenticated
using ((select auth.uid()) = user_id);

create index if not exists conversations_user_updated_idx
on public.conversations (user_id, updated_at desc);
