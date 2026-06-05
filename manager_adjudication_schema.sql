create table if not exists public.exam_manager_reviews (
  id uuid primary key default gen_random_uuid(),
  exam_id uuid not null references public.exams(id) on delete cascade,
  manager_id uuid not null references public.profiles(id) on delete cascade,
  review_status text not null default 'in_progress'
    check (review_status = any (array['in_progress'::text, 'completed'::text])),
  observer_agreement_level text null
    check (
      observer_agreement_level = any (
        array['Full agreement'::text, 'Partial agreement'::text, 'No agreement'::text]
      )
      or observer_agreement_level is null
    ),
  observer_comment text null,
  overall_model_grade smallint null
    check (overall_model_grade >= 1 and overall_model_grade <= 10 or overall_model_grade is null),
  overall_comment text null,
  exported_payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  completed_at timestamptz null,
  unique (exam_id, manager_id)
);

create table if not exists public.exam_manager_review_annotations (
  id uuid primary key default gen_random_uuid(),
  review_id uuid not null references public.exam_manager_reviews(id) on delete cascade,
  exam_id uuid not null references public.exams(id) on delete cascade,
  t0_seconds numeric not null,
  t1_seconds numeric not null,
  primary_label text null,
  occurrence_type text null
    check (
      occurrence_type = any (
        array[
          'Model correct'::text,
          'Model partially correct'::text,
          'False positive (model)'::text,
          'False negative (model)'::text,
          'Wrong type'::text,
          'Temporal extent mismatch'::text,
          'Disagreement between doctors'::text,
          'Uncertain'::text
        ]
      )
      or occurrence_type is null
    ),
  model_marked text[] not null default '{}'::text[],
  clinician_marked text[] not null default '{}'::text[],
  lateralization text null
    check (
      lateralization = any (
        array['Looks lateralized'::text, 'Looks generalized'::text, 'Bilateral'::text, 'Not sure'::text]
      )
      or lateralization is null
    ),
  model_grade smallint null
    check (model_grade >= 1 and model_grade <= 10 or model_grade is null),
  comment text null,
  extra_payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.exam_manager_reviews enable row level security;
alter table public.exam_manager_review_annotations enable row level security;

drop policy if exists manager_reviews_insert_own on public.exam_manager_reviews;
create policy manager_reviews_insert_own
on public.exam_manager_reviews
for insert
to authenticated
with check (
  manager_id = auth.uid()
  and (
    is_admin()
    or exists (
      select 1
      from public.profiles p
      where p.id = auth.uid()
        and p.role = 'manager'
    )
  )
);

drop policy if exists manager_reviews_select_own_or_admin on public.exam_manager_reviews;
create policy manager_reviews_select_own_or_admin
on public.exam_manager_reviews
for select
to authenticated
using (
  manager_id = auth.uid()
  or is_admin()
  or exists (
    select 1
    from public.profiles p
    where p.id = auth.uid()
      and p.role = 'manager'
  )
);

drop policy if exists manager_reviews_update_own_or_admin on public.exam_manager_reviews;
create policy manager_reviews_update_own_or_admin
on public.exam_manager_reviews
for update
to authenticated
using (
  manager_id = auth.uid()
  or is_admin()
)
with check (
  manager_id = auth.uid()
  or is_admin()
);

drop policy if exists manager_review_annotations_select_via_review on public.exam_manager_review_annotations;
create policy manager_review_annotations_select_via_review
on public.exam_manager_review_annotations
for select
to authenticated
using (
  is_admin()
  or exists (
    select 1
    from public.exam_manager_reviews r
    where r.id = exam_manager_review_annotations.review_id
      and r.manager_id = auth.uid()
  )
  or exists (
    select 1
    from public.profiles p
    where p.id = auth.uid()
      and p.role = 'manager'
  )
);

drop policy if exists manager_review_annotations_insert_via_review on public.exam_manager_review_annotations;
create policy manager_review_annotations_insert_via_review
on public.exam_manager_review_annotations
for insert
to authenticated
with check (
  is_admin()
  or exists (
    select 1
    from public.exam_manager_reviews r
    where r.id = exam_manager_review_annotations.review_id
      and r.manager_id = auth.uid()
  )
);

drop policy if exists manager_review_annotations_update_via_review on public.exam_manager_review_annotations;
create policy manager_review_annotations_update_via_review
on public.exam_manager_review_annotations
for update
to authenticated
using (
  is_admin()
  or exists (
    select 1
    from public.exam_manager_reviews r
    where r.id = exam_manager_review_annotations.review_id
      and r.manager_id = auth.uid()
  )
)
with check (
  is_admin()
  or exists (
    select 1
    from public.exam_manager_reviews r
    where r.id = exam_manager_review_annotations.review_id
      and r.manager_id = auth.uid()
  )
);

drop policy if exists manager_review_annotations_delete_via_review on public.exam_manager_review_annotations;
create policy manager_review_annotations_delete_via_review
on public.exam_manager_review_annotations
for delete
to authenticated
using (
  is_admin()
  or exists (
    select 1
    from public.exam_manager_reviews r
    where r.id = exam_manager_review_annotations.review_id
      and r.manager_id = auth.uid()
  )
);
