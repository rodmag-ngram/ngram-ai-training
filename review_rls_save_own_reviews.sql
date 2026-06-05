drop policy if exists human_reviews_insert_own_if_assigned on public.exam_human_reviews;

create policy human_reviews_insert_own_if_assigned
on public.exam_human_reviews
for insert
to authenticated
with check (
  reviewer_id = auth.uid()
);
