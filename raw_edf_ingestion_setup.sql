insert into storage.buckets (id, name, public)
values ('eeg-raw-edf', 'eeg-raw-edf', false)
on conflict (id) do nothing;

create policy "Admins can upload raw EDFs" on storage.objects
for insert to authenticated
with check (bucket_id = 'eeg-raw-edf' and is_admin());

create policy "Admins can update raw EDFs" on storage.objects
for update to authenticated
using (bucket_id = 'eeg-raw-edf' and is_admin())
with check (bucket_id = 'eeg-raw-edf' and is_admin());

create policy "Admins can read raw EDFs" on storage.objects
for select to authenticated
using (bucket_id = 'eeg-raw-edf' and is_admin());

create policy "Admins can delete raw EDFs" on storage.objects
for delete to authenticated
using (bucket_id = 'eeg-raw-edf' and is_admin());

create policy "ai_reviews_admin_manage" on public.exam_ai_reviews
for all to authenticated
using (is_admin())
with check (is_admin());
