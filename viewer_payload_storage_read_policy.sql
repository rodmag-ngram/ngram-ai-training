create policy "Authenticated users can read exam payloads"
on storage.objects
for select
to authenticated
using (
  bucket_id = 'eeg-viewer-payloads'
  and name like 'exams/%'
);
