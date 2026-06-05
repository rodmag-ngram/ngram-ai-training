create policy "Authenticated users can upload their own manager review exports" on storage.objects
for insert to authenticated
with check (
  bucket_id = 'eeg-viewer-payloads'
  and name like ('manager-reviews/' || auth.uid()::text || '/%')
);

create policy "Authenticated users can update their own manager review exports" on storage.objects
for update to authenticated
using (
  bucket_id = 'eeg-viewer-payloads'
  and name like ('manager-reviews/' || auth.uid()::text || '/%')
)
with check (
  bucket_id = 'eeg-viewer-payloads'
  and name like ('manager-reviews/' || auth.uid()::text || '/%')
);

create policy "Authenticated users can read their own manager review exports" on storage.objects
for select to authenticated
using (
  bucket_id = 'eeg-viewer-payloads'
  and name like ('manager-reviews/' || auth.uid()::text || '/%')
);
