"""Retrieve completed artifacts through persistent Volumes, including on Windows."""
import hashlib,json,time,zipfile
from pathlib import Path


def download(volume,remote_path,local_path,expected_sha=None,expected_size=None,timeout=90):
    local_path=Path(local_path);local_path.parent.mkdir(parents=True,exist_ok=True)
    partial=local_path.with_suffix(local_path.suffix+'.partial')
    deadline=time.monotonic()+timeout
    while True:
        try:
            digest=hashlib.sha256();size=0
            with partial.open('wb') as f:
                for block in volume.read_file(remote_path):
                    f.write(block);digest.update(block);size+=len(block)
            if expected_sha is not None and digest.hexdigest()!=expected_sha:raise ValueError('Downloaded SHA256 differs')
            if expected_size is not None and size!=expected_size:raise ValueError('Downloaded size differs')
            partial.replace(local_path)
            return {'path':str(local_path),'sha256':digest.hexdigest(),'bytes':size}
        except Exception:
            partial.unlink(missing_ok=True)
            if time.monotonic()>=deadline:raise
            time.sleep(min(2,max(0,deadline-time.monotonic())))


def retrieve_export(volume,operation_id,output,timeout=90):
    output=Path(output)
    metadata_path=output/'export.json'
    download(volume,f'exports/{operation_id}/export.json',metadata_path,timeout=timeout)
    metadata=json.loads(metadata_path.read_text())
    expected_path=f'exports/{operation_id}/AgentRoute_results.zip'
    if metadata['operation_id']!=operation_id or metadata['path']!=expected_path or metadata['status']!='ready':
        raise ValueError('Export metadata belongs to a different operation')
    result=download(volume,expected_path,output/'AgentRoute_results.zip',metadata['sha256'],metadata['bytes'],timeout)
    with zipfile.ZipFile(result['path']) as archive:
        if archive.testzip() is not None:raise ValueError('Export ZIP failed CRC validation')
        manifest=json.loads(archive.read('EXPORT_MANIFEST.sha256.json'))
        for name,sha in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=sha:raise ValueError('Export member hash differs: '+name)
        actual=set(archive.namelist())-{'EXPORT_MANIFEST.sha256.json'}
        if actual!=set(manifest):raise ValueError('Export manifest is incomplete')
    return result
