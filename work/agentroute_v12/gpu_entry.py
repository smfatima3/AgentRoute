"""Capture existing V1 engine timestamps without changing scheduler/model settings."""
import dataclasses,hashlib,json,os,runpy,time
from pathlib import Path
from vllm.v1.metrics.stats import IterationStats
root=Path(os.environ['V12_GPU_LOG_DIR']);root.mkdir(parents=True,exist_ok=True)
original=IterationStats.update_from_output
(root/'instrumentation.json').write_text(json.dumps({'target':'vllm.v1.metrics.stats.IterationStats.update_from_output',
  'original_source_sha256':hashlib.sha256(Path(original.__code__.co_filename).read_bytes()).hexdigest(),
  'behavior':'call original then copy existing measured events; no scheduler decisions changed'}))
def observe(self,output,engine_core_timestamp,is_prefilling,prompt_len,req_stats,lora_stats):
    value=original(self,output,engine_core_timestamp,is_prefilling,prompt_len,req_stats,lora_stats)
    if output.events or output.finish_reason is not None or is_prefilling:
        row={'request_id':output.request_id,'observed_unix_s':time.time(),'observed_monotonic_s':time.monotonic(),
          'engine_core_monotonic_s':engine_core_timestamp,'is_first_token':is_prefilling,'finished':output.finish_reason is not None,
          'finish_reason':str(output.finish_reason),'prompt_tokens':prompt_len,'state':dataclasses.asdict(req_stats),
          'events':[{'type':str(e.type),'monotonic_s':e.timestamp} for e in (output.events or [])]}
        with (root/'request_spans.jsonl').open('a',buffering=1) as f:f.write(json.dumps(row)+'\n')
    return value
IterationStats.update_from_output=observe
if __name__=='__main__':
    runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')
