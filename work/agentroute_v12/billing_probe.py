"""Read-only provider billing observation; never modifies account budgets."""
import asyncio,json,time
from datetime import datetime,timezone
from pathlib import Path
from google.protobuf.json_format import MessageToDict
from google.protobuf.timestamp_pb2 import Timestamp
from modal.client import _Client
from modal_proto import api_pb2
async def main():
    client=await _Client.from_env();now=datetime.now(timezone.utc)
    stamp=Timestamp();stamp.FromDatetime(now.replace(day=1,hour=0,minute=0,second=0,microsecond=0))
    result={'observed_utc':now.isoformat(),'read_only':True}
    for name,request in [('WorkspaceBillingSummary',api_pb2.WorkspaceBillingSummaryRequest(start_timestamp=stamp)),('WorkspaceBillingRates',api_pb2.WorkspaceBillingRatesRequest())]:
        try:result[name]=MessageToDict(await getattr(client.stub,name)(request),preserving_proto_field_name=True)
        except Exception as exc:result[name]={'error_type':type(exc).__name__,'message':str(exc)[:300]}
    path=Path(__file__).resolve().parents[2]/'outputs/agentroute_v12_completion/BILLING_OBSERVATION.json'
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':asyncio.run(main())
