import concurrent.futures,pathlib,sys,threading,time,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from run_campaign import Coordinator
class CreditTests(unittest.TestCase):
    def test_concurrent_launches_reserve_before_spending(self):
        c=object.__new__(Coordinator);c.lock=threading.RLock();c.registry={'resources':[]};c.pending_reservations={}
        def attempt(_):
            try:return c.reserve_check(7,3600)
            except RuntimeError:return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:results=list(pool.map(attempt,range(10)))
        self.assertEqual(sum(x is not None for x in results),4);self.assertEqual(c.cost(True),28)
    def test_live_future_and_failed_consumption_count(self):
        c=object.__new__(Coordinator);c.lock=threading.RLock();c.pending_reservations={}
        now=time.time();c.registry={'resources':[{'created_unix_s':now-3600,'stopped_unix_s':now,'timeout_s':7200,'rate_usd_per_hour':7},
          {'created_unix_s':now,'timeout_s':7200,'rate_usd_per_hour':7}]}
        self.assertEqual(c.cost(True),21)
        c.reserve_check(7,3600)
        with self.assertRaises(RuntimeError):c.reserve_check(.01,3600)
if __name__=='__main__':unittest.main()
