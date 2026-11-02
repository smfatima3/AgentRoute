import hashlib,json,pathlib,sys,tempfile,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from coordinator_store import Store

class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name);self.store=Store(self.root/'queue.sqlite')
        self.task=dict(task_id='q1',group='main_mmlu',fingerprint='a'*64,dataset='mmlu',method='single',seed=17)
        self.store.initialize([self.task]);self.lease=self.store.claim('worker-a',['main_mmlu'])
        (self.root/'raw.json').write_text('{}')
        self.receipt=dict(task_id='q1',group='main_mmlu',fingerprint='a'*64,status='measured',epoch=self.lease['epoch'],
          evidence=[dict(path='raw.json',sha256=hashlib.sha256(b'{}').hexdigest())])
    def tearDown(self):self.tmp.cleanup()
    def test_no_lease_steal_on_expiry(self):
        with self.assertRaises(ValueError):self.store.confirmed_dead(self.lease['id'],self.lease['epoch'],None)
        self.assertIsNone(self.store.claim('worker-b',['main_mmlu']))
    def test_fenced_receipt_rejected_after_recovery(self):
        self.store.confirmed_dead(self.lease['id'],self.lease['epoch'],137)
        new=self.store.claim('worker-b',['main_mmlu']);self.assertGreater(new['epoch'],self.lease['epoch'])
        with self.assertRaises(ValueError):self.store.accept(self.receipt,self.root)
    def test_corrupt_artifact_rejected(self):
        (self.root/'raw.json').write_text('modified')
        with self.assertRaises(ValueError):self.store.accept(self.receipt,self.root)
    def test_accept_is_idempotent_but_not_replaceable(self):
        self.assertTrue(self.store.accept(self.receipt,self.root));self.assertFalse(self.store.accept(self.receipt,self.root))
        changed=dict(self.receipt,outcome='correct')
        with self.assertRaises(ValueError):self.store.accept(changed,self.root)
    def test_bounded_retries(self):
        lease=self.lease
        for i in range(3):
            self.store.confirmed_dead(lease['id'],lease['epoch'],1)
            lease=self.store.claim('worker-b',['main_mmlu'])
            if i<2:self.assertIsNotNone(lease)
        self.assertIsNone(lease);self.assertEqual(self.store.status()['counts'][0]['state'],'blocked')
    def test_manifest_cannot_change_on_resume(self):
        with self.assertRaises(ValueError):self.store.initialize([dict(self.task,fingerprint='b'*64)])
if __name__=='__main__':unittest.main()
