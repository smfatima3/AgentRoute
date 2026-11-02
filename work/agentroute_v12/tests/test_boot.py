import pathlib,sys,tempfile,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from cloud_boot import safe_destination

class MountedRootTests(unittest.TestCase):
    def test_traversal_and_absolute_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=pathlib.Path(temp)/'mounted';root.mkdir()
            self.assertEqual(safe_destination(root,'outputs/result.json'),root/'outputs/result.json')
            for name in ['../escape',str(pathlib.Path(temp)/'escape')]:
                with self.assertRaises(RuntimeError):safe_destination(root,name)
    def test_symlink_mount_allowed_but_escape_not(self):
        with tempfile.TemporaryDirectory() as temp:
            real=pathlib.Path(temp)/'real';real.mkdir();alias=pathlib.Path(temp)/'alias'
            try:alias.symlink_to(real,target_is_directory=True)
            except OSError as exc:self.skipTest(str(exc))
            self.assertEqual(safe_destination(alias,'outputs/a'),real/'outputs/a')
            with self.assertRaises(RuntimeError):safe_destination(alias,'../escape')
if __name__=='__main__':unittest.main()
