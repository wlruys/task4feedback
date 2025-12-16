import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(project_root))

from runner.launcher import parse_overrides, launch

class TestLauncher(unittest.TestCase):
    def test_parse_overrides_single(self):
        args = ["+size=b8"]
        overrides = parse_overrides(args)
        self.assertEqual(len(overrides), 1)
        self.assertEqual(overrides[0], ["+size=b8"])

    def test_parse_overrides_sweep(self):
        args = ["+size=b8,b4", "+entropy=e0.001"]
        overrides = parse_overrides(args)
        self.assertEqual(len(overrides), 2)
        # Order might vary but we expect (b8, e0.001) and (b4, e0.001)
        expected = [
            set(["+size=b8", "+entropy=e0.001"]),
            set(["+size=b4", "+entropy=e0.001"])
        ]
        actual = [set(o) for o in overrides]
        for e in expected:
            self.assertIn(e, actual)

    def test_parse_overrides_multi_sweep(self):
        args = ["+a=1,2", "+b=3,4"]
        overrides = parse_overrides(args)
        self.assertEqual(len(overrides), 4) # 2x2

    @patch("runner.launcher.submitit.AutoExecutor")
    @patch("runner.launcher.compose")
    @patch("runner.launcher.initialize_config_dir")
    @patch("runner.launcher.OmegaConf.resolve")
    def test_launch(self, mock_resolve, mock_init, mock_compose, mock_executor_cls):
        # Mock sys.argv
        with patch.object(sys, "argv", ["launcher.py", "+size=b8,b4"]):
            # Mock compose to return a dummy config
            mock_compose.return_value = {"dummy": "config"}
            
            # Mock executor
            mock_executor = MagicMock()
            mock_executor_cls.return_value = mock_executor
            mock_executor.map_array.return_value = [MagicMock(job_id="123_0"), MagicMock(job_id="123_1")]
            
            launch()
            
            # Check if compose was called twice
            self.assertEqual(mock_compose.call_count, 2)
            
            # Check if map_array was called
            mock_executor.map_array.assert_called_once()
            args, _ = mock_executor.map_array.call_args
            self.assertEqual(len(args[1]), 2) # 2 configs

if __name__ == "__main__":
    unittest.main()
