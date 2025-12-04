import unittest
from unittest.mock import MagicMock, patch
import json
import io
import sys
import multiprocessing

# Import the script
import grinder_worker

class TestGrinderWorker(unittest.TestCase):
    @patch('sys.stdin', new_callable=io.StringIO)
    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('google.cloud.firestore.Client')
    @patch('multiprocessing.cpu_count', return_value=2)
    @patch('multiprocessing.Process')
    @patch('multiprocessing.Queue')
    @patch('multiprocessing.Event')
    def test_main(self, mock_event, mock_queue, mock_process, mock_cpu_count, mock_firestore_client, mock_stdout, mock_stdin):
        # Setup stdin
        mock_stdin.write(json.dumps({
            "job_id": "test_job_123",
            "prefix": "A",
            "suffix": ""
        }))
        mock_stdin.seek(0)

        # Mock Queue behavior
        # When main calls result_queue.get(), it should return a dummy result
        mock_q_instance = mock_queue.return_value
        dummy_pubkey = "A123456789"
        dummy_secret = list(range(64))
        mock_q_instance.get.return_value = (dummy_pubkey, dummy_secret)

        # Mock Event behavior
        mock_event_instance = mock_event.return_value
        mock_event_instance.is_set.return_value = False

        # Mock Firestore client and document update
        mock_db = MagicMock()
        mock_firestore_client.return_value = mock_db
        mock_doc_ref = MagicMock()
        mock_db.collection.return_value.document.return_value = mock_doc_ref

        # Run main
        grinder_worker.main()

        # Check cpu_count called
        mock_cpu_count.assert_called_once()

        # Check Process creation (should be called 2 times since cpu_count=2)
        self.assertEqual(mock_process.call_count, 2)

        # Check start and join
        mock_process_instance = mock_process.return_value
        self.assertEqual(mock_process_instance.start.call_count, 2)
        self.assertEqual(mock_process_instance.join.call_count, 2)

        # Check if Firestore was updated
        mock_db.collection.assert_called_with("vanity_jobs")
        mock_db.collection.return_value.document.assert_called_with("test_job_123")
        mock_doc_ref.update.assert_called_once()
        args, _ = mock_doc_ref.update.call_args
        update_data = args[0]
        self.assertEqual(update_data['status'], 'COMPLETED')
        self.assertEqual(update_data['public_key'], dummy_pubkey)

        # Check stdout for secret key array
        output = mock_stdout.getvalue()
        secret_key = json.loads(output)
        self.assertEqual(secret_key, dummy_secret)

    def test_worker_task(self):
        # Test the worker logic independently
        mock_stop_event = MagicMock()
        mock_stop_event.is_set.side_effect = [False, False, False, True] # Run loop a few times or until match

        mock_queue = MagicMock()

        # We need a predictable Keypair or just check that it runs until match
        # Since Keypair is random, we can't easily force a match unless we mock Keypair
        # or use empty prefix/suffix

        prefix = ""
        suffix = ""

        # With empty prefix/suffix, it should match immediately
        mock_stop_event.is_set.return_value = False # First check

        grinder_worker.worker_task(prefix, suffix, mock_stop_event, mock_queue)

        # Should have set stop_event
        mock_stop_event.set.assert_called()
        # Should have put result in queue
        mock_queue.put.assert_called_once()

        # Inspect result
        args, _ = mock_queue.put.call_args
        result = args[0]
        pubkey, secret = result
        self.assertIsInstance(pubkey, str)
        self.assertIsInstance(secret, list)
        self.assertEqual(len(secret), 64)

if __name__ == '__main__':
    unittest.main()
